import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_squared_error, mean_absolute_error


# 1. Reproducibility

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# 2. Synthetic time-series generator

def generate_series(
    length=5000,
    period=24,
    amplitude=1.0,
    trend_slope=0.0,
    noise_std=0.0,
    phase=0.0,
    seed=42
):
    if length <= 0:
        raise ValueError("length must be greater than 0.")

    if period <= 0:
        raise ValueError("period must be greater than 0.")

    if noise_std < 0:
        raise ValueError("noise_std must be >= 0.")

    rng = np.random.default_rng(seed)

    t = np.arange(
        length,
        dtype=np.float32
    )

    trend = (
        trend_slope * t
    ).astype(np.float32)

    periodic = (
        amplitude
        * np.sin(
            2 * np.pi * t / period
            + phase
        )
    ).astype(np.float32)

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=length
    ).astype(np.float32)

    series = (
        trend
        + periodic
        + noise
    ).astype(np.float32)

    return (
        t,
        series,
        trend,
        periodic,
        noise
    )

# 3. Sliding window

def create_windows(
    series,
    lookback,
    horizon
):
    series = np.asarray(
        series,
        dtype=np.float32
    )

    if series.ndim != 1:
        raise ValueError(
            "series must be a 1D array."
        )

    if lookback <= 0:
        raise ValueError(
            "lookback must be greater than 0."
        )

    if horizon <= 0:
        raise ValueError(
            "horizon must be greater than 0."
        )

    n_samples = (
        len(series)
        - lookback
        - horizon
        + 1
    )

    if n_samples <= 0:
        raise ValueError(
            "Series is too short for the given "
            "lookback and horizon."
        )

    X = []
    y = []

    for start in range(n_samples):

        input_end = (
            start
            + lookback
        )

        target_end = (
            input_end
            + horizon
        )

        X.append(
            series[
                start:input_end
            ]
        )

        y.append(
            series[
                input_end:target_end
            ]
        )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.float32
    )

    return X, y

# 4. Linear

class Linear(nn.Module):

    def __init__(
        self,
        lookback,
        horizon
    ):
        super().__init__()

        self.lookback = lookback
        self.horizon = horizon

        self.linear = nn.Linear(
            lookback,
            horizon
        )

    def forward(self, x):

        if x.ndim != 2:
            raise ValueError(
                "Linear expects input shape "
                "[batch_size, lookback]."
            )

        output = self.linear(x)

        return output

# 5. Moving average

class MovingAverage(nn.Module):

    def __init__(
        self,
        kernel_size=5
    ):
        super().__init__()

        if kernel_size <= 0:
            raise ValueError(
                "kernel_size must be greater than 0."
            )

        if kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size should be odd "
                "so that the output length is preserved."
            )

        self.kernel_size = kernel_size

        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=0
        )

    def forward(self, x):

        if x.ndim != 2:
            raise ValueError(
                "MovingAverage expects input shape "
                "[batch_size, lookback]."
            )

        padding = (
            self.kernel_size - 1
        ) // 2

        front = x[:, :1].repeat(
            1,
            padding
        )

        end = x[:, -1:].repeat(
            1,
            padding
        )

        padded_x = torch.cat(
            [
                front,
                x,
                end
            ],
            dim=1
        )

        padded_x = padded_x.unsqueeze(1)

        trend = self.avg_pool(
            padded_x
        )

        trend = trend.squeeze(1)

        return trend

# 6. DLinear

class DLinear(nn.Module):

    def __init__(
        self,
        lookback,
        horizon,
        kernel_size=5
    ):
        super().__init__()

        self.lookback = lookback
        self.horizon = horizon

        self.moving_average = MovingAverage(
            kernel_size=kernel_size
        )

        self.linear_seasonal = nn.Linear(
            lookback,
            horizon
        )

        self.linear_trend = nn.Linear(
            lookback,
            horizon
        )

    def forward(self, x):

        if x.ndim != 2:
            raise ValueError(
                "DLinear expects input shape "
                "[batch_size, lookback]."
            )

        trend = self.moving_average(
            x
        )

        seasonal = (
            x - trend
        )

        seasonal_output = (
            self.linear_seasonal(
                seasonal
            )
        )

        trend_output = (
            self.linear_trend(
                trend
            )
        )

        output = (
            seasonal_output
            + trend_output
        )

        return output

# 7. NLinear

class NLinear(nn.Module):

    def __init__(
        self,
        lookback,
        horizon
    ):
        super().__init__()

        self.lookback = lookback
        self.horizon = horizon

        self.linear = nn.Linear(
            lookback,
            horizon
        )

    def forward(self, x):

        if x.ndim != 2:
            raise ValueError(
                "NLinear expects input shape "
                "[batch_size, lookback]."
            )

        last_value = (
            x[:, -1:]
            .detach()
        )

        normalized_x = (
            x - last_value
        )

        output = self.linear(
            normalized_x
        )

        output = (
            output
            + last_value
        )

        return output

# 8. Train one epoch

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion=None,
    device=None
):
    if criterion is None:
        criterion = nn.MSELoss()

    if device is None:
        device = next(
            model.parameters()
        ).device

    model.train()

    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(
            device,
            dtype=torch.float32
        )

        y_batch = y_batch.to(
            device,
            dtype=torch.float32
        )

        optimizer.zero_grad()

        y_pred = model(
            X_batch
        )

        loss = criterion(
            y_pred,
            y_batch
        )

        loss.backward()

        optimizer.step()

        batch_size = X_batch.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

    if total_samples == 0:
        raise ValueError(
            "train_loader contains no samples."
        )

    average_loss = (
        total_loss
        / total_samples
    )

    return average_loss

# 9. Validation

def validate(
    model,
    val_loader,
    criterion=None,
    device=None
):
    if criterion is None:
        criterion = nn.MSELoss()

    if device is None:
        device = next(
            model.parameters()
        ).device

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(
                device,
                dtype=torch.float32
            )

            y_batch = y_batch.to(
                device,
                dtype=torch.float32
            )

            y_pred = model(
                X_batch
            )

            loss = criterion(
                y_pred,
                y_batch
            )

            batch_size = X_batch.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += (
                batch_size
            )

    if total_samples == 0:
        raise ValueError(
            "val_loader contains no samples."
        )

    average_loss = (
        total_loss
        / total_samples
    )

    return average_loss

# 10. Full training loop

def train_model(
    model,
    train_loader,
    val_loader,
    epochs=50,
    learning_rate=1e-3,
    device=None,
    verbose=True
):
    if epochs <= 0:
        raise ValueError(
            "epochs must be greater than 0."
        )

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be greater than 0."
        )

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model = model.to(
        device
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )

        val_loss = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )

        train_losses.append(
            train_loss
        )

        val_losses.append(
            val_loss
        )

        if verbose:

            should_print = (
                epoch == 0
                or (epoch + 1) % 10 == 0
                or (epoch + 1) == epochs
            )

            if should_print:

                print(
                    f"Epoch {epoch + 1:3d}/{epochs} | "
                    f"Train MSE: {train_loss:.6f} | "
                    f"Val MSE: {val_loss:.6f}"
                )

    return (
        train_losses,
        val_losses
    )

# 11. Evaluation

def evaluate_model(
    model,
    test_loader,
    device=None
):
    if device is None:
        device = next(
            model.parameters()
        ).device

    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            X_batch = X_batch.to(
                device,
                dtype=torch.float32
            )

            y_batch = y_batch.to(
                device,
                dtype=torch.float32
            )

            y_pred = model(
                X_batch
            )

            all_predictions.append(
                y_pred.cpu().numpy()
            )

            all_targets.append(
                y_batch.cpu().numpy()
            )

    if len(all_predictions) == 0:
        raise ValueError(
            "test_loader contains no samples."
        )

    predictions = np.concatenate(
        all_predictions,
        axis=0
    )

    targets = np.concatenate(
        all_targets,
        axis=0
    )

    mse = mean_squared_error(
        targets.reshape(-1),
        predictions.reshape(-1)
    )

    mae = mean_absolute_error(
        targets.reshape(-1),
        predictions.reshape(-1)
    )

    return (
        float(mse),
        float(mae),
        predictions,
        targets
    )

# 12. run_experiment

def run_experiment(
    model_name,
    period,
    lookback,
    horizon,
    trend_slope=0.0,
    noise_std=0.0,
    series_length=5000,
    kernel_size=5,
    epochs=50,
    learning_rate=1e-3,
    seed=42
):
    set_seed(seed)

    _, series, _, _, _ = generate_series(
        length=series_length,
        period=period,
        trend_slope=trend_slope,
        noise_std=noise_std
    )

    train_loader, val_loader, test_loader = prepare_data(
        series,
        lookback=lookback,
        horizon=horizon
    )

    model_classes = {
        "Linear": Linear,
        "DLinear": DLinear,
        "NLinear": NLinear
    }

    if model_name not in model_classes:
        raise ValueError(f"Unknown model: {model_name}")

    if model_name == "DLinear":
        model = DLinear(
            lookback=lookback,
            horizon=horizon,
            kernel_size=kernel_size
        )
    else:
        model = model_classes[model_name](
            lookback=lookback,
            horizon=horizon
        )

    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        learning_rate=learning_rate
    )

    test_mse = evaluate_model(
        model,
        test_loader
    )

    return {
        "Model": model_name,
        "Period": period,
        "Lookback": lookback,
        "Horizon": horizon,
        "L/P": lookback / period,
        "T/P": horizon / period,
        "Trend": trend_slope,
        "Noise": noise_std,
        "Seed": seed,
        "MSE": test_mse
    }


# 13. Data preparation
def prepare_data(
    series,
    lookback,
    horizon,
    batch_size=32,
    train_ratio=0.7,
    val_ratio=0.15
):
    series = np.asarray(series, dtype=np.float32).reshape(-1)

    n = len(series)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    if lookback < 1 or horizon < 1:
        raise ValueError("lookback and horizon must be positive")

    if not (0 < train_ratio < 1 and
            0 < val_ratio < 1 and
            train_ratio + val_ratio < 1):
        raise ValueError("Invalid train/validation ratios")

    def make_loader(start, end, shuffle=False):
        X, y = [], []

        for target_start in range(start, end - horizon + 1):
            input_start = target_start - lookback

            if input_start < 0:
                continue

            X.append(series[input_start:target_start])
            y.append(series[target_start:target_start + horizon])

        if not X:
            raise ValueError(
                "Not enough data for the selected lookback and horizon"
            )

        X = torch.tensor(np.stack(X), dtype=torch.float32)
        y = torch.tensor(np.stack(y), dtype=torch.float32)

        dataset = TensorDataset(X, y)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

    train_loader = make_loader(
        lookback, train_end, shuffle=True
    )

    val_loader = make_loader(
        train_end, val_end
    )

    test_loader = make_loader(
        val_end, n
    )

    return train_loader, val_loader, test_loader
