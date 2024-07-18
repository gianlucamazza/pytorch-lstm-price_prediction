import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loading import load_or_get_historical_data
from src.model import load_model
from src.logger import setup_logger
from src.config import load_config

# Set up logger
logger = setup_logger("predict_logger", "logs/predict.log")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")


def predict(
    _model: nn.Module,
    _x: np.ndarray,
    scaler_prices: StandardScaler,
    scaler_volume: MinMaxScaler,
    future_days: int,
    _features: List[str],
    _targets: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    _model.eval()
    num_targets = len(_targets)

    logger.info(
        f"Starting prediction for {num_targets} targets over {future_days} future days."
    )

    with torch.no_grad():
        logger.info("Making predictions on the input data.")
        _X_tensor = torch.tensor(_x, dtype=torch.float32).to(device)
        predictions = _model(_X_tensor).cpu().numpy()

        logger.info(f"Initial predictions shape: {predictions.shape}")

        predictions = inverse_transform_predictions(
            predictions, scaler_prices, scaler_volume, _features, num_targets
        )
        logger.info(f"Inverse transformed predictions: {predictions.shape}")

        # Forecast future prices
        last_x = _x[-1]
        future_predictions = []

        for day in range(future_days):
            _X_tensor = torch.tensor(last_x[np.newaxis, :], dtype=torch.float32).to(device)
            future_pred = _model(_X_tensor).cpu().numpy()[0]
            future_predictions.append(future_pred)

            new_row = np.zeros((1, last_x.shape[1]))
            new_row[0, :num_targets] = future_pred

            if new_row.shape[1] > num_targets:
                new_row[0, num_targets:] = last_x[-1, num_targets:]

            last_x = np.vstack((last_x[1:], new_row))

        future_predictions = np.array(future_predictions)
        logger.debug(f"Future predictions array shape: {future_predictions.shape}")

        future_predictions_reshaped = np.zeros(
            (future_days, len(_features) + num_targets)
        )
        future_predictions_reshaped[:, :num_targets] = future_predictions
        logger.debug(
            f"Future predictions reshaped: {future_predictions_reshaped.shape}"
        )

        padding_width = len(scaler_prices.scale_) - future_predictions_reshaped.shape[1]
        if padding_width > 0:
            future_predictions_reshaped = np.pad(
                future_predictions_reshaped, ((0, 0), (0, padding_width)), "constant"
            )

        future_predictions = scaler_prices.inverse_transform(
            future_predictions_reshaped[:, : num_targets - 1]
        )
        future_predictions = np.hstack(
            (
                future_predictions,
                scaler_volume.inverse_transform(
                    future_predictions_reshaped[:, num_targets - 1 : num_targets]
                ),
            )
        )
        logger.info(
            f"Inverse transformed future predictions: {future_predictions.shape}"
        )

    logger.info("Prediction completed.")
    return predictions, future_predictions


def make_initial_predictions(model: nn.Module, x: np.ndarray) -> np.ndarray:
    logger.info("Making predictions on the input data.")
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    predictions = model(x_tensor).cpu().numpy()
    logger.info(f"Initial predictions shape: {predictions.shape}")
    return predictions


def inverse_transform_predictions(
    predictions, scaler_prices, scaler_volume, features, num_targets
):
    predictions_reshaped = np.zeros((predictions.shape[0], len(features) + num_targets))
    predictions_reshaped[:, :num_targets] = predictions

    predictions_reshaped[:, : num_targets - 1] = scaler_prices.inverse_transform(
        predictions[:, : num_targets - 1]
    )
    predictions_reshaped[:, num_targets - 1 : num_targets] = (
        scaler_volume.inverse_transform(predictions[:, num_targets - 1 : num_targets])
    )

    return predictions_reshaped[:, :num_targets]


def update_input_data(
    x: np.ndarray, future_pred: np.ndarray, num_targets: int
) -> np.ndarray:
    new_row = np.zeros((1, x.shape[2]))
    new_row[0, :num_targets] = future_pred
    new_row[0, num_targets:] = x[-1, -1, num_targets:]
    x = np.append(x, [np.vstack((x[-1][1:], new_row))], axis=0)
    logger.debug(f"Updated input data shape: {x.shape}")
    return x


def create_candles(
    predictions: np.ndarray, freq: str, start_date: pd.Timestamp
) -> pd.DataFrame:
    columns = ["Open", "High", "Low", "Close"]
    if predictions.shape[1] == 5:
        columns.append("Volume")
    df = pd.DataFrame(predictions, columns=columns)
    df.index = pd.date_range(start=start_date, periods=len(df), freq=freq)
    return df


def plot_predictions(
    symbol: str,
    filename: str,
    candles: pd.DataFrame,
    predictions: np.ndarray,
    future_predictions: np.ndarray,
    freq: str,
    interval: str,
) -> None:
    logger.info("Plotting predictions")

    start_date = candles.index[-1]

    fig = go.Figure()
    add_candlestick_trace(fig, candles, "Actual Candlestick")

    historical_candles = create_candles(
        predictions, freq, candles.index[-len(predictions)]
    )
    add_candlestick_trace(fig, historical_candles, "Predicted Candlestick")

    future_candles = create_candles(future_predictions, freq, start_date)
    add_candlestick_trace(
        fig,
        future_candles,
        "Future Predicted Candlestick",
        increasing_color="blue",
        decreasing_color="orange",
    )

    update_layout(fig, symbol, interval)

    fig.write_html(filename)
    logger.info(f"Predictions plot saved to {filename}")


def add_candlestick_trace(
    fig: go.Figure,
    candles: pd.DataFrame,
    name: str,
    increasing_color="green",
    decreasing_color="red",
) -> None:
    logger.debug(f"Adding candlestick trace for {name}")
    fig.add_trace(
        go.Candlestick(
            x=candles.index,
            open=candles["Open"],
            high=candles["High"],
            low=candles["Low"],
            close=candles["Close"],
            name=name,
            increasing=dict(line=dict(color=increasing_color)),
            decreasing=dict(line=dict(color=decreasing_color)),
        )
    )


def update_layout(fig: go.Figure, symbol: str, interval: str) -> None:
    fig.update_layout(
        title=f"{symbol} - Predictions",
        xaxis_title="Date" if "d" in interval else "Date/Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        xaxis=dict(tickformat="%Y-%m-%d %H:%M" if "h" in interval else "%Y-%m-%d"),
    )


def main(config_path: str) -> None:
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")
    logger.info(f"Starting prediction for {config.ticker}")

    historical_data = load_or_get_historical_data(config)

    # load csv file from config.data_settings["historical_data_path"]
    historical_data = pd.read_csv(config.data_settings["historical_data_path"])
    logger.info(f"Loaded historical data from {config.data_settings['historical_data_path']}")
    logger.info(f"Data shape: {historical_data.shape}")

    features = config.data_settings["selected_features"]
    feature_data = historical_data[features]

    # scale target data
    scaler_prices = StandardScaler()
    scaler_volume = MinMaxScaler()

    x = feature_data.values
    x = scaler_prices.fit_transform(x)

    model = load_model(
        config.data_settings["symbol"], config.training_settings["model_dir"], len(features), config.model_settings
    )
    predictions, future_predictions = predict(
        _model=model,
        _x=x,
        scaler_prices=scaler_prices,
        scaler_volume=scaler_volume,
        future_days=config.look_forward,
        _features=features,
        _targets=config.targets,
    )
    candles = historical_data[["Open", "High", "Low", "Close"]]
    plot_predictions(
        config.symbol,
        f"docs/{config.symbol}_{features}_predictions.html",
        candles,
        predictions,
        future_predictions,
        config.data_resampling_frequency,
        config.data_sampling_interval,
    )

    save_predictions_report(
        future_predictions,
        config.targets,
        candles.index[-1],
        config.data_resampling_frequency,
        config.symbol,
    )
    logger.info(f"Prediction for {config.symbol} completed")


def save_predictions_report(
    predictions: np.ndarray,
    targets: List[str],
    start_date: pd.Timestamp,
    freq: str,
    symbol: str,
) -> None:
    if predictions.shape[1] < len(targets):
        padding_width = len(targets) - predictions.shape[1]
        predictions = np.pad(predictions, ((0, 0), (0, padding_width)), "constant")

    report = pd.DataFrame(data=predictions, columns=targets)
    report.index = pd.date_range(start=start_date, periods=len(predictions), freq=freq)
    report.to_csv(f"reports/{symbol}_predictions.csv", index=False)
    logger.info(f"Predictions report saved to reports/{symbol}_predictions.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration JSON file"
    )
    args = parser.parse_args()
    main(args.config)
