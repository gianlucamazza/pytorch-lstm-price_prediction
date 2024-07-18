from typing import List, Tuple, Dict
import os
import time
import pandas as pd
import yfinance as yf
from src.feature_engineering import calculate_technical_indicators
from src.logger import setup_logger

logger = setup_logger("data_loader_logger", "logs/data_loader.log")


def load_or_get_historical_data(config):
    """Load historical data from file or download it if not available."""
    if os.path.exists(config.data_settings["historical_data_path"]):
        historical_data = pd.read_csv(config.data_settings["historical_data_path"])
        logger.info("Historical data loaded from file.")
    else:
        historical_data, _ = get_data(
            config.data_settings["ticker"],
            config.data_settings["symbol"],
            config.data_settings["asset_type"],
            config.data_settings["start_date"],
            time.strftime("%Y-%m-%d"),
            config.data_settings["technical_indicators"],
            config.data_settings["data_sampling_interval"],
            config.data_settings["data_resampling_frequency"]
        )
        save_historical_data(config.data_settings["symbol"], config.data_settings["data_sampling_interval"], historical_data)
        logger.info("Historical data downloaded and saved.")
    return historical_data


def get_data(_ticker: str, symbol: str, asset_type: str, start: str, end: str, 
             windows: Dict[str, int], data_sampling_interval: str, 
             data_resampling_frequency: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Download historical stock data and calculate technical indicators.
    """
    logger.info(f"Downloading data for {_ticker} from {start} to {end}")
    historical_data = yf.download(_ticker, start=start, end=end, interval=data_sampling_interval)
    historical_data, features = calculate_technical_indicators(historical_data, windows=windows, 
                                                               asset_type=asset_type, frequency=data_resampling_frequency)
    save_historical_data(symbol, data_sampling_interval, historical_data)
    return historical_data, features


def save_historical_data(symbol: str, interval: str, historical_data: pd.DataFrame) -> None:
    """
    Save historical data to CSV.
    """
    historical_data.to_csv(f"data/{symbol}_{interval}.csv")
    logger.info(f"Data for {symbol} saved to data/{symbol}.csv")