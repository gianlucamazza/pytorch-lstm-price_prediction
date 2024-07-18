from typing import List, Tuple
import numpy as np
import pandas as pd
from src.logger import setup_logger

logger = setup_logger("data_loader_logger", "logs/data_loader.log")


def log_preprocessing_params(targets: List[str], look_back: int, look_forward: int, features: List[str]) -> None:
    """
    Log preprocessing parameters.
    """
    logger.info(f"Targets: {targets}")
    logger.info(f"Look back: {look_back}, Look forward: {look_forward}")
    logger.info(f"Selected features: {features}")


def split_data_into_targets_and_features(historical_data: pd.DataFrame, targets: List[str],
                                         features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split historical data into targets and features.
    """
    target_data = historical_data[targets]
    feature_data = historical_data[features]
    return target_data, feature_data


def save_scaled_data(symbol: str, interval: str, scaled_features: np.ndarray, scaled_targets: np.ndarray,
                     features: List[str], targets: List[str]) -> None:
    """
    Save scaled data to CSV.
    """
    scaled_data = np.concatenate((scaled_features, scaled_targets), axis=1)
    scaled_df = pd.DataFrame(scaled_data, columns=features + targets)
    scaled_df.to_csv(f"data/{symbol}_{interval}_scaled_data.csv", index=False)
    logger.info(f"Scaled dataset saved to data/{symbol}_scaled_data.csv")


def validate_data(_x: np.ndarray, _y: np.ndarray) -> None:
    """
    Validate dataset to ensure there are no NaN or infinite values.
    """
    if np.any(np.isnan(_x)) or np.any(np.isnan(_y)):
        logger.error("NaN values found in input data.")
        raise ValueError("NaN values found in input data.")
    if np.any(np.isinf(_x)) or np.any(np.isinf(_y)):
        logger.error("Infinite values found in input data.")
        raise ValueError("Infinite values found in input data.")
