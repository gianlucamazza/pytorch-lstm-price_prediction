from typing import List, Tuple, Any
import numpy as np
import pandas as pd
from numpy import ndarray, dtype
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.logger import setup_logger
from src.data_utils import log_preprocessing_params, split_data_into_targets_and_features, save_scaled_data, validate_data

logger = setup_logger("data_loader_logger", "logs/data_loader.log")

def preprocess_data(symbol: str, data_sampling_interval: str, historical_data: pd.DataFrame, targets: List[str],
                    look_back: int = 60, look_forward: int = 30, features: List[str] = None, 
                    disabled_features: List[str] = None, selected_features: List[str] = None) -> Tuple[
                        ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], StandardScaler, StandardScaler, 
                        MinMaxScaler, List[str]]:
    """
    Preprocess the historical stock data for training the model.
    """
    logger.info("Starting preprocessing of data")
    log_preprocessing_params(targets, look_back, look_forward, features)

    features = [f for f in features if f not in targets]
    if disabled_features:
        features = [f for f in features if f not in disabled_features]

    target_data, feature_data = split_data_into_targets_and_features(historical_data, targets, features)
    scaled_targets, scaler_prices, scaler_volume = scale_targets(target_data)
    scaled_features, scaler_features = scale_features(feature_data)

    save_scaled_data(symbol, data_sampling_interval, scaled_features, scaled_targets, features, targets)

    _X, _y = create_dataset(scaled_features, scaled_targets, look_back, look_forward, targets)
    validate_data(_X, _y)

    if selected_features:
        _X, _y, scaler_features, scaler_prices, scaler_volume, features = get_selected_features(
            _X, _y, selected_features, features, scaled_features, scaler_features, scaler_prices, scaler_volume
        )

    return _X, _y, scaler_features, scaler_prices, scaler_volume, features


def scale_targets(target_data: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler, MinMaxScaler]:
    """
    Scale target data.
    """
    scaler_prices = StandardScaler()
    scaled_prices = scaler_prices.fit_transform(target_data.drop(columns=["Volume"]))

    scaler_volume = MinMaxScaler()
    scaled_volume = scaler_volume.fit_transform(target_data[["Volume"]])

    scaled_targets = np.concatenate((scaled_prices, scaled_volume), axis=1)
    return scaled_targets, scaler_prices, scaler_volume


def scale_features(feature_data: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """
    Scale feature data.
    """
    scaler_features = StandardScaler()
    scaled_features = scaler_features.fit_transform(feature_data)
    return scaled_features, scaler_features


def create_dataset(scaled_features: np.ndarray, scaled_targets: np.ndarray, look_back: int, 
                   look_forward: int, targets: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create dataset from scaled features and targets.
    """
    _X, _y = [], []
    for i in range(look_back, len(scaled_features) - look_forward):
        _X.append(scaled_features[i - look_back: i])
        _y.append(scaled_targets[i + look_forward - 1, : len(targets)])

    _X = np.array(_X)
    _y = np.array(_y).reshape(-1, len(targets))
    logger.info(f"Shape of _X: {_X.shape}")
    logger.info(f"Shape of _y: {_y.shape}")
    return _X, _y


def get_selected_features(_x: np.ndarray, _y: np.ndarray, selected_features: List[str], 
                             features: List[str], scaled_features: np.ndarray, scaler_features: StandardScaler, 
                             scaler_prices: StandardScaler, scaler_volume: MinMaxScaler) -> Tuple[
                                np.ndarray, np.ndarray, StandardScaler, StandardScaler, MinMaxScaler, List[str]]:
    """
    Get selected features from the dataset
    """
    logger.info(f"Using predefined best features: {selected_features}")
    feature_indices = [features.index(feature) for feature in selected_features]
    validate_feature_indices(feature_indices, scaled_features.shape[1])
    _X_selected = _x[:, :, feature_indices]
    logger.info(f"Shape of _X_selected: {_X_selected.shape}")
    return _X_selected, _y, scaler_features, scaler_prices, scaler_volume, selected_features


def validate_feature_indices(feature_indices: List[int], max_index: int) -> None:
    """
    Validate feature indices to ensure they are within bounds.
    """
    if any(idx >= max_index for idx in feature_indices):
        logger.error("One or more feature indices are out of bounds.")
        raise ValueError("One or more feature indices are out of bounds.")
