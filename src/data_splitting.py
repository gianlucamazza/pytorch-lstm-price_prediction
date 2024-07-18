from typing import List, Tuple
import numpy as np
import torch
from numpy import ndarray
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from torch.utils.data import DataLoader, TensorDataset
from src.logger import setup_logger

logger = setup_logger("data_loader_logger", "logs/data_loader.log")

def create_timeseries_window(_x: np.ndarray, _y: np.ndarray, n_splits: int) -> List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Create k-fold cross-validation windows using TimeSeriesSplit.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for train_index, val_index in tscv.split(_x):
        x_train, x_val = _x[train_index], _x[val_index]
        y_train, y_val = _y[train_index], _y[val_index]
        train_data = (torch.tensor(x_train).float(), torch.tensor(y_train).float())
        val_data = (torch.tensor(x_val).float(), torch.tensor(y_val).float())
        splits.append((train_data, val_data))
    return splits


def split_data(_x: np.ndarray, _y: np.ndarray, batch_size: int, test_size: float = 0.15) -> Tuple[DataLoader, DataLoader]:
    """
    Split data into training and validation sets.
    """
    logger.info("Splitting data into training and validation sets")
    x_train, x_val, y_train, y_val = train_test_split(_x, _y, test_size=test_size, random_state=42)
    logger.info(f"Training data shape: {x_train.shape}, Validation data shape: {x_val.shape}")

    train_data = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_data = TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    return DataLoader(train_data, batch_size=batch_size, shuffle=True), DataLoader(val_data, batch_size=batch_size, shuffle=False)


def create_dataloader(data: Tuple[torch.Tensor, torch.Tensor], batch_size: int = 32) -> DataLoader:
    """
    Create a DataLoader from given data.
    """
    dataset = TensorDataset(*data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
