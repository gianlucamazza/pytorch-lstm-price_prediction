import os
import torch
from src.logger import setup_logger
import matplotlib.pyplot as plt

logger = setup_logger("train_logger", "logs/train.log")

def save_model_checkpoint(symbol, model, checkpoint_dir, epoch):
    """Save a checkpoint of the given model."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{symbol}_checkpoint_{epoch}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Model checkpoint saved to {checkpoint_path}")

def evaluate_model(model, data_loader, loss_fn, _device):
    """Evaluate the model on the given data loader."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(_device), y_batch.to(_device)
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def plot_evaluation(symbol, y_pred, y_true, dates):
    """Plot the evaluation results."""
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true, label="True")
    plt.plot(dates, y_pred, label="Predicted")
    plt.title(f"{symbol} - Model Evaluation")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"plots/{symbol}_evaluation.png")