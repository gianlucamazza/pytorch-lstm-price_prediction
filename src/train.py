import argparse
import os
import sys
import time
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_loading import load_or_get_historical_data
from data_splitting import create_timeseries_window, split_data, create_dataloader
from preprocessing import preprocess_data
from model import PricePredictor, init_weights
from early_stopping import EarlyStopping
from logger import setup_logger
from config import load_config, update_config
from model_utils import run_training_epoch, run_validation_epoch
from train_utils import evaluate_model, plot_evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = setup_logger("train_logger", "logs/train.log")


def initialize_model(config):
    """Initialize the model with the given configuration."""
    hidden_size = config.model_settings.get("hidden_size", 64)
    num_layers = config.model_settings.get("num_layers", 2)
    dropout = config.model_settings.get("dropout", 0.2)
    input_size = len(config.data_settings["selected_features"])
    fc_output_size = len(config.data_settings["targets"])

    model = PricePredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        fc_output_size=fc_output_size,
    ).to(device)

    model.apply(init_weights)

    return model


def parse_arguments():
    """Parse command-line arguments."""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", type=str, required=True, help="Path to configuration JSON file")
    arg_parser.add_argument("--rebuild-features", action="store_true", help="Rebuild features")
    return arg_parser.parse_args()


def rebuild_features_if_needed(config, historical_data, args):
    """Rebuild features if needed and update the configuration."""
    if args.rebuild_features or not config.data_settings["selected_features"]:
        logger.info("Rebuilding features...")
        _, _, _, _, _, selected_features = preprocess_data(
            config.data_settings["symbol"],
            config.data_settings["data_sampling_interval"],
            historical_data,
            config.data_settings["targets"],
            look_back=config.training_settings["look_back"],
            look_forward=config.training_settings["look_forward"],
            features=config.data_settings["all_features"],
            disabled_features=config.data_settings.get("disabled_features", []),
            selected_features=None
        )
        update_config(config, "data_settings.selected_features", selected_features)
        config.save(args.config)
        logger.info("Selected features saved to configuration.")


def preprocess_and_create_dataloaders(config, historical_data):
    """Preprocess data and create DataLoaders for training and validation."""
    X, y, _, scaler_prices, _, _ = preprocess_data(
        config.data_settings["symbol"],
        config.data_settings["data_sampling_interval"],
        historical_data,
        config.data_settings["targets"],
        look_back=config.training_settings["look_back"],
        look_forward=config.training_settings["look_forward"],
        features=config.data_settings["all_features"],
        disabled_features=config.data_settings.get("disabled_features", []),
        selected_features=config.data_settings["selected_features"]
    )

    if config.training_settings.get("use_time_series_split", False):
        logger.info("Using k-fold cross-validation with TimeSeriesSplit")
        splits = create_timeseries_window(X, y, config.training_settings["time_series_splits"])
    else:
        splits = [(split_data(X, y, config.training_settings["batch_size"]))]

    dataloaders = [(create_dataloader(train_data, config.training_settings["batch_size"]),
                    create_dataloader(val_data, config.training_settings["batch_size"]))
                   for train_data, val_data in splits]
    return dataloaders, scaler_prices


def train_model(symbol, model, train_loader, val_loader, epochs, learning_rate, model_dir, weight_decay, device, fold_idx):
    """Train the model."""
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=10, verbose=True, path=f"{model_dir}/{symbol}_best_model.pth")

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = run_training_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_validation_epoch(model, val_loader, criterion, device)
        end_time = time.time()

        logger.info(f"Fold {fold_idx}, Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {end_time - start_time:.2f}s")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break

    logger.info(f"Training completed for fold {fold_idx}")
    logger.info(f"Best model saved to {model_dir}/{symbol}_best_model.pth")
    

def main():
    """Main function to run the training and evaluation."""
    args = parse_arguments()
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")

    # Load or get historical data
    historical_data = load_or_get_historical_data(config)

    # Rebuild features if needed
    rebuild_features_if_needed(config, historical_data, args)

    # Preprocess data and create DataLoader(s)
    dataloaders, scaler_prices = preprocess_and_create_dataloaders(config, historical_data)
    model_dir = config.training_settings["model_dir"]
    os.makedirs(model_dir, exist_ok=True)

    for fold_idx, (train_loader, val_loader) in enumerate(dataloaders, 1):
        model = initialize_model(config)
        train_model(
            config.data_settings["symbol"],
            model,
            train_loader,
            val_loader,
            config.training_settings["epochs"],
            config.model_settings["learning_rate"],
            model_dir,
            config.model_settings["weight_decay"],
            device,
            fold_idx,
        )

        # Load the best model and evaluate
        model.load_state_dict(torch.load(f"{model_dir}/{config.data_settings['symbol']}_best_model.pth"))
        val_loss = evaluate_model(model, val_loader, torch.nn.MSELoss(), device)
        logger.info(f"Validation loss for fold {fold_idx}: {val_loss:.4f}")

    logger.info("Model training completed.")

    # Model evaluation on test data (if available)
    logger.info("Evaluating model on test data...")
    X_test, y_test, _, _, _, _ = preprocess_data(
        config.data_settings["symbol"],
        config.data_settings["data_sampling_interval"],
        historical_data,
        config.data_settings["targets"],
        look_back=config.training_settings["look_back"],
        look_forward=config.training_settings["look_forward"],
        features=config.data_settings["all_features"],
        disabled_features=config.data_settings.get("disabled_features", []),
        selected_features=config.data_settings["selected_features"],
        test=True
    )

    test_loader = create_dataloader((X_test, y_test), config.training_settings["batch_size"])
    test_loss = evaluate_model(model, test_loader, torch.nn.MSELoss(), device)
    logger.info(f"Test loss: {test_loss:.4f}")

    logger.info("Model evaluation completed.")
    logger.info("Saving model evaluation plot...")

    # Plot evaluation results
    model.eval()
    for batch in test_loader:
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_pred = model(x_batch)
        break

    y_pred = scaler_prices.inverse_transform(y_pred.detach().cpu().numpy())
    y_true = scaler_prices.inverse_transform(y_batch.detach().cpu().numpy())
    dates = historical_data["Date"].values

    plot_evaluation(config.data_settings["symbol"], y_pred, y_true, dates)

    logger.info("Model evaluation plot saved.")
    logger.info("Training and evaluation completed.")
    logger.info("Exiting...")
    logger.handlers.clear()


if __name__ == "__main__":
    main()
