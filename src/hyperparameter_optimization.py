import os
import sys
import argparse
import optuna
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from optuna.trial import TrialState

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import PricePredictor
from src.early_stopping import EarlyStopping
from src.config import load_config, update_config
from src.logger import setup_logger
from src.model_utils import run_training_epoch, run_validation_epoch
from train import initialize_model, load_or_get_historical_data, preprocess_and_create_dataloaders, train_model, evaluate_model

train_logger = setup_logger("train_logger", "logs/train.log")
optuna_logger = setup_logger("optuna_logger", "logs/optuna.log")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def objective(optuna_trial, config):
    hidden_size = optuna_trial.suggest_int("hidden_size", 32, 256)
    num_layers = optuna_trial.suggest_int("num_layers", 1, 5)
    dropout = optuna_trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = optuna_trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = optuna_trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    config.model_settings.update({
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay
    })

    optuna_logger.info(f"Starting Trial {optuna_trial.number}")
    optuna_logger.info(f"Trial {optuna_trial.number}: hidden_size={hidden_size}, num_layers={num_layers}, "
                       f"dropout={dropout}, learning_rate={learning_rate}, weight_decay={weight_decay}")

    try:
        # Load and preprocess the data
        historical_data = load_or_get_historical_data(config)
        dataloaders, _ = preprocess_and_create_dataloaders(config, historical_data)

        fold_val_losses = []
        for fold_idx, (train_loader, val_loader) in enumerate(dataloaders):
            model = PricePredictor(
                input_size=len(config.data_settings["selected_features"]),
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                fc_output_size=len(config.data_settings["targets"])
            ).to(device)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            early_stopping = EarlyStopping(patience=10, delta=0.001, verbose=True, path=f"models/optuna/model_{optuna_trial.number}_fold_{fold_idx}.pt")

            model.train()
            val_loss = float('inf')
            for epoch in range(config.training_settings["epochs"]):
                train_loss = run_training_epoch(model, train_loader, criterion, optimizer, device)
                optuna_logger.info(f"Trial {optuna_trial.number}, Fold {fold_idx}, Epoch {epoch + 1}/{config.training_settings['epochs']}, "
                                   f"Train Loss: {train_loss:.4f}")
                val_loss = run_validation_epoch(model, val_loader, criterion, device)
                optuna_logger.info(f"Trial {optuna_trial.number}, Fold {fold_idx}, Epoch {epoch + 1}/{config.training_settings['epochs']}, "
                                   f"Validation Loss: {val_loss:.4f}")

                if early_stopping(val_loss, model):
                    optuna_logger.info(f"Early stopping triggered for trial {optuna_trial.number} at epoch {epoch + 1}")
                    break

            optuna_logger.info(f"Trial {optuna_trial.number} Fold {fold_idx} completed with Validation Loss: {val_loss:.4f}")
            fold_val_losses.append(val_loss)

        avg_val_loss = np.mean(fold_val_losses)
        optuna_logger.info(f"Trial {optuna_trial.number} completed with Average Validation Loss: {avg_val_loss:.4f}")

    except Exception as e:
        optuna_logger.error(f"Error during trial {optuna_trial.number}: {e}")
        raise e

    return avg_val_loss


def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", type=str, required=True, help="Path to configuration JSON file")
    return arg_parser.parse_args()


def rebuild_features(config, config_path="config.json", n_trials=100):
    """Rebuild features using Optuna for feature selection."""
    def feature_selection_objective(optuna_trial):
        selected_features = []
        all_features = config.data_settings["all_features"]

        # Create a boolean parameter for each feature to decide if it's included or not
        for feature in all_features:
            if optuna_trial.suggest_categorical(f"use_{feature}", [False, True]):
                selected_features.append(feature)

        if len(selected_features) == 0:
            return float('inf')  # Penalize trials with no features selected

        # Load and preprocess data with the selected features
        historical_data = load_or_get_historical_data(config)
        config.data_settings["selected_features"] = selected_features
        dataloaders, _ = preprocess_and_create_dataloaders(config, historical_data)

        fold_val_losses = []
        for fold_idx, (train_loader, val_loader) in enumerate(dataloaders):
            model = PricePredictor(
                input_size=len(selected_features),  # Update input_size based on selected features
                hidden_size=config.model_settings['hidden_size'],
                num_layers=config.model_settings['num_layers'],
                dropout=config.model_settings['dropout'],
                fc_output_size=len(config.data_settings['targets'])
            ).to(device)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=config.model_settings['learning_rate'], weight_decay=config.model_settings['weight_decay'])
            early_stopping = EarlyStopping(patience=10, delta=0.001, verbose=True, path=f"models/optuna/feature_selection_{optuna_trial.number}_fold_{fold_idx}.pt")

            model.train()
            val_loss = float('inf')
            for _ in range(config.training_settings["epochs"]):
                _ = run_training_epoch(model, train_loader, criterion, optimizer, device)
                val_loss = run_validation_epoch(model, val_loader, criterion, device)

                if early_stopping(val_loss, model):
                    break

            fold_val_losses.append(val_loss)

        avg_val_loss = np.mean(fold_val_losses)
        return avg_val_loss

    # Create an Optuna study to optimize feature selection
    # The objective function will select the best features based on the validation loss
    study = optuna.create_study(direction="minimize")
    study.optimize(feature_selection_objective, n_trials)

    # Get the best trial and selected features
    best_trial = study.best_trial
    selected_features = [feature for feature in config.data_settings["all_features"] if best_trial.params[f"use_{feature}"]]

    # Update the configuration with the best features
    config.data_settings["selected_features"] = selected_features
    config.save(config_path)

    train_logger.info(f"Rebuilt features using Optuna. Best features: {selected_features}")


def main():
    args = parse_arguments()

    config = load_config(args.config)
    optuna_logger.info(f"Loaded configuration from {args.config}")

    if args.rebuild_features:
        historical_data = load_or_get_historical_data(config)
        rebuild_features(config, args.config, n_trials=100)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, config), n_trials=100)

    best_params = study.best_trial.params
    optuna_logger.info(f"Best hyperparameters: {best_params}")

    config.model_settings.update(best_params)
    update_config(config, "model_settings", config.model_settings)

    historical_data = load_or_get_historical_data(config)
    dataloaders, scaler_prices = preprocess_and_create_dataloaders(config, historical_data)

    for fold_idx, (train_loader, val_loader) in enumerate(dataloaders, 1):
        model = initialize_model(config)
        train_model(
            config.data_settings["symbol"],
            model,
            train_loader,
            val_loader,
            num_epochs=config.training_settings["epochs"],
            learning_rate=config.model_settings.get("learning_rate", 0.001),
            model_dir=config.training_settings["model_dir"],
            weight_decay=config.model_settings.get("weight_decay", 0.0),
            _device=device,
            fold_idx=fold_idx
        )

        model.load_state_dict(torch.load(f"{config.training_settings['model_dir']}/{config.data_settings['symbol']}_best_model.pth"))
        val_loss = evaluate_model(model, val_loader, torch.nn.MSELoss(), device)
        optuna_logger.info(f"Validation loss for fold {fold_idx}: {val_loss:.4f}")

    trials_df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    trials_df.to_csv("reports/optuna_trials.csv", index=False)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    optuna_logger.info("Study statistics: ")
    optuna_logger.info(f"  Number of finished trials: {len(study.trials)}")
    optuna_logger.info(f"  Number of pruned trials: {len(pruned_trials)}")
    optuna_logger.info(f"  Number of complete trials: {len(complete_trials)}")

    optuna_logger.info("Best trial:")
    trial = study.best_trial

    optuna_logger.info(f"  Value: {trial.value}")
    optuna_logger.info(f"  Params: ")
    for key, value in trial.params.items():
        optuna_logger.info(f"    {key}: {value}")


if __name__ == "__main__":
    main()
