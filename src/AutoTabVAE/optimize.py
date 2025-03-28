import optuna
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np

from AutoTabVAE.model import TabNetVAE
from AutoTabVAE.dataset import TabularDataset
from AutoTabVAE.train import train_model

def suggest(trial, name, method, default, **kwargs):
    if method == "int":
        return trial.suggest_int(name, kwargs.get("low"), kwargs.get("high"), step=kwargs.get("step", 1))
    elif method == "float":
        return trial.suggest_float(name, kwargs.get("low"), kwargs.get("high"), step=kwargs.get("step"), log=kwargs.get("log", False))
    elif method == "categorical":
        return trial.suggest_categorical(name, kwargs.get("choices"))
    else:
        return default

def objective(trial, param_config=None, train_settings=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = param_config or {}
    train_cfg = train_settings or {}

    # Suggestions
    n_d = suggest(trial, "n_d", "int", 16, **cfg.get("n_d", {"low": 8, "high": 64, "step": 8}))
    n_a = suggest(trial, "n_a", "int", 16, **cfg.get("n_a", {"low": 8, "high": 64, "step": 8}))
    n_steps = suggest(trial, "n_steps", "int", 3, **cfg.get("n_steps", {"low": 3, "high": 10}))
    gamma = suggest(trial, "gamma", "float", 1.3, **cfg.get("gamma", {"low": 1.0, "high": 2.0, "step": 0.1}))
    latent_dim = suggest(trial, "latent_dim", "int", 4, **cfg.get("latent_dim", {"low": 2, "high": 8, "step": 2}))

    # Loss weights
    recon_weight = suggest(trial, "recon", "float", 1.0, **cfg.get("recon", {"low": 0.1, "high": 1.0, "step": 0.1}))
    kl_weight = suggest(trial, "kl", "float", 1e-3, **cfg.get("kl", {"low": 1e-5, "high": 1e-2, "log": True}))
    reg_weight = suggest(trial, "reg", "float", 1.0, **cfg.get("reg", {"low": 0.1, "high": 2.0, "step": 0.1}))
    sparse_weight = suggest(trial, "sparse", "float", 1e-3, **cfg.get("sparse", {"low": 1e-5, "high": 1e-2, "log": True}))

    # Training hyperparams
    learning_rate = suggest(trial, "lr", "float", 1e-3, **cfg.get("lr", {"low": 1e-4, "high": 1e-2, "log": True}))
    batch_size = suggest(trial, "batch_size", "categorical", 64, **cfg.get("batch_size", {"choices": [32, 64, 128]}))

    max_neurons = suggest(trial, "max_reg_neurons", "categorical", 64, **cfg.get("max_reg_neurons", {"choices": [16, 32, 64, 128]}))
    num_layers = suggest(trial, "num_reg_layers", "int", 2, **cfg.get("num_reg_layers", {"low": 1, "high": 5}))
    hidden_sizes = [max_neurons // (2 ** i) for i in range(num_layers) if max_neurons // (2 ** i) >= 8]

    # Data
    X, y = objective.X, objective.y
    dataset = TabularDataset(X, y)
    input_dim = X.shape[1]
    output_dim = y.shape[1]

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    val_losses = []

    for train_idx, val_idx in kfold.split(dataset):
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

        model = TabNetVAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            hidden_sizes=hidden_sizes
        )

        config = {
            "epochs": train_cfg.get("epochs", 200),
            "lr": learning_rate,
            "patience": train_cfg.get("patience", 20),
            "loss_weights": {
                "recon": recon_weight,
                "kl": kl_weight,
                "reg": reg_weight,
                "sparse": sparse_weight
            }
        }

        _, val_loss = train_model(model, train_loader, val_loader, config, device)
        val_losses.append(val_loss)

    return np.mean(val_losses)

def run_optuna(X, y, n_trials=50, param_config=None, train_settings=None):
    objective.X = X
    objective.y = y
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, param_config, train_settings), n_trials=n_trials)
    return study
