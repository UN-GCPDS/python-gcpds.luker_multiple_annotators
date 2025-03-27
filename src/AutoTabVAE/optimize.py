import optuna
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np

from .model import TabNetVAE
from .dataset import TabularDataset
from .train import train_model


def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Suggest hyperparameters
    n_d = trial.suggest_int("n_d", 8, 64, step=8)
    n_a = trial.suggest_int("n_a", 8, 64, step=8)
    n_steps = trial.suggest_int("n_steps", 3, 10)
    gamma = trial.suggest_float("gamma", 1.0, 2.0, step=0.1)
    latent_dim = trial.suggest_int("latent_dim", 2, 8, step=2)

    recon_weight = trial.suggest_float("recon", 0.1, 1.0, step=0.1)
    kl_weight = trial.suggest_float("kl", 1e-5, 1e-2, log=True)
    reg_weight = trial.suggest_float("reg", 0.1, 2.0, step=0.1)
    sparse_weight = trial.suggest_float("sparse", 1e-5, 1e-2, log=True)

    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    max_neurons = trial.suggest_categorical("max_reg_neurons", [16, 32, 64, 128])
    num_layers = trial.suggest_int("num_reg_layers", 1, 5)
    hidden_sizes = [max_neurons // (2 ** i) for i in range(num_layers) if max_neurons // (2 ** i) >= 8]

    # Load your dataset (must be assigned beforehand)
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
            "epochs": 200,
            "lr": learning_rate,
            "patience": 20,
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


def run_optuna(X, y, n_trials=50):
    """
    Run Optuna hyperparameter optimization.

    Parameters
    ----------
    X : np.ndarray
        Input features.
    y : np.ndarray
        Regression targets.
    n_trials : int
        Number of optimization trials.
    """
    objective.X = X
    objective.y = y

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study
