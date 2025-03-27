import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


def train_model(model, train_loader, val_loader, config, device):
    """
    Train the AutoTabVAE model for a given number of epochs with early stopping.

    Parameters
    ----------
    model : nn.Module
        An instance of TabNetVAE.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    config : dict
        Dictionary containing hyperparameters and training configuration.
    device : torch.device
        Device to run training on ("cpu" or "cuda").

    Returns
    -------
    model : nn.Module
        Trained model with best validation performance.
    best_val_loss : float
        Best validation loss achieved.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    early_stop_counter = 0
    patience = config.get("patience", 20)

    for epoch in range(config["epochs"]):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            reconstructed_x, regression_output, mu, logvar, M_loss = model(x_batch)
            total_loss, *_ = model.loss_function(
                x_batch, reconstructed_x, y_batch, regression_output,
                mu, logvar, M_loss, **config["loss_weights"]
            )

            total_loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                reconstructed_x, regression_output, mu, logvar, M_loss = model(x_val)
                total_loss, *_ = model.loss_function(
                    x_val, reconstructed_x, y_val, regression_output,
                    mu, logvar, M_loss, **config["loss_weights"]
                )
                val_loss += total_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            break

    model.load_state_dict(best_model_state)
    return model, best_val_loss
