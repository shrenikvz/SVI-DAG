import numpy as np
import jax.numpy as jnp

from src.jax_utils import save_checkpoint


def compute_loss_reconstruction(model, loader):
    model.eval()
    loss = 0.0
    X_pred_all = None
    X_all = None

    for batch_index, X in enumerate(loader):
        X = jnp.asarray(X, dtype=jnp.float32)
        model.update_mask(type="deterministic")
        X_pred = model(X)
        flat_pred = np.asarray(X_pred.reshape(-1))
        flat_true = np.asarray(X.reshape(-1))
        if batch_index == 0:
            X_pred_all = flat_pred
            X_all = flat_true
        else:
            X_pred_all = np.concatenate([X_pred_all, flat_pred], axis=0)
            X_all = np.concatenate([X_all, flat_true], axis=0)
        loss += X.shape[0] * float(model.grad_loss)

    loss = loss / X_pred_all.shape[0]
    reconstruction = float(np.mean((X_all - X_pred_all) ** 2))
    model.train()
    return loss, reconstruction


def train_autoencoder(model, train_loader, val_loader, max_epochs=200, frequency=2, patience=50, model_path="saved_model", full_config_dict=None):
    if full_config_dict is None:
        full_config_dict = {}

    model.to(None)
    model.train()
    train_losses, val_losses, train_mses, val_mses = [], [], [], []
    best_val_loss = float("Inf")

    for epoch in range(max_epochs):
        for X_train in train_loader:
            X_train = jnp.asarray(X_train, dtype=jnp.float32)
            model.update_mask()
            model(X_train)
            model.step()

        if epoch % frequency == 0:
            train_loss, train_mse = compute_loss_reconstruction(model, train_loader)
            train_losses.append(round(train_loss, 3))
            train_mses.append(round(train_mse, 3))

            val_loss, val_mse = compute_loss_reconstruction(model, val_loader)
            val_losses.append(val_loss)
            val_mses.append(val_mse)

            print(f"Epoch {epoch} -> Val loss {round(val_losses[-1], 3)} | Val MSE.: {round(val_mses[-1], 3)}")

            if best_val_loss > val_losses[-1]:
                best_val_loss = val_losses[-1]
                save_checkpoint(
                    model_path,
                    {
                        "epoch": epoch,
                        "model_config_dict": full_config_dict,
                        "model_state_dict": model.state_dict(),
                        "loss": best_val_loss,
                    },
                )
                print("Model saved")

            if np.isnan(val_losses[-1]):
                print("Detected NaN Loss")
                break

            if int(epoch / frequency) > patience and val_losses[-patience] <= min(val_losses[-patience:]):
                print("Early Stopping.")
                break

    return train_losses, val_losses, train_mses, val_mses

