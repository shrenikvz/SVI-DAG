import numpy as np
import jax
import jax.numpy as jnp

from src.jax_utils import save_checkpoint


def train(model, true_dag_adj, max_epochs=30000, frequency=10, patience=30000, model_path="saved_model", full_config_dict=None):
    if full_config_dict is None:
        full_config_dict = {}

    model.to(None)
    true_dag_adj = jnp.asarray(true_dag_adj, dtype=jnp.float32)
    model.train()
    prob_abs_losses = []
    sampled_mse_losses = []
    best_losses = float("Inf")

    for epoch in range(max_epochs):
        sample_key = model.next_key()

        def loss_fn(params):
            sampled_dag_adj = model.sample(params=params, key=sample_key)
            return jnp.sum((true_dag_adj - sampled_dag_adj) ** 2)

        sampled_mse_loss, grads = jax.value_and_grad(loss_fn)(model.params)
        model.apply_gradients(grads)

        if epoch % frequency == 0:
            prob_mask = model.get_prob_mask()
            prob_abs_loss = jnp.sum(jnp.abs(true_dag_adj - prob_mask))
            prob_abs_losses.append(float(prob_abs_loss))
            sampled_mse_losses.append(float(sampled_mse_loss))

            print(f"Epoch {epoch} -> prob_abs_loss {prob_abs_losses[-1]} | sampled_mse_loss {sampled_mse_losses[-1]}")

            if best_losses > prob_abs_losses[-1]:
                best_losses = prob_abs_losses[-1]
                save_checkpoint(
                    model_path,
                    {
                        "epoch": epoch,
                        "model_config_dict": full_config_dict,
                        "model_state_dict": model.state_dict(),
                        "loss": best_losses,
                    },
                )
                print("Model saved")

            if np.isnan(prob_abs_losses[-1]):
                print("Detected NaN Loss")
                break

            if int(epoch / frequency) > patience and prob_abs_losses[-patience] <= min(prob_abs_losses[-patience:]):
                print("Early Stopping.")
                break

    return model, prob_abs_losses, sampled_mse_losses

