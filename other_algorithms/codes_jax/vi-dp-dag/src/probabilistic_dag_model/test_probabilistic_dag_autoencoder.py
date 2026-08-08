import numpy as np
import jax.numpy as jnp

from src.metrics.dag_metrics import edge_apr, edge_auroc


def test_autoencoder(model, true_dag_adj, train_loader, test_loader, result_path, seed_dataset):
    del train_loader, result_path, seed_dataset
    model.eval()

    if model.pd_initial_adj is None:
        prob_mask = model.probabilistic_dag.get_prob_mask()
    else:
        prob_mask = model.pd_initial_adj

    metrics = {
        "undirected_edge_auroc": edge_auroc(pred_edges=prob_mask + prob_mask.T, true_edges=true_dag_adj + true_dag_adj.T),
        "undirected_edge_apr": edge_apr(pred_edges=prob_mask + prob_mask.T, true_edges=true_dag_adj + true_dag_adj.T),
        "reverse_edge_auroc": edge_auroc(pred_edges=prob_mask, true_edges=true_dag_adj.T),
        "reverse_edge_apr": edge_apr(pred_edges=prob_mask, true_edges=true_dag_adj.T),
        "edge_auroc": edge_auroc(pred_edges=prob_mask, true_edges=true_dag_adj),
        "edge_apr": edge_apr(pred_edges=prob_mask, true_edges=true_dag_adj),
    }

    X_pred_all = None
    X_all = None
    for batch_index, X in enumerate(test_loader):
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
    metrics["reconstruction"] = float(np.mean((X_all - X_pred_all) ** 2))
    return metrics
