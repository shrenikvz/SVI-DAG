import numpy as np

from src.metrics import SHD, SID, edge_apr, edge_auroc, edge_fn_fp_rev


def test(model, true_dag_adj):
    model.eval()
    prob_mask = model.get_prob_mask()
    metrics = {
        "undirected_edge_auroc": edge_auroc(pred_edges=prob_mask + prob_mask.T, true_edges=true_dag_adj + true_dag_adj.T),
        "reverse_edge_auroc": edge_auroc(pred_edges=prob_mask, true_edges=true_dag_adj.T),
        "edge_auroc": edge_auroc(pred_edges=prob_mask, true_edges=true_dag_adj),
        "undirected_edge_apr": edge_apr(pred_edges=prob_mask + prob_mask.T, true_edges=true_dag_adj + true_dag_adj.T),
        "reverse_edge_apr": edge_apr(pred_edges=prob_mask, true_edges=true_dag_adj.T),
        "edge_apr": edge_apr(pred_edges=prob_mask, true_edges=true_dag_adj),
    }

    for threshold in np.linspace(0.01, 0.99, 99):
        pred_dag_adj = model.get_threshold_mask(threshold=threshold)
        fn, fp, rev = edge_fn_fp_rev(pred_dag_adj, true_dag_adj)
        metrics[f"fn_{threshold}"] = fn
        metrics[f"fp_{threshold}"] = fp
        metrics[f"rev_{threshold}"] = rev
        metrics[f"shd_{threshold}"] = SHD(pred_dag_adj, true_dag_adj)
        metrics[f"sid_{threshold}"] = SID(pred_dag_adj, true_dag_adj)

    return metrics

