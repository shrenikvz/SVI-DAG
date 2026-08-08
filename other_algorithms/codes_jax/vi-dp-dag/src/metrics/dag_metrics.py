import numpy as np
from sklearn import metrics

try:
    from cdt.metrics import SHD as _cdt_shd
    from cdt.metrics import SID as _cdt_sid
except ImportError:  # pragma: no cover - depends on local environment
    _cdt_shd = None
    _cdt_sid = None


def _as_numpy(value):
    return np.asarray(value)


def edge_auroc(pred_edges, true_edges):
    pred_edges = _as_numpy(pred_edges)
    true_edges = _as_numpy(true_edges)
    if true_edges.min() < 0 or true_edges.max() > 1:
        print("Groundtruth is CPDAG")
        true_edges = np.clip(true_edges, 0, 1)
    fpr, tpr, _ = metrics.roc_curve(true_edges.reshape(-1), pred_edges.reshape(-1))
    return metrics.auc(fpr, tpr)


def edge_apr(pred_edges, true_edges):
    pred_edges = _as_numpy(pred_edges)
    true_edges = _as_numpy(true_edges)
    if true_edges.min() < 0 or true_edges.max() > 1:
        print("Groundtruth is CPDAG")
        true_edges = np.clip(true_edges, 0, 1)
    true_edges = np.clip(true_edges, 0, 1)
    return metrics.average_precision_score(true_edges.reshape(-1), pred_edges.reshape(-1))


def edge_fn_fp_rev(pred_edges, true_edges):
    pred_edges = _as_numpy(pred_edges)
    true_edges = _as_numpy(true_edges)
    diff = true_edges - pred_edges
    rev = np.sum(((diff + diff.T) == 0) & (diff != 0)) / 2.0
    fn = np.sum(diff == 1) - rev
    fp = np.sum(diff == -1) - rev
    return fn, fp, rev


def SHD(pred_edges, true_edges):
    pred_edges = _as_numpy(pred_edges)
    true_edges = _as_numpy(true_edges)
    if _cdt_shd is not None:
        return _cdt_shd(pred_edges, true_edges)
    diff = pred_edges - true_edges
    return float(np.sum(np.abs(diff)))


def SID(pred_edges, true_edges):
    pred_edges = _as_numpy(pred_edges)
    true_edges = _as_numpy(true_edges)
    if _cdt_sid is None:
        raise ImportError("cdt is required for SID evaluation, matching the original implementation.")
    return _cdt_sid(pred_edges, true_edges)
