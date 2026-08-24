#!/usr/bin/env python
"""
Case 4: shared utilities
========================

Benchmarks SVIDAG against 6 baselines on the Sachs dataset across 10 equal
random train/test splits.  This module holds everything shared across the
SVIDAG runner and the baseline runners:

    * Sachs data loader (bypasses ``load_sachs_dataset`` so we control the
      splitting instead of using the default ``last-n_test_samples`` strategy).
    * 10-fold split generator.
    * Relaxed-adjacency thresholding / diagonal cleaning.
    * Convention-normalisation helpers (every baseline's output is converted
      to SVIDAG's ``A[i,j]=1 means j -> i`` convention before metrics).
    * Metric computation for BOTH families -- DAG-level (wraps
      ``compute_expected_metrics``) and CPDAG-level (wraps
      ``compute_expected_metrics_cpdag``).  ``evaluate_samples_all_modes``
      scores one set of posterior samples under both in a single pass, which
      is what lets case 4 emit a DAG table and a CPDAG table from one run.
    * Mean / standard-error aggregation across splits.

Author: Shrenik Zinage
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import networkx as nx

# ---------------------------------------------------------------------------
# Path plumbing: make the svidag package importable whether this file is
# executed from the repo root, from case_4/, or from a job scheduler.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from svidag.utils import (  # noqa: E402  (path set above)
    compute_expected_metrics,
    compute_expected_metrics_cpdag,
    dag_to_cpdag,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_SPLITS_DEFAULT = 10
NUM_POSTERIOR_SAMPLES_DEFAULT = 100   # Uniform across all algorithms for fairness.
THRESHOLD_DEFAULT = 0.5               # For baselines' relaxed outputs only;
                                      # SVIDAG samples are already binary DAGs.


# ===========================================================================
# Sachs data loading (split-aware)
# ===========================================================================
@dataclass
class SachsData:
    """Container for the full Sachs dataset BEFORE any train/test split."""
    X: np.ndarray                # [N_total, d] original (unscaled) observations
    true_adj: np.ndarray         # [d, d] ground-truth DAG, SVIDAG convention (j->i)
    true_cpdag: np.ndarray       # [d, d] ground-truth CPDAG, SVIDAG convention
    node_names: List[str]
    num_nodes: int


def load_sachs_full(data_path: Optional[str] = None) -> SachsData:
    """
    Load the full Sachs dataset without applying any train/test split.

    We replicate the loading logic in svidag.data.load_sachs_dataset but
    deliberately DO NOT strip off the last ``n_test_samples`` rows, because
    case 4 uses its own 10 random splits.

    ``data_path=None`` (the default) reads the 7,466-row full dataset bundled
    with cdt -- all perturbation conditions pooled -- which is the setting
    every published case-4 number was produced under (see this directory's
    README).  Passing a path instead reads that tab-separated file; note that
    the 853-row observational subset in ``data/sachs/`` is a *different*
    dataset, not another route to the same one.

    This used to default to the string ``"sachs.data.txt"`` and rely on a
    ``FileNotFoundError`` to reach the cdt data.  That worked only because no
    such file exists at the repository root -- and would have silently
    switched datasets for anyone who ran from a directory where one did.

    Ground truth convention: SVIDAG uses ``A[i,j]=1 means j -> i``.
    NetworkX's ``to_numpy_array`` uses ``A[i,j]=1 means i -> j``, so we
    transpose once at load time.
    """
    # Prefer cdt.data.load_dataset when cdt imports cleanly.  If cdt's
    # top-level import fails (e.g. torch was uninstalled -- cdt.utils.io
    # does `from torch.utils.data import Dataset` unconditionally at module
    # load time), fall back to reading the Sachs resource files that ship
    # inside the cdt package directly.  This keeps the loader working in
    # torch-free environments without reinstalling the 300MB torch wheel.
    def _cdt_load_sachs_direct():
        """Return (dataframe, nx.DiGraph) exactly like cdt.data.load_dataset('sachs')."""
        import importlib.util as _ilu
        spec = _ilu.find_spec("cdt")
        if spec is None or spec.submodule_search_locations is None:
            raise ImportError("cdt package not found on sys.path")
        cdt_dir = Path(list(spec.submodule_search_locations)[0])
        data_csv = cdt_dir / "data" / "resources" / "cyto_full_data.csv"
        target_csv = cdt_dir / "data" / "resources" / "cyto_full_target.csv"
        df = pd.read_csv(data_csv)
        edges = pd.read_csv(target_csv)  # columns: Cause, Effect
        g = nx.DiGraph()
        g.add_nodes_from(df.columns.tolist())
        for _, row in edges.iterrows():
            g.add_edge(row["Cause"], row["Effect"])
        return df, g

    try:
        from cdt.data import load_dataset as _cdt_load
    except ImportError:
        _cdt_load = lambda name: _cdt_load_sachs_direct() if name == "sachs" else (_ for _ in ()).throw(ValueError(f"Unsupported dataset: {name}"))

    # -- observations --
    df_cdt, graph = _cdt_load("sachs")
    df_cdt.dropna(inplace=True)
    node_names = df_cdt.columns.tolist()
    if data_path is None:
        X = df_cdt.to_numpy(dtype=np.float32)
    else:
        tbl = pd.read_csv(data_path, sep="\t")
        X = tbl.to_numpy(dtype=np.float32)

    # -- ground truth DAG (SVIDAG convention: j -> i) --
    true_adj = nx.to_numpy_array(graph, dtype=np.float32).T.astype(np.int32)

    # -- ground truth CPDAG (computed once so CPDAG metrics are cheap) --
    # dag_to_cpdag expects i->j convention, so transpose in and out.
    true_cpdag = dag_to_cpdag(true_adj.T).T.astype(np.int32)

    num_nodes = true_adj.shape[0]
    assert X.shape[1] == num_nodes, (
        f"Data/node count mismatch: {X.shape[1]} vs {num_nodes}"
    )

    return SachsData(
        X=X,
        true_adj=true_adj,
        true_cpdag=true_cpdag,
        node_names=node_names,
        num_nodes=num_nodes,
    )


# ===========================================================================
# 10-fold split generator
# ===========================================================================
@dataclass
class Split:
    """One train/test split."""
    index: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def make_splits(
    n_total: int,
    num_splits: int = NUM_SPLITS_DEFAULT,
    seed: int = 0,
    mode: str = "kfold",
) -> List[Split]:
    """
    Build ``num_splits`` train/test index pairs over ``n_total`` samples.

    mode="kfold":
        Shuffle once, then partition into ``num_splits`` equal folds.  Each
        fold in turn becomes the test set while the remaining data is the
        training set (standard K-fold CV -- "10 equal splits of the data").
    mode="random_90_10":
        ``num_splits`` independent random 90/10 shuffled splits.
    mode="disjoint":
        Shuffle once, then partition into ``num_splits`` equal *disjoint*
        chunks of ~n_total/num_splits samples each.  Each chunk in turn is
        used as the **training** set and ``test_idx`` is empty -- there is
        no held-out test partition.  This is the right mode when the
        downstream metrics are *structural* (compared against a known
        ground-truth DAG, not a held-out predictive likelihood) and we want
        the variance across splits to reflect *training-set sampling
        uncertainty*: each split fits the model on a different ~1/k slice
        of the data, and we average the structural metrics over splits.

    SVIDAG's Dataset uses ``train_data`` for fitting and ``test_data_scaled``
    only for predictive evaluation; for DAG metrics we really just need the
    *training* subset to differ across splits so the variance we report
    reflects genuine sampling uncertainty.
    """
    rng = np.random.default_rng(seed)

    if mode == "kfold":
        perm = rng.permutation(n_total)
        fold_sizes = np.full(num_splits, n_total // num_splits, dtype=int)
        fold_sizes[: n_total % num_splits] += 1
        splits: List[Split] = []
        start = 0
        for k, size in enumerate(fold_sizes):
            test_idx = perm[start : start + size]
            train_idx = np.concatenate([perm[:start], perm[start + size :]])
            splits.append(
                Split(index=k, train_idx=np.sort(train_idx), test_idx=np.sort(test_idx))
            )
            start += size
        return splits

    if mode == "random_90_10":
        splits = []
        for k in range(num_splits):
            perm = rng.permutation(n_total)
            n_test = max(1, n_total // 10)
            test_idx = np.sort(perm[:n_test])
            train_idx = np.sort(perm[n_test:])
            splits.append(Split(index=k, train_idx=train_idx, test_idx=test_idx))
        return splits

    if mode == "disjoint":
        # Shuffle once for reproducibility-with-seed, then partition into
        # ``num_splits`` equal disjoint chunks.  The first ``n_total %
        # num_splits`` chunks get one extra sample so every index is used
        # exactly once and chunk sizes differ by at most 1.
        perm = rng.permutation(n_total)
        chunk_sizes = np.full(num_splits, n_total // num_splits, dtype=int)
        chunk_sizes[: n_total % num_splits] += 1
        splits = []
        start = 0
        for k, size in enumerate(chunk_sizes):
            train_idx = np.sort(perm[start : start + size])
            test_idx = np.empty(0, dtype=train_idx.dtype)  # no test partition
            splits.append(
                Split(index=k, train_idx=train_idx, test_idx=test_idx)
            )
            start += size
        return splits

    raise ValueError(f"Unknown split mode: {mode!r}")


# ===========================================================================
# Adjacency convention handling & thresholding
# ===========================================================================
def binarize_and_clean(
    A_relaxed: np.ndarray,
    threshold: float = THRESHOLD_DEFAULT,
) -> np.ndarray:
    """
    Threshold continuous [0,1] adjacency samples and zero the diagonal.

    Works for inputs shaped [S, d, d] or [d, d].  Returns int32.
    """
    A = np.asarray(A_relaxed)
    A_bin = (A >= threshold).astype(np.int32)
    if A_bin.ndim == 3:
        S, d, _ = A_bin.shape
        diag = np.arange(d)
        A_bin[:, diag, diag] = 0
    elif A_bin.ndim == 2:
        d = A_bin.shape[0]
        A_bin[np.arange(d), np.arange(d)] = 0
    else:
        raise ValueError(f"Unexpected adjacency shape: {A_bin.shape}")
    return A_bin


def normalise_convention(
    A_samples: np.ndarray,
    source_convention: str,
) -> np.ndarray:
    """
    Convert posterior samples to SVIDAG's ``A[i,j]=1 means j -> i`` convention.

    source_convention:
        "j_to_i"  -- already SVIDAG-compatible, no transpose.
        "i_to_j"  -- transpose the last two axes.

    Works for [S, d, d] or [d, d].
    """
    A = np.asarray(A_samples)
    if source_convention == "j_to_i":
        return A
    if source_convention == "i_to_j":
        if A.ndim == 3:
            return np.transpose(A, (0, 2, 1))
        if A.ndim == 2:
            return A.T
        raise ValueError(f"Unexpected adjacency shape: {A.shape}")
    raise ValueError(f"Unknown convention: {source_convention!r}")


def posterior_mean(A_samples: np.ndarray) -> np.ndarray:
    """Soft posterior mean (for Brier / AUROC on probabilistic predictions)."""
    A = np.asarray(A_samples, dtype=np.float32)
    return A.mean(axis=0)


# ===========================================================================
# Metric helpers (DAG-level and CPDAG-level -- both are reported)
# ===========================================================================
# Case 4 emits one table per metric family.  These are the canonical mode keys
# used everywhere downstream (result JSONs, LaTeX filenames, console headers).
METRIC_MODES: Tuple[str, ...] = ("dag", "cpdag")
METRIC_MODE_LABELS: Dict[str, str] = {"dag": "DAG", "cpdag": "CPDAG"}


def _prepare_for_metrics(
    A_relaxed_samples: np.ndarray,
    source_convention: str,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shared front-half of the evaluation pipeline.

    Normalises the adjacency convention, thresholds to binary samples, and
    builds the diagonal-free soft posterior mean.  Factored out so the DAG and
    CPDAG metric families can be computed from *one* preprocessing pass over
    the same posterior samples (guaranteeing the two tables describe literally
    the same draws, not two independent re-evaluations).

    Returns ``(A_bin [S, d, d] int32, pred_probs [d, d] float32)``.
    """
    A = normalise_convention(A_relaxed_samples, source_convention)
    A_bin = binarize_and_clean(A, threshold=threshold)
    pred_probs = posterior_mean(A)                # soft mean (for Brier / AUROC)
    # Zero the diagonal of the mean too, so it can't inflate probabilistic metrics.
    d = pred_probs.shape[0]
    pred_probs = pred_probs.copy()
    pred_probs[np.arange(d), np.arange(d)] = 0.0
    return A_bin, pred_probs


def _metrics_for_mode(
    A_bin: np.ndarray,
    pred_probs: np.ndarray,
    sachs: SachsData,
    mode: str,
) -> Dict[str, float]:
    """
    Score already-preprocessed samples against the ground truth for one mode.

    mode="dag"   -> oriented-DAG metrics vs ``sachs.true_adj``.
    mode="cpdag" -> Markov-equivalence-class metrics vs ``sachs.true_cpdag``
                    (each posterior DAG sample is converted to its CPDAG first).
    """
    if mode == "cpdag":
        m = compute_expected_metrics_cpdag(
            A_bin, sachs.true_cpdag.astype(int), pred_probs=pred_probs
        )
    elif mode == "dag":
        m = compute_expected_metrics(
            A_bin, sachs.true_adj.astype(int), pred_probs=pred_probs
        )
    else:
        raise ValueError(f"Unknown metric mode: {mode!r}. Valid: {METRIC_MODES}")

    # Unify shape / None handling.  AUROC can be None if only one class appears
    # in the ground truth; we surface as NaN so aggregation stays numeric.
    return {
        "Brier": float(m["Brier"]) if m["Brier"] is not None else float("nan"),
        "E_SHD": float(m["E_SHD"]),
        "E_F1": float(m["E_F1"]),
        "AUROC": float(m["AUROC"]) if m["AUROC"] is not None else float("nan"),
    }


def evaluate_samples(
    A_relaxed_samples: np.ndarray,
    sachs: SachsData,
    source_convention: str,
    threshold: float = THRESHOLD_DEFAULT,
    use_cpdag: bool = False,
) -> Dict[str, float]:
    """
    Compute Brier / E[SHD] / E[F1] / AUROC for ONE metric family on one split.

    Kept for callers that want a single family; the case-4 orchestrators use
    ``evaluate_samples_all_modes`` instead so both tables come from one run.

    Inputs:
        A_relaxed_samples : [S, d, d] RELAXED samples in [0, 1] (or already-binary;
                            thresholding is still safe and cheap in that case).
        sachs             : SachsData container (carries true_adj, true_cpdag).
        source_convention : "j_to_i" or "i_to_j" -- describes the baseline's native
                            convention so we can re-express it as SVIDAG's j->i.
        threshold         : binarisation cutoff for baselines' relaxed samples
                            (SVIDAG samples arrive already binary; the cutoff
                            is then a no-op).
        use_cpdag         : False -> DAG-level metrics, True -> CPDAG-level.

    Returns a dict with keys: Brier, E_SHD, E_F1, AUROC (floats, AUROC in [0,1]).
    """
    A_bin, pred_probs = _prepare_for_metrics(
        A_relaxed_samples, source_convention, threshold
    )
    return _metrics_for_mode(
        A_bin, pred_probs, sachs, mode="cpdag" if use_cpdag else "dag"
    )


def evaluate_samples_all_modes(
    A_relaxed_samples: np.ndarray,
    sachs: SachsData,
    source_convention: str,
    threshold: float = THRESHOLD_DEFAULT,
    modes: Sequence[str] = METRIC_MODES,
) -> Dict[str, Dict[str, float]]:
    """
    Score one algorithm/split under EVERY metric family in ``modes``.

    Training is what costs hours here; scoring the resulting samples twice is
    nearly free (the CPDAG pass dominates and is unchanged, the DAG pass is a
    cheap per-sample confusion-matrix count).  Evaluating both families from a
    single fit is therefore strictly better than running case 4 twice: it
    halves the compute AND guarantees the DAG and CPDAG tables are derived
    from identical posterior draws.

    Returns ``{mode: {"Brier": ..., "E_SHD": ..., "E_F1": ..., "AUROC": ...}}``.
    """
    A_bin, pred_probs = _prepare_for_metrics(
        A_relaxed_samples, source_convention, threshold
    )
    return {
        mode: _metrics_for_mode(A_bin, pred_probs, sachs, mode=mode)
        for mode in modes
    }


# ===========================================================================
# Aggregation
# ===========================================================================
def aggregate_splits(
    per_split_metrics: Sequence[Dict[str, float]],
) -> Dict[str, Tuple[float, float]]:
    """
    Reduce a list of per-split metric dicts to (mean, standard_error) pairs.

    Standard error of the mean uses ``std(ddof=1) / sqrt(n)``.
    With n=1 the SE is reported as 0.0.
    """
    keys = per_split_metrics[0].keys()
    out: Dict[str, Tuple[float, float]] = {}
    for k in keys:
        vals = np.array([m[k] for m in per_split_metrics], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            out[k] = (float("nan"), float("nan"))
            continue
        mean = float(np.mean(vals))
        if vals.size > 1:
            se = float(np.std(vals, ddof=1) / np.sqrt(vals.size))
        else:
            se = 0.0
        out[k] = (mean, se)
    return out


# ===========================================================================
# Small print helper
# ===========================================================================
def fmt_mean_se(mean: float, se: float, pct: bool = False, decimals: int = 2) -> str:
    """Render "mean ± se" for the LaTeX / console tables."""
    if np.isnan(mean):
        return "--"
    if pct:
        return f"{100 * mean:.{decimals}f} $\\pm$ {100 * se:.{decimals}f}"
    return f"{mean:.{decimals}f} $\\pm$ {se:.{decimals}f}"


# Backwards-compatible alias for any caller that imported the SD-named
# helper from the previous revision.
fmt_mean_std = fmt_mean_se


# ===========================================================================
# LaTeX table generation
# ===========================================================================
def build_latex_table(
    aggregated: Dict[str, Dict[str, Tuple[float, float]]],
    mode: str,
    row_order: Sequence[str],
    num_splits: int = NUM_SPLITS_DEFAULT,
) -> str:
    """
    Render one paper table (mean ± SE per algorithm/metric cell).

    Lives here rather than in ``run_case4.py`` so ``make_tables.py`` can reuse
    it without importing the baseline wrappers (and their torch/jax deps).

    Parameters
    ----------
    aggregated : {row_label: {metric: (mean, se)}}
        Only labels present here get numbers; the rest render as "--".
    mode : "dag" | "cpdag"
        Drives the caption wording and the ``\\label``, so both tables can sit
        in the same LaTeX document without a duplicate-label clash.
    row_order : sequence of row labels
        Table row order (the paper's algorithm ordering).
    """
    metric_label = METRIC_MODE_LABELS[mode]
    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Performance on sachs dataset. The averages and standard "
        rf"errors are measured over {num_splits} splits of the data. The best "
        rf"value of each {metric_label} metric is indicated in bold.}}"
    )
    lines.append(rf"\label{{tab:sachs_{mode}}}")
    lines.append(r"\small")
    lines.append(r"\begin{tabularx}{\linewidth}{XXXXX}")
    lines.append(r"\toprule")
    lines.append(r" & Brier score & Exp. SHD & Exp. F1 score (\%) & AUROC (\%) \\")
    lines.append(r"\midrule")

    # Identify the best (min for Brier, SHD; max for F1, AUROC) for bolding.
    def best(key: str, higher_is_better: bool) -> str:
        means = {
            label: vals[key][0]
            for label, vals in aggregated.items()
            if key in vals and not np.isnan(vals[key][0])
        }
        if not means:
            return ""
        if higher_is_better:
            return max(means, key=means.get)
        return min(means, key=means.get)

    best_labels = {
        "Brier": best("Brier", higher_is_better=False),
        "E_SHD": best("E_SHD", higher_is_better=False),
        "E_F1": best("E_F1", higher_is_better=True),
        "AUROC": best("AUROC", higher_is_better=True),
    }

    for row_label in row_order:
        if row_label not in aggregated:
            cells = ["--"] * 4
        else:
            vals = aggregated[row_label]

            def _fmt(key, pct, decimals=2, _vals=vals, _row=row_label):
                mean, se = _vals[key]
                s = fmt_mean_se(mean, se, pct=pct, decimals=decimals)
                if _row == best_labels.get(key, "") and s != "--":
                    s = r"\textbf{" + s + r"}"
                return s

            cells = [
                _fmt("Brier", pct=False, decimals=3),
                _fmt("E_SHD", pct=False, decimals=2),
                _fmt("E_F1",  pct=True,  decimals=2),
                _fmt("AUROC", pct=True,  decimals=2),
            ]
        lines.append(rf"\texttt{{{row_label}}} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)
