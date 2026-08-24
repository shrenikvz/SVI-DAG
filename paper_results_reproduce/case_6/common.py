#!/usr/bin/env python
"""
Case 6: shared utilities (synthetic nonlinear ER DAGs, DAG metrics)
====================================================================

Benchmarks SVI-DAG against 5 baselines on synthetic *nonlinear* DAGs.  The
only configuration reported here is

    * ER, p = 50, s = 80    ("ER_p50_s80")  with a paper-spec nonlinear SEM
      (1-hidden-layer MLP with 10 ReLU units, weights ~ U([-0.7,-0.3] U
      [0.3,0.7]); see ``svidag.data.simulate_sem`` ``sem_type='nonlinear'``)
      and N(0, 1) Gaussian noise.

For this configuration we sweep five sample sizes
``n in {10^2, 10^2.5, 10^3, 10^3.5, 10^4} = {100, 316, 1000, 3162, 10000}`` and
generate ``NUM_REPLICATES = 5`` independent (graph, data) pairs.  Per
(algorithm, n, replicate) cell we report four **DAG-level** metrics --
Brier, E[SHD], E[F1], AUROC -- so the box plots in the paper can be drawn
directly from per-replicate rows.

(Case 2 reports the same four metrics in **CPDAG** space because the linear-
Gaussian SEM is structurally Markov-equivalent within a CPDAG.  For
*nonlinear* SEMs the DAG itself is identifiable from observational data, so
DAG-level metrics are the correct comparison and we use them here.)

Graph generation matches the LaTeX spec: ``nx.gnm_random_graph(p, s)`` for
exactly s undirected edges, then orient each edge along a uniformly-random
topological ordering.

Outputs (long-form CSV) are written to
``paper_results_reproduce/case_6/case_6_results_<suffix>.csv`` so each
parallel job writes independently.  An aggregate JSON with mean ± SE per
(n, algorithm) cell is dumped alongside.

Author: Shrenik Zinage
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import networkx as nx

# ---------------------------------------------------------------------------
# Path plumbing -- make svidag and case_4's wrappers importable.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
_SRC_ROOT = _REPO_ROOT / "src"
_CASE4_DIR = _REPO_ROOT / "paper_results_reproduce" / "case_4"
for _p in (str(_REPO_ROOT), str(_SRC_ROOT), str(_CASE4_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# Force the case-local dir to sys.path[0] (see _single_algo.py for why).
_local = str(_THIS_DIR)
if _local in sys.path:
    sys.path.remove(_local)
sys.path.insert(0, _local)

from svidag.data import simulate_sem  # noqa: E402
from svidag.utils import compute_expected_metrics  # noqa: E402  (DAG-level)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
THRESHOLD_DEFAULT = 0.5

# Sample-size grid: n = round(10**[2, 2.5, 3, 3.5, 4, 1, 1.5])
#                     = [100, 316, 1000, 3162, 10000, 10, 32].
#
# ORDER IS LOAD-BEARING -- do not sort this list.  ``_single_algo._cell_index``
# derives the seed offset from ``SAMPLE_SIZES.index(n)``, so the position of an
# entry, not its value, fixes the RNG stream for every (n, rep, algo) cell.
# The two half-decades below 10^2 were added after the original five had been
# run, and are therefore APPENDED rather than sorted into ascending place:
# prepending them would shift n_idx for 100..10000 and silently invalidate the
# committed results for those cells.  Consumers that need ascending order
# (plotting, x-axes) must sort at the point of use.
SAMPLE_SIZES: List[int] = [100, 316, 1000, 3162, 10000, 10, 32]
SAMPLE_SIZE_LOG10: List[float] = [2.0, 2.5, 3.0, 3.5, 4.0, 1.0, 1.5]

NUM_REPLICATES_DEFAULT = 5


@dataclass(frozen=True)
class GraphScenario:
    """One graph configuration."""
    label: str
    p: int
    s: int
    sem_type: str = "nonlinear"
    weight_range: Tuple[float, float] = (0.3, 0.7)
    noise_scale: float = 1.0


# Single nonlinear scenario reported in case 6 (ER, p = 50, s = 80,
# nonlinear MLP SEM, N(0, 1) Gaussian noise).
SCENARIOS: List[GraphScenario] = [
    GraphScenario(label="ER_p50_s80", p=50, s=80, sem_type="nonlinear"),
]


# ---------------------------------------------------------------------------
# Synthetic dataset container
# ---------------------------------------------------------------------------
@dataclass
class SyntheticDataset:
    """One (graph, observational data) pair, drawn iid for one replicate."""
    X: np.ndarray                # [N, p] float32 observations
    true_adj: np.ndarray         # [p, p] int32 ground-truth DAG (j->i convention)
    node_names: List[str]
    num_nodes: int
    scenario_label: str
    num_samples: int
    replicate: int


# ---------------------------------------------------------------------------
# Random-DAG generator with a fixed edge count
# ---------------------------------------------------------------------------
def _random_er_dag_fixed_edges(p: int, s: int, rng_seed: int) -> np.ndarray:
    """
    Sample an Erdos-Renyi graph with **exactly s undirected edges** on p
    nodes (``nx.gnm_random_graph``), then orient each edge along a uniformly
    random topological ordering of the p nodes.

    Returns the adjacency matrix in SVIDAG convention: ``A[i, j] = 1``
    iff ``j -> i``.
    """
    if s < 0 or s > p * (p - 1) // 2:
        raise ValueError(
            f"Cannot place {s} undirected edges on {p} nodes "
            f"(max = {p * (p - 1) // 2})."
        )
    G_undirected = nx.gnm_random_graph(p, s, seed=int(rng_seed))

    rng = np.random.default_rng(int(rng_seed))
    perm = list(rng.permutation(p))
    order = {node: i for i, node in enumerate(perm)}

    G_dag = nx.DiGraph()
    G_dag.add_nodes_from(range(p))
    for u, v in G_undirected.edges():
        if order[u] < order[v]:
            G_dag.add_edge(u, v)
        else:
            G_dag.add_edge(v, u)
    if not nx.is_directed_acyclic_graph(G_dag):
        raise RuntimeError("Generated graph is not a DAG (should be impossible).")

    # NetworkX: A[i,j] = 1 means i -> j.  SVIDAG: A[i,j] = 1 means j -> i.
    adj_nx = nx.to_numpy_array(G_dag, dtype=np.int32)
    return adj_nx.T.astype(np.int32)


def generate_dataset(
    scenario: GraphScenario,
    num_samples: int,
    replicate: int,
) -> SyntheticDataset:
    """
    Draw one (graph, data) replicate for the given scenario and sample size.

    A unique seed is built from the (scenario_label, num_samples, replicate)
    triple so every cell of the case_6 grid is reproducible while different
    cells see independent draws.  We deliberately use a different seed prefix
    ("case6|...") than case_3 to avoid accidental seed collisions across
    cases.
    """
    seed_str = f"case6|{scenario.label}|n={num_samples}|rep={replicate}"
    # NOTE: a *deterministic* digest, not builtin hash() -- Python salts str
    # hashes per process (PYTHONHASHSEED), which would hand every parallel
    # algorithm job a different graph/dataset for the same cell and make the
    # paired cross-algorithm comparison meaningless.
    rng_seed = int.from_bytes(
        hashlib.sha256(seed_str.encode("utf-8")).digest()[:4], "little"
    )

    adj = _random_er_dag_fixed_edges(scenario.p, scenario.s, rng_seed=rng_seed)

    # Decorrelate graph and data RNGs (matches generate_benchmark_dataset).
    data, _W = simulate_sem(
        adj,
        num_samples=num_samples,
        sem_type=scenario.sem_type,            # 'nonlinear' here
        noise_scale=scenario.noise_scale,
        weight_range=scenario.weight_range,
        rng_seed=rng_seed + 1,
    )

    return SyntheticDataset(
        X=data.astype(np.float32),
        true_adj=adj.astype(np.int32),
        node_names=[f"x{i}" for i in range(scenario.p)],
        num_nodes=scenario.p,
        scenario_label=scenario.label,
        num_samples=num_samples,
        replicate=replicate,
    )


# ---------------------------------------------------------------------------
# Adjacency convention handling, thresholding, posterior mean
# ---------------------------------------------------------------------------
def binarize_and_clean(
    A_relaxed: np.ndarray,
    threshold: float = THRESHOLD_DEFAULT,
) -> np.ndarray:
    A = np.asarray(A_relaxed)
    A_bin = (A >= threshold).astype(np.int32)
    if A_bin.ndim == 3:
        d = A_bin.shape[1]
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
    return np.asarray(A_samples, dtype=np.float32).mean(axis=0)


# ---------------------------------------------------------------------------
# DAG-level metric evaluation
# ---------------------------------------------------------------------------
def evaluate_samples(
    A_relaxed_samples: np.ndarray,
    dataset: SyntheticDataset,
    source_convention: str,
    threshold: float = THRESHOLD_DEFAULT,
) -> Dict[str, float]:
    """
    Compute Brier / E[SHD] / E[F1] / AUROC at **DAG level** (case 6 reports
    DAG metrics because the nonlinear SEM is identifiable from observational
    data, so the DAG itself -- not just its CPDAG -- is the correct target).
    """
    A = normalise_convention(A_relaxed_samples, source_convention)
    A_bin = binarize_and_clean(A, threshold=threshold)
    pred_probs = posterior_mean(A).copy()
    d = pred_probs.shape[0]
    pred_probs[np.arange(d), np.arange(d)] = 0.0

    m = compute_expected_metrics(
        A_bin, dataset.true_adj.astype(int), pred_probs=pred_probs
    )
    return {
        "Brier": float(m["Brier"]) if m["Brier"] is not None else float("nan"),
        "E_SHD": float(m["E_SHD"]),
        "E_F1": float(m["E_F1"]),
        "AUROC": float(m["AUROC"]) if m["AUROC"] is not None else float("nan"),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_replicates(
    per_replicate_metrics: Sequence[Dict[str, float]],
) -> Dict[str, Tuple[float, float]]:
    """Reduce a list of per-replicate metric dicts to (mean, SE) pairs.

    Standard error of the mean uses ``std(ddof=1) / sqrt(n)``; with n=1
    the SE is reported as 0.0.
    """
    if not per_replicate_metrics:
        return {}
    keys = per_replicate_metrics[0].keys()
    out: Dict[str, Tuple[float, float]] = {}
    for k in keys:
        vals = np.array([m[k] for m in per_replicate_metrics], dtype=np.float64)
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
