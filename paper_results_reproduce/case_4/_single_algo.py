#!/usr/bin/env python
"""
Shared per-algorithm driver for case-4 parallel jobs.
=====================================================

``run_case4.py`` runs all 9 algorithm rows in one job.  This helper lets each
algorithm (or a related group, e.g. the 3 SVIDAG prior scenarios) be launched
as its own SLURM job, so the 6 baselines + SVIDAG can run in parallel.

Each ``run_<algo>_only.py`` is a thin wrapper that calls
``run_subset(labels=[...], output_suffix="...")``.  The results are written to
``paper_results_reproduce/case_4/case_4_results_<suffix>.json``; once all jobs
finish, ``make_tables.py`` merges those files into the two LaTeX tables.

Every fit is scored under BOTH metric families (oriented-DAG and CPDAG) in one
pass, so the per-algorithm JSONs carry the numbers for both tables and no
algorithm ever needs to be retrained to switch metric family.

Config knobs (``NUM_SPLITS``, ``NUM_POSTERIOR_SAMPLES``, ``THRESHOLD``,
``METRIC_MODES_ACTIVE``, ``SVIDAG_NUM_ITERS``, ``SEED``, ``SPLIT_MODE``) are
kept in sync with ``run_case4.py`` so the parallel jobs produce numbers
comparable to the monolithic run.

Author: Shrenik Zinage
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
for p in (str(_REPO_ROOT), str(_REPO_ROOT / "src"), str(_THIS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from sklearn.preprocessing import StandardScaler

from common import (
    METRIC_MODE_LABELS,
    METRIC_MODES,
    SachsData,
    aggregate_splits,
    evaluate_samples_all_modes,
    load_sachs_full,
    make_splits,
)
from svidag_runner import run_svidag
from baselines import (
    prodag_wrapper,
    dibs_wrapper,
    bayesdag_wrapper,
    bcd_wrapper,
    dds_wrapper,
)

# ---------------------------------------------------------------------------
# Config (must stay in sync with run_case4.py).
# ---------------------------------------------------------------------------
# Env overrides let a single sbatch task narrow the grid (e.g. a 1-split
# SVI-DAG sweep) without touching the committed defaults the already-run
# baseline jobs used.  Unset => paper-spec values.
NUM_SPLITS = int(os.environ.get("CASE4_NUM_SPLITS", "10"))
NUM_POSTERIOR_SAMPLES = int(os.environ.get("CASE4_POSTERIOR_SAMPLES", "1000"))
# "kfold": shuffle once, partition into 10 equal folds, and use the union
# of the other 9 folds (~6720 rows on Sachs) as the training set for each
# split.  This puts each fit in the data regime the BayesDAG paper tuned its
# default hyperparameters on, so it can produce non-degenerate posteriors
# with non-zero standard error across splits.  The
# metrics are still structural (computed against the known Sachs ground
# truth, not a held-out predictive likelihood) -- the unused test split is
# simply not consumed by ``_run_one``.
SPLIT_MODE = "kfold"
SEED = int(os.environ.get("CASE4_SEED", "0"))
THRESHOLD = 0.5
# Score each fit under both metric families; one table per entry is produced
# downstream by make_tables.py.  Must match run_case4.py.
METRIC_MODES_ACTIVE = METRIC_MODES    # ("dag", "cpdag")
SVIDAG_NUM_ITERS = int(os.environ.get("SVIDAG_NUM_ITERS", "60000"))
SKIP_UNIMPLEMENTED = bool(int(os.environ.get("SKIP_UNIMPLEMENTED", "1")))


# ---------------------------------------------------------------------------
# Algorithm registry (must match run_case4.py's ALGORITHMS order).
# ---------------------------------------------------------------------------
ALL_ALGORITHMS: List[Tuple[str, Dict]] = [
    ("SVI-DAG (noninformative)",   {"kind": "svidag", "scenario": "noninformative"}),
    ("ProDAG",                     {"kind": "baseline", "fn": prodag_wrapper.run}),
    ("BayesDAG",                   {"kind": "baseline", "fn": bayesdag_wrapper.run}),
    ("DDS",                        {"kind": "baseline", "fn": dds_wrapper.run}),
    ("BCD Nets",                   {"kind": "baseline", "fn": bcd_wrapper.run}),
    ("DiBS",                       {"kind": "baseline", "fn": dibs_wrapper.run}),
]


# ---------------------------------------------------------------------------
# Per-(algorithm, split) driver -- identical body to run_case4.py:run_one.
# ---------------------------------------------------------------------------
def _run_one(
    row_label: str,
    spec: Dict,
    sachs: SachsData,
    train_idx: np.ndarray,
    split_index: int,
) -> Dict[str, Dict[str, float]]:
    """Fit one algorithm on one split; return ``{mode: metric_dict}``."""
    # Fair-comparison standardisation (same as run_case4.py).
    _scaler = StandardScaler().fit(sachs.X[train_idx])
    X_train = _scaler.transform(sachs.X[train_idx]).astype(np.float32)
    X_test_placeholder = _scaler.transform(sachs.X[train_idx[:1]]).astype(np.float32)

    t0 = time.time()
    if spec["kind"] == "svidag":
        A_samples, convention = run_svidag(
            X_train=X_train,
            X_test=X_test_placeholder,
            true_adj=sachs.true_adj,
            node_names=sachs.node_names,
            scenario=spec["scenario"],
            split_index=split_index,
            num_posterior_samples=NUM_POSTERIOR_SAMPLES,
            num_iters=SVIDAG_NUM_ITERS,
            seed=SEED,
            verbose=(split_index == 0),
        )
    else:
        # BayesDAG implements its paper-spec lambda grid search inside the
        # wrapper and needs the true sparsity level (number of edges in the
        # ground-truth DAG) to pick the closest match.  We pass it as an
        # extra kwarg only for that algorithm so the call signature for the
        # other 5 baselines stays narrow.
        extra = {}
        if row_label == "BayesDAG":
            extra["true_sparsity"] = int(np.sum(sachs.true_adj != 0))
        A_samples, convention = spec["fn"](
            X_train=X_train,
            num_nodes=sachs.num_nodes,
            num_posterior_samples=NUM_POSTERIOR_SAMPLES,
            seed=SEED + split_index,
            **extra,
        )
    dt = time.time() - t0

    metrics_by_mode = evaluate_samples_all_modes(
        A_relaxed_samples=A_samples,
        sachs=sachs,
        source_convention=convention,
        threshold=THRESHOLD,
        modes=METRIC_MODES_ACTIVE,
    )
    # Carry the wall time on every mode's dict so it survives into the result
    # JSON alongside the metrics and can be aggregated the same way. It is the
    # time for the whole fit + posterior draw on this split, measured once.
    for mode in METRIC_MODES_ACTIVE:
        metrics_by_mode[mode]["time_sec"] = float(dt)

    for mode in METRIC_MODES_ACTIVE:
        m = metrics_by_mode[mode]
        print(
            f"    split {split_index}/{NUM_SPLITS - 1:d} | {row_label:32s}"
            f" | {METRIC_MODE_LABELS[mode]:5s}"
            f" | Brier={m['Brier']:.3f}"
            f" | E[SHD]={m['E_SHD']:.2f}"
            f" | E[F1]={m['E_F1']:.3f}"
            f" | AUROC={m['AUROC']:.3f}"
            f" | {dt:.1f}s"
        )
    return metrics_by_mode


# ---------------------------------------------------------------------------
# Public entry point called from run_<algo>_only.py wrappers.
# ---------------------------------------------------------------------------
def run_subset(labels: List[str], output_suffix: str) -> Path:
    """
    Run the specified subset of ``ALL_ALGORITHMS`` over all ``NUM_SPLITS``
    splits of Sachs, save a JSON with per-split + aggregated metrics, and
    print a short console summary.  Returns the JSON path.

    Parameters
    ----------
    labels : list[str]
        Algorithm row labels to include, matching ``ALL_ALGORITHMS`` entries.
    output_suffix : str
        Result file suffix -- saved as
        ``paper_results_reproduce/case_4/case_4_results_{output_suffix}.json``.
    """
    spec_map = {lbl: spec for lbl, spec in ALL_ALGORITHMS}
    unknown = [l for l in labels if l not in spec_map]
    if unknown:
        raise ValueError(
            f"Unknown algorithm labels: {unknown}. "
            f"Valid: {list(spec_map.keys())}"
        )
    algorithms: List[Tuple[str, Dict]] = [(l, spec_map[l]) for l in labels]

    print("=" * 80)
    print(f"  Case 4 (parallel job): suffix = {output_suffix!r}")
    print(f"    algorithms     = {[l for l, _ in algorithms]}")
    print(f"    NUM_SPLITS     = {NUM_SPLITS}   SPLIT_MODE = {SPLIT_MODE}")
    print(f"    num_samples    = {NUM_POSTERIOR_SAMPLES}")
    print(f"    threshold      = {THRESHOLD}   metrics = {list(METRIC_MODES_ACTIVE)}")
    print(f"    svidag_num_iters = {SVIDAG_NUM_ITERS}   seed = {SEED}")
    print("=" * 80)

    sachs = load_sachs_full()
    print(f"  Sachs: N={sachs.X.shape[0]}, d={sachs.num_nodes}, "
          f"nodes={sachs.node_names}")

    splits = make_splits(
        n_total=sachs.X.shape[0],
        num_splits=NUM_SPLITS,
        seed=SEED,
        mode=SPLIT_MODE,
    )

    # per_split[mode][algorithm_label] = list of metric dicts (one per split).
    per_split: Dict[str, Dict[str, List[Dict[str, float]]]] = {
        mode: {lbl: [] for lbl, _ in algorithms} for mode in METRIC_MODES_ACTIVE
    }

    for split in splits:
        print(f"\n  ── split {split.index} (train={len(split.train_idx)}, "
              f"test={len(split.test_idx)}) ──")
        for row_label, spec in algorithms:
            try:
                metrics_by_mode = _run_one(
                    row_label=row_label, spec=spec,
                    sachs=sachs, train_idx=split.train_idx,
                    split_index=split.index,
                )
                for mode in METRIC_MODES_ACTIVE:
                    per_split[mode][row_label].append(metrics_by_mode[mode])
            except NotImplementedError as e:
                if SKIP_UNIMPLEMENTED:
                    print(f"    split {split.index} | {row_label:32s} | "
                          f"SKIPPED (stub): {e.args[0].splitlines()[0]}")
                    continue
                raise
            except Exception as e:  # noqa: BLE001
                print(f"    split {split.index} | {row_label:32s} | "
                      f"FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()

    # ── aggregate (per metric mode) ──────────────────────────────────────
    aggregated: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {}
    for mode in METRIC_MODES_ACTIVE:
        aggregated[mode] = {
            row_label: aggregate_splits(per_split[mode][row_label])
            for row_label, _ in algorithms
            if per_split[mode][row_label]
        }

    # ── save JSON ────────────────────────────────────────────────────────
    json_path = _THIS_DIR / f"case_4_results_{output_suffix}.json"
    dump = {
        "config": {
            "num_splits": NUM_SPLITS,
            "num_posterior_samples": NUM_POSTERIOR_SAMPLES,
            "split_mode": SPLIT_MODE,
            "threshold": THRESHOLD,
            "metric_modes": list(METRIC_MODES_ACTIVE),
            "svidag_num_iters": SVIDAG_NUM_ITERS,
            "seed": SEED,
            "algorithms_in_this_run": [l for l, _ in algorithms],
        },
        # Nested by metric mode: per_split["dag"]["ProDAG"] = [ {...}, ... ].
        "per_split": {
            mode: {lbl: rows for lbl, rows in per_split[mode].items()}
            for mode in METRIC_MODES_ACTIVE
        },
        "aggregated": {
            mode: {
                lbl: {k: {"mean": m, "se": s} for k, (m, s) in vals.items()}
                for lbl, vals in aggregated[mode].items()
            }
            for mode in METRIC_MODES_ACTIVE
        },
    }
    with open(json_path, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"\n  Saved JSON -> {json_path}")

    # ── console summary (one block per metric mode) ─────────────────────
    for mode in METRIC_MODES_ACTIVE:
        print("\n" + "=" * 80)
        print(f"  RESULTS for '{output_suffix}' -- {METRIC_MODE_LABELS[mode]} "
              f"metrics  (mean ± SE over splits)")
        print("=" * 80)
        header = (f"  {'Algorithm':32s} | {'Brier':>18s} | {'E[SHD]':>18s}"
                  f" | {'E[F1]%':>18s} | {'AUROC%':>18s}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for row_label, _ in algorithms:
            if row_label not in aggregated[mode]:
                print(f"  {row_label:32s} | {'--':>18s} | {'--':>18s} | "
                      f"{'--':>18s} | {'--':>18s}")
                continue
            v = aggregated[mode][row_label]

            def _c(key, pct, dec=3, _v=v):
                m, s = _v[key]
                if np.isnan(m):
                    return "--"
                if pct:
                    return f"{100 * m:6.2f} ± {100 * s:5.2f}"
                return f"{m:8.{dec}f} ± {s:6.{dec}f}"

            print(
                f"  {row_label:32s} | {_c('Brier', False):>18s} | "
                f"{_c('E_SHD', False, 2):>18s} | "
                f"{_c('E_F1', True):>18s} | {_c('AUROC', True):>18s}"
            )

    print("\n  Merge all per-algorithm JSONs into the two LaTeX tables with:")
    print("      python paper_results_reproduce/case_4/make_tables.py")

    return json_path
