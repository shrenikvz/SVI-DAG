#!/usr/bin/env python
"""
Case 4: Sachs benchmark (SVIDAG vs. 6 baselines)
=================================================

Runs SVIDAG (under the noninformative prior) and 6 baseline causal-discovery
algorithms on 10 equal random splits of the Sachs dataset (7,466 samples,
11 nodes).

For each (algorithm, split) pair we:
    1.  fit the algorithm on the training portion of the split;
    2.  draw ``NUM_POSTERIOR_SAMPLES`` relaxed adjacency matrices from the
        posterior;
    3.  threshold at ``THRESHOLD`` (default 0.5, matches svidag.config) and
        zero the diagonal;
    4.  score the SAME samples under BOTH metric families -- oriented-DAG
        metrics against the Sachs ground-truth DAG, and CPDAG metrics against
        its Markov-equivalence class (see ``METRIC_MODES`` below).

Both families are computed from a single fit, so the two tables describe
identical posterior draws and cost one training run, not two.

Outputs
-------
    case_4_results.json     -- per-split metrics + aggregated mean/SE, nested
                               under "dag" and "cpdag".
    case_4_table_dag.tex    -- ready-to-paste LaTeX table (DAG metrics).
    case_4_table_cpdag.tex  -- ready-to-paste LaTeX table (CPDAG metrics).
    case_4_table.tex        -- both tables concatenated, for convenience.

Usage
-----
    cd SVI-DAG-fast
    python paper_results_reproduce/case_4/run_case4.py

    # Or skip the unimplemented baselines (default when stubs raise):
    SKIP_UNIMPLEMENTED=1 python paper_results_reproduce/case_4/run_case4.py

Author: Shrenik Zinage
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Tuple

# Plumb sys.path -- this script should run from the repo root or anywhere.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
for p in (str(_REPO_ROOT), str(_REPO_ROOT / "src"), str(_THIS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from sklearn.preprocessing import StandardScaler

import common
from common import (
    METRIC_MODE_LABELS,
    METRIC_MODES,
    SachsData,
    aggregate_splits,
    build_latex_table,
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

# ===========================================================================
# Configuration
# ===========================================================================
NUM_SPLITS = 10
NUM_POSTERIOR_SAMPLES = 1000
SPLIT_MODE = "kfold"                 # "kfold" -> 10-fold CV (~6720 rows train / ~747 test). Best for BayesDAG, which was tuned on full Sachs and needs similar N to avoid posterior collapse. Other modes: "disjoint", "random_90_10".
SEED = 0
THRESHOLD = 0.5                      # baselines' relaxed outputs only (SVIDAG is binary)
# Metric families to score every fit under.  One LaTeX table is emitted per
# entry.  Scoring is post-hoc and cheap relative to training, so keeping both
# costs one extra confusion-matrix pass per (algorithm, split) -- not a second
# training run.  Narrow to e.g. ("cpdag",) only if you deliberately want a
# single table.
METRIC_MODES_ACTIVE = METRIC_MODES   # ("dag", "cpdag")
SVIDAG_NUM_ITERS = 60_000            # same as svidag.config.num_iters
SVIDAG_SCENARIOS = [
    "noninformative",
]
SKIP_UNIMPLEMENTED = bool(int(os.environ.get("SKIP_UNIMPLEMENTED", "1")))


# ===========================================================================
# Algorithm registry
# ===========================================================================
# Each entry maps a row label (as it appears in the LaTeX table) to:
#   - kind: "svidag" or "baseline"
#   - For "svidag": the scenario name
#   - For "baseline": a callable that takes (X_train, num_nodes, **kwargs)
#     and returns (A_samples [S, d, d], convention: "j_to_i" | "i_to_j")
#
# The ordering here matches the requested table.
# ---------------------------------------------------------------------------
ALGORITHMS: List[Tuple[str, Dict]] = [
    ("SVI-DAG (noninformative)",   {"kind": "svidag", "scenario": "noninformative"}),
    ("ProDAG",                     {"kind": "baseline", "fn": prodag_wrapper.run}),
    ("BayesDAG",                   {"kind": "baseline", "fn": bayesdag_wrapper.run}),
    ("DDS",                        {"kind": "baseline", "fn": dds_wrapper.run}),
    ("BCD Nets",                   {"kind": "baseline", "fn": bcd_wrapper.run}),
    ("DiBS",                       {"kind": "baseline", "fn": dibs_wrapper.run}),
]


# ===========================================================================
# Per-(algorithm, split) driver
# ===========================================================================
def run_one(
    row_label: str, spec: Dict,
    sachs: SachsData,
    train_idx: np.ndarray,
    split_index: int,
) -> Dict[str, Dict[str, float]]:
    """
    Train one algorithm on one split and score it under every active metric mode.

    Returns ``{mode: {"Brier": ..., "E_SHD": ..., "E_F1": ..., "AUROC": ...}}``.
    """
    # Fair-comparison step: standardise (z-score) the training split ONCE here
    # so every algorithm consumes inputs with the same per-feature scale.
    # Without this, SVIDAG's svidag_runner.dataset_from_arrays would be the
    # only pathway that z-scores -- the baselines (BCD, VI-DP-DAG in
    # particular) would see raw Sachs features whose variances differ by
    # orders of magnitude from their algorithm defaults.  This is a
    # pre-processing choice only; the adjacency metric is scale-invariant.
    #
    # SVIDAG's runner will re-fit a StandardScaler on already-standardised
    # data.  That re-fit is an idempotent no-op (mean ~ 0, std ~ 1), so the
    # SVIDAG result is unchanged relative to main.py's load_sachs_dataset.
    _scaler = StandardScaler().fit(sachs.X[train_idx])
    X_train = _scaler.transform(sachs.X[train_idx]).astype(np.float32)
    # We don't need a test split for the structure-learning metrics we report;
    # reuse the first row as a 1-sample placeholder so SVIDAG's Dataset stays
    # well-formed.  Transform with the SAME scaler to avoid leakage.
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
    else:  # baseline
        # BayesDAG runs its paper-spec lambda grid search inside the wrapper
        # and needs the true sparsity level (= ground-truth edge count) to
        # pick the closest-matching lambda.
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


# ===========================================================================
# LaTeX table generator
# ===========================================================================
# ``build_latex_table`` now lives in common.py so that make_tables.py can reuse
# it without importing this module (and with it every baseline's torch/jax
# dependency).  It is imported above; this shim keeps the call site here short.
def _table(aggregated_for_mode, mode: str) -> str:
    return build_latex_table(
        aggregated_for_mode,
        mode=mode,
        row_order=[lbl for lbl, _ in ALGORITHMS],
        num_splits=NUM_SPLITS,
    )


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 80)
    print(f"  Case 4: Sachs benchmark  (splits={NUM_SPLITS}, "
          f"S={NUM_POSTERIOR_SAMPLES}, threshold={THRESHOLD}, "
          f"metrics={list(METRIC_MODES_ACTIVE)})")
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

    # Accumulator: per_split[mode][algorithm_label] = list of metric dicts.
    per_split: Dict[str, Dict[str, List[Dict[str, float]]]] = {
        mode: {lbl: [] for lbl, _ in ALGORITHMS} for mode in METRIC_MODES_ACTIVE
    }

    for split in splits:
        print(f"\n  ── split {split.index} (train={len(split.train_idx)}, "
              f"test={len(split.test_idx)}) ──")
        for row_label, spec in ALGORITHMS:
            try:
                metrics_by_mode = run_one(
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
            for row_label, _ in ALGORITHMS
            if per_split[mode][row_label]
        }

    # ── save json ────────────────────────────────────────────────────────
    json_path = _THIS_DIR / "case_4_results.json"
    dump = {
        "config": {
            "num_splits": NUM_SPLITS,
            "num_posterior_samples": NUM_POSTERIOR_SAMPLES,
            "split_mode": SPLIT_MODE,
            "threshold": THRESHOLD,
            "metric_modes": list(METRIC_MODES_ACTIVE),
            "svidag_num_iters": SVIDAG_NUM_ITERS,
            "seed": SEED,
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
    print(f"\n  Saved JSON  -> {json_path}")

    # ── save LaTeX (one table per metric mode) ───────────────────────────
    tex_by_mode = {
        mode: _table(aggregated[mode], mode=mode)
        for mode in METRIC_MODES_ACTIVE
    }
    for mode, tex in tex_by_mode.items():
        tex_path = _THIS_DIR / f"case_4_table_{mode}.tex"
        with open(tex_path, "w") as f:
            f.write(tex)
        print(f"  Saved LaTeX -> {tex_path}")

    # Convenience: both tables in one file (also overwrites the pre-existing
    # single-table case_4_table.tex so no stale CPDAG-only copy is left behind).
    combined_path = _THIS_DIR / "case_4_table.tex"
    with open(combined_path, "w") as f:
        f.write("\n\n".join(tex_by_mode[m] for m in METRIC_MODES_ACTIVE) + "\n")
    print(f"  Saved LaTeX -> {combined_path}  (both tables)")

    # ── console summary (one block per metric mode) ─────────────────────
    for mode in METRIC_MODES_ACTIVE:
        print("\n" + "=" * 80)
        print(f"  RESULTS -- {METRIC_MODE_LABELS[mode]} metrics  "
              f"(mean ± SE over splits)")
        print("=" * 80)
        header = (f"  {'Algorithm':32s} | {'Brier':>18s} | {'E[SHD]':>18s}"
                  f" | {'E[F1]%':>18s} | {'AUROC%':>18s}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for row_label, _ in ALGORITHMS:
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
                    return f"{100*m:6.2f} ± {100*s:5.2f}"
                return f"{m:8.{dec}f} ± {s:6.{dec}f}"
            print(
                f"  {row_label:32s} | {_c('Brier', False):>18s} | "
                f"{_c('E_SHD', False, 2):>18s} | "
                f"{_c('E_F1', True):>18s} | {_c('AUROC', True):>18s}"
            )

    for mode in METRIC_MODES_ACTIVE:
        print("\n" + "=" * 80)
        print(f"  LaTeX table -- {METRIC_MODE_LABELS[mode]} metrics")
        print("=" * 80)
        print(tex_by_mode[mode])


if __name__ == "__main__":
    main()
