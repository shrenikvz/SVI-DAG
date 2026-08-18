#!/usr/bin/env python
"""
Case 3: per-algorithm driver (nonlinear synthetic DAGs, DAG metrics)
=====================================================================

Each ``run_<algo>_only.py`` calls ``run_subset(labels=[...], output_suffix=...)``
to fit one (or a related group of) algorithm(s) on every cell of the case 3
grid and write a long-form CSV (plus an aggregated JSON) with the four
**DAG-level** metrics (Brier, E[SHD], E[F1], AUROC).

Grid (defined in ``common.py``)
    scenarios   : ER_p25_s40  (1; nonlinear MLP SEM, paper-spec)
    sample sizes: 100, 316, 1000, 3162, 10000, 10, 32
                  (7; round(10**[2, 2.5, 3, 3.5, 4, 1, 1.5]) -- see common.py,
                   the list order is load-bearing and must not be sorted)
    replicates  : 0..NUM_REPLICATES-1                (5)
    algorithms  : 9 rows (3 SVIDAG + 6 baselines)

A task may own a slice of that grid rather than all of it -- see the grid
subsetting block below (``CASE3_SAMPLE_SIZES`` / ``CASE3_REP_START`` /
``CASE3_REP_END``), which is how ``run_case3.sh`` gives each array task one
(algorithm, n) block.

Each cell fits a fresh model on a fresh (graph, data) replicate.
Standardisation is applied before training (sklearn ``StandardScaler``) for
parity with case_2 / case_4.

Output (under ``paper_results_reproduce/case_3/``)
    case_3_results_<suffix>.csv  -- long-form rows for plotting (one per
                                    algorithm/scenario/n/replicate cell).
    case_3_results_<suffix>.json -- aggregated mean ± SE over replicates.

Author: Shrenik Zinage
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
_CASE4_DIR = _REPO_ROOT / "paper_results_reproduce" / "case_4"
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src"), str(_CASE4_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# Force the case-local dir to sys.path[0] so its ``common.py`` /
# ``svidag_runner.py`` shadow the case_4 ones (same module names exist in
# both directories; without this the conditional ``if _p not in sys.path``
# leaves _CASE4_DIR ahead of _THIS_DIR and the wrong ``common`` is imported).
_local = str(_THIS_DIR)
if _local in sys.path:
    sys.path.remove(_local)
sys.path.insert(0, _local)

import numpy as np
from sklearn.preprocessing import StandardScaler

from common import (
    SCENARIOS,
    SAMPLE_SIZES,
    SAMPLE_SIZE_LOG10,
    NUM_REPLICATES_DEFAULT,
    GraphScenario,
    SyntheticDataset,
    aggregate_replicates,
    evaluate_samples,
    generate_dataset,
)
from svidag_runner import run_svidag_synthetic

# Reuse the case_4 baseline wrappers as-is.
from baselines import (
    prodag_wrapper,
    dibs_wrapper,
    bayesdag_wrapper,
    bcd_wrapper,
    dds_wrapper,
)


# ---------------------------------------------------------------------------
# Config (must stay in sync across all run_<algo>_only.py wrappers)
# ---------------------------------------------------------------------------
# Env overrides let a single sbatch task narrow the grid (e.g. a 1-replicate
# SVI-DAG sweep) without touching the committed defaults that the already-run
# baseline jobs used.  Unset => paper-spec values.
NUM_REPLICATES = int(os.environ.get("CASE3_NUM_REPLICATES", NUM_REPLICATES_DEFAULT))
NUM_POSTERIOR_SAMPLES = int(os.environ.get("CASE3_POSTERIOR_SAMPLES", "1000"))


# ---------------------------------------------------------------------------
# Grid subsetting: one Slurm array task owns one (algorithm, n) block.
#
# Every baseline whose per-step cost is linear in the sample size (DDS, BCD,
# nonlinear DiBS, nonlinear ProDAG) costs 10-100x more at n = 10^4 / 10^5 than
# it did on the old 10..1000 grid, so one task per algorithm no longer fits in
# any walltime.  CASE3_SAMPLE_SIZES narrows the n loop and CASE3_REP_START/
# REP_END narrow the replicate loop; ``run_subset`` folds the narrowing into
# the output suffix so concurrent tasks never write the same CSV.  All unset
# => the full grid, i.e. exactly the previous single-task behaviour.
# ---------------------------------------------------------------------------
def _selected_sample_sizes() -> List[int]:
    raw = os.environ.get("CASE3_SAMPLE_SIZES", "").strip()
    if not raw:
        return list(SAMPLE_SIZES)
    want = {int(tok) for tok in raw.replace(",", " ").split()}
    unknown = sorted(want - set(SAMPLE_SIZES))
    if unknown:
        raise ValueError(
            f"CASE3_SAMPLE_SIZES={raw!r} requests {unknown}, which are not on "
            f"the case-3 grid {SAMPLE_SIZES}."
        )
    return [n for n in SAMPLE_SIZES if n in want]  # canonical order


SELECTED_SAMPLE_SIZES = _selected_sample_sizes()
REP_START = max(0, int(os.environ.get("CASE3_REP_START", "0")))
REP_END = min(NUM_REPLICATES, int(os.environ.get("CASE3_REP_END", str(NUM_REPLICATES))))
SELECTED_REPLICATES = list(range(REP_START, REP_END))


def _decorate_suffix(suffix: str) -> str:
    """Append the grid narrowing to the CSV suffix (no-op on the full grid)."""
    parts = [suffix]
    if SELECTED_SAMPLE_SIZES != list(SAMPLE_SIZES):
        parts.append("n" + "-".join(str(n) for n in SELECTED_SAMPLE_SIZES))
    if SELECTED_REPLICATES != list(range(NUM_REPLICATES)):
        parts.append(f"r{REP_START}-{REP_END - 1}")
    return "_".join(parts)
THRESHOLD = 0.5
USE_CPDAG = False                             # case_3 reports DAG metrics
SVIDAG_NUM_ITERS = int(os.environ.get("SVIDAG_NUM_ITERS", "60000"))
SEED = int(os.environ.get("CASE3_SEED", "0"))
SKIP_UNIMPLEMENTED = bool(int(os.environ.get("SKIP_UNIMPLEMENTED", "1")))


# ---------------------------------------------------------------------------
# Algorithm registry (matches case_4 / case_2 ordering)
#
# case_3's SEM is *nonlinear* (1-hidden-layer MLP, 10 ReLU units), so every
# baseline whose library ships a nonlinear mode gets it via "kwargs":
#   - ProDAG   -> ProDAG.fit_mlp        (hidden_layers=(10,), ReLU — paper spec)
#   - DiBS     -> JointDiBS + DenseNonlinearGaussian, hidden_layers=(10,)
# SVI-DAG (Bayesian MLP node models) and BayesDAG (nonlinear ICGNN default)
# are already nonlinear.  Two baselines have NO nonlinear option and remain
# knowingly misspecified on this case:
#   - BCD Nets: linear SEM by construction (no library option).
#   - DDS: the vendored vi-dp-dag JAX port raises NotImplementedError for
#     any ``ma_architecture != "linear"`` (the nonlinear autoencoder was
#     never ported), so the linear autoencoder is used.
# ---------------------------------------------------------------------------
ALL_ALGORITHMS: List[Tuple[str, Dict]] = [
    ("SVI-DAG (strong incorrect)", {"kind": "svidag", "scenario": "strong_incorrect"}),
    ("SVI-DAG (noninformative)",   {"kind": "svidag", "scenario": "noninformative"}),
    ("SVI-DAG (strong correct)",   {"kind": "svidag", "scenario": "strong_correct"}),
    ("ProDAG",                     {"kind": "baseline", "fn": prodag_wrapper.run,
                                    "kwargs": {"mode": "mlp"}}),
    ("BayesDAG",                   {"kind": "baseline", "fn": bayesdag_wrapper.run}),
    ("DDS",                        {"kind": "baseline", "fn": dds_wrapper.run}),
    ("BCD Nets",                   {"kind": "baseline", "fn": bcd_wrapper.run}),
    ("DiBS",                       {"kind": "baseline", "fn": dibs_wrapper.run,
                                    "kwargs": {"model": "nonlinear", "hidden_layers": (10,)}}),
]


# ---------------------------------------------------------------------------
# One-cell driver
# ---------------------------------------------------------------------------
def _run_one(
    row_label: str,
    spec: Dict,
    dataset: SyntheticDataset,
    cell_index: int,
    verbose_svidag: bool,
) -> Dict[str, float]:
    """Standardise the replicate's data, fit the algorithm, return metrics."""
    scaler = StandardScaler().fit(dataset.X)
    X_train_scaled = scaler.transform(dataset.X).astype(np.float32)

    t0 = time.time()
    if spec["kind"] == "svidag":
        A_samples, convention = run_svidag_synthetic(
            X_train_scaled=X_train_scaled,
            true_adj=dataset.true_adj,
            node_names=dataset.node_names,
            scenario=spec["scenario"],
            cell_index=cell_index,
            num_posterior_samples=NUM_POSTERIOR_SAMPLES,
            num_iters=SVIDAG_NUM_ITERS,
            seed=SEED,
            verbose=verbose_svidag,
        )
    else:
        # BayesDAG implements its paper-spec lambda grid search inside the
        # wrapper and needs the true sparsity level (number of edges in the
        # ground-truth DAG) to pick the closest-matching lambda.
        extra = dict(spec.get("kwargs", {}))
        if row_label == "BayesDAG":
            extra["true_sparsity"] = int(np.sum(dataset.true_adj != 0))
        A_samples, convention = spec["fn"](
            X_train=X_train_scaled,
            num_nodes=dataset.num_nodes,
            num_posterior_samples=NUM_POSTERIOR_SAMPLES,
            seed=SEED + cell_index,
            **extra,
        )
    dt = time.time() - t0

    metrics = evaluate_samples(
        A_relaxed_samples=A_samples,
        dataset=dataset,
        source_convention=convention,
        threshold=THRESHOLD,
    )
    metrics["time_sec"] = float(dt)

    print(
        f"    [{dataset.scenario_label} | n={dataset.num_samples:4d} | "
        f"rep={dataset.replicate}] "
        f"{row_label:32s} | Brier={metrics['Brier']:.3f} "
        f"| E[SHD]={metrics['E_SHD']:6.2f} "
        f"| E[F1]={metrics['E_F1']:.3f} "
        f"| AUROC={metrics['AUROC']:.3f} | {dt:.1f}s"
    )
    return metrics


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_subset(labels: List[str], output_suffix: str) -> Tuple[Path, Path]:
    """Run the given algorithms over the full case 3 grid and write CSV + JSON."""
    spec_map = {lbl: spec for lbl, spec in ALL_ALGORITHMS}
    unknown = [l for l in labels if l not in spec_map]
    if unknown:
        raise ValueError(f"Unknown algorithm labels: {unknown}. Valid: {list(spec_map)}")
    algorithms: List[Tuple[str, Dict]] = [(l, spec_map[l]) for l in labels]
    output_suffix = _decorate_suffix(output_suffix)

    print("=" * 80)
    print(f"  Case 3 (parallel job): suffix = {output_suffix!r}")
    print(f"    algorithms          = {[l for l, _ in algorithms]}")
    print(f"    scenarios           = {[s.label for s in SCENARIOS]}")
    print(f"    sample_sizes        = {SAMPLE_SIZES}  (10**{SAMPLE_SIZE_LOG10})")
    print(f"    this task runs n    = {SELECTED_SAMPLE_SIZES}")
    print(f"    this task runs reps = {SELECTED_REPLICATES}")
    print(f"    num_replicates      = {NUM_REPLICATES}")
    print(f"    num_samples (post)  = {NUM_POSTERIOR_SAMPLES}")
    print(f"    threshold = {THRESHOLD}   use_cpdag = {USE_CPDAG}  "
          f"(case_3 -> DAG metrics)")
    print(f"    svidag_num_iters = {SVIDAG_NUM_ITERS}   seed = {SEED}")
    print("=" * 80)

    csv_path = _THIS_DIR / f"case_3_results_{output_suffix}.csv"
    json_path = _THIS_DIR / f"case_3_results_{output_suffix}.json"

    csv_columns = [
        "scenario", "p", "s", "num_samples", "sample_size_log10",
        "replicate", "algorithm", "Brier", "E_SHD", "E_F1", "AUROC", "time_sec",
    ]
    raw_rows: List[Dict] = []

    # ``cell_index`` seeds the baselines (``seed=SEED + cell_index``) and the
    # SVI-DAG runner, so it MUST be a pure function of the cell rather than a
    # running counter -- otherwise a task that runs only n=10^5 would seed that
    # cell differently from the full-grid task that used to run it, and the
    # paired cross-algorithm comparison would silently drift.  With one
    # scenario and one algorithm per job this reproduces the old counter
    # exactly (n_idx * NUM_REPLICATES + rep).
    def _cell_index(sc_idx: int, n_idx: int, rep: int, algo_idx: int) -> int:
        idx = sc_idx * len(SAMPLE_SIZES) + n_idx
        idx = idx * NUM_REPLICATES + rep
        return idx * len(algorithms) + algo_idx

    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=csv_columns)
        writer.writeheader()

        for sc_idx, sc in enumerate(SCENARIOS):
            for n in SELECTED_SAMPLE_SIZES:
                n_idx = SAMPLE_SIZES.index(n)
                for rep in SELECTED_REPLICATES:
                    print(
                        f"\n  ── {sc.label} | n={n} | rep={rep} "
                        f"(p={sc.p}, s={sc.s}, sem={sc.sem_type}) ──"
                    )
                    dataset = generate_dataset(sc, num_samples=n, replicate=rep)

                    for algo_idx, (row_label, spec) in enumerate(algorithms):
                        try:
                            metrics = _run_one(
                                row_label=row_label, spec=spec,
                                dataset=dataset,
                                cell_index=_cell_index(sc_idx, n_idx, rep, algo_idx),
                                verbose_svidag=(rep == REP_START and n_idx == 0),
                            )
                        except NotImplementedError as e:
                            if SKIP_UNIMPLEMENTED:
                                print(f"    {row_label:32s} | SKIPPED (stub): "
                                      f"{e.args[0].splitlines()[0]}")
                                continue
                            raise
                        except Exception as e:  # noqa: BLE001
                            print(f"    {row_label:32s} | FAILED: "
                                  f"{type(e).__name__}: {e}")
                            traceback.print_exc()
                            continue

                        row = {
                            "scenario": sc.label,
                            "p": sc.p,
                            "s": sc.s,
                            "num_samples": n,
                            "sample_size_log10": SAMPLE_SIZE_LOG10[n_idx],
                            "replicate": rep,
                            "algorithm": row_label,
                            **{k: metrics[k] for k in
                               ("Brier", "E_SHD", "E_F1", "AUROC", "time_sec")},
                        }
                        writer.writerow(row)
                        f_csv.flush()
                        raw_rows.append(row)

    # ── aggregate per (algorithm, scenario, n) ──
    aggregated: Dict[str, Dict[str, Dict[int, Dict[str, Tuple[float, float]]]]] = {}
    for row_label, _ in algorithms:
        aggregated[row_label] = {}
        for sc in SCENARIOS:
            aggregated[row_label][sc.label] = {}
            for n in SELECTED_SAMPLE_SIZES:
                cell_rows = [
                    r for r in raw_rows
                    if r["algorithm"] == row_label
                    and r["scenario"] == sc.label
                    and r["num_samples"] == n
                ]
                if not cell_rows:
                    continue
                metrics_list = [
                    {k: r[k] for k in ("Brier", "E_SHD", "E_F1", "AUROC")}
                    for r in cell_rows
                ]
                aggregated[row_label][sc.label][n] = aggregate_replicates(metrics_list)

    dump = {
        "config": {
            "num_replicates": NUM_REPLICATES,
            "sample_sizes": SAMPLE_SIZES,
            "sample_size_log10": SAMPLE_SIZE_LOG10,
            "sample_sizes_in_this_run": SELECTED_SAMPLE_SIZES,
            "replicates_in_this_run": SELECTED_REPLICATES,
            "scenarios": [
                {"label": s.label, "p": s.p, "s": s.s,
                 "sem_type": s.sem_type, "weight_range": list(s.weight_range),
                 "noise_scale": s.noise_scale}
                for s in SCENARIOS
            ],
            "num_posterior_samples": NUM_POSTERIOR_SAMPLES,
            "threshold": THRESHOLD,
            "use_cpdag": USE_CPDAG,
            "svidag_num_iters": SVIDAG_NUM_ITERS,
            "seed": SEED,
            "algorithms_in_this_run": [l for l, _ in algorithms],
        },
        "aggregated": {
            algo: {
                sc_label: {
                    str(n): {k: {"mean": m, "se": s} for k, (m, s) in vals.items()}
                    for n, vals in by_n.items()
                }
                for sc_label, by_n in by_sc.items()
            }
            for algo, by_sc in aggregated.items()
        },
    }
    with open(json_path, "w") as f:
        json.dump(dump, f, indent=2)

    print(f"\n  Saved CSV  -> {csv_path}")
    print(f"  Saved JSON -> {json_path}")

    # ── short console summary ──
    print("\n" + "=" * 80)
    print(f"  CASE 3 SUMMARY for '{output_suffix}'  (mean ± SE over "
          f"{NUM_REPLICATES} replicates; DAG metrics)")
    print("=" * 80)
    for algo in [l for l, _ in algorithms]:
        for sc in SCENARIOS:
            print(f"\n  {algo}   |   {sc.label}")
            print(f"  {'n':>6s}  | {'Brier':>14s} | {'E[SHD]':>14s} | "
                  f"{'E[F1]%':>14s} | {'AUROC%':>14s}")
            for n in SELECTED_SAMPLE_SIZES:
                cell = aggregated.get(algo, {}).get(sc.label, {}).get(n)
                if cell is None:
                    print(f"  {n:>6d}  | {'--':>14s} | {'--':>14s} | "
                          f"{'--':>14s} | {'--':>14s}")
                    continue

                def _c(key, pct, dec=3):
                    m, se = cell[key]
                    if np.isnan(m):
                        return "--"
                    if pct:
                        return f"{100 * m:5.2f} ± {100 * se:4.2f}"
                    return f"{m:6.{dec}f} ± {se:6.{dec}f}"

                print(f"  {n:>6d}  | {_c('Brier', False):>14s} | "
                      f"{_c('E_SHD', False, 2):>14s} | "
                      f"{_c('E_F1', True):>14s} | {_c('AUROC', True):>14s}")
    return csv_path, json_path
