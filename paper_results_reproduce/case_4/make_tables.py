#!/usr/bin/env python
"""
Case 4: merge per-algorithm result JSONs into the two LaTeX tables.
===================================================================

The parallel workflow (``run_<algo>_only.py`` -> one SLURM job per algorithm)
writes ``case_4_results_<suffix>.json`` files but no tables, because no single
job knows about all the rows.  This script collects whatever result files are
present and emits:

    case_4_table_dag.tex    -- oriented-DAG metrics
    case_4_table_cpdag.tex  -- CPDAG (Markov-equivalence-class) metrics
    case_4_table.tex        -- both, concatenated

Algorithms with no completed splits render as "--" rows, so a partial sweep
still produces compilable tables.

Deliberately imports only ``common`` (numpy / pandas / networkx / svidag.utils)
and never the baseline wrappers, so it runs on a login node without torch,
jax-cuda, or the baselines' conflicting environments.

Usage
-----
    python paper_results_reproduce/case_4/make_tables.py
    python paper_results_reproduce/case_4/make_tables.py --results-dir /some/dir

Author: Shrenik Zinage
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
for p in (str(_REPO_ROOT), str(_REPO_ROOT / "src"), str(_THIS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from common import (
    METRIC_MODE_LABELS,
    METRIC_MODES,
    build_latex_table,
    aggregate_splits,
)

# ---------------------------------------------------------------------------
# Row order -- must match ALGORITHMS in run_case4.py / ALL_ALGORITHMS in
# _single_algo.py.  Kept as plain strings so this module stays import-light.
# ---------------------------------------------------------------------------
ROW_ORDER: List[str] = [
    "SVI-DAG (noninformative)",
    "ProDAG",
    "BayesDAG",
    "DDS",
    "BCD Nets",
    "DiBS",
]

METRIC_KEYS = ("Brier", "E_SHD", "E_F1", "AUROC")

# Canonical output suffixes, one per ``run_<algo>_only.py`` entry point (the
# array-task names in run_case4_all.sh).  Only these files feed the tables.
#
# An explicit allow-list rather than a bare ``case_4_results_*.json`` glob: ad
# hoc sweeps write files like ``case_4_results_svidag_gate.json`` into the same
# directory, and because loading is ordered by filename such a file sorts
# AFTER ``case_4_results_svidag.json`` and would silently overwrite the real
# SVI-DAG rows with tuning-run numbers.
CANONICAL_SUFFIXES = ("svidag", "prodag", "bayesdag", "dds", "dibs", "bcd")


# ---------------------------------------------------------------------------
# Loading (handles both the current nested schema and the legacy flat one)
# ---------------------------------------------------------------------------
def _modes_in_file(payload: dict) -> List[str]:
    """
    Determine which metric families a result file carries.

    Current schema: ``per_split``/``aggregated`` are keyed by mode first
    ("dag"/"cpdag").  Legacy schema (single-family runs written before case 4
    reported both): keyed by algorithm label directly, with the family recorded
    as ``config.use_cpdag``.  We detect by inspecting the top-level keys.
    """
    agg = payload.get("aggregated", {})
    per = payload.get("per_split", {})
    probe = agg if agg else per
    if probe and all(k in METRIC_MODES for k in probe):
        return [k for k in METRIC_MODES if k in probe]
    # Legacy: infer the single family from the config flag.
    legacy_mode = "cpdag" if payload.get("config", {}).get("use_cpdag") else "dag"
    return [f"legacy:{legacy_mode}"]


def load_results(results_dir: Path) -> Tuple[
    Dict[str, Dict[str, Dict[str, Tuple[float, float]]]], List[str]
]:
    """
    Merge every ``case_4_results_*.json`` in ``results_dir``.

    Returns ``(aggregated[mode][row_label][metric] = (mean, se), notes)``.
    Metrics are recomputed from ``per_split`` when available so the mean/SE
    always match the splits actually stored in the file; the file's own
    ``aggregated`` block is used only as a fallback.
    """
    aggregated: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {
        mode: {} for mode in METRIC_MODES
    }
    notes: List[str] = []

    found = sorted(results_dir.glob("case_4_results_*.json"))
    files = [p for p in found
             if p.stem[len("case_4_results_"):] in CANONICAL_SUFFIXES]
    for p in found:
        if p not in files:
            notes.append(
                f"{p.name}: SKIPPED (not one of the canonical per-algorithm "
                f"outputs {CANONICAL_SUFFIXES}) -- ad hoc sweep files are "
                f"ignored so they cannot overwrite a real row"
            )
    if not files:
        raise FileNotFoundError(
            f"No canonical case_4_results_<algo>.json files found in "
            f"{results_dir} (looked for suffixes {CANONICAL_SUFFIXES}). "
            f"Run the per-algorithm jobs first (e.g. run_svidag_only.py)."
        )

    for path in files:
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            notes.append(f"{path.name}: SKIPPED (malformed JSON: {e})")
            continue

        for raw_mode in _modes_in_file(payload):
            legacy = raw_mode.startswith("legacy:")
            mode = raw_mode.split(":", 1)[1] if legacy else raw_mode

            per_split = payload.get("per_split", {})
            file_agg = payload.get("aggregated", {})
            if not legacy:
                per_split = per_split.get(mode, {})
                file_agg = file_agg.get(mode, {})

            if legacy:
                notes.append(
                    f"{path.name}: legacy single-family file, counted as "
                    f"{METRIC_MODE_LABELS[mode]} only"
                )

            for label, rows in per_split.items():
                if label not in ROW_ORDER:
                    notes.append(f"{path.name}: unknown row label {label!r}, ignored")
                    continue
                if not rows:
                    notes.append(
                        f"{path.name}: {label} has 0 completed splits "
                        f"({METRIC_MODE_LABELS[mode]}) -- row will render as '--'"
                    )
                    continue
                if label in aggregated[mode]:
                    notes.append(
                        f"{path.name}: {label} ({METRIC_MODE_LABELS[mode]}) "
                        f"already loaded from an earlier file -- overwriting"
                    )
                aggregated[mode][label] = aggregate_splits(rows)
                if len(rows) < payload.get("config", {}).get("num_splits", len(rows)):
                    notes.append(
                        f"{path.name}: {label} ({METRIC_MODE_LABELS[mode]}) "
                        f"has only {len(rows)}/"
                        f"{payload['config']['num_splits']} splits"
                    )

            # Fallback for files that carry aggregates but no per-split rows.
            for label, vals in file_agg.items():
                if label in aggregated[mode] or label not in ROW_ORDER:
                    continue
                aggregated[mode][label] = {
                    k: (v["mean"], v["se"]) for k, v in vals.items()
                }

    return aggregated, notes


# ---------------------------------------------------------------------------
# Console rendering
# ---------------------------------------------------------------------------
def print_summary(aggregated: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]):
    for mode in METRIC_MODES:
        print("\n" + "=" * 80)
        print(f"  {METRIC_MODE_LABELS[mode]} metrics  (mean ± SE over splits)")
        print("=" * 80)
        header = (f"  {'Algorithm':32s} | {'Brier':>18s} | {'E[SHD]':>18s}"
                  f" | {'E[F1]%':>18s} | {'AUROC%':>18s}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for label in ROW_ORDER:
            if label not in aggregated[mode]:
                print(f"  {label:32s} | {'--':>18s} | {'--':>18s} | "
                      f"{'--':>18s} | {'--':>18s}")
                continue
            v = aggregated[mode][label]

            def _c(key, pct, dec=3, _v=v):
                m, s = _v[key]
                if np.isnan(m):
                    return "--"
                if pct:
                    return f"{100 * m:6.2f} ± {100 * s:5.2f}"
                return f"{m:8.{dec}f} ± {s:6.{dec}f}"

            print(
                f"  {label:32s} | {_c('Brier', False):>18s} | "
                f"{_c('E_SHD', False, 2):>18s} | "
                f"{_c('E_F1', True):>18s} | {_c('AUROC', True):>18s}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_runtime_table(
    aggregated: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]],
    row_order: List[str],
    num_splits: int,
) -> str:
    """
    LaTeX table of wall-clock time per algorithm: mean +/- SE over splits.

    ``time_sec`` is recorded once per (algorithm, split) in ``_single_algo.py``
    and carried on both metric-mode dicts, so either mode gives the same
    numbers; DAG is read here. The fastest algorithm is bolded.

    Times are hardware-dependent and are reported only for relative comparison
    -- they are the one quantity in this benchmark that will not reproduce
    exactly on different hardware.
    """
    src = aggregated.get("dag", {})
    present = [r for r in row_order if r in src and "time_sec" in src[r]]

    best = None
    if present:
        best = min(present, key=lambda r: src[r]["time_sec"][0])

    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\caption{Wall-clock time per train/evaluate run on the Sachs dataset. "
        r"Mean and standard error over " + str(num_splits) + r" splits. "
        r"Lower is better; the fastest is in bold. Timings are hardware "
        r"dependent and are given for relative comparison only.}",
        r"\label{tab:sachs_runtime}",
        r"\small",
        r"\begin{tabularx}{\linewidth}{Xr}",
        r"\toprule",
        r" & Time per run (s) \\",
        r"\midrule",
    ]
    for row in row_order:
        vals = src.get(row)
        if not vals or "time_sec" not in vals:
            lines.append(r"\texttt{" + row + r"} & -- \\")
            continue
        m, se = vals["time_sec"]
        cell = f"{m:.1f} $\\pm$ {se:.1f}"
        if row == best:
            cell = r"\textbf{" + cell + r"}"
        lines.append(r"\texttt{" + row + r"} & " + cell + r" \\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-dir", type=Path, default=_THIS_DIR,
        help="Directory holding case_4_results_*.json (default: this script's dir)",
    )
    ap.add_argument(
        "--num-splits", type=int, default=10,
        help="Split count quoted in the table caption (default: 10)",
    )
    args = ap.parse_args()

    aggregated, notes = load_results(args.results_dir)

    print("=" * 80)
    print(f"  Case 4: building tables from {args.results_dir}")
    print("=" * 80)
    for n in notes:
        print(f"  NOTE  {n}")

    tex_by_mode = {
        mode: build_latex_table(
            aggregated[mode], mode=mode,
            row_order=ROW_ORDER, num_splits=args.num_splits,
        )
        for mode in METRIC_MODES
    }

    for mode, tex in tex_by_mode.items():
        out = args.results_dir / f"case_4_table_{mode}.tex"
        out.write_text(tex)
        print(f"\n  Saved -> {out}")

    runtime_tex = build_runtime_table(
        aggregated, row_order=ROW_ORDER, num_splits=args.num_splits
    )
    runtime_out = args.results_dir / "case_4_table_runtime.tex"
    runtime_out.write_text(runtime_tex)
    print(f"  Saved -> {runtime_out}")

    combined = args.results_dir / "case_4_table.tex"
    combined.write_text(
        "\n\n".join(tex_by_mode[m] for m in METRIC_MODES) + "\n\n" + runtime_tex + "\n"
    )
    print(f"  Saved -> {combined}  (both metric tables + runtime)")

    print_summary(aggregated)

    for mode in METRIC_MODES:
        print("\n" + "=" * 80)
        print(f"  LaTeX table -- {METRIC_MODE_LABELS[mode]} metrics")
        print("=" * 80)
        print(tex_by_mode[mode])

    print("\n" + "=" * 80)
    print("  LaTeX table -- computational time")
    print("=" * 80)
    print(runtime_tex)


if __name__ == "__main__":
    main()
