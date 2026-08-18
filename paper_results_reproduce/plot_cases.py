#!/usr/bin/env python
"""
Metrics figures for the synthetic benchmark cases (2, 3, 5, 6)
==============================================================

Loads every ``case_<N>_results_*.csv`` written by the per-algorithm jobs and
renders one 4-panel figure per (case, scenario):

    Brier score | Expected SHD | Expected F1 (%) | AUROC (%)

versus sample size (log axis, n = 10^2 .. 10^4 in half-decades), one line per algorithm with
mean ± standard-error bars over replicates — the paper's benchmark figure
layout (Computer Modern serif, square markers, marker-only legend along the
bottom; see ``_PAPER_RCPARAMS`` and ``SERIES_STYLE``).

Usage
-----
    python paper_results_reproduce/plot_cases.py                  # all cases
    python paper_results_reproduce/plot_cases.py --cases 3 6      # subset
    python paper_results_reproduce/plot_cases.py --all-svidag     # also plot
                                                # the correct/incorrect priors

Outputs, per case: ``paper_results_reproduce/case_<N>/<scenario>_metrics.pdf``
(and ``.png``).  Writing is atomic (tmp file + rename) so the figure is never
half-written even when several SLURM array tasks finish simultaneously —
each ``run_case<N>_all.sh`` array task re-runs this script on completion, so
the figure always reflects every algorithm whose CSV exists at that moment,
and the last task to finish leaves the complete plot.

Cases 1 and 4 are intentionally out of scope (2-node prior study; Sachs
table).

Author: Shrenik Zinage
"""

from __future__ import annotations

import argparse
import functools
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless nodes
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent

# Cases 5 and 6 are the p=50 / s=80 counterparts of Cases 2 and 3. They use
# the same long-form CSV schema, so they plot through the same path.
PLOTTABLE_CASES = [2, 3, 5, 6]

# Canonical output suffixes, one per ``run_<algo>_only.py`` entry point (the
# array-task names in run_case<N>_all.sh).  Only these CSVs feed a figure.
#
# An explicit allow-list rather than a bare ``case_<N>_results_*.csv`` glob:
# ad hoc files written next to the real ones (e.g. a
# ``case_<N>_results_<algo>_BACKUP.csv`` saved during a sweep) also match that
# glob, and since frames are concatenated in filename order and de-duplicated
# with ``keep="last"``, such a file sorts AFTER the canonical one ('.' < '_')
# and silently REPLACES every one of its rows in the figure.  This has bitten
# a committed figure before; keep the allow-list.
CANONICAL_ALGOS = ("svidag", "prodag", "bayesdag", "dds", "dibs", "bcd")

# A suffix is canonical if it is one of the algorithm names above, optionally
# followed by the grid-narrowing tags ``_single_algo._decorate_suffix`` appends
# when one Slurm task owns a slice of the grid:
#     case_2_results_dds.csv                 (whole grid, one task)
#     case_2_results_dds_n100000.csv         (one (algo, n) block)
#     case_2_results_dds_n100000_r0-4.csv    (n block split by replicate)
CANONICAL_SUFFIX_RE = re.compile(
    r"^(?:" + "|".join(CANONICAL_ALGOS) + r")"
    r"(?:_n\d+(?:-\d+)*)?"
    r"(?:_r\d+-\d+)?$"
)

# ---------------------------------------------------------------------------
# Figure styling
#
# Everything below reproduces the paper's benchmark figure exactly: Computer
# Modern serif text (matplotlib ships cmr10 + the `cm` mathtext set, so no
# LaTeX install is needed), square markers on solid lines, a dotted grey grid,
# no per-panel titles, and a single-row marker-only legend under the panels.
# ---------------------------------------------------------------------------
_PAPER_RCPARAMS = {
    # cmr10 is metric-compatible with the LaTeX body font; the `cm` mathtext
    # set makes $10^{1}$ / $F_1$ render in the same face as the surrounding
    # text instead of DejaVu.
    "font.family": "serif",
    "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    # cmr10 has no U+2212 MINUS SIGN glyph; without this every negative tick
    # label emits a missing-glyph warning and renders as a box.
    "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": True,
    "font.size": 18,
    "axes.labelsize": 21,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 20,
    # Borderless panels: no frame and no tick marks, so the dotted grid is the
    # only structure behind the series.
    "axes.spines.left": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.spines.bottom": False,
    "xtick.major.size": 0.0,
    "ytick.major.size": 0.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "grid.color": "#b0b0b0",
    "grid.linestyle": ":",
    "grid.linewidth": 0.7,
    "grid.alpha": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}

# (display label, color, marker, linestyle) per CSV algorithm label.
# Order here fixes the legend order.  Colours are the paper figure's palette.
SERIES_STYLE: Dict[str, Tuple[str, str, str, str]] = {
    # Legend shows plain "SVI-DAG" -- no prior named, by user request.
    "SVI-DAG (noninformative)":   ("SVI-DAG",            "#d6392f", "s", "-"),
    "SVI-DAG (strong correct)":   ("SVIDAG (Corr.)",     "#8b0000", "s", "--"),
    "SVI-DAG (strong incorrect)": ("SVIDAG (Inc.)",      "#f4a582", "s", ":"),
    "ProDAG":                     ("ProDAG",             "#2e4a9b", "s", "-"),
    "BayesDAG":                   ("BayesDAG",           "#7b57a6", "s", "-"),
    "DiBS":                       ("DiBS",               "#4ec3d9", "s", "-"),
    "DDS":                        ("DDS",                "#7f7f7f", "s", "-"),
    "BCD Nets":                   ("BCD Nets",           "#c7ae3e", "s", "-"),
}
DEFAULT_SERIES = [
    "SVI-DAG (noninformative)", "ProDAG",
    "BayesDAG", "DiBS", "DDS", "BCD Nets",
]

# (csv column, y label, plot as percentage, y-axis scale)
#
# Expected SHD is drawn on a LOG y-axis.  DDS starts an order of magnitude
# above everyone else in the small-sample regime (672 at n=10 against 85-177
# for the other five) and decays to 81, so a linear axis has to span 0-700 and
# squeezes the other five algorithms into the bottom ~17% of the panel, where
# their curves overlap into a single band.  A log axis gives that band ~45% of
# the panel height instead and keeps DDS's decay on the same plot -- the range
# is only 10.9x (62.3 to 677.3), so one decade of log covers everything with no
# risk of a zero or negative value (expected SHD is a positive count).
PANELS = [
    ("Brier", "Brier score",              False, "linear"),
    ("E_SHD", "Expected SHD (log scale)", False, "log"),
    ("E_F1",  "Expected $F_1$ score (%)", True,  "linear"),
    ("AUROC", "AUROC (%)",                True,  "linear"),
]

# All four sweep cases (2, 3, 5, 6) were extended downward by two half-decades
# to 10^1 and 10^1.5, so the grid is global again.  Ascending, as the x-axis
# requires -- unlike case_<N>/common.py, where the list ORDER fixes the RNG
# seed per cell and must not be sorted.
SAMPLE_SIZES = [10, 32, 100, 316, 1000, 3162, 10000]
XTICK_LABELS = ["$10^{1.0}$", "$10^{1.5}$", "$10^{2.0}$", "$10^{2.5}$",
                "$10^{3.0}$", "$10^{3.5}$", "$10^{4.0}$"]


def sample_grid(case_num: int):
    """(sample sizes, x-tick labels) for one case, ascending."""
    return SAMPLE_SIZES, XTICK_LABELS


# Candidate mantissas for log-axis ticks, densest first.  _log_tick_values
# walks these and takes the first that yields few enough ticks to label at the
# panel's full font size.
_LOG_TICK_LADDERS = (
    (1, 1.5, 2, 3, 4, 5, 7),
    (1, 1.5, 2, 3, 5, 7),
    (1, 2, 3, 5, 7),
    (1, 2, 4, 7),
    (1, 2, 5),
    (1, 3),
    (1,),
)


def _log_tick_values(lo: float, hi: float, max_ticks: int = 6) -> List[float]:
    """
    Tick values for a log axis spanning [lo, hi], as plain numbers.

    Derived from the data rather than hardcoded: expected SHD spans a different
    range in every case (roughly 22-101 in case 2, 62-677 in case 6), so one
    fixed ladder cannot serve all of them.  Returns at most ``max_ticks``
    values, which is what keeps the labels legible at the panel's full font
    size instead of needing a smaller one.
    """
    if not (lo > 0 and hi > lo):
        return []
    k0 = int(np.floor(np.log10(lo)))
    k1 = int(np.ceil(np.log10(hi)))
    chosen: List[float] = []
    for subs in _LOG_TICK_LADDERS:
        ticks = sorted(s * 10.0 ** k for k in range(k0, k1 + 1) for s in subs)
        ticks = [t for t in ticks if lo <= t <= hi]
        chosen = ticks or chosen
        if 2 <= len(ticks) <= max_ticks:
            return ticks
    return chosen


def _fmt_tick(v: float, _pos=None) -> str:
    """Plain integer when the value is one ('70', '150'), else one decimal."""
    return f"{v:.0f}" if float(v).is_integer() else f"{v:g}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_case_results(case_num: int) -> Optional[pd.DataFrame]:
    """Concatenate every results CSV of one case; None if none exist yet."""
    case_dir = _THIS_DIR / f"case_{case_num}"
    prefix = f"case_{case_num}_results_"
    found = sorted(case_dir.glob(f"{prefix}*.csv"))
    csv_paths = [p for p in found
                 if CANONICAL_SUFFIX_RE.match(p.stem[len(prefix):])]
    for p in found:
        if p not in csv_paths:
            print(f"  [case {case_num}] ignoring non-canonical {p.name} "
                  f"(suffix does not match {CANONICAL_SUFFIX_RE.pattern})")
    if not csv_paths:
        return None

    frames = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:  # noqa: BLE001  (a half-written CSV must not kill the plot)
            print(f"  [case {case_num}] skipping unreadable {p.name}: {e}")
            continue
        needed = {"scenario", "num_samples", "replicate", "algorithm",
                  "Brier", "E_SHD", "E_F1", "AUROC"}
        if not needed.issubset(df.columns):
            print(f"  [case {case_num}] skipping {p.name}: missing columns")
            continue
        frames.append(df)
    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    # If the same (algorithm, cell) appears in several CSVs (e.g. a rerun
    # alongside an old file), keep the last occurrence.
    df = df.drop_duplicates(
        subset=["scenario", "num_samples", "replicate", "algorithm"],
        keep="last",
    )
    return df


def mean_se(vals: np.ndarray) -> Tuple[float, float]:
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
    return mean, se


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _with_paper_style(fn):
    """Render under the paper rcParams without mutating matplotlib globally."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with matplotlib.rc_context(_PAPER_RCPARAMS):
            return fn(*args, **kwargs)
    return wrapper


@_with_paper_style
def plot_case_scenario(
    df: pd.DataFrame,
    case_num: int,
    scenario: str,
    series: List[str],
    out_dir: Path,
) -> Optional[Path]:
    """Render the 4-panel figure for one (case, scenario); None if no data."""
    sub = df[df["scenario"] == scenario]
    present = [s for s in series if s in set(sub["algorithm"])]
    if not present:
        return None

    sample_sizes, xtick_labels = sample_grid(case_num)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))

    labels: List[str] = []
    handles: List[Line2D] = []
    for ax, (col, ylabel, as_pct, yscale) in zip(axes, PANELS):
        for algo in present:
            label, color, marker, ls = SERIES_STYLE[algo]
            xs, ys, es = [], [], []
            for n in sample_sizes:
                vals = sub[(sub["algorithm"] == algo)
                           & (sub["num_samples"] == n)][col].to_numpy(dtype=float)
                if vals.size == 0:
                    continue
                m, se = mean_se(vals)
                if np.isnan(m):
                    continue
                scale = 100.0 if as_pct else 1.0
                xs.append(n)
                ys.append(m * scale)
                es.append(se * scale)
            if not xs:
                continue
            # `capsize` draws the horizontal end caps on each error bar.
            #
            # Do NOT pass `markeredgewidth` here.  Matplotlib builds the caps
            # as `_`-marker artists -- i.e. pure edge strokes -- and an
            # explicit `markeredgewidth` in the errorbar kwargs is applied to
            # them too, overriding `capthick`.  Setting it to 0 for the square
            # data markers therefore renders every cap invisible.
            ax.errorbar(
                xs, ys, yerr=es,
                color=color, marker=marker, linestyle=ls,
                markersize=6.0, linewidth=1.6,
                capsize=4.0, elinewidth=1.2, capthick=1.2, zorder=3,
            )
            if label not in labels:
                # Marker-only proxy: the paper legend shows a bare square per
                # algorithm, not a line-through-marker sample.
                handles.append(Line2D(
                    [], [], color=color, marker=marker, linestyle="none",
                    markersize=10.0, markeredgewidth=0.0,
                ))
                labels.append(label)

        ax.set_xscale("log")
        ax.set_xticks(sample_sizes)
        ax.set_xticklabels(xtick_labels)
        ax.minorticks_off()
        ax.set_xlabel("Sample size")
        ax.set_ylabel(ylabel)

        if yscale == "log":
            ax.set_yscale("log")
            # minorticks_off() above cleared BOTH axes; a log y-axis spanning
            # barely one decade needs its minor ticks back or the panel shows
            # only 10^2, leaving the 62-180 cluster unlabelled.  Plain integers
            # rather than 10^x: these are SHD counts, and readers compare them
            # to the edge count (s=80), not to powers of ten.
            # Every LABELLED tick is a major tick, so it inherits
            # ytick.labelsize exactly like the three linear panels -- nothing
            # here overrides the font size.  A plain LogLocator would put a
            # single major at 100 and force the rest onto minor ticks, which
            # would then need a smaller size to fit.
            #
            # Tick values come from the plotted range, not a fixed list: these
            # panels span 22-101 in case 2 but 62-677 in case 6, and a ladder
            # picked for one case leaves the other with two labels.  Plain
            # integers, not "7 x 10^1" -- readers compare expected SHD against
            # the edge count s.
            ax.yaxis.set_major_locator(
                mticker.FixedLocator(_log_tick_values(*ax.get_ylim()))
            )
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_tick))
            # Unlabelled minors purely so the dotted grid still reads as a log
            # axis between the labelled values.
            ax.yaxis.set_minor_locator(
                mticker.LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1))
            )
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
            # _PAPER_RCPARAMS zeroes ytick.major.size for borderless panels, but
            # not the minor size (matplotlib defaults it to 2.0), which would
            # put tick marks on this panel alone.
            ax.tick_params(axis="y", which="minor", length=0.0)
            # Grid on minor ticks too, else the band between decades reads flat.
            ax.grid(True, which="minor", axis="y", linewidth=0.4, alpha=0.5,
                    zorder=0)

        ax.grid(True, which="major", zorder=0)
        ax.set_axisbelow(True)

    fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, -0.12),
        ncol=len(labels), frameon=False,
        handletextpad=0.6, columnspacing=3.2,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{scenario}_metrics"
    final_pdf = out_dir / f"{stem}.pdf"
    for ext in ("pdf", "png"):
        final = out_dir / f"{stem}.{ext}"
        tmp = out_dir / f".{stem}.{ext}.tmp"
        fig.savefig(tmp, format=ext, bbox_inches="tight", dpi=200)
        os.replace(tmp, final)  # atomic: concurrent array tasks never corrupt it
    plt.close(fig)
    return final_pdf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--cases", type=int, nargs="+", default=PLOTTABLE_CASES,
        help=f"Cases to plot (default: {PLOTTABLE_CASES}). Cases 1 and 4 are not supported.",
    )
    parser.add_argument(
        "--all-svidag", action="store_true",
        help="Also plot the strong-correct / strong-incorrect SVI-DAG priors "
             "(default: only the noninformative prior, as in the paper figure).",
    )
    args = parser.parse_args()

    bad = [c for c in args.cases if c not in PLOTTABLE_CASES]
    if bad:
        print(f"Unsupported case(s) {bad}; choose from {PLOTTABLE_CASES}.")
        return 2

    series = list(SERIES_STYLE) if args.all_svidag else list(DEFAULT_SERIES)

    made_any = False
    for case_num in args.cases:
        df = load_case_results(case_num)
        if df is None or df.empty:
            print(f"[case {case_num}] no results CSVs yet -- skipped")
            continue
        out_dir = _THIS_DIR / f"case_{case_num}"
        for scenario in sorted(df["scenario"].unique()):
            path = plot_case_scenario(df, case_num, scenario, series, out_dir)
            if path is None:
                print(f"[case {case_num}] {scenario}: no plottable series -- skipped")
                continue
            algos = sorted(set(df[df['scenario'] == scenario]['algorithm']))
            print(f"[case {case_num}] {scenario}: wrote {path} "
                  f"({len(algos)} algorithms present)")
            made_any = True
    return 0 if made_any else 1


if __name__ == "__main__":
    sys.exit(main())
