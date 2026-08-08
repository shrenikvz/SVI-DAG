# Case 4 — Sachs benchmark (SVIDAG vs. 5 baselines)

Reproduces Table *Performance on sachs dataset* from the paper: mean ± standard
error over 10 equal random splits of the full Sachs dataset (~7,466 samples,
11 nodes), for SVI-DAG under the `noninformative` prior and five competing
methods.

**Prior scenario.** `noninformative` is p=0.5 on every off-diagonal entry —
purely data-driven, no domain knowledge. It is the only prior case 4 reports.

The ground-truth-derived `strong_correct` / `strong_incorrect` prior-sensitivity
rows are no longer part of case 4; that study lives in case 1. The uniform
graph-agnostic sparse prior (`p_ij = 0.1`, selected by `SVIDAG_PRIOR_P0`) has
been removed from the codebase entirely.

**Two tables are produced from every run** — one scored with oriented-DAG
metrics, one with CPDAG (Markov-equivalence-class) metrics. Both come from the
*same* fits: training is what costs hours, and re-scoring the stored posterior
samples under a second metric family is nearly free, so there is never a reason
to run case 4 twice.

## Files

| File | Purpose |
|------|---------|
| `common.py`             | Sachs loader (no default holdout), 10-fold split generator, thresholding + convention-normalisation helpers, DAG **and** CPDAG metric wrappers (`evaluate_samples_all_modes`), mean/SE aggregation, LaTeX table builder. |
| `svidag_runner.py`      | Trains SVIDAG on one split under one scenario and returns relaxed posterior samples in SVIDAG convention (`A[i,j]=1 ⇒ j → i`). |
| `baselines/prodag_wrapper.py` | ProDAG — clean Python API, implementation complete. |
| `baselines/dibs_wrapper.py`   | DiBS — marginal BGe inference, implementation complete. |
| `baselines/bayesdag_wrapper.py` | BayesDAG — **stub**; see its docstring for the call pattern to wire in. |
| `baselines/bcd_wrapper.py`    | BCD Nets — **stub**; requires refactoring `bcd/main.py`'s argparse body into a callable. |
| `baselines/dds_wrapper.py`    | DDS (VI-DP-DAG) — **stub**; call `train_dag.train` directly. |
| `run_case4.py`          | Top-level orchestrator (all 7 rows in one job) + LaTeX table generator. |
| `_single_algo.py` + `run_<algo>_only.py` | Per-algorithm drivers, one SLURM job each, so the baselines run in parallel. Each writes `case_4_results_<suffix>.json`. |
| `make_tables.py`        | Merges the per-algorithm JSONs into both LaTeX tables. Run this after a parallel sweep. |

## Adjacency convention

SVIDAG uses `A[i,j] = 1 ⇒ j → i` (column causes row); most baselines use the
standard `A[i,j] = 1 ⇒ i → j`. Every wrapper declares its native convention
via the second return value (`"j_to_i"` or `"i_to_j"`), and
`common.evaluate_samples` transposes as needed before metric computation so
**all algorithms are scored against the same oriented ground truth**.

## Fair-comparison recipe (same threshold for everyone)

Every wrapper returns `[S, d, d]` float samples in `[0, 1]`. The orchestrator
then:

1. normalises to SVIDAG convention,
2. thresholds at `0.5` (matches `svidag.config.threshold_A`),
3. zeros the diagonal,
4. computes DAG metrics via `svidag.utils.compute_expected_metrics`.

Methods that natively emit binary samples (DiBS, BCD, VI-DP-DAG) pass
through thresholding unchanged.

## DAG vs. CPDAG metrics

Both families are computed for every (algorithm, split) by
`common.evaluate_samples_all_modes`, which normalises the convention,
thresholds, and builds the soft posterior mean **once**, then scores that one
set of samples twice:

| Mode | Scored against | Function |
|------|----------------|----------|
| `"dag"`   | `sachs.true_adj` (oriented ground-truth DAG) | `compute_expected_metrics` |
| `"cpdag"` | `sachs.true_cpdag` (its Markov equivalence class; each posterior sample is converted to its CPDAG first) | `compute_expected_metrics_cpdag` |

The ground-truth CPDAG is pre-computed once in `common.load_sachs_full`, so the
CPDAG pass costs only the per-sample `dag_to_cpdag` conversion — which was
already being paid before; the added DAG pass is a cheap confusion-matrix count.

`METRIC_MODES_ACTIVE` in `run_case4.py` and `_single_algo.py` controls which
families are reported. Both default to `("dag", "cpdag")`; narrow it to a
single-element tuple only if you deliberately want one table.

## Running

Monolithic (all 7 rows in one job):

```bash
python paper_results_reproduce/case_4/run_case4.py
```

Parallel (one job per algorithm, then merge):

```bash
python paper_results_reproduce/case_4/run_svidag_only.py
python paper_results_reproduce/case_4/make_tables.py
```

By default the scripts skip any baseline whose wrapper is still a stub and
report results only for the implemented ones. To fail loudly on missing
baselines instead:

```bash
SKIP_UNIMPLEMENTED=0 python paper_results_reproduce/case_4/run_case4.py
```

## Outputs

- `case_4_results.json` / `case_4_results_<algo>.json` — per-split metric dicts
  + aggregated mean/SE, **nested by metric mode** (`per_split["dag"][label]`,
  `aggregated["cpdag"][label]`), plus the configuration snapshot.
- `case_4_table_dag.tex`   — DAG-metric table (`\label{tab:sachs_dag}`).
- `case_4_table_cpdag.tex` — CPDAG-metric table (`\label{tab:sachs_cpdag}`).
- `case_4_table.tex`       — both of the above concatenated, for convenience.

Best value per metric is bolded independently within each table.

`make_tables.py` also reads the older single-family result files (which carry
`config.use_cpdag` instead of `config.metric_modes`) and files them under the
family they were produced with, printing a `NOTE` for each — so a partially
re-run sweep still yields compilable tables.

## Implementing the remaining baselines

Each stub wrapper's docstring spells out:

1. the exact entry-point file and function to call,
2. the expected return signature,
3. the declared adjacency convention.

The simplest way to complete one is to copy the pattern from
`prodag_wrapper.py`: set up `sys.path`, call the algorithm's training
function on `X_train`, collect posterior samples into shape `[S, d, d]`,
return `(A_samples, convention)`.
