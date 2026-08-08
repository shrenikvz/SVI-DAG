# Welcome to SVI-DAG!

This repo corresponds to my paper: SVI-DAG: A Structured Variational Inference Approach to Bayesian Causal Discovery.

## Quick start

You need an NVIDIA GPU (CUDA 12.4-capable driver) and
[conda](https://docs.conda.io/en/latest/miniconda.html). Everything else is
installed into a **new, isolated conda environment** — nothing already on your
machine is touched or upgraded.

```bash
git clone https://github.com/shrenikvz/SVI-DAG.git
cd SVI-DAG
bash setup_env.sh
```

Then, in every new shell:

```bash
conda activate svidag
```

Confirm the whole pipeline works end to end (a few minutes):

```bash
./run_local.sh 1
```

And reproduce any of the six paper cases with one command each:

```bash
./run_local.sh 4
```

That is the whole workflow. The rest of this section explains what those
commands do and how to run a subset when a full case is too long.

---

## Installation

### Prerequisites

* **conda** — Miniconda or Miniforge. `setup_env.sh` refuses to run without it
  and prints the download link.
* **Python** is supplied by the conda environment (3.11), so no system Python
  version is required.
* **An NVIDIA GPU** with a driver new enough for **CUDA 12.4** — that is driver
  `>= 550.54.14`, or anything whose `nvidia-smi` header reports
  `CUDA Version: 12.4` or higher. Check with:

  ```bash
  nvidia-smi
  ```

  The pinned wheels target the CUDA **12.4** family specifically. A newer wheel
  set desyncs the JAX plugin from the runtime libraries and produces
  `Bus error` or `CUDNN_STATUS_INTERNAL_ERROR` at JAX startup; that is an ABI
  mismatch, and the fix is to reinstall exactly the pinned versions.
* **Disk**: roughly 8 GB for the environment (the CUDA wheels dominate).
* No GPU? See [CPU-only](#cpu-only) below. It works, but expect a full
  benchmark case to take days rather than hours.

### The one-command install

```bash
bash setup_env.sh
```

This detects whether an NVIDIA GPU is present, creates the matching conda
environment (`svidag` for GPU, `svidag-cpu` otherwise), installs the pinned
package set, installs SVI-DAG itself, and verifies the result. Useful flags:

```bash
bash setup_env.sh --gpu           # force the GPU environment
bash setup_env.sh --cpu           # force the CPU-only environment
bash setup_env.sh --force         # delete and rebuild an existing environment
bash setup_env.sh --name myenv    # use a different environment name
```

### Or install by hand

The script is a convenience wrapper; these are the same four steps:

```bash
conda env create -f environment.yml      # environment-cpu.yml for CPU-only
conda activate svidag
pip install -e . --no-deps
python scripts/check_env.py
```

`--no-deps` matters. The lockfile is the authority on versions, and
`pyproject.toml` lists its dependencies *unpinned* precisely so that an
accidental plain `pip install -e .` cannot float them. The editable install is
in fact optional — `main.py`, `run_local.sh` and everything under
`paper_results_reproduce/` put `src/` on `sys.path` themselves — it only makes
`import svidag` work from any working directory.

### What gets pinned, and where

| file | contents |
|---|---|
| [`requirements.txt`](./requirements.txt) | complete lock, direct **and** transitive, platform independent |
| [`requirements-cuda12.txt`](./requirements-cuda12.txt) | the NVIDIA CUDA 12.4 wheels, GPU installs only |
| [`requirements-direct.txt`](./requirements-direct.txt) | annotated list of *direct* dependencies, for humans — **do not install this one** |

Together the first two are the exact package set that produced every number in
`paper_results_reproduce/`, whenever you install them and whatever PyPI has
published since. To change a dependency, follow
[`scripts/lock_requirements.md`](./scripts/lock_requirements.md).

**Do not `pip install` the bundled baselines.** ProDAG, DiBS, VI-DP-DAG,
BayesDAG and BCD Nets under `other_algorithms/codes_jax/` are reached by
`sys.path` injection from their wrappers in
`paper_results_reproduce/case_4/baselines/`, so no install is required. Their
upstream `setup.py` files declare unpinned dependencies (`jax>=0.3.17`,
`jupyter`, …) that would pull packages outside the lockfile and float the JAX
pins — which is exactly what the lockfile exists to prevent.

### Verifying

```bash
python scripts/check_env.py
```

Checks the Python version, which devices JAX can see, that the load-bearing
versions are the pinned ones, that `svidag` and all five baseline wrappers
import, and that the Sachs ground truth loads. A missing GPU is reported as a
warning, not a failure. Then run the unit tests:

```bash
pytest -q
```

### CPU-only

```bash
bash setup_env.sh --cpu
conda activate svidag-cpu
./run_local.sh 1
./run_local.sh 2 --quick --cpu
```

`run_local.sh` refuses to start a full case when JAX reports no GPU, because
the runtimes grow by one to two orders of magnitude. Pass `--cpu` to override
that check once you know what you are asking for. Apple Silicon and
AMD/ROCm are not covered by the pinned wheels; both fall back to the CPU
backend.

---

## Running SVI-DAG on your own data

[`main.py`](./main.py) is the interactive entry point. Point `CUSTOM_DATA_BUILDER`
at a generator or a loader and run:

```bash
python main.py
```

It trains SVI-DAG under the three prior scenarios (`noninformative`,
`strong_correct`, `strong_incorrect`) and writes plots to `plots_svidag/`. The
file ships with several commented-out example generators — 2- and 3-node linear
and nonlinear SCMs, Erdős–Rényi and scale-free benchmark graphs, and the Sachs
loader — so adapting it is mostly a matter of uncommenting one line.

Note that `main.py` uses the committed `config.py` defaults, which are **not**
the per-case values required to reproduce the paper. See
[Hyperparameters](#hyperparameters).

---

## Reproducing the paper's six cases

One command per case:

```bash
./run_local.sh 1     # prior effect, 2-node graph
./run_local.sh 2     # benchmark, synthetic linear-Gaussian,   p=25
./run_local.sh 3     # benchmark, synthetic nonlinear,          p=25
./run_local.sh 4     # benchmark, real data (Sachs), 10 splits
./run_local.sh 5     # benchmark, synthetic linear-Gaussian,   p=50
./run_local.sh 6     # benchmark, synthetic nonlinear,          p=50
```

Each command loads that case's hyperparameters, walks the case's
grid sequentially on your GPU, writes one result file per cell, and regenerates
the case's figure or tables at the end. Console output for each cell is also
saved under `logs/case_<N>/`.

### What each case produces

| case | what it shows | grid | outputs under `paper_results_reproduce/case_N/` |
|---|---|---|---|
| 1 | effect of the domain-informed prior, 2-node graph, 10 000 hard posterior DAG draws | 3 priors × 2 generators | `case_1_results.json`, `case_1_table.tex`, `case_1_figure_data.csv` |
| 2 | SVI-DAG vs 5 baselines, linear-Gaussian ER (p=25, s=40), CPDAG metrics | 6 algos × 5 n × 5 reps | `case_2_results_<algo>_n<N>.{csv,json}`, `ER_p25_s40_metrics.{pdf,png}` |
| 3 | same on a nonlinear SEM, DAG metrics (the DAG is identifiable there) | 6 algos × 5 n × 5 reps | `case_3_results_<algo>_n<N>.{csv,json}`, `ER_p25_s40_metrics.{pdf,png}` |
| 4 | SVI-DAG vs 5 baselines on Sachs | 6 algos × 10 splits | `case_4_table_dag.tex`, `case_4_table_cpdag.tex`, `case_4_table_runtime.tex`, `case_4_table.tex` |
| 5 | case 2 at twice the graph size (p=50, s=80) | 6 algos × 5 n × 5 reps | `case_5_results_<algo>_n<N>.{csv,json}`, `ER_p50_s80_metrics.{pdf,png}` |
| 6 | case 3 at twice the graph size (p=50, s=80) | 6 algos × 5 n × 5 reps | `case_6_results_<algo>_n<N>.{csv,json}`, `ER_p50_s80_metrics.{pdf,png}` |

Cases 2, 3, 5 and 6 sweep `n ∈ {100, 316, 1000, 3162, 10000}` — that is
`10^{2, 2.5, 3, 3.5, 4}`, half-decades. Case 4 reports three tables:
oriented-DAG metrics, Markov-equivalence-class (CPDAG) metrics, and wall-clock
time per run as mean ± standard error over the 10 splits.

### Running part of a case

Smoke-test the whole path for a case in a few minutes:

```bash
./run_local.sh 2 --quick
```

`--quick` runs 1 replicate at `n=100` with ~100 training iterations and small
posterior sample counts. It exercises every code path — SVI-DAG, all five
baselines, metrics, figure generation. It does **not** reproduce the published
numbers, and says so on every run.

Run only SVI-DAG, skipping the baselines. The baselines' results are already
committed, so the figure regenerates with your fresh SVI-DAG numbers against
the published baseline numbers:

```bash
./run_local.sh 6 --algo svidag
```

Run a single cell:

```bash
./run_local.sh 3 --algo dds --n 10000
```

Resume an interrupted sweep — cells that already have result files are skipped:

```bash
./run_local.sh 5 --resume
```

Inspect the work plan before starting:

```bash
./run_local.sh 6 --list         # the cells, in order, then exit
./run_local.sh 6 --dry-run      # same plus the exact command per cell
```

### All `run_local.sh` options

| flag | meaning |
|---|---|
| `--algo A[,B]` | restrict to these algorithms (`svidag prodag bayesdag dds dibs bcd`) |
| `--n N[,M]` | restrict to these sample sizes (cases 2, 3, 5, 6) |
| `--reps a-b` | restrict to replicates `[a, b)` (cases 2, 3, 5, 6) |
| `--resume` | skip cells whose result file already exists |
| `--quick` | minutes-long smoke test; **not** the published numbers |
| `--list` | print the work plan and exit |
| `--dry-run` | `--list` plus the exact command per cell |
| `--no-figures` | skip figure/table regeneration at the end |
| `--cpu` | proceed even though JAX reports no GPU |

A failing cell does not abort the sweep: the failure is logged, the run
continues, and the exit summary lists what failed and the command to retry it.

### Rebuilding figures and tables without recomputing anything

The per-cell CSVs **are** the figure data, and the case-4 tables are built from
the result JSONs. Both steps are idempotent and refit nothing:

```bash
python paper_results_reproduce/plot_cases.py --cases 2 3 5 6
python paper_results_reproduce/case_4/make_tables.py
```

Every run of `run_local.sh` does this at the end anyway, from whatever results
exist at that moment — so a partial sweep still renders, with missing
algorithms shown as `--`.

---

## Hyperparameters

Each case requires a specific set of hyperparameters. They are listed in full
below, and stored as [`profiles/case<N>.env`](./profiles). `run_local.sh <N>`
and `run_case<N>.sh` source the matching file automatically, so under normal
use you never set any of these by hand — the values in effect are echoed at the
top of every run.

**Any variable not listed for a case takes its committed value from
[`src/svidag/config.py`](src/svidag/config.py).**

### SVI-DAG

| variable | case 1 | case 2 | case 3 | case 4 | case 5 | case 6 |
|---|---|---|---|---|---|---|
| `SVIDAG_LR` | `1e-3` | `3e-3` | `3e-3` | `3e-3` | `3e-3` | `3e-3` |
| `SVIDAG_GRAD_CLIP` | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` |
| `SVIDAG_NUM_ITERS` | `6000` | `1500` | `1500` | `2500` | `1500` | `1500` |
| `SVIDAG_BATCH_SIZE` | — | `64` | `64` | — | `64` | `64` |
| `SVIDAG_N_PARTICLES` | `20` | `20` | `20` | `20` | `20` | `20` |
| `SVIDAG_ETA_R` | `1e-3` | `1e-1` | `1e-1` | `1e-1` | `1e-1` | `1e-1` |
| `SVIDAG_PRIOR_R_SIGMA` | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` |
| `SVIDAG_PARTICLE_CLIP_MODE` | `norm` | `norm` | `norm` | `norm` | `norm` | `norm` |
| `SVIDAG_PARTICLE_CLIP` | — | `10.0` | `10.0` | — | `10.0` | `10.0` |
| `SVIDAG_SVGD_REP_RATIO` | `1.0` | — | — | — | — | — |
| `SVIDAG_FLOW_HIDDEN` | `5` | `64` | `64` | `64` | `64` | `64` |
| `SVIDAG_FLOW_BLOCKS` | — | `5` | `5` | — | `5` | `5` |
| `SVIDAG_FLOW_TYPE` | — | `nsf_coupling` | `nsf_coupling` | — | `nsf_coupling` | `nsf_coupling` |
| `SVIDAG_NSF_BINS` | — | — | `8` | — | — | `8` |
| `SVIDAG_HIDDEN_DIM` | — | — | `32` | — | — | `32` |
| `SVIDAG_SINKHORN_ITERS` | `100` | `100` | `100` | `100` | `100` | `100` |
| `SVIDAG_SCALE_INV` | `0` | `1` | `1` | `1` | `1` | `1` |
| `SVIDAG_T_B` | — | `0.3` | `0.3` | — | `0.3` | `0.3` |
| `SVIDAG_TAU_START` | `0.1` | `0.1` | `0.1` | `20` | `0.1` | `0.1` |
| `SVIDAG_TAU_END` | `0.1` | `0.1` | `0.1` | `0.1` | `0.1` | `0.1` |
| `SVIDAG_TAU_ANNEAL_FRAC` | — | `1.0` | `1.0` | — | `1.0` | `1.0` |
| `SVIDAG_ST_WARMUP` | `0.0` | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` |
| `SVIDAG_ROW_ONLY` | `0` | `1` | `1` | `1` | `1` | `1` |
| `SVIDAG_PRIOR_P0` | — | `0.025` | `0.05` | `0.15` | `0.025` | `0.05` |
| `SVIDAG_PRIOR_NU` | — | `20` | `20` | `20` | `20` | `20` |
| `SVIDAG_LEARN_NOISE` | — | `0` | `0` | — | `0` | `0` |
| `SVIDAG_OBS_NOISE` | — | `0.5` | `0.5` | — | `0.5` | `0.5` |
| `SVIDAG_KL_THETA` | — | `0.01` | `0.01` | — | `0.01` | `0.01` |
| `SVIDAG_MC_SAMPLES` | — | — | `1` | — | — | `1` |
| `SVIDAG_POSTERIOR_BIAS_INTERCEPT` | — | — | `-1.0` | — | — | `-1.0` |
| `SVIDAG_POSTERIOR_BIAS_LOG10_SLOPE` | — | `0.75` | `0.0` | — | `0.75` | `0.0` |
| `SVIDAG_POSTERIOR_BIAS_REFERENCE_N` | — | `100` | `100` | — | `100` | `100` |
| `SVIDAG_POSTERIOR_BIAS_FLOOR` | — | `-1.5` | — | — | `-1.5` | — |
| `SVIDAG_EVAL_EVERY` | `100` | `100` | `100` | `100` | `100` | `100` |
| `SVIDAG_PATIENCE` | `100000` | `100000` | `100000` | `100000` | `100000` | `100000` |

### BayesDAG

BayesDAG is the only baseline configured through the environment. The same
values apply to every case in which it runs (2, 3, 4, 5, 6); case 1 trains
SVI-DAG alone.

| variable | value |
|---|---|
| `BAYESDAG_EPOCHS` | `150` |
| `BAYESDAG_GRID_EPOCHS` | `25` |
| `BAYESDAG_NLAMBDA` | `4` |
| `BAYESDAG_GRID_SAMPLES` | `64` |

Setting `BAYESDAG_PAPER_SPEC=1` overrides all four with BayesDAG's own upstream
defaults.

### ProDAG, DiBS, DDS, BCD Nets

These four take no environment configuration. Their settings are fixed in their
wrappers under
[`paper_results_reproduce/case_4/baselines/`](paper_results_reproduce/case_4/baselines),
which every case reuses.

### Data generation

| variable | case 1 | case 2 | case 3 | case 4 | case 5 | case 6 |
|---|---|---|---|---|---|---|
| `CASE<N>_SEED` | — | `0` | `0` | `0` | `0` | `0` |
| `CASE<N>_NUM_REPLICATES` | — | `5` | `5` | — | `5` | `5` |
| sample sizes | — | `100 316 1000 3162 10000` | same | — | same | same |
| splits | — | — | — | `10` | — | — |

## Reproducibility

Everything that affects the numbers is pinned:

* `PYTHONHASHSEED=0` — the per-case seed derivation hashes strings, so without
  this the generated `(graph, data)` pairs differ between runs.
* `PYTHONNOUSERSITE=1` — keeps `~/.local` off the import path so the pinned
  package set is what actually runs.
* `CASE<N>_SEED=0` and `CASE<N>_NUM_REPLICATES=5` — fixed data generation.
* The full SVI-DAG hyperparameter profile is exported explicitly rather than
  left to `config.py` defaults.
* `XLA_PYTHON_CLIENT_PREALLOCATE=false` — JAX otherwise grabs 75 % of VRAM on
  first use, which makes a sequential six-algorithm sweep fail on the second
  algorithm on consumer cards.

`run_local.sh` sets all of these for you. Given the same package versions,
rerunning any case on a different machine reproduces the published numbers. The
single exception is the case-4 runtime table, which is hardware dependent by
construction.

Three smaller notes carried over from the original code: `config.seed` controls
deterministic initialization for a direct `main.py` run, flow permutations use a
fixed seed (42), and results may vary in the last digits across JAX and hardware
versions.

---

## Running on a Slurm cluster

The six `run_case<N>.sh` scripts are the cluster counterpart of `run_local.sh`.
Both source the same [`profiles/case<N>.env`](./profiles), so the two paths
cannot drift apart.

### Two things to fill in first

**1. Slurm directives.** Account strings, partition and QOS names, and the
syntax for requesting a GPU are all site specific. Open any `run_case<N>.sh`;
the lines you are most likely to need are at the top, commented out with a
leading `##`. Uncomment and fill in whichever your site requires:

```bash
##SBATCH --account=your_account_here          # if your site requires one
##SBATCH --partition=your_gpu_partition_here  # a partition that has GPUs
##SBATCH --qos=normal                         # if your site requires one
##SBATCH --mail-user=you@example.com          # optional notifications
```

The already-active directives request 1 node, 1 task, 8 CPUs, 64 GB and 1 GPU.
If `--gpus-per-node=1` is rejected, your site probably uses the older syntax —
replace it with `#SBATCH --gres=gpu:1`. Check with `sinfo` and
`scontrol show partition` if unsure.

**2. The Python environment.** Set `SVIDAG_ENV_ACTIVATE` to the activate script
of an environment holding the pinned packages. The scripts exit with a clear
error if it is unset. Either edit the default in the script, or pass it at
submit time:

```bash
SVIDAG_ENV_ACTIVATE=$(conda info --base)/envs/svidag/bin/activate sbatch run_case4.sh
```

If your cluster needs modules loaded first, list them in `SVIDAG_MODULES`:

```bash
SVIDAG_MODULES="anaconda cuda/12.4" \
SVIDAG_ENV_ACTIVATE=/path/to/env/bin/activate \
    sbatch run_case2.sh
```

Build the environment on the cluster exactly as you would locally
(`bash setup_env.sh`), typically from a login node or an interactive job.

### Submitting

```bash
sbatch run_case1.sh                 # single job
sbatch run_case4.sh                 # job array 0-5,  one task per algorithm
sbatch run_case2.sh                 # job array 0-29, one task per (algo, n)
sbatch --array=0-4 run_case2.sh     # SVI-DAG only, every n
sbatch --array=19 run_case3.sh      # DDS at n=10^4 only
```

For cases 2, 3, 5 and 6 the task id decodes as

```
task = (algo_idx * 5 + n_idx) * NCHUNK + chunk_idx      # 6 * 5 * 1 = 30 tasks
```

with `algo_idx` over `svidag prodag bayesdag dds dibs bcd` and `n_idx` over
`[100, 316, 1000, 3162, 10000]`, so tasks `0-4` are SVI-DAG, `5-9` ProDAG, and
so on. To split finer, lower `REPS_PER_TASK` in the script and widen `--array`
to `6*5*NCHUNK-1`; slices then carry an `_r<a>-<b>` tag and still merge.
`CASE2_SAMPLE_SIZES`, `CASE2_REP_START` and `CASE2_REP_END` (likewise `CASE3_*`,
`CASE5_*`, `CASE6_*`) override the task-id decode for a single cell.

Each task writes its own result file, so tasks never contend, and every task
regenerates the case's figure or tables from whatever results exist — the last
task to finish leaves a complete set.

## Project structure

```
SVI-DAG/
├── setup_env.sh            # one-command install into a fresh conda env
├── run_local.sh            # run any case on a single local GPU
├── environment.yml         # conda env, GPU (CUDA 12.4)
├── environment-cpu.yml     # conda env, CPU only
├── requirements.txt        # complete lockfile, platform independent
├── requirements-cuda12.txt # NVIDIA CUDA 12.4 wheels (GPU installs)
├── requirements-direct.txt # annotated direct dependencies (documentation)
├── pyproject.toml          # package metadata, requires-python
├── main.py                 # interactive entry point for your own data
├── profiles/
│   ├── README.md           # how these files are used
│   └── case1.env … case6.env   # exact hyperparameters, one file per case
├── run_case1.sh … run_case6.sh # Slurm counterparts of run_local.sh
├── data/sachs/
│   └── sachs.data.txt      # Sachs, 853-row observational subset (see below)
├── src/svidag/
│   ├── __init__.py         # package initialization
│   ├── config.py           # hyperparameters and settings (ORIGINAL defaults)
│   ├── data.py             # dataset loading and preprocessing
│   ├── model.py            # main SVIDAG model architecture
│   ├── train.py            # training loop with SVGD + ELBO
│   ├── flows.py            # normalizing flows (MAF, NSF)
│   ├── bayesian.py         # Bayesian neural networks
│   ├── eval.py             # posterior evaluation and sampling
│   ├── plots.py            # visualization utilities
│   ├── utils.py            # helper functions (Sinkhorn, metrics, etc.)
│   └── runner.py           # experiment orchestration
├── paper_results_reproduce/
│   ├── case_1/ … case_6/   # per-case drivers, results, tables, figures
│   └── plot_cases.py       # rebuilds the case 2/3/5/6 figures from their CSVs
├── other_algorithms/       # 5 vendored baselines (upstream + JAX ports)
├── scripts/
│   ├── check_env.py        # environment verification / smoke test
│   └── lock_requirements.md  # how to regenerate the lockfile
├── logs/                   # per-cell logs from run_local.sh (gitignored)
└── tests/                  # unit tests
```

## Testing

```bash
pytest -q            # the whole suite
pytest tests/ -v     # verbose
```


## Metrics

**DAG-Level:**
- Expected SHD (Structural Hamming Distance)
- Expected TPR (True Positive Rate)
- Expected F1 Score
- Brier Score, AUROC

**CPDAG-Level (Markov Equivalence Class):**
- Same metrics computed after DAG → CPDAG conversion

**Uncertainty Metrics:**
- Posterior entropy over structures
- `P(True DAG)` / `P(True CPDAG)` in posterior
- Top-k coverage (is true structure in top-k most probable?)

## Example Output

```
======================================================================
  STRUCTURE LEARNING METRICS - Scenario: strong_correct
  (Expected metrics computed over 1000 posterior samples)
======================================================================

--- DAG-Level Metrics ---
  Expected SHD:    2.1340
  Expected TPR:    0.8523
  Expected F1:     0.7891
  Brier Score:     0.1234
  AUROC:           0.9456

--- Epistemic Uncertainty Metrics (DAG) ---
  Posterior Entropy: 3.2145
  Prob(True DAG):    0.0234
  True in Top-1:     0
  True in Top-5:     1
  True in Top-10:    1
======================================================================
```


## Citation

If you use this code in your research, please cite:

```bibtex
@article{zinage2026svidag,
  title         = {SVI-DAG: A Structured Variational Inference Approach to
                   Bayesian Causal Discovery},
  author        = {Zinage, Shrenik},
  journal       = {arXiv preprint arXiv:2608.04930},
  year          = {2026},
  eprint        = {2608.04930},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2608.04930}
}
```

Paper: [arXiv:2608.04930](https://arxiv.org/abs/2608.04930)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE)
file for details.

The baselines bundled under `other_algorithms/` are third-party code and remain
under their own upstream licenses; see the license file in each subdirectory.

