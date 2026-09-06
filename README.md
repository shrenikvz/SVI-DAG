# SVI-DAG: Structured Variational Inference for Bayesian Causal Discovery


Official implementation of **"SVI-DAG: A Structured Variational Inference Approach to Bayesian Causal Discovery"** ([arXiv:2608.04930](https://arxiv.org/abs/2608.04930)).

> [!IMPORTANT]
>
> This code base is refactored by AI (Claude), and AI can make mistakes.
> Please feel free to open up any issue when you have any trouble or questions regarding the code.

## 🛠️ Installation

### Prerequisites

- **conda** (Miniconda or Miniforge). `setup_env.sh` refuses to run without it and prints the download link.
- **An NVIDIA GPU** with a driver new enough for **CUDA 12.4** (driver `>= 550.54.14`, or `nvidia-smi` reporting `CUDA Version: 12.4` or higher). Python 3.11 is supplied by the conda environment.
- **Disk**: roughly 8 GB for the environment (the CUDA wheels dominate).
- No GPU? See [CPU-only](#cpu-only). It works, but a full benchmark case takes days instead of hours.

### Environment Setup

Everything is installed into a **new, isolated conda environment**; nothing already on your machine is touched.

```bash
git clone https://github.com/shrenikvz/SVI-DAG.git
cd SVI-DAG
bash setup_env.sh
conda activate svidag
```

`setup_env.sh` detects whether an NVIDIA GPU is present, creates the matching environment (`svidag` for GPU, `svidag-cpu` otherwise), installs the pinned package set, installs SVI-DAG itself, and verifies the result. Useful flags:

```bash
bash setup_env.sh --gpu           # force the GPU environment
bash setup_env.sh --cpu           # force the CPU-only environment
bash setup_env.sh --force         # delete and rebuild an existing environment
bash setup_env.sh --name myenv    # use a different environment name
```

### Manual Setup (Optional)

The script is a convenience wrapper around these four steps:

```bash
conda env create -f environment.yml      # environment-cpu.yml for CPU-only
conda activate svidag
pip install -e . --no-deps
python scripts/check_env.py
```

`--no-deps` matters. The lockfile is the authority on versions, and `pyproject.toml` lists its dependencies *unpinned* so that an accidental `pip install -e .` cannot float them. The editable install is optional: `main.py`, `run_local.sh` and everything under `paper_results_reproduce/` put `src/` on `sys.path` themselves.

| file | contents |
|---|---|
| [`requirements.txt`](./requirements.txt) | complete lock, direct **and** transitive, platform independent |
| [`requirements-cuda12.txt`](./requirements-cuda12.txt) | the NVIDIA CUDA 12.4 wheels, GPU installs only |
| [`requirements-direct.txt`](./requirements-direct.txt) | annotated list of *direct* dependencies, for humans. **Do not install this one** |

To change a dependency, follow [`scripts/lock_requirements.md`](./scripts/lock_requirements.md).

### Baselines Setup

No extra install is needed. ProDAG, DiBS, VI-DP-DAG (DDS), BayesDAG and BCD Nets are vendored under `other_algorithms/codes_jax/` and reached by `sys.path` injection from their wrappers in `paper_results_reproduce/case_4/baselines/`.

> [!WARNING]
>
> **Do not `pip install` the bundled baselines.** Their upstream `setup.py` files declare unpinned dependencies (`jax>=0.3.17`, `jupyter`, …) that would pull packages outside the lockfile and float the JAX pins, which is exactly what the lockfile exists to prevent.

### ⚙️ Verifying the Environment

```bash
python scripts/check_env.py     # Python version, JAX devices, pinned versions, all imports, Sachs ground truth
pytest -q                       # unit tests
```

A missing GPU is reported as a warning, not a failure. If JAX fails at startup with `Bus error` or `CUDNN_STATUS_INTERNAL_ERROR`, the CUDA wheels have desynced from the pinned versions; reinstall exactly the pinned set.

### CPU-only

```bash
bash setup_env.sh --cpu
conda activate svidag-cpu
./run_local.sh 1
./run_local.sh 2 --quick --cpu
```

`run_local.sh` refuses to start a full case when JAX reports no GPU, because runtimes grow by one to two orders of magnitude. Pass `--cpu` to override that check. Apple Silicon and AMD/ROCm are not covered by the pinned wheels; both fall back to the CPU backend.

## 📦 Repository Structure

```
├── README.md
├── setup_env.sh                # one-command install into a fresh conda env
├── run_local.sh                # run any paper case on a single local GPU
├── run_case1.sh … run_case6.sh # Slurm counterparts of run_local.sh
├── main.py                     # 🧭 entry point for running SVI-DAG on your own data
├── environment.yml             # conda env, GPU (CUDA 12.4)
├── environment-cpu.yml         # conda env, CPU only
├── requirements.txt            # complete lockfile, platform independent
├── requirements-cuda12.txt     # NVIDIA CUDA 12.4 wheels (GPU installs)
├── requirements-direct.txt     # annotated direct dependencies (documentation)
├── pyproject.toml              # package metadata, requires-python
├── profiles/
│   └── case1.env … case6.env   # exact hyperparameters, one file per case
├── data/sachs/
│   └── sachs.data.txt          # Sachs, 853-row observational subset
├── src/svidag/
│   ├── config.py               # hyperparameters and settings (defaults)
│   ├── data.py                 # synthetic generators, benchmark graphs, Sachs loader
│   ├── model.py                # 🧭 SVI-DAG model: flow over edges, Sinkhorn orderings
│   ├── flows.py                # normalizing flows (MAF, NSF coupling)
│   ├── bayesian.py             # Bayesian neural networks for the node models
│   ├── train.py                # training loop with SVGD + ELBO
│   ├── eval.py                 # posterior evaluation and sampling
│   ├── utils.py                # Sinkhorn, DAG/CPDAG metrics, helpers
│   ├── plots.py                # visualization utilities
│   └── runner.py               # experiment orchestration
├── paper_results_reproduce/
│   ├── case_1/ … case_6/       # per-case drivers, committed results, tables, figures
│   ├── case_4/baselines/       # wrappers for the five baselines (shared by every case)
│   ├── ablation/               # component ablation (prior / flow / SVGD) + MEC coverage
│   └── plot_cases.py           # rebuilds the case 2/3/5/6 figures from their CSVs
├── other_algorithms/
│   ├── original_codes/         # upstream ProDAG, DiBS, VI-DP-DAG, BayesDAG, BCD Nets
│   └── codes_jax/              # JAX mirrors / ports of the above
├── scripts/
│   ├── check_env.py            # environment verification / smoke test
│   └── lock_requirements.md    # how to regenerate the lockfile
├── benchmarks/                 # train-step micro-benchmark
└── tests/                      # unit tests
```

## 🚀 Quick Start

### Run SVI-DAG on your own data

[`main.py`](./main.py) is the interactive entry point. Point `CUSTOM_DATA_BUILDER` at a generator or a loader and run:

```bash
python main.py
```

It trains SVI-DAG under three prior scenarios (`noninformative`, `strong_correct`, `strong_incorrect`) and writes plots to `plots_svidag/`. The file ships with several commented-out example generators (2- and 3-node linear and nonlinear SCMs, Erdős–Rényi and scale-free benchmark graphs, and the Sachs loader), so adapting it is mostly a matter of uncommenting one line. Note that `main.py` uses the committed `config.py` defaults, which are **not** the per-case values needed to reproduce the paper; see [Hyperparameters](#️-hyperparameters).

### Reproduce a paper case

Confirm the whole pipeline works end to end (a few minutes), then reproduce any of the six paper cases with one command each:

```bash
./run_local.sh 1     # prior effect, 2-node graph
./run_local.sh 2     # benchmark, synthetic linear-Gaussian,   p=25
./run_local.sh 3     # benchmark, synthetic nonlinear,          p=25
./run_local.sh 4     # benchmark, real data (Sachs), 10 splits
./run_local.sh 5     # benchmark, synthetic linear-Gaussian,   p=50
./run_local.sh 6     # benchmark, synthetic nonlinear,          p=50
```

Each command loads that case's hyperparameters, walks the case's grid sequentially on your GPU, writes one result file per cell, and regenerates the case's figure or tables at the end. Console output for each cell is saved under `logs/case_<N>/`. See [Reproducing the Paper](#-reproducing-the-paper) for partial runs and the full option list.

### Example output

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

## 📊 Baselines Comparison

| Method | Type | Inference | Key Features |
|--------|------|-----------|--------------|
| **SVI-DAG** | VI | Normalizing flow + SVGD | Edge-dependent flow posterior `q(γ \| r)`, Sinkhorn orderings, domain-informed prior |
| ProDAG | VI | Projected distributions | Projects a continuous distribution onto the DAG space |
| BayesDAG | SG-MCMC + VI | Low-rank node potentials | Unconstrained optimization over DAGs; linear and nonlinear models |
| DDS (VI-DP-DAG) | VI | Differentiable DAG sampling | Gumbel-Sinkhorn permutations with a probabilistic edge model |
| DiBS | SVGD | Latent graph embeddings | Joint or marginal inference with the BGe score |
| BCD Nets | VI | Permutation + weighted adjacency | Bayesian causal discovery nets for linear-Gaussian SEMs |

The five baselines run through a common harness: every wrapper returns `[S, d, d]` posterior samples in `[0, 1]` and declares its adjacency convention, and `paper_results_reproduce/case_4/common.py` normalizes, thresholds at 0.5, and scores every method against the same oriented ground truth with both DAG and CPDAG metrics. BayesDAG is the only baseline configured through the environment (`BAYESDAG_*` in the profiles); the other four take their settings from their wrappers.

## 📈 Metrics

All metrics are expectations over posterior samples, computed at the **DAG level** and, after DAG → CPDAG conversion, at the **Markov-equivalence-class level**:

- **Accuracy**: Expected SHD (Structural Hamming Distance), Expected TPR, Expected F1
- **Calibration**: Brier score and AUROC of the edge marginals
- **Uncertainty**: posterior entropy over structures, `P(True DAG)` / `P(True CPDAG)`, top-k coverage (is the true structure among the k most probable?)

## 📂 Data Availability

Every dataset used in the paper is either generated on the fly from `src/svidag/data.py` or shipped in this repository. No download is required.

### Datasets

| Case | Dataset | Nodes / Edges | Sample sizes | Replicates | Metrics | Generator |
|------|---------|---------------|--------------|------------|---------|-----------|
| 1 | 2-node linear and nonlinear SCMs, three priors | 2 | fixed | 10 000 posterior draws | prior effect | `generate_synthetic_dataset` |
| 2 | Linear-Gaussian Erdős–Rényi | p=25, s=40 | 100, 316, 1 000, 3 162, 10 000 | 5 | CPDAG | `generate_benchmark_dataset` |
| 3 | Nonlinear (MLP) Erdős–Rényi | p=25, s=40 | same | 5 | DAG | `generate_benchmark_dataset` |
| 4 | Sachs protein signaling (real) | p=11 | 7 466 rows, 10 random splits | 10 splits | DAG + CPDAG + runtime | `load_sachs_dataset` |
| 5 | Linear-Gaussian Erdős–Rényi | p=50, s=80 | same as case 2 | 5 | CPDAG | `generate_benchmark_dataset` |
| 6 | Nonlinear (MLP) Erdős–Rényi | p=50, s=80 | same as case 2 | 5 | DAG | `generate_benchmark_dataset` |
| Ablation | Nonlinear ER (main) and linear ER (MEC companion) | p=20, s=40 and p=10, s=10 | 300 and 1 000 | 10 seeds | Brier, E-SHD, E-F1, AUROC, MEC-cov | see `paper_results_reproduce/ablation/` |

Cases 2, 3, 5 and 6 sweep `n ∈ {100, 316, 1000, 3162, 10000}`, that is `10^{2, 2.5, 3, 3.5, 4}` in half-decades. Linear cases report CPDAG metrics because only the equivalence class is identifiable; nonlinear cases report DAG metrics because the DAG is. Every `(graph, data)` pair is derived from `CASE<N>_SEED=0`, so a rerun draws the same graphs and samples.

### How to Load the Data

Each loader returns a `Dataset` dataclass holding the scaled train/test splits on device, the original arrays, the ground-truth adjacency and node names:

```python
from svidag.data import generate_benchmark_dataset, load_sachs_dataset

# 25-node linear-Gaussian ER graph, 1000 samples (train + held-out test)
ds = generate_benchmark_dataset(num_nodes=25, graph_type="er", sem_type="linear",
                                num_samples=1000, edge_prob=0.3, rng_seed=0)
X_train  = ds.train_data       # [N_train, m], standardized, on device
X_test   = ds.test_data_scaled # [N_test, m]
A_true   = ds.true_adj_np      # [m, m], A[i, j] = 1  ⇒  j → i

# Sachs, the 7,466-row full dataset bundled with cdt (the case-4 setting)
sachs = load_sachs_dataset()
```

`generate_benchmark_dataset` accepts `graph_type ∈ {"er", "sf"}` and `sem_type ∈ {"linear", "mlp", "mim"}`. For arbitrary structural causal models, `generate_synthetic_dataset(generator_fn)` takes any function returning `(obs_data, true_adj, node_names)`; see the examples in `main.py`.

> **Note on the adjacency convention.** SVI-DAG uses `A[i, j] = 1 ⇒ j → i` (column causes row); most baselines use `A[i, j] = 1 ⇒ i → j`. The benchmark harness transposes as needed before scoring, so all methods are evaluated against the same oriented ground truth.

### Original Dataset Sources

- **Sachs** — the 7,466-row full dataset is loaded through the [`cdt`](https://github.com/FenTechSolutions/CausalDiscoveryToolbox) package, whose consensus network is used as ground truth. The 853-row observational subset shipped in `data/sachs/` can be selected with `svidag.data.OBSERVATIONAL_SACHS_PATH`; note that this changes the results, it is not merely a different route to the same data.
- **Synthetic graphs** — Erdős–Rényi and scale-free DAGs with linear, MLP and multiplicative-interaction SEMs follow the standard NOTEARS-style simulators reimplemented in `src/svidag/data.py`.

## 🔁 Reproducing the Paper

### What each case produces

| case | what it shows | grid | outputs under `paper_results_reproduce/case_N/` |
|---|---|---|---|
| 1 | effect of the domain-informed prior, 2-node graph | 3 priors × 2 generators | `case_1_results.json`, `case_1_table.tex`, `case_1_figure_data.csv` |
| 2 | SVI-DAG vs 5 baselines, linear-Gaussian ER (p=25, s=40), CPDAG metrics | 6 algos × 5 n × 5 reps | `case_2_results_<algo>_n<N>.{csv,json}`, `ER_p25_s40_metrics.{pdf,png}` |
| 3 | same on a nonlinear SEM, DAG metrics | 6 algos × 5 n × 5 reps | `case_3_results_<algo>_n<N>.{csv,json}`, `ER_p25_s40_metrics.{pdf,png}` |
| 4 | SVI-DAG vs 5 baselines on Sachs | 6 algos × 10 splits | `case_4_table_dag.tex`, `case_4_table_cpdag.tex`, `case_4_table_runtime.tex` |
| 5 | case 2 at twice the graph size (p=50, s=80) | 6 algos × 5 n × 5 reps | `case_5_results_<algo>_n<N>.{csv,json}`, `ER_p50_s80_metrics.{pdf,png}` |
| 6 | case 3 at twice the graph size (p=50, s=80) | 6 algos × 5 n × 5 reps | `case_6_results_<algo>_n<N>.{csv,json}`, `ER_p50_s80_metrics.{pdf,png}` |

### Running part of a case

```bash
./run_local.sh 2 --quick                 # minutes-long smoke test of every code path (NOT the published numbers)
./run_local.sh 6 --algo svidag           # SVI-DAG only; figure regenerates against the committed baseline results
./run_local.sh 3 --algo dds --n 10000    # a single cell
./run_local.sh 5 --resume                # skip cells whose result file already exists
./run_local.sh 6 --list                  # print the work plan and exit
./run_local.sh 6 --dry-run               # --list plus the exact command per cell
```

| flag | meaning |
|---|---|
| `--algo A[,B]` | restrict to these algorithms (`svidag prodag bayesdag dds dibs bcd`) |
| `--n N[,M]` | restrict to these sample sizes (cases 2, 3, 5, 6) |
| `--reps a-b` | restrict to replicates `[a, b)` (cases 2, 3, 5, 6) |
| `--resume` | skip cells whose result file already exists |
| `--quick` | minutes-long smoke test; **not** the published numbers |
| `--list` / `--dry-run` | print the work plan (and the exact commands) and exit |
| `--no-figures` | skip figure/table regeneration at the end |
| `--cpu` | proceed even though JAX reports no GPU |

A failing cell does not abort the sweep: the failure is logged, the run continues, and the exit summary lists what failed and the command to retry it.

### Rebuilding figures and tables without recomputing

The per-cell CSVs **are** the figure data, and the case-4 tables are built from the result JSONs. Both steps are idempotent and refit nothing:

```bash
python paper_results_reproduce/plot_cases.py --cases 2 3 5 6
python paper_results_reproduce/case_4/make_tables.py
python paper_results_reproduce/ablation/make_table.py
```

### Ablation

`paper_results_reproduce/ablation/` removes SVI-DAG's components one at a time (domain-informed prior, flow posterior, SVGD inference, and flow + SVGD jointly) and reports Brier, E-SHD, E-F1, AUROC and exact Markov-equivalence-class coverage. See its [README](paper_results_reproduce/ablation/README.md) for the Slurm submission pipeline and cluster notes.

### Reproducibility

Everything that affects the numbers is pinned, and `run_local.sh` sets all of it for you:

- `PYTHONHASHSEED=0`: the per-case seed derivation hashes strings
- `PYTHONNOUSERSITE=1`: keeps `~/.local` off the import path so the pinned package set is what runs
- `CASE<N>_SEED=0` and `CASE<N>_NUM_REPLICATES=5`: fixed data generation
- The full SVI-DAG hyperparameter profile is exported explicitly rather than left to `config.py` defaults
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`: JAX otherwise grabs 75 % of VRAM on first use

Given the same package versions, rerunning any case on a different machine reproduces the published numbers. The single exception is the case-4 runtime table, which is hardware dependent by construction. Results may differ in the last digits across JAX and hardware versions.

## ⚙️ Hyperparameters

Each case requires a specific set of hyperparameters, stored as [`profiles/case<N>.env`](./profiles). `run_local.sh <N>` and `run_case<N>.sh` source the matching file automatically, and the values in effect are echoed at the top of every run. **Any variable not listed for a case takes its committed value from [`src/svidag/config.py`](src/svidag/config.py).**

<details>
<summary><b>SVI-DAG per-case values</b> (click to expand)</summary>

| variable | case 1 | case 2 | case 3 | case 4 | case 5 | case 6 |
|---|---|---|---|---|---|---|
| `SVIDAG_LR` | `1e-3` | `3e-3` | `3e-3` | `3e-3` | `3e-3` | `3e-3` |
| `SVIDAG_GRAD_CLIP` | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` |
| `SVIDAG_NUM_ITERS` | `6000` | `3000` | `1500` | `2500` | `1500` | `1500` |
| `SVIDAG_BATCH_SIZE` | — | `64` | `64` | — | `64` | `64` |
| `SVIDAG_N_PARTICLES` | `20` | `20` | `40` | `20` | `40` | `20` |
| `SVIDAG_N_PARTICLES_ABOVE` | — | — | — | — | `20` | — |
| `SVIDAG_N_PARTICLES_THRESHOLD_N` | — | — | — | — | `3162` | — |
| `SVIDAG_ETA_R` | `1e-3` | `1e-1` | `1e-1` | `1e-1` | `1e-1` | `1e-1` |
| `SVIDAG_PRIOR_R_SIGMA` | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` | `1.0` |
| `SVIDAG_PARTICLE_CLIP_MODE` | `norm` | `norm` | `norm` | `norm` | `norm` | `norm` |
| `SVIDAG_PARTICLE_CLIP` | — | `10.0` | `10.0` | — | `10.0` | `10.0` |
| `SVIDAG_SVGD_REP_RATIO` | `1.0` | — | `1.0` | — | `1.0` | — |
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
| `SVIDAG_POSTERIOR_BIAS_INTERCEPT` | — | `-1.0` | `-2.0` | — | `0.0` | `-4.0` |
| `SVIDAG_POSTERIOR_BIAS_LOG10_SLOPE` | — | `2.0` | `4.0` | — | `4.0` | `4.0` |
| `SVIDAG_POSTERIOR_BIAS_REFERENCE_N` | — | `100` | `100` | — | `100` | `100` |
| `SVIDAG_POSTERIOR_BIAS_FLOOR` | — | `-4.5` | `-4.0` | — | `-8.0` | `-4.0` |
| `SVIDAG_POSTERIOR_BIAS_CEILING` | — | `1.0` | `0.0` | — | `-1.0` | `0.0` |
| `SVIDAG_POSTERIOR_SCALE_INTERCEPT` | — | `2.0` | `3.0` | — | `3.0` | `5.0` |
| `SVIDAG_POSTERIOR_SCALE_LOG10_SLOPE` | — | `0.5` | `0.0` | — | — | `0.0` |
| `SVIDAG_POSTERIOR_SCALE_REFERENCE_N` | — | `100` | `100` | — | — | `100` |
| `SVIDAG_POSTERIOR_SCALE_FLOOR` | — | `1.0` | — | — | — | — |
| `SVIDAG_POSTERIOR_SCALE_CEILING` | — | `2.0` | — | — | — | — |
| `SVIDAG_POSTERIOR_Z_SCALE_INTERCEPT` | — | `-1.0` | `0.0` | — | `0.0` | `0.0` |
| `SVIDAG_POSTERIOR_Z_SCALE_LOG10_SLOPE` | — | `-2.0` | `0.0` | — | — | `0.0` |
| `SVIDAG_POSTERIOR_Z_SCALE_REFERENCE_N` | — | `100` | `100` | — | — | `100` |
| `SVIDAG_POSTERIOR_Z_SCALE_FLOOR` | — | `0.0` | — | — | — | — |
| `SVIDAG_POSTERIOR_Z_SCALE_CEILING` | — | `1.0` | — | — | — | — |
| `SVIDAG_POSTERIOR_PARTICLE_TEMP` | — | — | `1.0` | — | — | `1.0` |
| `SVIDAG_POSTERIOR_PTEMP_INTERCEPT` | — | — | — | — | `2.0` | — |
| `SVIDAG_POSTERIOR_PTEMP_LOG10_SLOPE` | — | — | — | — | `-2.0` | — |
| `SVIDAG_POSTERIOR_PTEMP_REFERENCE_N` | — | — | — | — | `100` | — |
| `SVIDAG_POSTERIOR_PTEMP_FLOOR` | — | — | — | — | `-1.0` | — |
| `SVIDAG_POSTERIOR_PTEMP_CEILING` | — | — | — | — | `0.0` | — |
| `SVIDAG_EVAL_EVERY` | `100` | `100` | `100` | `100` | `100` | `100` |
| `SVIDAG_PATIENCE` | `100000` | `100000` | `100000` | `100000` | `100000` | `100000` |

</details>

<details>
<summary><b>BayesDAG and data generation</b> (click to expand)</summary>

BayesDAG is the only baseline configured through the environment. The same values apply to every case in which it runs (2, 3, 4, 5, 6); setting `BAYESDAG_PAPER_SPEC=1` overrides all four with BayesDAG's upstream defaults.

| variable | value |
|---|---|
| `BAYESDAG_EPOCHS` | `150` |
| `BAYESDAG_GRID_EPOCHS` | `25` |
| `BAYESDAG_NLAMBDA` | `4` |
| `BAYESDAG_GRID_SAMPLES` | `64` |

ProDAG, DiBS, DDS and BCD Nets take no environment configuration; their settings are fixed in their wrappers under [`paper_results_reproduce/case_4/baselines/`](paper_results_reproduce/case_4/baselines).

| variable | case 1 | case 2 | case 3 | case 4 | case 5 | case 6 |
|---|---|---|---|---|---|---|
| `CASE<N>_SEED` | — | `0` | `0` | `0` | `0` | `0` |
| `CASE<N>_NUM_REPLICATES` | — | `5` | `5` | — | `5` | `5` |
| sample sizes | — | `100 316 1000 3162 10000` | same | — | same | same |
| splits | — | — | — | `10` | — | — |

</details>

Profiles are plain `export`s, so anything exported *after* sourcing wins. A run with any value overridden no longer reproduces the published numbers:

```bash
SVIDAG_NUM_ITERS=200 ./run_local.sh 2 --algo svidag --n 100
```

## 🖥️ Running on a Slurm Cluster

The six `run_case<N>.sh` scripts are the cluster counterpart of `run_local.sh`. Both source the same [`profiles/case<N>.env`](./profiles), so the two paths cannot drift apart.

**1. Slurm directives.** Account, partition, QOS and the GPU request syntax are site specific. The lines you are most likely to need are at the top of each script, commented out with a leading `##`:

```bash
##SBATCH --account=your_account_here          # if your site requires one
##SBATCH --partition=your_gpu_partition_here  # a partition that has GPUs
##SBATCH --qos=normal                         # if your site requires one
##SBATCH --mail-user=you@example.com          # optional notifications
```

The active directives request 1 node, 1 task, 8 CPUs, 64 GB and 1 GPU. If `--gpus-per-node=1` is rejected, replace it with `#SBATCH --gres=gpu:1`.

**2. The Python environment.** Set `SVIDAG_ENV_ACTIVATE` to the activate script of an environment holding the pinned packages, and list any modules to load in `SVIDAG_MODULES`:

```bash
SVIDAG_MODULES="anaconda cuda/12.4" \
SVIDAG_ENV_ACTIVATE=$(conda info --base)/envs/svidag/bin/activate \
    sbatch run_case2.sh
```

**3. Submit.**

```bash
sbatch run_case1.sh                 # single job
sbatch run_case4.sh                 # job array 0-5,  one task per algorithm
sbatch run_case2.sh                 # job array 0-29, one task per (algo, n)
sbatch --array=0-4 run_case2.sh     # SVI-DAG only, every n
sbatch --array=19 run_case3.sh      # DDS at n=10^4 only
```

For cases 2, 3, 5 and 6 the task id decodes as `task = (algo_idx * 5 + n_idx) * NCHUNK + chunk_idx`, with `algo_idx` over `svidag prodag bayesdag dds dibs bcd` and `n_idx` over `[100, 316, 1000, 3162, 10000]`. Each task writes its own result file, so tasks never contend, and every task regenerates the case's figure or tables from whatever results exist; the last task to finish leaves a complete set.

## 🧪 Testing

```bash
pytest -q            # the whole suite
pytest tests/ -v     # verbose
```

## 📝 Citation

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

## 📄 License

This project is licensed under the Apache License 2.0; see the [LICENSE](LICENSE) file for details.

The baselines bundled under `other_algorithms/` are third-party code and remain under their own upstream licenses; see the license file in each subdirectory.
