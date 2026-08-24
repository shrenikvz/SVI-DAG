#!/bin/bash -l
# ===========================================================================
# EDIT BEFORE SUBMITTING.
#
# Slurm directives are site specific: account strings, partition and QOS names,
# and the syntax for requesting a GPU all differ between clusters. Check your
# site's documentation (or `sinfo`, `scontrol show partition`) and adjust the
# block below. The lines that are commented out are the ones most likely to be
# required at your site -- uncomment and fill them in.
#
# You must also tell the script which Python environment to use; see
# SVIDAG_ENV_ACTIVATE further down. See README.md, "Running on a Slurm
# cluster".
# ===========================================================================

## Uncomment and set if your cluster requires an allocation/account:
##SBATCH --account=your_account_here

## Uncomment and set to a partition that has GPUs:
##SBATCH --partition=your_gpu_partition_here

## Uncomment if your cluster requires an explicit QOS:
##SBATCH --qos=normal

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=96:00:00
#SBATCH --job-name=svidag_case6
#SBATCH --array=0-209
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# One GPU. Some clusters use --gres instead; if the line below is rejected,
# replace it with:   #SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1

## Optional email notifications -- uncomment and set your own address:
##SBATCH --mail-user=you@example.com
##SBATCH --mail-type=END,FAIL

# ===========================================================================
# Case 6 -- SVI-DAG vs 5 baselines on synthetic nonlinear data (ER, p=50, s=80),
# DAG-level metrics, 7 sample sizes (n = 10^1..10^4 in half-decades)
# x 5 replicates.
#
#   sbatch run_case6.sh                     # the whole grid (210 tasks)
#   sbatch --array=0-34 run_case6.sh        # SVI-DAG, every n, every replicate
#   sbatch --array=135-139 run_case6.sh     # DDS at n=10^4 only
#
# ONE ARRAY TASK PER (ALGORITHM, SAMPLE SIZE, REPLICATE):
#   6 algorithms x 7 sample sizes x 5 chunks of REPS_PER_TASK=1 = 210 tasks
#   task = (algo_idx * 7 + n_idx) * NCHUNK + chunk_idx
# so tasks 0-34 are svidag (5 consecutive tasks per n, one replicate each),
# tasks 35-69 are prodag, and so on.
#
# Why per-n rather than one task per algorithm: ProDAG runs its nonlinear MLP
# mode and DiBS runs the joint nonlinear model with a full-batch likelihood, so
# their largest cells can be expensive and one task covering all seven sizes is
# a poor unit of work. Splitting per n also isolates a timeout to one sample
# size. ProDAG (tasks 35-69) is by far the most expensive row at this size
# (~5 h for the worst single replicate) -- watch it.
#
# To coarsen, raise REPS_PER_TASK back to 5 and narrow --array to
# 6*7*NCHUNK-1 accordingly; the suffix carries the narrowing so the slices
# still merge. Or narrow one cell by hand:
#   CASE6_SAMPLE_SIZES=10000 CASE6_REP_START=0 CASE6_REP_END=1 \
#       sbatch --array=0 run_case6.sh
# (the explicit env vars win; the task-id decode below only sets defaults).
#
# Each task writes its own slice (the suffix carries the narrowing, see
# _single_algo._decorate_suffix), so tasks never contend for a file:
#   paper_results_reproduce/case_6/case_6_results_<algo>_n<N>.csv
#   paper_results_reproduce/case_6/case_6_results_<algo>_n<N>.json
# then regenerates the figure from every slice that exists so far:
#   paper_results_reproduce/case_6/ER_p50_s80_metrics.{pdf,png}
# The CSVs ARE the plot data -- plot_cases.py reads them directly, so the
# figure can always be rebuilt without rerunning any algorithm:
#   python paper_results_reproduce/plot_cases.py --cases 6
# ===========================================================================

ALGOS=(svidag prodag bayesdag dds dibs bcd)
# Must cover every entry of case_6/common.py SAMPLE_SIZES (order here is free --
# the RNG seed comes from the position in that list, not in this one).
SIZES=(10 32 100 316 1000 3162 10000)
NREPS=5
# One REPLICATE per task (not 5). Each task is then a single
# (algorithm, n, replicate) cell of well under an hour, which both fits
# inside a short scheduling window and caps the work lost if a task is
# killed mid-flight. Set back to 5 to amortise the ~9 min XLA compile
# across a cell's replicates when walltime is not scarce; then also set
# --array=0-41 below.
REPS_PER_TASK=1
NCHUNK=$(( NREPS / REPS_PER_TASK ))

T=$SLURM_ARRAY_TASK_ID
CHUNK=$(( T % NCHUNK ))
NIDX=$(( (T / NCHUNK) % ${#SIZES[@]} ))
AIDX=$(( T / (NCHUNK * ${#SIZES[@]}) ))

ALGO=${ALGOS[$AIDX]}
NSAMP=${SIZES[$NIDX]}
# ${VAR:=default} so an explicitly exported override at submit time wins.
: "${CASE6_SAMPLE_SIZES:=$NSAMP}"
: "${CASE6_REP_START:=$(( CHUNK * REPS_PER_TASK ))}"
: "${CASE6_REP_END:=$(( CHUNK * REPS_PER_TASK + REPS_PER_TASK ))}"
export CASE6_SAMPLE_SIZES CASE6_REP_START CASE6_REP_END
echo "=== case 6 | array task $T -> $ALGO @ n=$CASE6_SAMPLE_SIZES "\
"reps [$CASE6_REP_START,$CASE6_REP_END) | $(hostname) ==="

# ---------------------------------------------------------------------------
# Python environment. REQUIRED -- there is no sensible default.
#
# Point SVIDAG_ENV_ACTIVATE at the activate script of an environment holding
# the pinned package set (the one setup_env.sh builds, or any equivalent).
# Either edit the default below, or export it at submit time:
#
#     SVIDAG_ENV_ACTIVATE=/path/to/env/bin/activate sbatch run_case6.sh
#
# For a conda environment named "svidag":
#
#     SVIDAG_ENV_ACTIVATE=$(conda info --base)/envs/svidag/bin/activate
#
# If your cluster needs modules loaded first, list them in SVIDAG_MODULES:
#
#     SVIDAG_MODULES="anaconda cuda/12.4" SVIDAG_ENV_ACTIVATE=... sbatch ...
# ---------------------------------------------------------------------------
: "${SVIDAG_MODULES:=}"
if [ -n "$SVIDAG_MODULES" ]; then
    # shellcheck disable=SC2086
    module load $SVIDAG_MODULES
fi

: "${SVIDAG_ENV_ACTIVATE:=}"
if [ -z "$SVIDAG_ENV_ACTIVATE" ]; then
    echo "error: SVIDAG_ENV_ACTIVATE is not set -- this script does not know" >&2
    echo "       which Python environment to use. Edit run_case6.sh or export" >&2
    echo "       it at submit time. See README.md, 'Running on a Slurm cluster'." >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$SVIDAG_ENV_ACTIVATE"

# ---------------------------------------------------------------------------
# Exact reproducibility.
#
# PYTHONHASHSEED pins str hashing, which the per-case seed derivation depends
# on; PYTHONNOUSERSITE keeps ~/.local out of the import path so the pinned
# package set in requirements.txt is what actually runs. Every algorithm seeds
# off CASE<N>_SEED (default 0), and the hyperparameters sourced below are
# fully explicit, so reruns are reproducible -- with the sole exception of the
# wall-clock timings, which are hardware dependent by nature.
# ---------------------------------------------------------------------------
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ---------------------------------------------------------------------------
# Hyperparameters for this case.
#
# profiles/case6.env holds the exact settings required to reproduce case 6.
# The same file is sourced by ./run_local.sh, so a Slurm run and a local run of
# case 6 use identical hyperparameters. Change a value there, not here --
# there is no second copy to keep in sync. See profiles/README.md.
#
# Sourcing it also clears any SVIDAG_* inherited from the submitting shell, so
# a stray export at submit time cannot silently alter the run.
# ---------------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR" || exit 1
# shellcheck disable=SC1091
source profiles/case6.env

if [ "$ALGO" = "svidag" ]; then
    echo "  [svidag] hyperparameters:"
    env | grep -E "^SVIDAG_" | sort | sed 's/^/    /'
fi

srun python -u "paper_results_reproduce/case_6/run_${ALGO}_only.py"
STATUS=$?

# Refresh the figure from every CSV present so far. Idempotent.
python -u paper_results_reproduce/plot_cases.py --cases 6 || true
exit $STATUS
