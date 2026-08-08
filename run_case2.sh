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
#SBATCH --job-name=svidag_case2
#SBATCH --array=0-29
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# One GPU. Some clusters use --gres instead; if the line below is rejected,
# replace it with:   #SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1

## Optional email notifications -- uncomment and set your own address:
##SBATCH --mail-user=you@example.com
##SBATCH --mail-type=END,FAIL

# ===========================================================================
# Case 2 -- SVI-DAG vs 5 baselines on synthetic linear-Gaussian data (ER, p=25, s=40),
# CPDAG-level metrics, 5 sample sizes (n = 10^2..10^4 in half-decades)
# x 5 replicates.
#
#   sbatch run_case2.sh                    # the whole grid (30 tasks)
#   sbatch --array=0-4 run_case2.sh        # SVI-DAG, every n
#   sbatch --array=19 run_case2.sh         # DDS at n=10^4 only
#
# ONE ARRAY TASK PER (ALGORITHM, SAMPLE SIZE):
#   6 algorithms x 5 sample sizes x 1 chunk of REPS_PER_TASK=5 = 30 tasks
#   task = (algo_idx * 5 + n_idx) * NCHUNK + chunk_idx
# so tasks 0-4 are svidag at n=100..10^4, tasks 5-9 are prodag, and so on.
#
# Why per-n rather than one task per algorithm: the baselines whose per-step
# cost is linear in the sample size (DDS, BCD, the nonlinear DiBS/ProDAG
# models) still cost ~10x more at n=10^4 than at n=10^3, so one task covering
# all five sizes is a poor unit of work -- and a timeout in the largest cell
# would take the cheap cells with it. Splitting per-n bounds that blast radius.
#
# To split further, drop REPS_PER_TASK (e.g. to 1) and widen --array to
# 6*5*NCHUNK-1 accordingly; the suffix carries the narrowing so the slices
# still merge. Or narrow one cell by hand:
#   CASE2_SAMPLE_SIZES=10000 CASE2_REP_START=0 CASE2_REP_END=1 \
#       sbatch --array=0 run_case2.sh
# (the explicit env vars win; the task-id decode below only sets defaults).
#
# Each task writes its own slice (the suffix carries the narrowing, see
# _single_algo._decorate_suffix), so tasks never contend for a file:
#   paper_results_reproduce/case_2/case_2_results_<algo>_n<N>.csv
#   paper_results_reproduce/case_2/case_2_results_<algo>_n<N>.json
# then regenerates the figure from every slice that exists so far:
#   paper_results_reproduce/case_2/ER_p25_s40_metrics.{pdf,png}
# The CSVs ARE the plot data -- plot_cases.py reads them directly, so the
# figure can always be rebuilt without rerunning any algorithm:
#   python paper_results_reproduce/plot_cases.py --cases 2
# ===========================================================================

ALGOS=(svidag prodag bayesdag dds dibs bcd)
SIZES=(100 316 1000 3162 10000)
NREPS=5
REPS_PER_TASK=5
NCHUNK=$(( NREPS / REPS_PER_TASK ))

T=$SLURM_ARRAY_TASK_ID
CHUNK=$(( T % NCHUNK ))
NIDX=$(( (T / NCHUNK) % ${#SIZES[@]} ))
AIDX=$(( T / (NCHUNK * ${#SIZES[@]}) ))

ALGO=${ALGOS[$AIDX]}
NSAMP=${SIZES[$NIDX]}
# ${VAR:=default} so an explicitly exported override at submit time wins.
: "${CASE2_SAMPLE_SIZES:=$NSAMP}"
: "${CASE2_REP_START:=$(( CHUNK * REPS_PER_TASK ))}"
: "${CASE2_REP_END:=$(( CHUNK * REPS_PER_TASK + REPS_PER_TASK ))}"
export CASE2_SAMPLE_SIZES CASE2_REP_START CASE2_REP_END
echo "=== case 2 | array task $T -> $ALGO @ n=$CASE2_SAMPLE_SIZES "\
"reps [$CASE2_REP_START,$CASE2_REP_END) | $(hostname) ==="

# ---------------------------------------------------------------------------
# Python environment. REQUIRED -- there is no sensible default.
#
# Point SVIDAG_ENV_ACTIVATE at the activate script of an environment holding
# the pinned package set (the one setup_env.sh builds, or any equivalent).
# Either edit the default below, or export it at submit time:
#
#     SVIDAG_ENV_ACTIVATE=/path/to/env/bin/activate sbatch run_case2.sh
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
    echo "       which Python environment to use. Edit run_case2.sh or export" >&2
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
# fully explicit, so a rerun on any machine reproduces these numbers --
# with the sole exception of the wall-clock timings, which are hardware
# dependent by nature.
# ---------------------------------------------------------------------------
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ---------------------------------------------------------------------------
# Hyperparameters for this case.
#
# profiles/case2.env holds the exact settings required to reproduce case 2.
# The same file is sourced by ./run_local.sh, so a Slurm run and a local run of
# case 2 use identical hyperparameters. Change a value there, not here --
# there is no second copy to keep in sync. See profiles/README.md.
#
# Sourcing it also clears any SVIDAG_* inherited from the submitting shell, so
# a stray export at submit time cannot silently alter the run.
# ---------------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR" || exit 1
# shellcheck disable=SC1091
source profiles/case2.env

if [ "$ALGO" = "svidag" ]; then
    echo "  [svidag] hyperparameters:"
    env | grep -E "^SVIDAG_" | sort | sed 's/^/    /'
fi

srun python -u "paper_results_reproduce/case_2/run_${ALGO}_only.py"
STATUS=$?

# Refresh the figure from every CSV present so far. Idempotent.
python -u paper_results_reproduce/plot_cases.py --cases 2 || true
exit $STATUS
