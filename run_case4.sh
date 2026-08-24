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
#SBATCH --job-name=svidag_case4
#SBATCH --array=0-5
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# One GPU. Some clusters use --gres instead; if the line below is rejected,
# replace it with:   #SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1

## Optional email notifications -- uncomment and set your own address:
##SBATCH --mail-user=you@example.com
##SBATCH --mail-type=END,FAIL

# ===========================================================================
# Case 4 -- SVI-DAG vs 5 baselines on the Sachs dataset, 10 splits.
#
#   sbatch run_case4.sh                 # all six algorithms
#   sbatch --array=0 run_case4.sh       # SVI-DAG only
#
# One array task per algorithm, each on its own GPU: run sequentially the
# total exceeds any single-job walltime (BayesDAG and DDS alone are hours).
# Each task writes paper_results_reproduce/case_4/case_4_results_<algo>.json,
# then rebuilds the tables from every result file present, so the last task to
# finish leaves a complete set:
#     case_4_table_dag.tex        oriented-DAG metrics
#     case_4_table_cpdag.tex      Markov-equivalence-class metrics
#     case_4_table_runtime.tex    wall-clock time, mean +/- SE per algorithm
#     case_4_table.tex            all three concatenated
# ===========================================================================

ALGOS=(svidag prodag bayesdag dds dibs bcd)
ALGO=${ALGOS[$SLURM_ARRAY_TASK_ID]}
echo "=== case 4 | array task $SLURM_ARRAY_TASK_ID -> $ALGO | $(hostname) ==="

# ---------------------------------------------------------------------------
# Python environment. REQUIRED -- there is no sensible default.
#
# Point SVIDAG_ENV_ACTIVATE at the activate script of an environment holding
# the pinned package set (the one setup_env.sh builds, or any equivalent).
# Either edit the default below, or export it at submit time:
#
#     SVIDAG_ENV_ACTIVATE=/path/to/env/bin/activate sbatch run_case4.sh
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
    echo "       which Python environment to use. Edit run_case4.sh or export" >&2
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
# profiles/case4.env holds the exact settings required to reproduce case 4.
# The same file is sourced by ./run_local.sh, so a Slurm run and a local run of
# case 4 use identical hyperparameters. Change a value there, not here --
# there is no second copy to keep in sync. See profiles/README.md.
#
# Sourcing it also clears any SVIDAG_* inherited from the submitting shell, so
# a stray export at submit time cannot silently alter the run.
# ---------------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR" || exit 1
# shellcheck disable=SC1091
source profiles/case4.env

if [ "$ALGO" = "svidag" ]; then
    echo "  [svidag] hyperparameters:"
    env | grep -E "^SVIDAG_" | sort | sed 's/^/    /'
fi

srun python -u "paper_results_reproduce/case_4/run_${ALGO}_only.py"
STATUS=$?

# Rebuild all three tables from whatever result files exist. Idempotent, so
# every finishing task refreshes them; algorithms not yet done render as "--".
python -u paper_results_reproduce/case_4/make_tables.py || true
exit $STATUS
