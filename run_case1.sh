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
#SBATCH --time=06:00:00
#SBATCH --job-name=svidag_case1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# One GPU. Some clusters use --gres instead; if the line below is rejected,
# replace it with:   #SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1

## Optional email notifications -- uncomment and set your own address:
##SBATCH --mail-user=you@example.com
##SBATCH --mail-type=END,FAIL

# ===========================================================================
# Case 1 -- effect of the domain-informed prior on a 2-node graph.
#
#   sbatch run_case1.sh
#
# Trains SVI-DAG under the three prior scenarios (incorrect / noninformative /
# correct) on both the linear and the nonlinear 2-node generator, draws 10000
# hard posterior DAG samples A = B (.) M(r) from the generative construction,
# and writes:
#     paper_results_reproduce/case_1/case_1_results.json
#     paper_results_reproduce/case_1/case_1_table.tex
#     paper_results_reproduce/case_1/case_1_figure_data.csv
# ===========================================================================

# ---------------------------------------------------------------------------
# Python environment. REQUIRED -- there is no sensible default.
#
# Point SVIDAG_ENV_ACTIVATE at the activate script of an environment holding
# the pinned package set (the one setup_env.sh builds, or any equivalent).
# Either edit the default below, or export it at submit time:
#
#     SVIDAG_ENV_ACTIVATE=/path/to/env/bin/activate sbatch run_case1.sh
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
    echo "       which Python environment to use. Edit run_case1.sh or export" >&2
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
# profiles/case1.env holds the exact settings required to reproduce case 1.
# The same file is sourced by ./run_local.sh, so a Slurm run and a local run of
# case 1 use identical hyperparameters. Change a value there, not here --
# there is no second copy to keep in sync. See profiles/README.md.
#
# Sourcing it also clears any SVIDAG_* inherited from the submitting shell, so
# a stray export at submit time cannot silently alter the run.
# ---------------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR" || exit 1
# shellcheck disable=SC1091
source profiles/case1.env

echo "=== case 1 | $(hostname) ==="
env | grep -E "^SVIDAG_" | sort | sed 's/^/  /'

srun python -u paper_results_reproduce/case_1/run_case1.py
