#!/bin/bash -l
# ===========================================================================
# One ablation cell: VARIANT over SEEDS at the graph/budget given by env.
#
# Normally submitted by submit_ablation.sh, which forwards every site-
# specific value via --export.  To submit it directly, supply at least:
#
#   sbatch -p <partition> --gres=gpu:1 -c 8 --mem=200G \
#          --export=ALL,VARIANT=full,P=20,S_EDGES=40,NTRAIN=300,ITERS=2000 \
#          ablation_cell.sbatch
#
# Required via --export: VARIANT.
# Optional: REPO (derived from SLURM_SUBMIT_DIR if unset), CONDA_SH
# (auto-detected), CONDA_ENV (svidag), P (10), S_EDGES (20), NTRAIN (300),
# ITERS (from profile), PSAMP (1000), SEEDS (0-9), BIAS (0), OUTFILE
# (default results/ path inside run_ablation.py).
#
# Resume-safe: run_ablation.py skips seeds already in its JSON when the
# config fingerprint matches, so a preemption/requeue redoes at most one seed.
# ===========================================================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -uo pipefail

# Repo root: forwarded by submit_ablation.sh, else derived from the directory
# the job was submitted from (this script lives at <repo>/paper_results_reproduce/ablation).
if [ -z "${REPO:-}" ]; then
    REPO="$(cd "${SLURM_SUBMIT_DIR:-$PWD}" && cd ../.. && pwd)" 2>/dev/null \
        || { echo "set REPO=<path to SVI-DAG checkout> via --export" >&2; exit 1; }
fi
[ -d "$REPO/paper_results_reproduce/ablation" ] \
    || { echo "REPO=$REPO is not an SVI-DAG checkout" >&2; exit 1; }
cd "$REPO/paper_results_reproduce/ablation" || exit 1

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU"

# Conda: forwarded by submit_ablation.sh, else auto-detected on this node.
if [ -z "${CONDA_SH:-}" ]; then
    CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
fi
[ -r "$CONDA_SH" ] || { echo "cannot read CONDA_SH=$CONDA_SH" >&2; exit 1; }
source "$CONDA_SH"
conda activate "${CONDA_ENV:-svidag}" || exit 1

export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
. "$REPO/profiles/case3.env"

# The dense-graph nonlinear ablation drops the case-3 sampling bias (a p=25
# benchmark calibration) unless the caller sets BIAS explicitly.
export SVIDAG_POSTERIOR_BIAS_INTERCEPT="${BIAS:-0}"

ARGS=(--variant "$VARIANT"
      --p-nodes "${P:-10}" --s-edges "${S_EDGES:-20}"
      --n-train "${NTRAIN:-300}" --num-iters "${ITERS:-1500}"
      --posterior-samples "${PSAMP:-1000}"
      --seeds "${SEEDS:-0-9}" --verbose)
[ -n "${OUTFILE:-}" ] && ARGS+=(--out "$OUTFILE")

echo "cell: variant=$VARIANT p=${P:-10} s=${S_EDGES:-20} n=${NTRAIN:-300}" \
     "iters=${ITERS:-1500} seeds=${SEEDS:-0-9} bias=${BIAS:-0}"

python -u run_ablation.py "${ARGS[@]}"
status=$?

echo "exit=$status  end=$(date)"
exit $status
