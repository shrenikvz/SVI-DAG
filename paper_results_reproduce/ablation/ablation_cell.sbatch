#!/bin/bash -l
# ===========================================================================
# One ablation cell: VARIANT over SEEDS at the graph/budget given by env.
#
# Required via --export: VARIANT.
# Optional: P (10), S_EDGES (20), NTRAIN (300), ITERS (from profile), SEEDS
# (0-9), BIAS (0), OUTFILE (default results/ path inside run_ablation.py).
#
# Resume-safe: run_ablation.py skips seeds already in its JSON when the
# config fingerprint matches, so a preemption/requeue redoes at most one seed.
# ===========================================================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -uo pipefail

REPO=/home/shrenik/SVI-DAG
cd "$REPO/paper_results_reproduce/ablation" || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID  start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU"

source /orcd/software/core/001/pkg/miniforge/25.11.0-0/etc/profile.d/conda.sh
conda activate svidag || exit 1

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
