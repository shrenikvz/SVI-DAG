#!/bin/bash
# ===========================================================================
# Reproduce the ablation table (ablation_table.tex) end to end on any
# Slurm cluster.
#
#   GPU_PARTITION=<gpu-partition> CPU_PARTITION=<cpu-partition> \
#       bash submit_ablation.sh
#
# Set the site-specific values in the CONFIGURATION block below, either by
# editing this file or by exporting them before running.  Nothing else in
# the ablation directory is site specific.
#
# Two studies feed the table:
#
#   MAIN (columns 1-4): nonlinear ER p=20 s=40, n=300 (240 train / 60
#     holdout inside run_ablation.py), 2000 iters, sampling bias 0, 10 seeds.
#     Stock-trainer variants (full, no_flow, no_prior) run on GPU and need
#     GPU_MEM>=200G: XLA's compile of svidag's train_step exceeds 64G host
#     RAM at p>=20, and an undersized cgroup shows up as a silent D-state
#     stall, not a clean OOM.  The Gaussian-r variants (no_svgd,
#     no_flow_no_svgd) run on CPU in 2-seed chunks: their train step stalls
#     XLA:GPU compilation indefinitely (see ablation_lib._make_gauss_step)
#     but compiles and runs fine on CPU.  Chunks write to results/chunks/
#     and merge_chunks.py assembles the canonical per-variant JSONs.
#
#   MEC COMPANION (column 5): linear ER p=10 s=10, n=1000, 1500 iters,
#     profile sampling bias (-1.0 -- deliberately NOT overridden; bias
#     removal zeroes exact-member coverage), 10 seeds, via run_mec_study.py.
#     Same GPU/CPU split by variant.
#
# A dependent job merges chunks and renders ablation_table.tex.
# ===========================================================================
set -uo pipefail

# --- CONFIGURATION: fill these in for your cluster -------------------------
# Required.  Names of the Slurm partitions to submit to (sinfo -s to list).
GPU_PARTITION="${GPU_PARTITION:-}"
CPU_PARTITION="${CPU_PARTITION:-}"

# Optional.  Slurm account / QoS, if your site requires one (-A / -q).
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_QOS="${SLURM_QOS:-}"

# GPU request string passed to --gres.
GPU_GRES="${GPU_GRES:-gpu:1}"

# Extra sbatch flags for the GPU jobs, e.g. "--requeue" on a preemptable
# partition.  Left empty by default.
GPU_SBATCH_EXTRA="${GPU_SBATCH_EXTRA:-}"

# Resources.  The GPU memory figure is a real constraint, not a guess --
# see the note above before lowering it.
GPU_MEM="${GPU_MEM:-200G}"
CPU_MEM="${CPU_MEM:-100G}"
MEC_MEM="${MEC_MEM:-64G}"
GPU_TIME="${GPU_TIME:-06:00:00}"
CPU_TIME="${CPU_TIME:-10:00:00}"
GPU_CPUS="${GPU_CPUS:-8}"
CPU_CPUS="${CPU_CPUS:-16}"

# Conda.  CONDA_SH is auto-detected from the active conda installation if
# left empty; set it explicitly if the compute nodes see a different path.
CONDA_SH="${CONDA_SH:-}"
CONDA_ENV="${CONDA_ENV:-svidag}"
# ---------------------------------------------------------------------------

# Repo layout is derived from this script's own location -- do not hardcode.
ABL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ABL/../.." && pwd)"
OUT="${LOG_DIR:-$REPO/logs/ablation_jobs}"

_die() { echo "error: $*" >&2; exit 1; }

[ -n "$GPU_PARTITION" ] || _die "GPU_PARTITION is unset. Run 'sinfo -s' to list
  partitions, then re-run as:
    GPU_PARTITION=<gpu-partition> CPU_PARTITION=<cpu-partition> bash $0"
[ -n "$CPU_PARTITION" ] || _die "CPU_PARTITION is unset (see GPU_PARTITION above)."

if [ -z "$CONDA_SH" ]; then
    _base="$(conda info --base 2>/dev/null)" \
        || _die "conda not found on PATH; set CONDA_SH to your conda.sh path."
    CONDA_SH="$_base/etc/profile.d/conda.sh"
fi
[ -r "$CONDA_SH" ] || _die "CONDA_SH does not exist or is unreadable: $CONDA_SH"
command -v sbatch >/dev/null || _die "sbatch not found; this script needs Slurm."

mkdir -p "$OUT" "$ABL/results/chunks"

# Optional account/QoS flags, expanded only when set.
EXTRA=()
[ -n "$SLURM_ACCOUNT" ] && EXTRA+=(-A "$SLURM_ACCOUNT")
[ -n "$SLURM_QOS" ] && EXTRA+=(-q "$SLURM_QOS")

CONDA_SETUP="source $CONDA_SH; \
conda activate $CONDA_ENV; \
export PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false; \
. $REPO/profiles/case3.env;"

# Forwarded to ablation_cell.sbatch so the cell script needs no site config.
CELL_ENV="REPO=$REPO,CONDA_SH=$CONDA_SH,CONDA_ENV=$CONDA_ENV"

echo "repo:       $REPO"
echo "partitions: gpu=$GPU_PARTITION cpu=$CPU_PARTITION"
echo "conda:      $CONDA_SH (env $CONDA_ENV)"
echo "logs:       $OUT"
echo

ALL_IDS=()

echo "=== main study: stock variants (GPU, $GPU_MEM) ==="
for v in full no_flow no_prior; do
    id=$(sbatch --parsable -p "$GPU_PARTITION" ${EXTRA[@]+"${EXTRA[@]}"} $GPU_SBATCH_EXTRA \
        -t "$GPU_TIME" --gres="$GPU_GRES" \
        -c "$GPU_CPUS" --mem="$GPU_MEM" -J "abl_${v}" \
        -o "$OUT/%x-%j.out" -e "$OUT/%x-%j.err" \
        --export="ALL,$CELL_ENV,VARIANT=$v,P=20,S_EDGES=40,NTRAIN=300,ITERS=2000,SEEDS=0-9,BIAS=0" \
        "$ABL/ablation_cell.sbatch") || _die "sbatch failed for $v"
    ALL_IDS+=("$id"); printf '  %-28s %s\n' "abl_$v" "$id"
done

echo "=== main study: Gaussian-r variants (CPU, 2-seed chunks) ==="
for v in no_svgd no_flow_no_svgd; do
    for c in 0 1 2 3 4; do
        a=$((c*2)); b=$((c*2+1))
        id=$(sbatch --parsable -p "$CPU_PARTITION" ${EXTRA[@]+"${EXTRA[@]}"} \
            -t "$CPU_TIME" -c "$CPU_CPUS" --mem="$CPU_MEM" \
            -J "abl_${v}_s${a}${b}" -o "$OUT/%x-%j.out" -e "$OUT/%x-%j.err" \
            --export="ALL,$CELL_ENV,VARIANT=$v,P=20,S_EDGES=40,NTRAIN=300,ITERS=2000,SEEDS=${a}-${b},BIAS=0,OUTFILE=$ABL/results/chunks/${v}_s${a}${b}.json" \
            "$ABL/ablation_cell.sbatch") || _die "sbatch failed for ${v} seeds ${a}-${b}"
        ALL_IDS+=("$id"); printf '  %-28s %s\n' "abl_${v}_s${a}${b}" "$id"
    done
done

echo "=== MEC companion study ==="
id=$(sbatch --parsable -p "$GPU_PARTITION" ${EXTRA[@]+"${EXTRA[@]}"} $GPU_SBATCH_EXTRA \
    -t "$GPU_TIME" --gres="$GPU_GRES" \
    -c "$GPU_CPUS" --mem="$MEC_MEM" -J abl_mec_gpu \
    -o "$OUT/%x-%j.out" -e "$OUT/%x-%j.err" \
    --wrap="$CONDA_SETUP cd $ABL; \
            python -u run_mec_study.py --variants full,no_flow,no_prior --seeds 0-9") \
    || _die "sbatch failed for the MEC GPU job"
ALL_IDS+=("$id"); printf '  %-28s %s\n' "abl_mec_gpu" "$id"
id=$(sbatch --parsable -p "$CPU_PARTITION" ${EXTRA[@]+"${EXTRA[@]}"} \
    -t "$CPU_TIME" -c "$CPU_CPUS" --mem="$MEC_MEM" -J abl_mec_cpu \
    -o "$OUT/%x-%j.out" -e "$OUT/%x-%j.err" \
    --wrap="$CONDA_SETUP cd $ABL; \
            python -u run_mec_study.py --variants no_svgd,no_flow_no_svgd --seeds 0-9") \
    || _die "sbatch failed for the MEC CPU job"
ALL_IDS+=("$id"); printf '  %-28s %s\n' "abl_mec_cpu" "$id"

echo "=== dependent merge + table job ==="
DEP=$(IFS=:; echo "${ALL_IDS[*]}")
TAB=$(sbatch --parsable -p "$CPU_PARTITION" ${EXTRA[@]+"${EXTRA[@]}"} \
    -t 00:10:00 -c 2 --mem=8G -J abl_table \
    -o "$OUT/%x-%j.out" -e "$OUT/%x-%j.err" --dependency="afterany:$DEP" \
    --wrap="$CONDA_SETUP cd $ABL; python merge_chunks.py && python make_table.py") \
    || _die "sbatch failed for the table job"
printf '  %-28s %s\n' "abl_table" "$TAB"

printf '%s\n' "${ALL_IDS[@]}" "$TAB" > "$REPO/logs/ablation.jobids"
echo
echo "submitted ${#ALL_IDS[@]} compute jobs + 1 table job -> $REPO/logs/ablation.jobids"
