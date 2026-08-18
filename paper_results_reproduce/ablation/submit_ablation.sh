#!/bin/bash
# ===========================================================================
# Reproduce the ablation table (ablation_table.tex) end to end.
#
#   bash submit_ablation.sh
#
# Two studies feed the table:
#
#   MAIN (columns 1-4): nonlinear ER p=20 s=40, n=300 (240 train / 60
#     holdout inside run_ablation.py), 2000 iters, sampling bias 0, 10 seeds.
#     Stock-trainer variants (full, no_flow, no_prior) run on GPU and need
#     --mem=200G: XLA's compile of svidag's train_step exceeds 64G host RAM
#     at p>=20, and an undersized cgroup shows up as a silent D-state stall,
#     not a clean OOM.  The Gaussian-r variants (no_svgd, no_flow_no_svgd)
#     run on CPU in 2-seed chunks: their train step stalls XLA:GPU
#     compilation indefinitely (see ablation_lib._make_gauss_step) but
#     compiles and runs fine on CPU.  Chunks write to results/chunks/ and
#     merge_chunks.py assembles the canonical per-variant JSONs.
#
#   MEC COMPANION (column 5): linear ER p=10 s=10, n=1000, 1500 iters,
#     profile sampling bias (-1.0 -- deliberately NOT overridden; bias
#     removal zeroes exact-member coverage), 10 seeds, via run_mec_study.py.
#     Same GPU/CPU split by variant.  p=10 is load-bearing; see the note in
#     run_mec_study.py before changing it.
#
# A dependent job merges chunks and renders ablation_table.tex.
# ===========================================================================
set -uo pipefail

REPO=/home/shrenik/SVI-DAG
ABL="$REPO/paper_results_reproduce/ablation"
OUT="$REPO/logs/ablation_jobs"
mkdir -p "$OUT" "$ABL/results/chunks"

CONDA_SETUP="source /orcd/software/core/001/pkg/miniforge/25.11.0-0/etc/profile.d/conda.sh; \
conda activate svidag; \
export PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false; \
. $REPO/profiles/case3.env;"

ALL_IDS=()

echo "=== main study: stock variants (GPU, 200G) ==="
for v in full no_flow no_prior; do
    id=$(sbatch --parsable -p mit_preemptable -t 06:00:00 --gres=gpu:1 --requeue \
        -c 8 --mem=200G -J "abl_${v}" -o "$OUT/%x-%j.out" -e "$OUT/%x-%j.err" \
        --export="ALL,VARIANT=$v,P=20,S_EDGES=40,NTRAIN=300,ITERS=2000,SEEDS=0-9,BIAS=0" \
        "$ABL/ablation_cell.sbatch")
    ALL_IDS+=("$id"); printf '  %-28s %s\n' "abl_$v" "$id"
done

echo "=== main study: Gaussian-r variants (CPU, 2-seed chunks) ==="
for v in no_svgd no_flow_no_svgd; do
    for c in 0 1 2 3 4; do
        a=$((c*2)); b=$((c*2+1))
        id=$(sbatch --parsable -p mit_normal -t 10:00:00 -c 16 --mem=100G \
            -J "abl_${v}_s${a}${b}" -o "$OUT/%x-%j.out" -e "$OUT/%x-%j.err" \
            --export="ALL,VARIANT=$v,P=20,S_EDGES=40,NTRAIN=300,ITERS=2000,SEEDS=${a}-${b},BIAS=0,OUTFILE=$ABL/results/chunks/${v}_s${a}${b}.json" \
            "$ABL/ablation_cell.sbatch")
        ALL_IDS+=("$id"); printf '  %-28s %s\n' "abl_${v}_s${a}${b}" "$id"
    done
done

echo "=== MEC companion study ==="
id=$(sbatch --parsable -p mit_preemptable -t 04:00:00 --gres=gpu:1 --requeue \
    -c 8 --mem=64G -J abl_mec_gpu -o "$OUT/%x-%j.out" -e "$OUT/%x-%j.err" \
    --wrap="$CONDA_SETUP cd $ABL; \
            python -u run_mec_study.py --variants full,no_flow,no_prior --seeds 0-9")
ALL_IDS+=("$id"); printf '  %-28s %s\n' "abl_mec_gpu" "$id"
id=$(sbatch --parsable -p mit_normal -t 10:00:00 -c 16 --mem=64G \
    -J abl_mec_cpu -o "$OUT/%x-%j.out" -e "$OUT/%x-%j.err" \
    --wrap="$CONDA_SETUP cd $ABL; \
            python -u run_mec_study.py --variants no_svgd,no_flow_no_svgd --seeds 0-9")
ALL_IDS+=("$id"); printf '  %-28s %s\n' "abl_mec_cpu" "$id"

echo "=== dependent merge + table job ==="
DEP=$(IFS=:; echo "${ALL_IDS[*]}")
TAB=$(sbatch --parsable -p mit_normal -t 00:10:00 -c 2 --mem=8G -J abl_table \
    -o "$OUT/%x-%j.out" -e "$OUT/%x-%j.err" --dependency="afterany:$DEP" \
    --wrap="$CONDA_SETUP cd $ABL; python merge_chunks.py && python make_table.py")
printf '  %-28s %s\n' "abl_table" "$TAB"

printf '%s\n' "${ALL_IDS[@]}" "$TAB" > "$REPO/logs/ablation.jobids"
echo
echo "submitted ${#ALL_IDS[@]} compute jobs + 1 table job -> $REPO/logs/ablation.jobids"
