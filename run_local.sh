#!/usr/bin/env bash
# ===========================================================================
# run_local.sh -- reproduce any case on a single local GPU.
#
# This is the local counterpart of the Slurm scripts run_case<N>.sh. Both
# source the SAME hyperparameter profile from profiles/case<N>.env, so a local
# run and a cluster run of the same case use identical settings. The only
# difference is scheduling: Slurm fans the grid out across array tasks, this
# script walks the same grid sequentially on one device.
#
#   ./run_local.sh 4                          # case 4, all six algorithms
#   ./run_local.sh 2 --algo svidag            # one algorithm, every n
#   ./run_local.sh 2 --algo svidag --n 1000   # a single cell
#   ./run_local.sh 3 --quick                  # minutes-long smoke test
#   ./run_local.sh 2 --list                   # print the work plan, run nothing
#   ./run_local.sh 5 --resume                 # skip cells already on disk
#
# Run ./run_local.sh --help for the full option list.
# ===========================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT" || exit 1

ALL_ALGOS="svidag prodag bayesdag dds dibs bcd"
ALL_SIZES="100 316 1000 3162 10000"
SWEEP_CASES=" 2 3 5 6 "        # cases with an (algorithm x n x replicate) grid

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
CASE=""
ALGOS=""
SIZES=""
REPS=""
QUICK=0
LIST_ONLY=0
DRY_RUN=0
RESUME=0
NO_FIGURES=0
CPU_OK=0

usage() {
    cat <<'EOF'
Usage: ./run_local.sh <case> [options]

  <case>   1..6

Options
  --algo A[,B,...]   Restrict to these algorithms. Default: all six, in the
                     order svidag prodag bayesdag dds dibs bcd.
                     Case 1 trains SVI-DAG only and ignores this flag.
  --n N[,M,...]      Restrict to these sample sizes (cases 2, 3, 5, 6 only).
                     Valid: 100 316 1000 3162 10000.
  --reps a-b         Restrict to replicates [a, b) (cases 2, 3, 5, 6 only).
                     Default: the case's full range, 0-5.
  --resume           Skip any (algorithm, n) cell whose result file already
                     exists. Use this to continue an interrupted sweep.
  --quick            Smoke test: 1 replicate, 1 split, n=100 only, ~100
                     training iterations, tiny posterior sample counts.
                     Finishes in minutes and exercises every code path.
                     *** Does NOT reproduce the published numbers. ***
  --list             Print the work plan and exit without running anything.
  --dry-run          Like --list, but also prints the exact command per cell.
  --no-figures       Skip the figure/table regeneration step at the end.
  --cpu              Proceed even if JAX reports no GPU (very slow).
  -h, --help         This message.

Examples
  ./run_local.sh 1                       Case 1 end to end (~10 min on a GPU).
  ./run_local.sh 4 --algo svidag         SVI-DAG on Sachs, all 10 splits.
  ./run_local.sh 2 --quick               Validate the whole case-2 path fast.
  ./run_local.sh 2 --algo dds --n 10000  Rerun just the cell that timed out.
  ./run_local.sh 3 --resume              Continue after an interruption.

Outputs land in paper_results_reproduce/case_<N>/ and per-cell logs in
logs/case_<N>/. See README.md, "Reproducing the paper's six cases".
EOF
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
if [ $# -eq 0 ]; then usage; exit 1; fi

case "$1" in
    -h|--help) usage; exit 0 ;;
    [1-6])     CASE="$1"; shift ;;
    *)         echo "error: first argument must be a case number 1..6 (got '$1')" >&2
               echo "try: ./run_local.sh --help" >&2; exit 1 ;;
esac

while [ $# -gt 0 ]; do
    case "$1" in
        --algo)       ALGOS="$(echo "$2" | tr ',' ' ')"; shift 2 ;;
        --n)          SIZES="$(echo "$2" | tr ',' ' ')"; shift 2 ;;
        --reps)       REPS="$2"; shift 2 ;;
        --quick)      QUICK=1; shift ;;
        --list)       LIST_ONLY=1; shift ;;
        --dry-run)    LIST_ONLY=1; DRY_RUN=1; shift ;;
        --resume)     RESUME=1; shift ;;
        --no-figures) NO_FIGURES=1; shift ;;
        --cpu)        CPU_OK=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        *)            echo "error: unknown option '$1'" >&2
                      echo "try: ./run_local.sh --help" >&2; exit 1 ;;
    esac
done

CASE_DIR="paper_results_reproduce/case_${CASE}"
PROFILE="profiles/case${CASE}.env"
LOG_DIR="logs/case_${CASE}"

[ -f "$PROFILE" ] || { echo "error: missing profile $PROFILE" >&2; exit 1; }
[ -d "$CASE_DIR" ] || { echo "error: missing $CASE_DIR" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Validate the algorithm / sample-size selections against what the case offers
# ---------------------------------------------------------------------------
is_sweep_case() { case "$SWEEP_CASES" in *" $CASE "*) return 0 ;; *) return 1 ;; esac; }

[ -n "$ALGOS" ] || ALGOS="$ALL_ALGOS"
for a in $ALGOS; do
    case " $ALL_ALGOS " in
        *" $a "*) ;;
        *) echo "error: unknown algorithm '$a'. Valid: $ALL_ALGOS" >&2; exit 1 ;;
    esac
    if [ "$CASE" != "1" ] && [ ! -f "$CASE_DIR/run_${a}_only.py" ]; then
        echo "error: case $CASE has no driver for '$a' ($CASE_DIR/run_${a}_only.py)" >&2
        exit 1
    fi
done

if is_sweep_case; then
    [ -n "$SIZES" ] || SIZES="$ALL_SIZES"
    for n in $SIZES; do
        case " $ALL_SIZES " in
            *" $n "*) ;;
            *) echo "error: sample size '$n' is not on the grid. Valid: $ALL_SIZES" >&2; exit 1 ;;
        esac
    done
else
    if [ -n "$SIZES" ]; then
        echo "error: --n applies to cases 2, 3, 5, 6 only (case $CASE has a single sample size)" >&2
        exit 1
    fi
    if [ -n "$REPS" ]; then
        echo "error: --reps applies to cases 2, 3, 5, 6 only" >&2
        exit 1
    fi
    SIZES=""
fi

if [ -n "$REPS" ]; then
    case "$REPS" in
        [0-9]*-[0-9]*) REP_START="${REPS%%-*}"; REP_END="${REPS##*-}" ;;
        *) echo "error: --reps expects the form a-b (e.g. 0-2), got '$REPS'" >&2; exit 1 ;;
    esac
fi

# ---------------------------------------------------------------------------
# Environment: reproducibility pins, then the case profile.
#
# PYTHONHASHSEED  -- the per-case seed derivation hashes strings, so without
#                    this the generated (graph, data) pairs differ per run.
# PYTHONNOUSERSITE-- keeps ~/.local off the import path so the pinned package
#                    set from requirements.txt is what actually runs.
# XLA_PYTHON_CLIENT_PREALLOCATE=false
#                 -- JAX otherwise grabs 75% of VRAM on first use, which makes
#                    a sequential six-algorithm sweep fail on the second
#                    algorithm on consumer cards.
# ---------------------------------------------------------------------------
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# shellcheck disable=SC1090
. "$PROFILE"

# --quick overrides come AFTER the profile so they win.
if [ "$QUICK" -eq 1 ]; then
    export SVIDAG_NUM_ITERS=100
    export SVIDAG_EVAL_EVERY=50
    export BAYESDAG_EPOCHS=5 BAYESDAG_GRID_EPOCHS=2
    export BAYESDAG_NLAMBDA=2 BAYESDAG_GRID_SAMPLES=8
    export "CASE${CASE}_POSTERIOR_SAMPLES=50"
    if is_sweep_case; then
        export "CASE${CASE}_NUM_REPLICATES=1"
        SIZES="100"
        REP_START=0
        REP_END=1
    else
        export CASE4_NUM_SPLITS=1
    fi
fi

if [ -n "${REP_START:-}" ]; then
    export "CASE${CASE}_REP_START=$REP_START"
    export "CASE${CASE}_REP_END=$REP_END"
fi

# ---------------------------------------------------------------------------
# Build the work plan
# ---------------------------------------------------------------------------
PLAN=""          # newline-separated "<algo>|<n>" ("" for n on non-sweep cases)
SKIPPED=""

case_output_exists() {
    # Mirrors _single_algo._decorate_suffix: the per-cell file carries the
    # narrowing, so a resumed sweep can tell which cells are already done.
    _algo="$1"; _n="$2"
    if [ -n "$_n" ]; then
        [ -f "$CASE_DIR/case_${CASE}_results_${_algo}_n${_n}.csv" ]
    else
        [ -f "$CASE_DIR/case_${CASE}_results_${_algo}.json" ]
    fi
}

if [ "$CASE" = "1" ]; then
    PLAN="svidag|"
else
    for a in $ALGOS; do
        if is_sweep_case; then
            for n in $SIZES; do
                if [ "$RESUME" -eq 1 ] && case_output_exists "$a" "$n"; then
                    SKIPPED="${SKIPPED}${a} n=${n}
"
                    continue
                fi
                PLAN="${PLAN}${a}|${n}
"
            done
        else
            if [ "$RESUME" -eq 1 ] && case_output_exists "$a" ""; then
                SKIPPED="${SKIPPED}${a}
"
                continue
            fi
            PLAN="${PLAN}${a}|
"
        fi
    done
fi

PLAN="$(printf '%s' "$PLAN" | sed '/^$/d')"
N_CELLS=$(printf '%s\n' "$PLAN" | sed '/^$/d' | wc -l | tr -d ' ')

# ---------------------------------------------------------------------------
# Report the plan
# ---------------------------------------------------------------------------
echo "==========================================================================="
echo "  SVI-DAG local run -- case $CASE"
echo "==========================================================================="
echo "  hyperparams  : $PROFILE"
echo "  outputs      : $CASE_DIR/"
echo "  logs         : $LOG_DIR/"
if [ "$CASE" = "1" ]; then
    echo "  work         : 1 run (3 prior scenarios x 2 generators, internally)"
else
    echo "  algorithms   : $ALGOS"
    if is_sweep_case; then
        echo "  sample sizes : $SIZES"
        echo "  replicates   : [${REP_START:-0}, ${REP_END:-5})"
    else
        echo "  splits       : ${CASE4_NUM_SPLITS:-10}"
    fi
    echo "  cells to run : $N_CELLS"
fi
[ "$QUICK" -eq 1 ] && echo "  MODE         : --quick (smoke test; NOT the published numbers)"
if [ -n "$SKIPPED" ]; then
    echo "  resumed, skipping already-complete cells:"
    printf '%s' "$SKIPPED" | sed 's/^/      /'
fi
echo "  SVI-DAG hyperparameters in effect:"
env | grep -E '^SVIDAG_' | sort | sed 's/^/      /'
echo "==========================================================================="

if [ "$N_CELLS" -eq 0 ]; then
    echo
    echo "Nothing to run (every selected cell already has results)."
    echo "Drop --resume to recompute them."
    exit 0
fi

# ---------------------------------------------------------------------------
# Device check
# ---------------------------------------------------------------------------
if [ "$LIST_ONLY" -eq 0 ]; then
    GPU_INFO="$(python -c 'import jax,sys; d=jax.devices(); sys.stdout.write(",".join(x.platform for x in d))' 2>/dev/null)"
    case "$GPU_INFO" in
        *gpu*|*cuda*|*rocm*)
            echo "  device: $(python -c 'import jax; print(jax.devices()[0])' 2>/dev/null)"
            ;;
        "")
            echo "error: could not import jax. Is the conda environment active?" >&2
            echo "       conda activate svidag" >&2
            echo "       (or run: bash setup_env.sh)" >&2
            exit 1
            ;;
        *)
            echo "WARNING: JAX reports no GPU -- it will run on CPU."
            echo "         Devices: $GPU_INFO"
            echo "         A full case on CPU can take days. Diagnose with:"
            echo "             python scripts/check_env.py"
            if [ "$CPU_OK" -eq 0 ]; then
                echo "         Pass --cpu to proceed anyway."
                exit 1
            fi
            echo "         Proceeding because --cpu was given."
            ;;
    esac
    echo
fi

if [ "$LIST_ONLY" -eq 1 ]; then
    echo
    if [ "$N_CELLS" -eq 1 ]; then
        echo "Work plan (1 cell):"
    else
        echo "Work plan ($N_CELLS cells, run in this order):"
    fi
    i=0
    printf '%s\n' "$PLAN" | while IFS='|' read -r a n; do
        i=$((i + 1))
        if [ -n "$n" ]; then
            label="$a  n=$n"
        else
            label="$a"
        fi
        printf '  %3d. %s\n' "$i" "$label"
        if [ "$DRY_RUN" -eq 1 ]; then
            if [ "$CASE" = "1" ]; then
                printf '       python -u %s/run_case1.py\n' "$CASE_DIR"
            else
                [ -n "$n" ] && printf '       CASE%s_SAMPLE_SIZES=%s \\\n' "$CASE" "$n"
                printf '       python -u %s/run_%s_only.py\n' "$CASE_DIR" "$a"
            fi
        fi
    done
    echo
    echo "Nothing was run (--list/--dry-run)."
    exit 0
fi

# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"
RUN_START=$(date +%s)
FAILED=""
DONE=0
IDX=0

# Read the plan from a file descriptor so the loop body does not run in a
# subshell (which would discard FAILED / DONE).
TMP_PLAN="$(mktemp)"
printf '%s\n' "$PLAN" | sed '/^$/d' > "$TMP_PLAN"

while IFS='|' read -r ALGO NSAMP; do
    IDX=$((IDX + 1))
    if [ -n "$NSAMP" ]; then
        CELL="${ALGO}_n${NSAMP}"
        export "CASE${CASE}_SAMPLE_SIZES=$NSAMP"
    else
        CELL="$ALGO"
    fi
    LOG="$LOG_DIR/${CELL}.log"

    echo "---------------------------------------------------------------------------"
    echo "[$IDX/$N_CELLS] case $CASE | $CELL | started $(date '+%H:%M:%S')"
    echo "            log -> $LOG"

    CELL_START=$(date +%s)
    if [ "$CASE" = "1" ]; then
        python -u "$CASE_DIR/run_case1.py" 2>&1 | tee "$LOG"
        STATUS=${PIPESTATUS[0]}
    else
        python -u "$CASE_DIR/run_${ALGO}_only.py" 2>&1 | tee "$LOG"
        STATUS=${PIPESTATUS[0]}
    fi
    CELL_SECS=$(( $(date +%s) - CELL_START ))

    if [ "$STATUS" -eq 0 ]; then
        DONE=$((DONE + 1))
        printf '[%d/%d] OK   %s  (%dm %ds)\n' \
            "$IDX" "$N_CELLS" "$CELL" $((CELL_SECS / 60)) $((CELL_SECS % 60))
    else
        FAILED="${FAILED}${CELL} (exit $STATUS, see $LOG)
"
        printf '[%d/%d] FAIL %s  (exit %d, %dm %ds) -- continuing\n' \
            "$IDX" "$N_CELLS" "$CELL" "$STATUS" $((CELL_SECS / 60)) $((CELL_SECS % 60))
    fi
done < "$TMP_PLAN"
rm -f "$TMP_PLAN"

# ---------------------------------------------------------------------------
# Figures and tables.
#
# Regenerated from whatever result files exist, so a partial sweep still
# renders (missing algorithms show as "--") and a completed one leaves a full
# set. Both steps are idempotent and read only committed result files, so they
# can be rerun at any time without refitting anything.
# ---------------------------------------------------------------------------
if [ "$NO_FIGURES" -eq 0 ]; then
    echo "---------------------------------------------------------------------------"
    case "$CASE" in
        1) echo "Case 1 writes its own table and figure data; nothing to regenerate." ;;
        4) echo "Rebuilding case-4 tables from every result file present ..."
           python -u "$CASE_DIR/make_tables.py" || echo "  (table rebuild failed; results are still on disk)" ;;
        *) echo "Rebuilding the case-$CASE figure from every CSV present ..."
           python -u paper_results_reproduce/plot_cases.py --cases "$CASE" \
               || echo "  (figure rebuild failed; the CSVs are still on disk)" ;;
    esac
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL=$(( $(date +%s) - RUN_START ))
echo "==========================================================================="
printf '  case %s finished in %dh %dm %ds\n' \
    "$CASE" $((TOTAL / 3600)) $(((TOTAL % 3600) / 60)) $((TOTAL % 60))
echo "  cells succeeded : $DONE / $N_CELLS"
if [ -n "$FAILED" ]; then
    echo "  cells failed    :"
    printf '%s' "$FAILED" | sed 's/^/      /'
    echo
    echo "  Rerun just the failures, e.g.:"
    echo "      ./run_local.sh $CASE --algo <algo> ${SIZES:+--n <n>}"
    echo "  Or continue the sweep, skipping what already succeeded:"
    echo "      ./run_local.sh $CASE --resume"
fi
echo "  results  -> $CASE_DIR/"
echo "  logs     -> $LOG_DIR/"
echo "==========================================================================="

[ -n "$FAILED" ] && exit 1
exit 0
