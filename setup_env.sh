#!/usr/bin/env bash
# ===========================================================================
# setup_env.sh -- one-command install into a fresh conda environment.
#
#     bash setup_env.sh              # auto-detect GPU, build the right env
#     bash setup_env.sh --cpu        # force the CPU-only environment
#     bash setup_env.sh --gpu        # force the GPU environment
#     bash setup_env.sh --force      # delete and rebuild an existing env
#     bash setup_env.sh --name myenv # use a different environment name
#
# Creates a NEW conda environment, so nothing already installed on the machine
# is touched or upgraded. Everything lands in that environment and disappears
# with `conda env remove -n <name>`.
#
# What it does:
#   1. finds conda and checks for an NVIDIA GPU
#   2. creates the environment from environment.yml / environment-cpu.yml
#      (Python 3.11 + the pinned lockfile)
#   3. installs svidag itself in editable mode with --no-deps
#   4. runs scripts/check_env.py to verify the result
# ===========================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT" || exit 1

MODE="auto"
FORCE=0
ENV_NAME=""

while [ $# -gt 0 ]; do
    case "$1" in
        --cpu)   MODE="cpu"; shift ;;
        --gpu)   MODE="gpu"; shift ;;
        --force) FORCE=1; shift ;;
        --name)  ENV_NAME="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "error: unknown option '$1' (try --help)" >&2; exit 1 ;;
    esac
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Locate conda
# ---------------------------------------------------------------------------
say "Locating conda"
# `conda` is frequently only a shell FUNCTION defined by conda init in the
# user's rc file, and functions are not exported into `bash setup_env.sh`.
# Fall back to $CONDA_EXE (which conda init does export) before giving up.
if ! command -v conda >/dev/null 2>&1; then
    if [ -n "${CONDA_EXE:-}" ] && [ -x "$CONDA_EXE" ]; then
        PATH="$(dirname "$CONDA_EXE"):$PATH"
        export PATH
    else
        fail "conda is not on PATH.
       Install Miniconda (or Miniforge) first:
           https://docs.conda.io/en/latest/miniconda.html
       then re-run: bash setup_env.sh
       If conda IS installed, run this from a shell where 'conda --version'
       works, or export CONDA_EXE=/path/to/conda/bin/conda first."
    fi
fi
echo "    $(conda --version)  ($(command -v conda))"

# `conda activate` needs the shell hook; `conda env create` alone does not.
CONDA_BASE="$(conda info --base)" || fail "could not run 'conda info --base'"
# shellcheck disable=SC1091
. "$CONDA_BASE/etc/profile.d/conda.sh" || fail "could not source conda.sh from $CONDA_BASE"

# ---------------------------------------------------------------------------
# 2. Decide GPU vs CPU
# ---------------------------------------------------------------------------
say "Selecting environment flavour"
if [ "$MODE" = "auto" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        MODE="gpu"
        echo "    NVIDIA GPU detected:"
        nvidia-smi -L | sed 's/^/      /'
    else
        MODE="cpu"
        echo "    No NVIDIA GPU found (nvidia-smi absent or reported none)."
        echo "    Falling back to the CPU environment. Force with --gpu."
    fi
fi

if [ "$MODE" = "gpu" ]; then
    ENV_FILE="environment.yml"
    [ -n "$ENV_NAME" ] || ENV_NAME="svidag"
    # The pinned wheels are CUDA 12.4; warn (do not block) on an older driver.
    if command -v nvidia-smi >/dev/null 2>&1; then
        DRIVER_CUDA="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)"
        if [ -n "$DRIVER_CUDA" ]; then
            echo "    Driver reports CUDA $DRIVER_CUDA (wheels target the 12.4 family)."
            case "$DRIVER_CUDA" in
                12.[4-9]*|12.1[0-9]*|1[3-9].*|[2-9][0-9].*) ;;
                *) echo "    WARNING: this is older than CUDA 12.4. JAX may fail to"
                   echo "             initialise the GPU. Update the NVIDIA driver, or"
                   echo "             re-run with --cpu." ;;
            esac
        fi
    fi
else
    ENV_FILE="environment-cpu.yml"
    [ -n "$ENV_NAME" ] || ENV_NAME="svidag-cpu"
fi
echo "    mode = $MODE   file = $ENV_FILE   env name = $ENV_NAME"
[ -f "$ENV_FILE" ] || fail "missing $ENV_FILE"

# ---------------------------------------------------------------------------
# 3. Create the environment
# ---------------------------------------------------------------------------
say "Creating conda environment '$ENV_NAME'"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    if [ "$FORCE" -eq 1 ]; then
        echo "    '$ENV_NAME' exists; removing it (--force)."
        conda env remove -n "$ENV_NAME" -y || fail "could not remove '$ENV_NAME'"
    else
        fail "conda environment '$ENV_NAME' already exists.
       Rebuild it:        bash setup_env.sh --force
       Use another name:  bash setup_env.sh --name svidag2
       Or just use it:    conda activate $ENV_NAME"
    fi
fi

# The env file names the environment; --name here keeps a custom --name working.
conda env create -f "$ENV_FILE" --name "$ENV_NAME" \
    || fail "environment creation failed.
       The pip step is the usual culprit -- scroll up for the failing package.
       On a slow or flaky connection, simply re-running often succeeds."

# ---------------------------------------------------------------------------
# 4. Install svidag itself
# ---------------------------------------------------------------------------
say "Installing svidag (editable, --no-deps)"
conda activate "$ENV_NAME" || fail "could not activate '$ENV_NAME'"
# --no-deps matters: the lockfile is the authority on versions, and
# pyproject.toml lists its dependencies unpinned so a plain editable install
# could float them.
pip install -e . --no-deps || fail "editable install of svidag failed"

# ---------------------------------------------------------------------------
# 5. Verify
# ---------------------------------------------------------------------------
say "Verifying the installation"
python scripts/check_env.py
CHECK_STATUS=$?

echo
echo "==========================================================================="
if [ "$CHECK_STATUS" -eq 0 ]; then
    echo "  Environment '$ENV_NAME' is ready."
else
    echo "  Environment '$ENV_NAME' was created, but the checks above FAILED."
    echo "  Fix those before trusting any results."
fi
echo
echo "  Activate it in every new shell:"
echo "      conda activate $ENV_NAME"
echo
echo "  Then try the fastest end-to-end run (a few minutes):"
echo "      ./run_local.sh 1"
echo
echo "  Or smoke-test a full benchmark case:"
echo "      ./run_local.sh 2 --quick"
echo
echo "  See README.md, \"Reproducing the paper's six cases\", for the rest."
echo "==========================================================================="
exit "$CHECK_STATUS"
