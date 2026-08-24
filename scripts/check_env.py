#!/usr/bin/env python
"""
Verify that an SVI-DAG environment is ready to reproduce the paper's cases.

    python scripts/check_env.py

Run by ``setup_env.sh`` after building the conda environment, and safe to run
by hand at any time. Checks, in order:

    1. Python version is one pyproject.toml supports.
    2. JAX imports and which devices it sees (GPU vs CPU).
    3. The versions that actually matter are the pinned ones.
    4. ``svidag`` imports.
    5. Every baseline wrapper imports (they reach the vendored code by
       ``sys.path`` injection, which is the step that breaks first).
    6. The Sachs ground truth loads.

Exit status is 0 when everything required passed. Missing GPU is a WARNING,
not a failure -- CPU is a supported (slow) configuration.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Match what every runner does, so this script tests the same import path the
# real runs use rather than a pip install that may or may not be present.
for _p in (REPO_ROOT / "src", REPO_ROOT / "paper_results_reproduce" / "case_4"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Versions the lockfile pins that a mismatch would silently change results or
# break the CUDA plugin. Not the whole lock -- just the load-bearing ones.
EXPECTED = {
    "jax": "0.5.0",
    "jaxlib": "0.5.0",
    "flax": "0.10.3",
    "optax": "0.2.4",
    "numpy": "2.2.0",
    "scipy": "1.14.1",
    "networkx": "3.4.2",
    "scikit-learn": "1.6.0",
}

failures: list[str] = []
warnings: list[str] = []


def ok(msg: str) -> None:
    print(f"  [ ok ] {msg}")


def warn(msg: str) -> None:
    print(f"  [warn] {msg}")
    warnings.append(msg)


def bad(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    failures.append(msg)


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


print("=" * 75)
print("  SVI-DAG environment check")
print("=" * 75)

# ---------------------------------------------------------------------------
section("1. Python")
# ---------------------------------------------------------------------------
v = sys.version_info
print(f"  interpreter: {sys.executable}")
if (3, 10) <= (v.major, v.minor) < (3, 13):
    ok(f"Python {v.major}.{v.minor}.{v.micro}")
    if v.minor != 11:
        warn(f"Python 3.11 is what the published runs used; this is 3.{v.minor}.")
else:
    bad(f"Python {v.major}.{v.minor} is outside the supported range (>=3.10, <3.13).")

if os.environ.get("CONDA_DEFAULT_ENV"):
    ok(f"conda environment: {os.environ['CONDA_DEFAULT_ENV']}")
else:
    warn("not inside a conda environment (CONDA_DEFAULT_ENV unset). "
         "That is fine if you installed another way.")

# ---------------------------------------------------------------------------
section("2. JAX backend and devices")
# ---------------------------------------------------------------------------
gpu_available = False
try:
    import jax

    devices = jax.devices()
    platforms = sorted({d.platform for d in devices})
    ok(f"jax imported, backend = {jax.default_backend()}")
    for d in devices:
        print(f"         device: {d}")
    if any(p in ("gpu", "cuda", "rocm") for p in platforms):
        gpu_available = True
        ok(f"{len(devices)} GPU device(s) visible to JAX")
    else:
        warn("no GPU visible to JAX -- everything will run on CPU.\n"
             "         SVI-DAG on CPU is fine for case 1 (2 nodes); the case "
             "2/3/5/6\n"
             "         benchmarks and the five baselines are far slower.\n"
             "         If you expected a GPU: check `nvidia-smi`, then confirm "
             "the CUDA\n"
             "         wheels are installed (pip install -r requirements-cuda12.txt).")
except Exception as e:  # noqa: BLE001
    bad(f"could not import jax / enumerate devices: {type(e).__name__}: {e}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
section("3. Pinned versions")
# ---------------------------------------------------------------------------
try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    for pkg, want in sorted(EXPECTED.items()):
        try:
            got = _pkg_version(pkg)
        except PackageNotFoundError:
            bad(f"{pkg} is not installed (lockfile pins {want})")
            continue
        if got == want:
            ok(f"{pkg}=={got}")
        else:
            warn(f"{pkg}=={got}, lockfile pins {want}. "
                 "Results may differ from the published numbers.")
except Exception as e:  # noqa: BLE001
    bad(f"version check failed: {type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
section("4. SVI-DAG")
# ---------------------------------------------------------------------------
try:
    import svidag
    from svidag import config  # noqa: F401
    from svidag.model import SVIDAGModel  # noqa: F401
    from svidag.train import make_model_and_state  # noqa: F401

    ok(f"svidag imported from {Path(svidag.__file__).parent}")
except Exception as e:  # noqa: BLE001
    bad(f"could not import svidag: {type(e).__name__}: {e}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
section("5. Baseline wrappers")
# ---------------------------------------------------------------------------
# These reach other_algorithms/codes_jax/ by sys.path injection rather than by
# being installed, so an import error here means the vendored tree is missing
# or a dependency of one baseline is absent -- the failure mode that shows up
# hours into a sweep if it is not caught now.
try:
    from baselines import (  # noqa: F401
        prodag_wrapper,
        dibs_wrapper,
        bayesdag_wrapper,
        bcd_wrapper,
        dds_wrapper,
    )

    ok("all five baseline wrappers imported")

    # The wrappers import their heavy dependencies lazily, so touch the one
    # import that a missing `imageio` would break.
    from dibs.inference import MarginalDiBS  # noqa: F401

    ok("dibs.inference.MarginalDiBS imported (lazy dependency check)")
except Exception as e:  # noqa: BLE001
    bad(f"baseline wrapper import failed: {type(e).__name__}: {e}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
section("6. Sachs data (case 4)")
# ---------------------------------------------------------------------------
# cdt's top-level import needs torch, which is deliberately NOT in the
# lockfile; the loader reads cdt's bundled resource CSVs directly instead.
try:
    from common import load_sachs_full

    sachs = load_sachs_full()
    n_edges = int(sachs.true_adj.sum())
    ok(f"Sachs loaded: X={sachs.X.shape}, ground truth has {n_edges} edges")
    if sachs.X.shape[0] != 7466:
        warn(f"expected the 7466-row pooled dataset, got {sachs.X.shape[0]} rows. "
             "Case 4's published numbers use the pooled one.")
except Exception as e:  # noqa: BLE001
    bad(f"could not load Sachs: {type(e).__name__}: {e}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
print()
print("=" * 75)
if failures:
    print(f"  {len(failures)} CHECK(S) FAILED")
    for f in failures:
        print(f"    - {f.splitlines()[0]}")
    print("\n  Fix these before running any case; results would be unreliable.")
elif warnings:
    print(f"  All required checks passed, with {len(warnings)} warning(s).")
    if not gpu_available:
        print("  No GPU: expect long runtimes. ./run_local.sh needs --cpu to proceed.")
else:
    print("  All checks passed.")

if not failures:
    print("\n  Next:")
    print("      pytest -q                  # unit tests (~1 min)")
    print("      ./run_local.sh 1           # smallest full case")
    print("      ./run_local.sh 2 --quick   # smoke-test a benchmark case")
print("=" * 75)

sys.exit(1 if failures else 0)
