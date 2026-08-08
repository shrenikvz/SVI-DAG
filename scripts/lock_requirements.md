# Regenerating `requirements.txt`

`requirements.txt` is a **complete lockfile**: every package, direct and
transitive, pinned to an exact version. That is what makes the numbers in
`paper_results_reproduce/` reproducible on a machine that installs them months
or years later. `requirements-direct.txt` is the annotated list of *direct*
dependencies and exists for humans; it is not an install target.

Regenerate the lock whenever a direct dependency is added, removed, or bumped.

## Procedure

1. Add/modify the entry in `requirements-direct.txt`, with a comment naming the
   component that needs it.

2. Build a clean environment from the direct list and let pip resolve:

   ```bash
   python3.11 -m venv /tmp/svidag-lock && source /tmp/svidag-lock/bin/activate
   pip install --upgrade pip
   pip install -r requirements-direct.txt
   ```

3. Freeze the result into the lockfile body, then restore the header comment
   and the CUDA section split from the previous `requirements.txt`:

   ```bash
   pip freeze > /tmp/frozen.txt
   ```

4. Verify the lockfile installs from scratch before committing it. A freeze can
   capture a package set pip will refuse to reinstall — this has happened here:

   ```bash
   pip install --dry-run --ignore-installed -r requirements.txt
   ```

   Exit status 0 is the requirement. If it reports `ResolutionImpossible`, the
   environment you froze was internally inconsistent (typically a package
   installed with `--no-deps`, or upgraded after its dependents were pinned).
   Find the offending pin from the reported line numbers, relax it to the
   version pip does resolve, and re-verify.

## Deliberate omissions

The following were in the original reference environment and are **not** in the
lockfile. Nothing under `src/`, `paper_results_reproduce/`, `tests/`, or
`other_algorithms/codes_jax/` imports them; they are leftovers from earlier
experiments, and two of them (`gpytorch`, `linear-operator`) carry pins that
make the set unresolvable.

```
gpytorch  linear-operator  torch-geometric  pyro-ppl  pyro-api
gym  gym-notices  jraph  seaborn  imageio-ffmpeg  sympy  mpmath
```

`jraph` supported the JSP-GFN baseline, which is no longer part of the
repository.

`torch` is intentionally absent. `cdt` declares it optional but does
`from torch.utils.data import Dataset` at import time, so `import cdt` fails —
both Sachs loaders (`svidag/data.py`, `paper_results_reproduce/case_4/common.py`)
detect that and read cdt's bundled resource CSVs directly instead. Do not add
torch to "fix" the import; it would pull a ~300 MB wheel that nothing uses.

## Packages held at the reference version

`equinox` and `lineax` differ from the original environment (`0.13.8` vs
`0.13.7`, `0.0.8` vs `0.1.0`). The recorded `lineax==0.1.0` requires
`jax>=0.6.1` while this project pins `jax==0.5.0`, so that combination cannot
be installed at all. Both packages are transitive-only — no file in the
repository imports either — so the resolvable versions are used.

Everything that *is* imported is pinned to the exact reference version:
`jax`, `jaxlib`, `flax`, `optax`, `chex`, `dm-haiku`, `tensorflow-probability`,
`ott-jax`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`,
`networkx`, `tqdm`, `cdt`, `igraph`, `imageio`, `pytest`.
