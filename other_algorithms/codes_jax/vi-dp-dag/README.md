# VI-DP-DAG JAX Port

This folder contains a JAX refactor of the original `vi-dp-dag` implementation
from `../original_codes/vi-dp-dag`.

Scope:

- probabilistic DAG sampler
- differentiable sorting utilities
- masked autoencoder variant
- dataset loading, training loops, and evaluation helpers

Compatibility notes:

- The hard Sinkhorn permutation path still relies on a host-side Hungarian
  assignment callback, matching the original SciPy-based implementation.
- `cdt` is still required for Sachs loading and SID evaluation, just as in the
  original codebase.
- The original `run_probabilistic_dag.py` omitted `dataset_directory` even
  though the loader requires it. The JAX port preserves the runner name and
  signature and resolves the dataset root from the `VI_DP_DAG_DATASET_DIRECTORY`
  environment variable when needed.

