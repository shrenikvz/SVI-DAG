## codes_jax

This directory is the destination for JAX implementations of the benchmark
algorithms stored in `../original_codes`.

Current status:

- `bcd`: mirrored from `original_codes/bcd` because the original implementation
  is already JAX-based.
- `dibs`: mirrored from `original_codes/dibs` because the original
  implementation is already JAX-based.
- `vi-dp-dag`: JAX port added under `codes_jax/vi-dp-dag` for the core dataset,
  model, training, and evaluation path. See its local `PORT_STATUS.md` for the
  remaining environment caveats.
- `bayesdag`: JAX port added under `codes_jax/bayesdag` for the continuous
  BayesDAG benchmark path, including linear and nonlinear models plus a simple
  runner and mirrored benchmark configs.
- `prodag`: JAX port added under `codes_jax/prodag` for the public `fit_linear`,
  `fit_mlp`, and `sample` API.

Each algorithm folder contains a `PORT_STATUS.md` file documenting whether the
folder is a direct JAX mirror or a translated JAX implementation with any
minimal compatibility notes.
