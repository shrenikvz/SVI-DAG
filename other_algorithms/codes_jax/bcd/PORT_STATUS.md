## Port Status

Status: complete mirror

Reason:

- The original `bcd` implementation is already JAX-based.
- This folder is a faithful mirror of
  `../../original_codes/bcd` to keep the JAX baselines under one common
  `codes_jax` location for benchmarking.

Compatibility notes:

- GPU execution follows the original JAX implementation and depends on the
  local JAX installation and any non-JAX optional dependencies used by the
  original code.
- Non-JAX helper artifacts bundled with the original repository, such as R
  scripts or C/Cython extensions, were preserved because they are part of the
  original benchmark package layout.
