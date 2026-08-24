## Port Status

Status: complete mirror

Reason:

- The original `dibs` implementation is already JAX-based.
- This folder is a faithful mirror of
  `../../original_codes/dibs` to place the benchmark under `codes_jax`
  without changing the algorithm code.

Compatibility notes:

- GPU execution uses the original JAX code paths.
- Example notebooks and docs were preserved because they do not alter the core
  algorithm implementation and help maintain one-to-one correspondence.
