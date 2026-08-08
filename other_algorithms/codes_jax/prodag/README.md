# ProDAG JAX Port

This folder contains a JAX refactor of the public ProDAG API from
`../original_codes/prodag`.

Ported surface:

- `fit_linear`
- `fit_mlp`
- `sample`

Compatibility notes:

- The optimizer interface is narrowed to the default Adam-style path used by
  the original implementation. The keyword is still accepted, but alternate
  Julia optimiser objects are not meaningful in Python/JAX.
- The acyclicity projection and its custom backward rule are implemented in
  JAX to preserve the original projected variational training flow.
- Output MLP samples are returned as Python callables operating on arrays with
  the same `p x n` orientation used by the original Julia code.

