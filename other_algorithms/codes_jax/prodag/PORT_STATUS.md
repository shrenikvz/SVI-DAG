## Port Status

Status: JAX port added for the public ProDAG API

Ported surface:

- `fit_linear`
- `fit_mlp`
- `sample`

Compatibility notes:

- The projection operator and its custom backward rule are reimplemented in
  JAX to preserve the original projected variational update flow.
- The Julia optimiser object interface is narrowed to the default Adam-style
  training path used by the original implementation.
- Sampled nonlinear models are returned as Python callables rather than Julia
  `Flux.Parallel` objects, while preserving the same input/output orientation.

Validation status:

- Syntax checked locally after the port.
- End-to-end execution could not be run in this environment because the local
  JAX installation fails to import on this machine.
