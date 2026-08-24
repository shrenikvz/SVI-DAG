## Port Status

Status: JAX port added for the core executable surface

Ported modules:

- dataset loading and train/val/test splitting
- differentiable sorting utilities (`sinkhorn` and `topk`)
- probabilistic DAG model
- masked autoencoder variant
- training and evaluation entrypoints

Minimal compatibility adjustments:

- Hard Sinkhorn uses a host callback for Hungarian assignment so the straight-through
  hard permutation path stays aligned with the original SciPy-based behavior.
- `cdt` remains an optional runtime dependency for Sachs loading and SID metrics,
  matching the original code.
- The original DAG-only runner omitted `dataset_directory`; the JAX port resolves it
  from `VI_DP_DAG_DATASET_DIRECTORY` when the dataset is not Sachs.

Validation status:

- Syntax checked locally after the port.
- End-to-end execution could not be run in this environment because the local JAX
  installation fails to import on this machine.
