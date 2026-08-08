# Per-case hyperparameters

Each `case<N>.env` lists **the exact hyperparameters required to reproduce case
N**. Any variable not listed in a file takes its committed value from
[`src/svidag/config.py`](../src/svidag/config.py).

The cases do not use the same values, so the files are not interchangeable —
see the full comparison in the main [README](../README.md#hyperparameters).

## How they are used

Both runners source these files, so you never select one by hand:

| runner | for | how it uses the file |
|---|---|---|
| [`run_local.sh`](../run_local.sh) | a local machine with one GPU | sources it, then walks the case's grid sequentially |
| [`run_case<N>.sh`](../run_case1.sh) | Slurm | sources it, then decodes `$SLURM_ARRAY_TASK_ID` into one grid cell |

Because both read the same file, a local run and a cluster run of the same case
use identical hyperparameters. There is no second copy to keep in sync.

## What is in each file

| block | applies to |
|---|---|
| `SVIDAG_*` | SVI-DAG |
| `CASE<N>_SEED`, `CASE<N>_NUM_REPLICATES` | data generation, all algorithms |
| `BAYESDAG_*` | BayesDAG only — read by `paper_results_reproduce/case_4/baselines/bayesdag_wrapper.py` |

The other four baselines (ProDAG, DiBS, DDS, BCD Nets) take no environment
configuration; their settings live in their wrappers under
`paper_results_reproduce/case_4/baselines/`.

Every file starts by clearing all `SVIDAG_*` from the environment, so sourcing
one fully determines the configuration regardless of what ran before it in the
same shell.

## Overriding a value

These are plain `export`s, so anything exported *after* sourcing wins.
`run_local.sh` sources the file first and applies its own flags afterwards,
which is how `--quick` shortens a run without editing anything:

```bash
SVIDAG_NUM_ITERS=200 ./run_local.sh 2 --algo svidag --n 100
```

A run with any value overridden no longer reproduces the published numbers.
