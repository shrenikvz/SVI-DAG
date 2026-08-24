# BayesDAG JAX Port

This folder contains a JAX refactor of the continuous BayesDAG benchmark path.

Ported surface:

- `BayesDAGLinear`
- `BayesDAGNonLinear`
- continuous-data training loop with helper-network VI and SG-MCMC updates for `p` and SEM weights
- adjacency and weighted-adjacency sampling
- an experiment runner that accepts either direct array paths or the original benchmark-style dataset directories with `train.csv` and `adj_matrix.csv`
- copied BayesDAG default config files under `src/configs`

Compatibility notes:

- This port targets the continuous benchmark setting used by synthetic DAG data.
- The broader Causica experiment framework and discrete-variable utilities are not reimplemented here.
- Hard Sinkhorn assignment uses a host callback around SciPy Hungarian matching to preserve the original straight-through permutation step.
