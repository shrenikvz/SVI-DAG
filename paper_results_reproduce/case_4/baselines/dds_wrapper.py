#!/usr/bin/env python
"""
Baseline wrapper: DDS (VI-DP-DAG / Differentiable DAG Sampling)
===============================================================

Thin adapter over ``dds_runner.fit_dds`` -- a dataloader-free shim around
VI-DP-DAG's autoencoder training path.  See ``dds_runner.py`` docstring
for a discussion of why we use the autoencoder path
(``train_probabilistic_dag_autoencoder.train_autoencoder`` +
``ProbabilisticDAGAutoencoder``) rather than ``train_dag.train`` (which
does not consume observational data and is therefore not the published
observational-data algorithm).

Adjacency convention
--------------------
``ProbabilisticDAG.sample`` returns ``permutation_inv @ mask @ permutation``
with upper-triangular ``mask`` -- i.e. ``A[i, j] = 1`` means ``i -> j``.
We declare ``"i_to_j"`` downstream.

Convergence fix
---------------
The DDS-specific defaults (topk ordering, corrected Bernoulli-KL sparsity
regularizer, regr=0.1) live in ``dds_runner.fit_dds``.  Before that fix DDS
returned a near-complete DAG (E[SHD] flat at ~190 for p=25); it now
decreases with n.  See ``dds_runner.py``'s module docstring.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


def run(
    X_train: np.ndarray,
    num_nodes: int,
    num_posterior_samples: int = 100,
    seed: int = 0,
    max_epochs: int = 100,
    patience: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    pd_lr: float = 1e-2,
    temperature: float = 1.0,
    hard: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, str]:
    """
    Fit VI-DP-DAG (autoencoder path) and draw ``num_posterior_samples``
    binary DAG samples.

    ``lr`` is the masked-autoencoder learning rate; ``pd_lr`` (default 1e-2,
    the authors' notebook value) is the separate learning rate for the
    permutation/edge parameters -- the two are decoupled because the DAG
    parameters need a faster rate than the reconstruction network.  All other
    DDS-specific hyper-parameters (topk ordering, corrected sparsity
    regularizer, prior_p, hidden dims) come from ``dds_runner.fit_dds``'s
    defaults; see that module's docstring for the convergence fix.

    Returns
    -------
    A_samples : [S, d, d] float32 (binary)
    convention : "i_to_j"
    """
    from dds_runner import fit_dds

    A = fit_dds(
        X_train=np.asarray(X_train, dtype=np.float32),
        num_nodes=int(num_nodes),
        num_posterior_samples=int(num_posterior_samples),
        seed=int(seed),
        max_epochs=int(max_epochs),
        patience=int(patience),
        batch_size=int(batch_size),
        ma_lr=float(lr),
        pd_lr=float(pd_lr),
        pd_temperature=float(temperature),
        pd_hard=bool(hard),
        verbose=verbose,
    )
    if A.ndim != 3 or A.shape[1] != num_nodes:
        raise RuntimeError(
            f"DDS returned unexpected adjacency shape {A.shape}, "
            f"expected (S, {num_nodes}, {num_nodes})."
        )
    return A.astype(np.float32), "i_to_j"
