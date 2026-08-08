#!/usr/bin/env python
"""
Baseline wrapper: BCD Nets
==========================

Thin adapter over ``bcd_runner.fit_bcd`` -- a function-style refactor of
``other_algorithms/codes_jax/bcd/main.py``.  The refactor is *local* to
``case_4/baselines/`` and does not modify BCD's source: it re-uses BCD's
own modules (``doubly_stochastic``, ``models``, ``utils``) and replicates
only the pieces that ``main.py`` runs at module scope (argparse-driven
globals + training loop).  See ``bcd_runner.py`` for the parity checklist.

Defaults here (``batch_size=256``, ``fixed_tau=None`` -> tau annealing
30 -> 1) follow the original repo's synthetic-benchmark configuration.  An
earlier version used the Sachs configuration (``20_000``/``64``/``fixed_tau=0.2``
plus a P-network L2 term inside ``fit_bcd``); on the p=25 standardised cases
that setup deterministically returned the empty graph (E[SHD] = #true edges,
E[F1] = 0, AUROC ~ 0.5 at every sample size) -- see ``bcd_runner.py``'s module
docstring for the full diagnosis.

Training budget (``num_steps=15_000``, was ``30_000``)
------------------------------------------------------
Halved from the repo's 30k to cut case-2 wall-time ~2x (from ~8.9 h to
~4.5 h, into the same ballpark as the DDS / DiBS baselines).  The tau
schedule (``bcd_runner.tau_schedule``) finishes annealing to its terminal
value (1.0) by step 10k and then holds flat, so steps beyond ~12k are pure
same-temperature refinement.  A GPU convergence sweep on the p=25 case-2
grid (per-checkpoint CPDAG metrics from one run; discarding the step-10k
checkpoint, which samples at tau=10 and is not comparable to the tau=1.0
posterior) showed:
  * n <= 100  -- fully converged by ~12k; 15k == 30k within replicate noise
    (e.g. n=100 rep0 AUROC 0.713@15k vs 0.706@30k).  ZERO accuracy cost.
  * n  = 1000 -- keeps refining out to 30k: AUROC ~0.91 at 12-20k then
    0.9425 at 30k (aggregate 0.9321 +/- 0.0062).  15k therefore trades a
    small, bounded AUROC drop (~0.03, ~5x SE; Brier 0.072 -> 0.082) at the
    LARGEST sample size only; E[SHD] (40.2 -> ~41.0) and E[F1] are within
    noise.
This is a deliberate accuracy/speed tradeoff, NOT an accuracy-neutral win:
BCD's large-n cells (which dominate cost) genuinely use the full budget, so
no lever -- fewer steps OR fewer ELBO particles (batch_size 128/64 gave the
same ~0.91 AUROC at n=1000) -- reaches DDS-level runtime for free.  Restore
``num_steps=30_000`` to recover the paper-exact large-n numbers; drop to
~12_000 to match DDS wall-time exactly at a marginally larger n=1000 cost.

Adjacency convention
--------------------
BCD's likelihood (``bcd_runner.log_prob_x``) builds
``precision = (I - W) @ D @ (I - W).T`` with ``D = diag(1/sigma^2)``.
For an SEM ``x = C x + eps`` the precision is ``(I - C).T @ D @ (I - C)``,
so matching terms gives ``C = W.T``: node ``x_j`` regresses on ``x_i``
whenever ``W[i, j] != 0``, i.e. ``i -> j``.  We therefore return
``"i_to_j"`` (the standard NOTEARS convention); ``common.normalise_convention``
transposes it into SVIDAG's native ``j -> i`` before metric computation.
(An earlier version declared ``"j_to_i"``, which fed every BCD sample into
the metrics transposed.)

Output scale
------------
``fit_bcd`` returns *raw signed* edge weights.  The shared evaluation
pipeline thresholds at 0.5 and treats the posterior mean as an edge
probability, so — exactly like ``prodag_wrapper`` — we take ``abs()`` and
max-normalise each sample into [0, 1].  Without this, every negative-weight
true edge (half of them, by the U([-0.7,-0.3] U [0.3,0.7]) weight spec) was
unconditionally invisible to the 0.5 threshold.
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
    num_steps: int = 15_000,  # halved from 30k: ~2x faster; see module docstring
    batch_size: int = 256,
    lr: float = 1e-3,
    fixed_tau: float | None = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, str]:
    """
    Fit BCD Nets and return ``num_posterior_samples`` adjacency samples
    drawn from the hard Gumbel-Sinkhorn variational posterior, as
    per-sample max-normalised weight magnitudes in [0, 1].

    Returns
    -------
    A_samples : [S, d, d] float32 in [0, 1]
    convention : "i_to_j"
    """
    from bcd_runner import fit_bcd

    W = fit_bcd(
        X_train=np.asarray(X_train, dtype=np.float32),
        num_nodes=int(num_nodes),
        num_posterior_samples=int(num_posterior_samples),
        num_steps=int(num_steps),
        batch_size=int(batch_size),
        lr=float(lr),
        fixed_tau=None if fixed_tau is None else float(fixed_tau),
        seed=int(seed),
        verbose=verbose,
    )
    if W.ndim != 3 or W.shape[1] != num_nodes:
        raise RuntimeError(
            f"BCD returned unexpected adjacency shape {W.shape}, "
            f"expected (S, {num_nodes}, {num_nodes})."
        )
    # Signed weights -> [0, 1] edge strengths (mirrors prodag_wrapper): the
    # shared 0.5 threshold and Brier/AUROC assume probabilities, and signed
    # values would hide every negative-weight edge.
    A_samples = np.abs(W.astype(np.float32))
    per_sample_max = np.max(A_samples, axis=(1, 2), keepdims=True)
    per_sample_max = np.where(per_sample_max > 0, per_sample_max, 1.0)
    A_samples = A_samples / per_sample_max
    return A_samples.astype(np.float32), "i_to_j"
