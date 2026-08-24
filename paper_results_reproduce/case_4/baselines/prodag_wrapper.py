#!/usr/bin/env python
"""
Baseline wrapper: ProDAG  (paper-spec hyperparameters)
========================================================

Thin adapter around ``other_algorithms/codes_jax/prodag/src/ProDAG.py`` that
materialises the **paper-specified hyperparameters** for ProDAG so every
benchmark case (case_2 / case_3 / case_4) trains the algorithm identically:

    * Prior on W̃: MV Gaussian, indep, mean 0, variance 1.
    * Variational posterior: MV Gaussian, indep, init at prior.
    * Optimiser: Adam, lr = 0.1, 100 posterior samples per iteration.
    * Projection GD lr: 1/p (linear) or 0.25/p (nonlinear).
    * Threshold for small |W| entries: 0.1.
    * Acyclicity path: mu^(1) = 1, mu^(t+1) = mu^(t) / 2, T = 10.
    * Sparsity grid for lambda: 10 values from lambda_min = 0 to
      lambda_max = average l1 ball of the posterior fit with lambda = inf.
    * lambda chosen on a held-out validation set of size floor(0.1 n) using
      the SEM reconstruction MSE (linear) or its forward-pass analogue
      (nonlinear).
    * Nonlinear network: single hidden layer, 10 ReLU units.

Most of these (mean/variance, optimiser, lr, n_sample, projection params,
threshold, acyclicity schedule, MLP shape) are honoured by the **defaults**
declared in ``ProDAG.fit_linear`` / ``ProDAG.fit_mlp`` -- this wrapper is
careful not to override them.  The two things that need orchestration here
are (i) the validation split and (ii) the 10-point lambda grid + selection
rule, both of which live in this file.

Adjacency convention
--------------------
ProDAG's weighted adjacency ``W[i, j]`` corresponds to the SEM equation
``x_j = sum_i W[i, j] * x_i + noise`` -- i.e. ``i -> j``.  We therefore pass
``source_convention="i_to_j"`` through to ``common.evaluate_samples``.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_CASE4_DIR = _THIS_DIR.parent
_REPO_ROOT = _CASE4_DIR.parent.parent
_PRODAG_SRC = _REPO_ROOT / "other_algorithms" / "codes_jax" / "prodag" / "src"
for p in (str(_PRODAG_SRC),):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Validation NLL: SEM reconstruction MSE on held-out rows.
#
# For ProDAG-linear the SEM is ``x_hat = X W`` -- ``xw_mult`` in
# ``ProDAG._train_linear`` is ``einsum("np,pqs->nqs", x, w)``, i.e. ``x @ w``,
# NOT ``x @ w.T``.
#
# For ProDAG-MLP that closed form is NOT a usable surrogate, and using it made
# the nonlinear benchmark return an empty graph at every sample size.  In MLP
# mode ``W`` does not hold regression coefficients: it holds the group 2-norms
# of each node's first-layer weights (``sqrt(ind_mat @ omega_fhl^2)``), so every
# entry is a non-negative magnitude.  Feeding those to a linear SEM is not an
# approximation of the nonlinear model, it is a different and meaningless
# predictor -- measured on an ER_p25_s40 cell, its MSE rose monotonically with
# lambda (0.93 at lambda=0 through 11.1 at lambda_max), so the lambda=0 end of
# the grid -- ``W = 0``, i.e. no edges -- always won.
#
# ``ProDAG.sample`` returns ``(W, models)`` for MLP fits, so the real forward
# IS available; ``_validation_mse_mlp`` below scores lambda with it, exactly
# the reconstruction term ``train_mlp!`` optimises.
# ---------------------------------------------------------------------------
def _validation_mse(W_pps: np.ndarray, X_val: np.ndarray) -> float:
    """W_pps has shape [p, p, S]; X_val has shape [n_val, p]."""
    n_val, p = X_val.shape
    assert W_pps.shape[0] == p and W_pps.shape[1] == p
    S = W_pps.shape[2]
    if n_val == 0 or S == 0:
        return float("inf")
    # Iterate over posterior samples -- S is bounded (=num_posterior_samples)
    # and each step is a single (n_val, p) @ (p, p) matmul, so this is cheap.
    mses = np.empty(S, dtype=np.float64)
    for s in range(S):
        W_s = np.asarray(W_pps[:, :, s], dtype=np.float64)
        x_hat = X_val @ W_s                                 # [n_val, p]; matches xw_mult
        mses[s] = float(np.mean((X_val.astype(np.float64) - x_hat) ** 2))
    return float(np.mean(mses))


def _validation_mse_mlp(models, X_val: np.ndarray, max_models: int = 100) -> float:
    """
    Held-out reconstruction MSE under ProDAG's own nonlinear forward.

    ``train_mlp!`` scores ``construct(omega)(x)`` against ``x`` with ``x`` laid
    out ``[p, n]``; this is that same term on the validation rows, averaged
    over posterior draws.  ``max_models`` caps how many draws are scored --
    this only picks one of 10 lambdas, and the mean over 100 draws is already
    far tighter than the grid spacing, so scoring all 1000 buys nothing.
    """
    import jax.numpy as jnp  # local: keeps the module importable without JAX

    if X_val.shape[0] == 0 or not models:
        return float("inf")
    xv = jnp.asarray(X_val.T, dtype=jnp.float32)            # [p, n_val]
    used = models[: int(max_models)]
    total = 0.0
    for model in used:
        total += float(jnp.mean((xv - model(xv)) ** 2))
    return total / len(used)


def run(
    X_train: np.ndarray,
    num_nodes: int,
    num_posterior_samples: int = 100,
    seed: int = 0,
    mode: str = "linear",
    elbo_n_sample: int = 100,   # paper spec / ProDAG default: 100 ELBO draws per iteration
    epoch_max: int = 1000,
    patience: int = 5,
    n_lambda_grid: int = 10,                  # paper: 10-point sparsity grid
    validation_frac: float = 0.1,             # paper: floor(0.1 n) held out
    verbose: bool = False,
) -> Tuple[np.ndarray, str]:
    """
    Fit ProDAG on ``X_train`` with the paper-spec hyperparameters and draw
    ``num_posterior_samples`` adjacency samples at the lambda chosen by
    held-out validation MSE on a 10-point grid.

    Returns:
        A_samples  : [S, d, d] RELAXED weights (per-sample magnitude
                     normalised to [0, 1] so the downstream 0.5 threshold
                     in ``common.evaluate_samples`` is meaningful).
        convention : "i_to_j"
    """
    import ProDAG  # resolved via sys.path injection above

    X_train = np.asarray(X_train, dtype=np.float32)
    n, p = X_train.shape
    assert p == num_nodes, f"X_train has {p} cols but num_nodes={num_nodes}"

    # ── 1. validation split ────────────────────────────────────────────────
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    # Paper: validation set size = floor(0.1 n).  When n < 10 we still hold
    # out at least 1 sample so the grid search has a defined criterion.
    n_val = max(1, int(np.floor(validation_frac * n)))
    n_val = min(n_val, n - 1)  # guarantee at least 1 training sample
    val_idx = perm[:n_val]
    fit_idx = perm[n_val:]
    X_fit = X_train[fit_idx]
    X_val = X_train[val_idx]

    # ── 2. fit posterior at lambda = inf ───────────────────────────────────
    #
    # ``n_sample`` here is the number of Monte-Carlo draws used to estimate the
    # ELBO *per training iteration* -- the paper spec (see this module's
    # docstring) and ProDAG's own default are both 100.  It is NOT the number
    # of posterior samples to report, which is what ``ProDAG.sample`` takes
    # below.  Passing ``num_posterior_samples`` (1000 in cases 2/3) conflated
    # the two and inflated every training iteration 10x: the DAG projection
    # inverts one p x p matrix per draw per inner step, so its cost is linear
    # in ``n_sample`` and it dominates the fit.
    fit_fn = ProDAG.fit_linear if mode == "linear" else ProDAG.fit_mlp
    fit = fit_fn(
        X_fit,
        epoch_max=epoch_max,
        patience=patience,
        n_sample=int(elbo_n_sample),
        verbose=verbose,
        # All other hparams (prior_mu=0, prior_sigma=1, optimiser='adam',
        # optimiser_args=(0.1,), n_sample=100 default, params with mu_path=1,
        # c=0.5, T=10, threshold=0.1, lr=1/p (linear) or 0.25/p (nonlinear),
        # hidden_layers=(10,), activation=ReLU) are honoured by the function
        # defaults -- see prodag_wrapper docstring for the audit.
    )

    def _sample_W(fit_obj):
        """``ProDAG.sample`` returns W for linear fits but ``(W, models)`` for
        MLP fits.  Returns ``(W [p, p, S], models or None)`` -- the models are
        what lets ``_validation_mse_mlp`` score lambda with the real nonlinear
        forward instead of a linear stand-in."""
        out = ProDAG.sample(fit_obj, n_sample=num_posterior_samples, guarantee_dag=True)
        if isinstance(out, tuple):
            return np.asarray(out[0], dtype=np.float32), out[1]
        return np.asarray(out, dtype=np.float32), None

    def _score(W_lam, models_lam) -> float:
        """Validation criterion for one lambda, matching the fitted model."""
        if models_lam is not None:
            return _validation_mse_mlp(models_lam, X_val)
        return _validation_mse(W_lam, X_val)

    # ── 3. lambda_max = average l_1 ball of W samples at lambda = inf ──────
    W_inf, _models_inf = _sample_W(fit)                        # [p, p, S]
    l1_per_sample = np.sum(np.abs(W_inf), axis=(0, 1))         # [S]
    lam_max = float(np.mean(l1_per_sample))
    if not np.isfinite(lam_max) or lam_max <= 0.0:
        # Degenerate posterior (e.g. n < p with no signal): skip the grid
        # search and return the lambda = inf samples directly.
        if verbose:
            print(f"[prodag_wrapper] degenerate lam_max={lam_max!r}; "
                  f"skipping lambda grid search.")
        best_W = W_inf
    else:
        # ── 4. 10-point grid from lambda_min = 0 to lambda_max ─────────────
        lam_grid = np.linspace(0.0, lam_max, int(n_lambda_grid)).astype(np.float32)
        if verbose:
            print(f"[prodag_wrapper] n={n}, n_val={n_val}, p={p}, "
                  f"lam_max={lam_max:.4g}; grid={lam_grid}")

        # ── 5. score each lambda by validation reconstruction MSE ──────────
        best_nll = float("inf")
        best_W = None
        for lam in lam_grid:
            # Override the dirac alpha so ``ProDAG.sample`` uses ``lam`` as
            # the l_1 ball radius instead of the trained ``fit.alpha = inf``.
            fit_lam = replace(fit, alpha=np.array([float(lam)], dtype=np.float32))
            W_lam, models_lam = _sample_W(fit_lam)             # [p, p, S]
            nll = _score(W_lam, models_lam)
            if verbose:
                print(f"[prodag_wrapper]   lam={lam:.4g}  val_MSE={nll:.6g}")
            if nll < best_nll:
                best_nll = nll
                best_W = W_lam
        assert best_W is not None
        if verbose:
            print(f"[prodag_wrapper] selected lambda gives val_MSE={best_nll:.6g}")

    # ── 6. format output ───────────────────────────────────────────────────
    # Reorder to [S, d, d].
    A_samples = np.transpose(best_W, (2, 0, 1)).astype(np.float32)
    # Take magnitude so the 0.5 threshold reads as "edge presence/strength".
    A_samples = np.abs(A_samples)
    # Per-sample max-normalise so the downstream 0.5 threshold has consistent
    # meaning across replicates / sample sizes (prevents the absolute scale of
    # the weighted adjacency from dominating which edges "count").
    per_sample_max = np.max(A_samples, axis=(1, 2), keepdims=True)
    per_sample_max = np.where(per_sample_max > 0, per_sample_max, 1.0)
    A_samples = A_samples / per_sample_max
    return A_samples, "i_to_j"
