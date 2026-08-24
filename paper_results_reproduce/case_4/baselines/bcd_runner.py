#!/usr/bin/env python
"""
BCD Nets trainer -- function-style refactor of ``bcd/main.py``.
===============================================================

``bcd/main.py`` is a monolithic argparse-driven script; its ~900 lines of
module-scope state cannot be called from a benchmark loop.  This module
replicates the *training* pieces of that script as a function -- the
algorithm (Gumbel-Sinkhorn permutation + lower-triangular L + Horseshoe
prior on edges + pathwise ELBO with ``num_outer=1``) and all hyper-parameter
defaults are kept identical to the Sachs configuration in main.py.  What is
dropped is only ancillary: Weights & Biases logging, GOLEM/pc/mmhc
baselines, ground-truth-aware metrics printed during training, and the
matplotlib/heatmap plumbing.

Algorithmic parity checklist (identical to ``bcd/main.py`` unless noted):

- ``GumbelSinkhorn(dim, "gumbel", tol=max_deviation=0.01)``
- L parameterisation: ``[means_{l_dim+noise_dim} | log_stds_{l_dim+noise_dim}]``
  initialised as ``concat(zeros(l_dim), zeros(noise_dim), zeros - 1)``
- Normal variational family on L (``L_dist = Normal``)
- ``log_stds_max=10.0`` soft-clip via tanh
- P network: 2-layer MLP, hidden_size=128, Gelu, output logits clipped via
  ``tanh(logits/logit_constraint)*logit_constraint``
- Optax chain: ``scale_by_belief(eps=1e-8)`` + ``scale(-lr)`` for both P and L
- ELBO: Horseshoe(l_dim) prior on L-lower, Gaussian(0, s_prior_std=3) on
  noise-scale, ``-logprob_P + log_P_prior`` per-sample, ``num_outer=1``
- Sampler: ``W = (P @ L @ P.T).T`` from hard Gumbel-Sinkhorn

Deliberate deviations from the Sachs configuration (defaults below), made
after the case_2/3/5-9 runs showed BCD stuck at the empty graph (E[SHD] ==
number of true edges, E[F1] == 0, AUROC ~ 0.5 at every sample size).
Diagnosis on the case_2 grid (p=25, standardised data):

1. **P-network L2 decay killed the permutation posterior.**  main.py adds
   ``grad(0.5*||P_params||^2)`` to the ELBO gradient before AdaBelief.
   Early in training the ELBO gradient w.r.t. P is ~0 (L ~ 0 makes the
   likelihood permutation-invariant), so AdaBelief normalises the L2 term
   into a full-size update: the P-MLP weights decay to ~0 within ~1k steps
   and ``logprob(P)`` stays frozen at ``-log(dim!)`` (uniform) for the rest
   of training -- verified empirically (even with L initialised at the true
   weights, P never moved).  Default here: ``p_l2_reg=0.0``.  Passing
   ``p_l2_reg=1.0`` restores main.py behaviour exactly.
2. **Fixed tau=0.2 saturates the straight-through Sinkhorn gradient.**
   The hard sampler's gradient flows through soft Sinkhorn at temperature
   tau; at 0.2 with d=25 it is exponentially small, so P receives no
   learning signal.  Default here: ``fixed_tau=None`` -> main.py's own
   ``tau_schedule`` (30 -> 10 -> 1 -> 0.5 -> 0.25), which is what the
   original repo provides for exactly this purpose.
3. **The n-dependent horseshoe scale over-shrinks weak standardised
   weights.**  With p=25, degree=1, n=100 the default formula gives
   tau_hs ~ 0.009 while the true (standardised) edge weights are 0.2-0.65;
   the prior wins and L collapses before P can concentrate.  Default here:
   ``use_alternative_horseshoe_tau=True`` -- the van der Pas et al. variant
   already implemented in main.py behind ``--use_alternative_horseshoe_tau``
   (tau_hs ~ 0.13 for d=25, n-independent).

With (1)+(2)+(3) the permutation posterior concentrates (logprob(P) rises
from -log(25!) ~ -56 to ~ -15 during the warm phase) and edge probabilities
separate on true vs. non-edges, instead of returning the empty graph.

Adjacency convention
--------------------
``log_prob_x`` builds ``precision = (I - W) @ D @ (I - W).T`` with
``D = diag(1/sigma^2)``.  For an SEM ``x = C x + eps`` the precision is
``(I - C).T @ D @ (I - C)``, so ``C = W.T``: node ``x_j`` regresses on
``x_i`` whenever ``W[i, j] != 0`` -- i.e. ``i -> j`` (standard NOTEARS
convention).  Samples returned here are therefore ``"i_to_j"``; the
wrapper declares that downstream.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as onp

# Suppress TF-Probability FutureWarning spam (matches main.py behaviour).
warnings.simplefilter(action="ignore", category=FutureWarning)

_THIS_DIR = Path(__file__).resolve().parent
_CASE4_DIR = _THIS_DIR.parent
_REPO_ROOT = _CASE4_DIR.parent.parent
_BCD_ROOT = _REPO_ROOT / "other_algorithms" / "codes_jax" / "bcd"
if str(_BCD_ROOT) not in sys.path:
    sys.path.insert(0, str(_BCD_ROOT))


def fit_bcd(
    X_train: onp.ndarray,
    num_nodes: int,
    num_posterior_samples: int = 100,
    num_steps: int = 20_000,
    batch_size: int = 64,
    lr: float = 1e-3,
    fixed_tau: Optional[float] = None,
    logit_constraint: float = 10.0,
    max_deviation: float = 0.01,
    degree: int = 1,
    do_ev_noise: bool = False,
    hidden_size: int = 128,
    num_perm_layers: int = 2,
    s_prior_std: float = 3.0,
    log_stds_max: float = 10.0,
    use_alternative_horseshoe_tau: bool = True,
    p_l2_reg: float = 0.0,
    seed: int = 0,
    verbose: bool = False,
    checkpoint_steps: Optional[Sequence[int]] = None,
    checkpoint_cb: Optional[Callable[[int, onp.ndarray], None]] = None,
) -> onp.ndarray:
    """
    Run BCD Nets on observational data and draw adjacency samples from the
    variational posterior.

    Parameters match the algorithmically-relevant arguments in ``main.py``.
    Defaults follow the repo's synthetic-benchmark configuration (tau
    schedule, alternative horseshoe scale, no P-network L2) rather than the
    Sachs run -- see the module docstring for why the Sachs configuration
    (fixed_tau=0.2, n-dependent horseshoe, P L2) fails to converge on the
    p=25 standardised benchmarks.

    Parameters
    ----------
    X_train : [N, d] float
        Standardised observational data (BCD's own Sachs helper normalises
        and centres; when this function is called directly pre-processing
        is the caller's responsibility).
    num_nodes : int
        d.
    num_posterior_samples : int
        S adjacency samples to draw.
    num_steps : int
        Optimiser steps.
    batch_size : int
        Monte-Carlo samples per ELBO evaluation.
    lr : float
        Learning rate for both P and L optimisers (``scale_by_belief``).
    fixed_tau : float or None
        Temperature for the Gumbel-Sinkhorn relaxation.  ``None`` (default)
        uses main.py's ``tau_schedule`` (30 -> 10 -> 1 -> 0.5 -> 0.25 over
        the run); a float trains at that fixed temperature (0.2 was the
        Sachs configuration, which does not converge on the p=25 synthetic
        benchmarks -- see module docstring).
    logit_constraint : float
        Tanh-based soft clip on permutation logits.
    max_deviation : float
        Sinkhorn doubly-stochastic tolerance.
    degree : int
        Prior expected in-degree -- controls the horseshoe scale.
    do_ev_noise : bool
        If True, use a single shared noise log-sigma across nodes
        (main.py default for Sachs is False).
    hidden_size, num_perm_layers : int
        P-network MLP width / depth.
    s_prior_std : float
        Std of the Gaussian prior on the log-noise-scale block of L.
    log_stds_max : float
        Tanh soft-clip on L log-stds.
    use_alternative_horseshoe_tau : bool
        If True (default), use the van der Pas et al. horseshoe scale
        (main.py's ``--use_alternative_horseshoe_tau``); if False, the
        n-dependent formula from main.py's default path.
    p_l2_reg : float
        Coefficient on the L2 gradient added to the P-network's ELBO
        gradient.  main.py uses 1.0; default 0.0 here because the L2 term
        zeroes the P-MLP before any learning signal arrives (see module
        docstring).
    seed : int
        JAX PRNG seed.

    Returns
    -------
    W_samples : [S, d, d] float32
        Weighted adjacency samples in the ``i_to_j`` convention (see
        module docstring).
    """
    # Heavy JAX imports are done lazily so that an import error of bcd's
    # dependencies surfaces only when the algorithm is actually run.
    import jax
    import jax.numpy as jnp
    import jax.random as rnd
    from jax import config, grad, jit, lax, pmap, value_and_grad, vmap
    from jax.tree_util import tree_map

    # bcd/main.py runs in float64 -- same here for numerical parity.
    config.update("jax_enable_x64", True)

    # Optax chain matches main.py lines 354-362.
    import optax

    from doubly_stochastic import GumbelSinkhorn
    from models import get_model, get_model_arrays
    from utils import lower, num_params, un_pmap, rk, ff2  # noqa: F401
    from tensorflow_probability.substrates.jax.distributions import (
        Horseshoe,
        Normal,
    )

    dim = int(num_nodes)
    n_data = int(X_train.shape[0])
    l_dim = dim * (dim - 1) // 2
    noise_dim = 1 if do_ev_noise else dim
    num_outer = 1
    num_devices = jax.device_count()

    # Horseshoe scale: both formulas exist in main.py (lines 204-215); the
    # alternative (van der Pas et al.) variant is the default here -- see
    # module docstring, deviation (3).
    if use_alternative_horseshoe_tau:
        p_n_over_n = min(2 * degree / (dim - 1), 1.0)
        horseshoe_tau = p_n_over_n * onp.sqrt(onp.log(1.0 / p_n_over_n))
    else:
        horseshoe_tau = (1.0 / onp.sqrt(n_data)) * (
            2 * degree / max((dim - 1) - 2 * degree, 1e-8)
        )
    if horseshoe_tau < 0:
        horseshoe_tau = 1.0 / (2 * dim)
    if verbose:
        print(f"[bcd_runner] horseshoe_tau={horseshoe_tau:.4g}")

    # main.py's tau_schedule (used when fixed_tau is None) -- see module
    # docstring, deviation (2).
    def tau_schedule(i: int) -> float:
        boundaries = onp.array([5_000, 10_000, 20_000, 60_000, 100_000])
        values = onp.array([30.0, 10.0, 1.0, 1.0, 0.5, 0.25])
        return float(values[int(onp.sum(boundaries < i))])

    ds = GumbelSinkhorn(dim, noise_type="gumbel", tol=max_deviation)

    # Optimisers -- exactly the main.py chain.
    opt_P = optax.chain(optax.scale_by_belief(eps=1e-8), optax.scale(-lr))
    opt_L = optax.chain(optax.scale_by_belief(eps=1e-8), optax.scale(-lr))

    _, p_model = get_model(
        dim, batch_size, num_perm_layers,
        hidden_size=hidden_size, do_ev_noise=do_ev_noise,
    )

    def init_parallel_params(rng_key):
        @pmap
        def init_params(rng_key):
            # L is [means | log_stds] as in main.py lines 454-462.
            L_params = jnp.concatenate((
                jnp.zeros(l_dim),
                jnp.zeros(noise_dim),
                jnp.zeros(l_dim + noise_dim) - 1,
            ))
            L_states = jnp.array([0.0])  # dummy; kept for pmap compatibility
            P_params = get_model_arrays(
                dim, batch_size, num_perm_layers, rng_key,
                hidden_size=hidden_size, do_ev_noise=do_ev_noise,
            )
            P_opt_state = opt_P.init(P_params)
            L_opt_state = opt_L.init(L_params)
            return P_params, L_params, L_states, P_opt_state, L_opt_state

        rng_keys = jnp.tile(rng_key[None, :], (num_devices, 1))
        return init_params(rng_keys)

    def get_P_logits(P_params, L_samples, rng_key):
        p_logits = p_model(P_params, rng_key, L_samples)
        if logit_constraint is not None:
            p_logits = jnp.tanh(p_logits / logit_constraint) * logit_constraint
        return p_logits.reshape((-1, dim, dim))

    def sample_L(L_params, rng_key):
        means, log_stds = L_params[: l_dim + noise_dim], L_params[l_dim + noise_dim :]
        if log_stds_max is not None:
            log_stds = jnp.tanh(log_stds / log_stds_max) * log_stds_max
        l_distribution = Normal(loc=means, scale=jnp.exp(log_stds))
        full_l_batch = l_distribution.sample(seed=rng_key, sample_shape=(batch_size,))
        full_log_prob_l = jnp.sum(l_distribution.log_prob(full_l_batch), axis=1)
        return full_l_batch, full_log_prob_l

    def log_prob_x(Xs, log_sigmas, P, L, rng_key):
        n, d = Xs.shape
        W = (P @ L @ P.T).T
        precision = (jnp.eye(d) - W) @ jnp.diag(jnp.exp(-2 * log_sigmas)) @ (jnp.eye(d) - W).T
        log_det_precision = -2 * jnp.sum(log_sigmas)

        def dp_exponent(x):
            return -0.5 * x.T @ precision @ x

        log_exponent = vmap(dp_exponent)(Xs)
        return (
            0.5 * n * (log_det_precision - d * jnp.log(2 * jnp.pi))
            + jnp.sum(log_exponent)
        )

    def elbo(P_params, L_params, L_states, Xs, rng_key, tau, n_outer, hard):
        l_prior = Horseshoe(scale=jnp.ones(l_dim + noise_dim) * horseshoe_tau)

        def outer_loop(rng_key):
            rng_key, rng_key_1 = rnd.split(rng_key, 2)
            full_l_batch, full_log_prob_l = sample_L(L_params, rng_key)
            w_noise = full_l_batch[:, -noise_dim:]
            l_batch = full_l_batch[:, :-noise_dim]
            batched_noises = jnp.ones((batch_size, dim)) * w_noise.reshape(
                (batch_size, noise_dim)
            )
            batched_lower_samples = vmap(lower, in_axes=(0, None))(l_batch, dim)
            batched_P_logits = get_P_logits(P_params, full_l_batch, rng_key_1)
            if hard:
                batched_P_samples = ds.sample_hard_batched_logits(
                    batched_P_logits, tau, rng_key
                )
            else:
                batched_P_samples = ds.sample_soft_batched_logits(
                    batched_P_logits, tau, rng_key
                )
            likelihoods = vmap(log_prob_x, in_axes=(None, 0, 0, 0, None))(
                Xs, batched_noises, batched_P_samples, batched_lower_samples, rng_key,
            )
            l_prior_probs = jnp.sum(l_prior.log_prob(full_l_batch)[:, :l_dim], axis=1)
            s_prior_probs = jnp.sum(
                full_l_batch[:, l_dim:] ** 2 / (2 * s_prior_std ** 2), axis=-1
            )
            KL_term_L = full_log_prob_l - l_prior_probs - s_prior_probs
            logprob_P = vmap(ds.logprob, in_axes=(0, 0, None))(
                batched_P_samples, batched_P_logits, 20
            )
            log_P_prior = -jnp.sum(jnp.log(onp.arange(dim) + 1))
            final_term = likelihoods - KL_term_L - logprob_P + log_P_prior
            return jnp.mean(final_term), L_states

        rng_keys = rnd.split(rng_key, n_outer)
        _, (elbos, out_L_states) = lax.scan(
            lambda _, rng_key: (None, outer_loop(rng_key)), None, rng_keys
        )
        return jnp.mean(elbos), tree_map(lambda x: x[-1], out_L_states)

    from functools import partial

    # NOTE on static_broadcasted_argnums: bcd/main.py originally marked ``Xs``
    # (arg 3) as static_broadcasted.  In modern JAX, static-broadcasted args
    # must be hashable -- a ``jaxlib.xla_extension.ArrayImpl`` is not, so that
    # pattern now raises ``ValueError: Non-hashable static arguments are not
    # supported``.  ``Xs`` is already declared as ``in_axes=None`` so it is
    # broadcast (unmapped) across devices, which is what the legacy code
    # actually needed.  Dropping the static annotation here preserves the
    # original numerical behaviour while making the call hashable.
    @partial(
        pmap,
        axis_name="i",
        in_axes=(0, 0, 0, None, 0, 0, 0, None),
    )
    def parallel_gradient_step(
        P_params, L_params, L_states, Xs, P_opt_state, L_opt_state, rng_key, tau,
    ):
        rng_key, rng_key_2 = rnd.split(rng_key, 2)
        tau_scaling = 1.0 / tau

        (_, L_states), grads = value_and_grad(elbo, argnums=(0, 1), has_aux=True)(
            P_params, L_params, L_states, Xs, rng_key, tau, num_outer, True,
        )
        elbo_grad_P, elbo_grad_L = tree_map(lambda x: -tau_scaling * x, grads)
        elbo_grad_P = lax.pmean(elbo_grad_P, axis_name="i")
        elbo_grad_L = lax.pmean(elbo_grad_L, axis_name="i")

        # P-network L2 (main.py behaviour at p_l2_reg=1.0).  Off by default:
        # with the ELBO gradient w.r.t. P near zero early in training,
        # AdaBelief turns this term into full-size weight decay that zeroes
        # the P-MLP -- see module docstring, deviation (1).
        if p_l2_reg > 0.0:
            l2_grad_P = grad(
                lambda p: p_l2_reg * 0.5 * sum(
                    jnp.sum(jnp.square(param)) for param in jax.tree_util.tree_leaves(p)
                )
            )(P_params)
            elbo_grad_P = tree_map(lambda x, y: x + y, elbo_grad_P, l2_grad_P)

        P_updates, P_opt_state = opt_P.update(elbo_grad_P, P_opt_state, P_params)
        P_params = optax.apply_updates(P_params, P_updates)
        L_updates, L_opt_state = opt_L.update(elbo_grad_L, L_opt_state, L_params)
        L_params = optax.apply_updates(L_params, L_updates)

        return P_params, L_params, L_states, P_opt_state, L_opt_state, rng_key_2

    # ----- init -----
    rng_key = rk(int(seed))
    P_params, L_params, L_states, P_opt_state, L_opt_state = init_parallel_params(rng_key)
    rng_key = rnd.split(rng_key, num_devices)
    Xs = jnp.asarray(X_train, dtype=jnp.float64)
    tau = fixed_tau if fixed_tau is not None else tau_schedule(0)

    if verbose:
        print(
            f"[bcd_runner] L #params={num_params(L_params)}, "
            f"P #params={num_params(P_params)}"
        )

    # ----- posterior sampling helper -----
    # main.py's eval_mean samples a batch of size `batch_size`; we emit exactly
    # `num_samples` draws by taking ceil(S/batch_size) batches and slicing.  The
    # tau passed in is the temperature to sample the hard permutations at (the
    # current schedule value at the point of sampling), matching main.py.
    def draw_posterior(P_params, L_params, tau_now, num_samples, base_seed):
        up_P, up_L = un_pmap(P_params), un_pmap(L_params)

        def sample_batch(key):
            full_l_batch, _ = sample_L(up_L, key)
            l_batch = full_l_batch[:, :-noise_dim]
            batched_lower = vmap(lower, in_axes=(0, None))(l_batch, dim)
            batched_P_logits = get_P_logits(up_P, full_l_batch, key)
            batched_P_samples = ds.sample_hard_batched_logits(
                batched_P_logits, tau_now, key
            )
            Ws = vmap(lambda L, P: (P @ L @ P.T).T)(batched_lower, batched_P_samples)
            return Ws  # [batch_size, d, d]

        n_batches = int(onp.ceil(num_samples / batch_size))
        sample_keys = rnd.split(rk(int(base_seed)), n_batches)
        all_Ws = [onp.asarray(jit(sample_batch)(k)) for k in sample_keys]
        return onp.concatenate(all_Ws, axis=0)[:num_samples].astype("float32")

    # ----- training -----
    checkpoint_set = {int(c) for c in checkpoint_steps} if checkpoint_steps else set()

    for i in range(int(num_steps)):
        # main.py updates tau from the schedule every 400 steps.
        if fixed_tau is None and i % 400 == 0:
            tau = tau_schedule(i)
        (
            P_params,
            L_params,
            L_states,
            P_opt_state,
            L_opt_state,
            rng_key,
        ) = parallel_gradient_step(
            P_params, L_params, L_states, Xs, P_opt_state, L_opt_state, rng_key, tau,
        )
        step = i + 1
        if step in checkpoint_set and checkpoint_cb is not None:
            # Diagnostic hook (default off): draw the posterior at this step so a
            # single training run yields the full accuracy-vs-steps curve.  Does
            # not perturb training state or the final result.
            W_ck = draw_posterior(
                P_params, L_params, tau, num_posterior_samples, int(seed) + 777
            )
            checkpoint_cb(step, W_ck)
        if verbose and i % 1_000 == 0:
            print(f"[bcd_runner] step {i}/{num_steps} (tau={tau:g})")

    # ----- posterior sampling -----
    W_all = draw_posterior(
        P_params, L_params, tau, num_posterior_samples, int(seed) + 777
    )
    return W_all.astype("float32")
