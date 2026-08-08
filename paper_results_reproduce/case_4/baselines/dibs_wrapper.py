#!/usr/bin/env python
"""
Baseline wrapper: DiBS
======================

Wraps ``other_algorithms/codes_jax/dibs/``'s JointDiBS inference on a linear
Gaussian model -- the standard Sachs-compatible setup from the DiBS paper.

We run the marginal inference path (``MarginalDiBS``) with a BGe scorer so
the output is a set of graph samples without needing to infer SEM parameters
separately.  This matches the Sachs evaluation in the original DiBS code.

Adjacency convention
--------------------
DiBS's ``u @ v.T`` scores are interpreted as ``A[i, j]`` with ``i -> j``.
We pass ``"i_to_j"`` downstream.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_CASE4_DIR = _THIS_DIR.parent
_REPO_ROOT = _CASE4_DIR.parent.parent
_DIBS_ROOT = _REPO_ROOT / "other_algorithms" / "codes_jax" / "dibs"
for p in (str(_DIBS_ROOT),):
    if p not in sys.path:
        sys.path.insert(0, p)


#: Peak per-step footprint of the parent-masked data tensor we aim to stay
#: under, in float32 elements.  6 GiB is the n=10^3 footprint, which ran
#: comfortably on the 40 GB A100s.
_GRAD_MC_TARGET_ELEMENTS = 6 * (2 ** 30) // 4

#: Candidate chunk sizes; must divide n_grad_mc_samples (128).
_GRAD_MC_CHUNKS = (128, 64, 32, 16, 8, 4, 2, 1)


def _grad_mc_chunk_for(n_obs: int, n_particles: int, d: int):
    """Largest chunk keeping ``n_particles * chunk * d * n_obs * d`` under the
    target, or None when the unchunked path already fits."""
    per_sample = n_particles * d * n_obs * d
    for chunk in _GRAD_MC_CHUNKS:
        if per_sample * chunk <= _GRAD_MC_TARGET_ELEMENTS:
            return None if chunk == _GRAD_MC_CHUNKS[0] else chunk
    return 1


def run(
    X_train: np.ndarray,
    num_nodes: int,
    num_posterior_samples: int = 100,
    seed: int = 0,
    num_particles: int = 20,
    steps: int = 2000,
    # "bge": marginal linear-Gaussian BGe score (default, MarginalDiBS).
    # "nonlinear": joint inference with a neural-net likelihood
    # (DenseNonlinearGaussian + JointDiBS) — used by case_3 whose SEM is a
    # 1-hidden-layer 10-unit ReLU MLP; `hidden_layers` matches that spec.
    model: str = "bge",
    hidden_layers: Tuple[int, ...] = (10,),
    verbose: bool = False,
) -> Tuple[np.ndarray, str]:
    """
    Run DiBS and return binary graph samples.

    Returns:
        A_samples  : [S, d, d] BINARY (already hard graphs from DiBS).
        convention : "i_to_j"
    """
    import jax
    import jax.numpy as jnp
    import jax.random as jrand
    from dibs.inference import JointDiBS, MarginalDiBS
    # MarginalDiBS requires a likelihood model that exposes
    # `interventional_log_marginal_prob` — that's the BGe (Bayesian
    # Gaussian equivalent) score, built by `make_linear_gaussian_equivalent_model`.
    # `make_linear_gaussian_model` returns a plain `LinearGaussian` that only
    # has the joint `log_prob` and is intended for JointDiBS.
    from dibs.target import make_linear_gaussian_equivalent_model as make_linear_gaussian_model
    from dibs.target import make_nonlinear_gaussian_model

    if model not in ("bge", "nonlinear"):
        raise ValueError(f"Unknown DiBS model {model!r}; use 'bge' or 'nonlinear'.")

    key = jrand.PRNGKey(int(seed))
    # Build the target manually: DiBS wants the log joint prob and graph
    # prior callables.  Both ported factories return the same
    # (data, graph_model, likelihood_model) triple with these already bound.
    key, subk = jrand.split(key)
    if model == "nonlinear":
        data, graph_model, likelihood_model = make_nonlinear_gaussian_model(
            key=subk, n_vars=num_nodes, graph_prior_str="er",
            hidden_layers=tuple(hidden_layers),
        )
    else:
        data, graph_model, likelihood_model = make_linear_gaussian_model(
            key=subk, n_vars=num_nodes, graph_prior_str="er",
        )

    # Override the observations with the actual training split.
    data = data._replace(x=jnp.asarray(X_train, dtype=jnp.float32))

    # The nonlinear likelihood's per-MC-sample working set is the parent-masked
    # data tensor [d, n, d], and the gradient estimators are vmapped over
    # n_grad_mc_samples INSIDE a vmap over particles -- so the default path
    # holds n_particles * n_grad_mc_samples of them at once.  At d=25 that is
    # 20 * 128 * 25 * n * 25 * 4 B, i.e. 6 GiB at n=10^3 but 19 GiB at n=10^3.5
    # and 60 GiB at n=10^4, which OOMs a 40 GB GPU (both cells failed with
    # RESOURCE_EXHAUSTED before this).  Chunking evaluates the MC samples
    # `grad_mc_chunk_size` at a time; it is the same estimator over the same
    # samples in the same order, only fewer intermediates are live at once.
    # Sized to keep the tensor near 6 GiB, i.e. the n=10^3 footprint that ran
    # fine, and left off entirely for small n / the BGe path so previously
    # completed cells reproduce bit-for-bit.
    extra = {}
    if model == "nonlinear":
        n_obs = int(data.x.shape[0])
        chunk = _grad_mc_chunk_for(n_obs, num_particles, int(data.x.shape[1]))
        if chunk is not None:
            extra["grad_mc_chunk_size"] = chunk
            if verbose:
                print(f"[dibs_wrapper] n={n_obs}: grad_mc_chunk_size={chunk}")

    dibs_cls = JointDiBS if model == "nonlinear" else MarginalDiBS
    dibs = dibs_cls(
        x=data.x,
        interv_mask=None,
        graph_model=graph_model,
        likelihood_model=likelihood_model,
        **extra,
    )
    key, subk = jrand.split(key)
    gs = dibs.sample(
        key=subk,
        n_particles=num_particles,
        steps=steps,
        callback_every=max(1, steps // 4) if verbose else steps + 1,
    )
    if model == "nonlinear":
        # JointDiBS.sample returns (graphs, thetas); we only score structure.
        gs = gs[0]
    gs_np = np.asarray(gs, dtype=np.int32)  # [n_particles, d, d]

    # Tile particles up to the requested sample count so downstream metrics
    # have the same S across algorithms (DiBS particles ARE the posterior
    # samples; repeating them is the fair apples-to-apples choice).
    if gs_np.shape[0] >= num_posterior_samples:
        idx = np.linspace(0, gs_np.shape[0] - 1, num_posterior_samples).astype(int)
        A_samples = gs_np[idx]
    else:
        reps = int(np.ceil(num_posterior_samples / gs_np.shape[0]))
        A_samples = np.tile(gs_np, (reps, 1, 1))[:num_posterior_samples]
    return A_samples.astype(np.float32), "i_to_j"
