#!/usr/bin/env python
"""
Case 4: SVIDAG runner
=====================

Trains SVIDAG under the noninformative prior on one train-split of Sachs and
returns posterior RELAXED adjacency samples (so downstream metric code in
``common.py`` handles thresholding uniformly across all algorithms for a
fair comparison).

Scenarios exposed here:
    * "noninformative"    -- p_ij = 0.5 everywhere (purely data-driven).
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

# sys.path plumbing so this script can be imported whether it lives as a
# module or is executed directly.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
for p in (str(_REPO_ROOT), str(_REPO_ROOT / "src"), str(_THIS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
from sklearn.preprocessing import StandardScaler

from common import SachsData  # noqa: F401  (re-exported for type hints)

from svidag import config
from svidag.data import Dataset
from svidag.train import (make_model_and_state, train_step_donated,
                          maybe_warmstart, maybe_resample)
from svidag.eval import sample_hard_adj_fast
from svidag.utils import (
    get_prior_matrix,
    compute_alpha_beta_from_prior,
    to_device,
    to_numpy,
)
from svidag.runner import _clone_train_state


# ---------------------------------------------------------------------------
# Hyperparameters (kept here so the runner is self-contained; override by
# editing this file or by monkey-patching svidag.config before import).
# ---------------------------------------------------------------------------
PATIENCE = 3_000           # Early-stopping patience for ELBO.


# ---------------------------------------------------------------------------
# Environment-driven hyperparameter profile.
#
# Mirrors the BayesDAG FAST-profile convention already used by
# ``run_case4_all.sh``: the sbatch script exports a handful of ``SVIDAG_*``
# variables and this module applies them to ``svidag.config`` before the model
# is built or ``train_step`` is traced.  With nothing exported, every value
# falls through to the committed ``config.py`` default, so the paper-spec
# behaviour is unchanged.
#
# Identical block to ``case_2``/``case_3``'s runners.  NOTE for case 4: the
# table's three prior-sensitivity rows must keep their own priors, so
# ---------------------------------------------------------------------------
_ENV_FLOAT = {
    "SVIDAG_LR": "lr",
    "SVIDAG_ETA_R": "eta_r",
    "SVIDAG_GRAD_CLIP": "grad_clip",
    "SVIDAG_PRIOR_R_SIGMA": "prior_r_sigma",
    "SVIDAG_T_B": ("T_B_start", "T_B_end"),
    "SVIDAG_TAU_SINK": ("tau_sink_start", "tau_sink_end"),
    "SVIDAG_OBS_NOISE": "obs_noise_scale",
    "SVIDAG_PR_PREC_KAPPA": "PR_PREC_KAPPA",
    "SVIDAG_PARTICLE_CLIP": "particle_grad_clip",
    "SVIDAG_ST_WARMUP": "st_warmup_frac",
    "SVIDAG_ETA_R_WARMUP": "eta_r_warmup_frac",
    "SVIDAG_TAU_START": "tau_sink_start",
    "SVIDAG_TAU_END": "tau_sink_end",
    "SVIDAG_TAU_ANNEAL_FRAC": "tau_anneal_frac",
    "SVIDAG_SVGD_REPULSION": "svgd_repulsion_weight",
    "SVIDAG_SVGD_REP_RATIO": "svgd_repulsion_max_ratio",
    "SVIDAG_RESAMPLE_TEMP": "particle_resample_temp",
    "SVIDAG_RESAMPLE_JITTER": "particle_resample_jitter",
    "SVIDAG_SVGD_REP_ANNEAL": "svgd_repulsion_anneal_frac",
    "SVIDAG_WARMSTART_FRAC": "particle_warmstart_frac",
    "SVIDAG_WARMSTART_JITTER": "particle_warmstart_jitter",
    "SVIDAG_KL_THETA": "kl_theta_weight",
    "SVIDAG_PRIOR_THETA_SIGMA": "prior_theta_sigma",
}
_ENV_STR = {
    "SVIDAG_PARTICLE_CLIP_MODE": "particle_grad_clip_mode",
    "SVIDAG_FLOW_TYPE": "flow_type",
}
_ENV_INT = {
    "SVIDAG_BATCH_SIZE": "batch_size",
    "SVIDAG_N_PARTICLES": "n_particles",
    "SVIDAG_SINKHORN_ITERS": "sinkhorn_iters",
    "SVIDAG_HIDDEN_DIM": "hidden_dim",
    "SVIDAG_FLOW_BLOCKS": "flow_n_blocks",
    "SVIDAG_NSF_BINS": "nsf_num_bins",
    "SVIDAG_RESAMPLE_EVERY": "particle_resample_every",
}

# Host-sync stride for the early-stopping ELBO check.  The original loop calls
# float(aux["elbo"]) every iteration, which blocks on a device sync each step
# and serialises the whole training loop against dispatch.
EVAL_EVERY = int(os.environ.get("SVIDAG_EVAL_EVERY", "1"))
PATIENCE = int(os.environ.get("SVIDAG_PATIENCE", str(PATIENCE)))


def _apply_env_overrides(verbose: bool = False) -> None:
    applied = {}
    for env_key, attr in _ENV_FLOAT.items():
        if env_key in os.environ:
            val = float(os.environ[env_key])
            for a in (attr if isinstance(attr, tuple) else (attr,)):
                setattr(config, a, val)
            applied[env_key] = val
    for env_key, attr in _ENV_INT.items():
        if env_key in os.environ:
            val = int(os.environ[env_key])
            setattr(config, attr, val)
            applied[env_key] = val
    for env_key, attr in _ENV_STR.items():
        if env_key in os.environ:
            val = os.environ[env_key]
            setattr(config, attr, val)
            applied[env_key] = val
    if "SVIDAG_MC_SAMPLES" in os.environ:
        v = max(1, int(os.environ["SVIDAG_MC_SAMPLES"]))
        config.elbo_mc_samples = v
        config.ELBO_MC_SAMPLES = v
        applied["SVIDAG_MC_SAMPLES"] = v
    if "SVIDAG_FLOW_HIDDEN" in os.environ:
        raw = os.environ["SVIDAG_FLOW_HIDDEN"]
        widths = [int(w) for w in raw.split(",") if w.strip()]
        if len(widths) == 1:
            widths = widths * 2
        config.flow_hidden = widths
        applied["SVIDAG_FLOW_HIDDEN"] = widths
    for env_key, attr in (
        ("SVIDAG_ROW_ONLY", "node_cond_row_only"),
        ("SVIDAG_SCALE_INV", "sinkhorn_scale_invariant"),
    ):
        if env_key in os.environ:
            setattr(config, attr, bool(int(os.environ[env_key])))
            applied[env_key] = getattr(config, attr)
    if "SVIDAG_LEARN_NOISE" in os.environ:
        config.learn_likelihood_noise = bool(int(os.environ["SVIDAG_LEARN_NOISE"]))
        applied["SVIDAG_LEARN_NOISE"] = config.learn_likelihood_noise
    if verbose and applied:
        print(f"      [svidag] env profile: {applied}", flush=True)


def _build_prior(scenario: str, dataset: Dataset):
    """Prior edge-probability matrix -> (alpha, beta).

    The scenario matrix comes from ``svidag.utils.get_prior_matrix``.

    ``SVIDAG_PRIOR_P0`` (optionally with ``SVIDAG_PRIOR_NU``) replaces the
    scenario matrix with the SAME probability on every ordered pair.  That is a
    sparsity prior, not domain knowledge: it says how many edges to expect, never
    which ones, and it never reads ``true_adj``.  It is the direct analogue of
    the Erdos-Renyi prior DiBS and JSP-GFN use and of BCD Nets' horseshoe.
    Only meaningful for the noninformative row -- the two prior-sensitivity rows
    must keep their own matrices, so it is ignored for them.
    """
    num_nodes = dataset.num_nodes
    p0 = os.environ.get("SVIDAG_PRIOR_P0")
    if p0 is not None and scenario == "noninformative":
        p_prior = jnp.full((num_nodes, num_nodes), float(p0), dtype=jnp.float32) * (
            1.0 - jnp.eye(num_nodes, dtype=jnp.float32))
        nu_env = os.environ.get("SVIDAG_PRIOR_NU")
        if nu_env is not None:
            nu = float(nu_env) * (1.0 - jnp.eye(num_nodes, dtype=jnp.float32))
            alpha_mat, beta_mat = nu * p_prior + 1.0, nu * (1.0 - p_prior) + 1.0
        else:
            alpha_mat, beta_mat = compute_alpha_beta_from_prior(p_prior)
    else:
        p_prior = get_prior_matrix(
            scenario, dataset.node_names, dataset.true_adj_np, num_nodes
        )
        alpha_mat, beta_mat = compute_alpha_beta_from_prior(p_prior)
    return to_device(p_prior), to_device(alpha_mat), to_device(beta_mat)


# ---------------------------------------------------------------------------
# Build a Dataset for a split without going through load_sachs_dataset
# ---------------------------------------------------------------------------
def dataset_from_arrays(
    X_train: np.ndarray,
    X_test: np.ndarray,
    true_adj_np: np.ndarray,
    node_names,
    dataset_name: str = "sachs_split",
) -> Dataset:
    """
    Replicate the tail of ``svidag.data.load_sachs_dataset`` (scaling + device
    placement + Dataset packing) but for a user-supplied train/test partition.
    """
    num_nodes = true_adj_np.shape[0]
    assert X_train.shape[1] == num_nodes
    assert X_test.shape[1] == num_nodes

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_data = to_device(jnp.array(X_train_scaled, dtype=jnp.float32))
    test_data_scaled = to_device(jnp.array(X_test_scaled, dtype=jnp.float32))
    dataset_size = train_data.shape[0]
    noise_scales = to_device(
        jnp.ones((num_nodes,), dtype=jnp.float32) * config.obs_noise_scale
    )
    true_adj_np_f = true_adj_np.astype(np.float32)
    true_DAG = jnp.array(true_adj_np_f)

    return Dataset(
        train_data=train_data,
        test_data_scaled=test_data_scaled,
        dataset_size=dataset_size,
        noise_scales=noise_scales,
        obs_train_orig=np.asarray(X_train, dtype=np.float32),
        obs_test_orig=np.asarray(X_test, dtype=np.float32),
        scaler=scaler,
        true_adj_np=true_adj_np_f,
        true_DAG=true_DAG,
        node_names=list(node_names),
        num_nodes=num_nodes,
        dataset_name=dataset_name,
    )


# ---------------------------------------------------------------------------
# Train one SVIDAG model on one split under one scenario
# ---------------------------------------------------------------------------
@dataclass
class _TrainedSVIDAG:
    state: object           # TrainState (opaque to this module)
    alpha_mat: jnp.ndarray
    beta_mat: jnp.ndarray


def _train_svidag(
    scenario: str,
    dataset: Dataset,
    seed: int,
    num_iters: int,
    verbose: bool = True,
) -> _TrainedSVIDAG:
    """
    Train SVIDAG to convergence (ELBO-based early stopping).

    Mirrors ``paper_results_reproduce/case_1/run_case1.py:train_model`` but
    takes explicit ``seed`` and ``num_iters`` so the caller controls split-to-split
    reproducibility and compute budget.
    """
    key = jrand.PRNGKey(seed)

    p_prior, alpha_mat, beta_mat = _build_prior(scenario, dataset)

    key, init_key = jrand.split(key)
    _model, state = make_model_and_state(
        init_key, dataset.train_data, p_prior, dataset.num_nodes,
        fixed_noise_scales=dataset.noise_scales,
    )

    best_elbo = -np.inf
    best_state = _clone_train_state(state)
    no_improve = 0
    stopped_early = False
    # Early stopping must not look at the relaxation warm-up: while
    # ``st_weight`` < 1 the likelihood is evaluated on the SOFT adjacency,
    # which fits strictly better than the hard DAG, so every warm-up iterate
    # scores a higher ELBO than anything that follows. Selecting on that would
    # reliably return a warm-up state rather than a trained one.
    warm_end = int(float(config.st_warmup_frac) * num_iters)
    t0 = time.time()

    for it in range(1, num_iters + 1):
        key, kb, ks = jrand.split(key, 3)
        idx = jrand.randint(kb, (config.batch_size,), 0, dataset.dataset_size)
        batch = dataset.train_data[idx]

        state, _, aux = train_step_donated(
            state, batch, ks, it, num_iters,
            alpha_mat, beta_mat, dataset.dataset_size,
            config.ELBO_MC_SAMPLES,
        )
        state, _ess = maybe_resample(state, batch, ks, it, num_iters, alpha_mat,
                                     beta_mat, dataset.dataset_size,
                                     config.ELBO_MC_SAMPLES, config.T_B_end,
                                     config.tau_sink_end)
        state, _warmstarted = maybe_warmstart(state, ks, it, num_iters, config.T_B_end)
        if _warmstarted and verbose:
            print(f"      warm-started SVGD particles @ iter {it}")

        # Host sync only every EVAL_EVERY iters.  float(aux["elbo"]) blocks
        # until the step's device work completes, so doing it every iteration
        # prevents JAX from ever queueing the next step while the current one
        # runs.  EVAL_EVERY=1 reproduces the original behaviour exactly.
        if it % EVAL_EVERY == 0 and it > warm_end:
            cur_elbo = float(aux["elbo"])
            if cur_elbo > best_elbo:
                best_elbo = cur_elbo
                best_state = _clone_train_state(state)
                no_improve = 0
            else:
                no_improve += EVAL_EVERY
                if no_improve >= PATIENCE:
                    if verbose:
                        print(f"      early-stop @ iter {it}")
                    state = best_state
                    stopped_early = True
                    break

        if verbose and it % config.print_every == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(
                f"      iter {it:6d}/{num_iters} | ELBO {aux['elbo']:.3f}"
                f" | ELL {aux['ell']:.3f} | KLγ {aux['kl_gamma']:.3f}"
                f" | {dt:.1f}s"
            )

    # Single-batch ELBO swings by ~4x between consecutive Sachs minibatches,
    # so "best ELBO ever seen" is close to a random iterate. Keep the final
    # state unless early stopping actually fired and restored the best one.
    final = best_state if stopped_early else state
    return _TrainedSVIDAG(state=final, alpha_mat=alpha_mat, beta_mat=beta_mat)


def _sample_posterior(
    trained: _TrainedSVIDAG,
    dataset: Dataset,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    """
    Draw ``num_samples`` hard (binary) posterior DAG samples A = B ⊙ M(r).

    Binary and acyclic by construction; no thresholding step is involved.
    Convention: SVIDAG's native ``A[i,j]=1 means j -> i`` -- no transpose needed
    when passed through ``common.evaluate_samples(source_convention="j_to_i")``.
    """
    key = jrand.PRNGKey(seed + 10_007)
    # A = B ⊙ M(r) depends only on the flow and the order potentials, so the
    # per-node Bayesian MLPs / noise posterior / Sinkhorn that the full
    # forward pass computes are all dead work here.  sample_hard_adj_fast
    # returns bit-identical draws without them.
    A_samples = sample_hard_adj_fast(
        trained.state.apply_fn,
        trained.state.params,
        trained.state.particles,
        key,
        config.T_B_end,
        num_samples=num_samples,
        distinct_particles=True,
    )
    return np.asarray(A_samples).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Public entry point for one (scenario, split)
# ---------------------------------------------------------------------------
def run_svidag(
    X_train: np.ndarray,
    X_test: np.ndarray,
    true_adj: np.ndarray,
    node_names,
    scenario: str,
    split_index: int,
    num_posterior_samples: int,
    num_iters: int,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[np.ndarray, str]:
    """
    Train SVIDAG for (scenario, split) and return relaxed posterior samples.

    ``scenario`` is "noninformative".

    Returns:
        A_relaxed : [S, d, d] float32, values in [0, 1], SVIDAG convention.
        convention: "j_to_i" (always, for this runner).
    """
    _apply_env_overrides(verbose=verbose)
    dataset = dataset_from_arrays(
        X_train, X_test, true_adj, node_names,
        dataset_name=f"sachs_split{split_index}_{scenario}",
    )
    trained = _train_svidag(
        scenario=scenario, dataset=dataset,
        seed=seed + split_index * 997,
        num_iters=num_iters, verbose=verbose,
    )
    A_relaxed = _sample_posterior(
        trained, dataset,
        num_samples=num_posterior_samples,
        seed=seed + split_index * 997,
    )
    return A_relaxed, "j_to_i"
