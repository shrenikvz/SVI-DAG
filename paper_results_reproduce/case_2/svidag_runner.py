#!/usr/bin/env python
"""
Case 2: SVIDAG runner (synthetic linear-DAG benchmarks)
=======================================================

Adapted from ``paper_results_reproduce/case_4/svidag_runner.py``.  Two key
differences vs the case_4 runner:

  1. Operates on a ``SyntheticDataset`` (not Sachs) -- node names are just
     ``["x0", ..., "x{p-1}"]`` and ``true_adj`` is the freshly generated
     synthetic DAG.
  2. Clamps the SVIDAG mini-batch size to ``min(config.batch_size, n)`` so
     that runs with very small ``n`` (e.g. ``n=10`` from the case_2 grid)
     do not request a 64-sample batch from a 10-sample dataset.

Scenarios exposed: ``"strong_correct"``, ``"noninformative"``,
``"strong_incorrect"`` -- identical to case_4.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Tuple

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# Force the case-local dir to sys.path[0] so its ``common.py`` shadows the
# one in case_4 (same module name exists in both directories).
_local = str(_THIS_DIR)
if _local in sys.path:
    sys.path.remove(_local)
sys.path.insert(0, _local)

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
from sklearn.preprocessing import StandardScaler

from svidag import config
from svidag.data import Dataset
from svidag.train import (make_model_and_state, train_step_donated,
                          maybe_warmstart, maybe_resample)
from svidag.eval import sample_hard_adj_fast
from svidag.utils import (
    build_mask_M_hard,
    center_order_potentials,
    get_prior_matrix,
    compute_alpha_beta_from_prior,
    logistic_concrete,
    to_device,
    vec_to_offdiag_matrix,
)
from svidag.runner import _clone_train_state


PATIENCE = 3_000  # Early-stopping patience on ELBO improvement.


# ---------------------------------------------------------------------------
# Environment-driven hyperparameter profile.
#
# Mirrors the BayesDAG FAST-profile convention already used by
# ``run_case2_all.sh``: the sbatch script exports a handful of ``SVIDAG_*``
# variables and this module applies them to ``svidag.config`` before the model
# is built or ``train_step`` is traced.  With nothing exported, every value
# falls through to the committed ``config.py`` default, so the paper-spec
# behaviour is unchanged.
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
    "SVIDAG_ST_WARMUP": "st_warmup_frac",
    "SVIDAG_TAU_START": "tau_sink_start",
    "SVIDAG_TAU_END": "tau_sink_end",
    "SVIDAG_TAU_ANNEAL_FRAC": "tau_anneal_frac",
    "SVIDAG_SVGD_REPULSION": "svgd_repulsion_weight",
    "SVIDAG_SVGD_REP_RATIO": "svgd_repulsion_max_ratio",
    "SVIDAG_RESAMPLE_TEMP": "particle_resample_temp",
    "SVIDAG_RESAMPLE_JITTER": "particle_resample_jitter",
    "SVIDAG_PARTICLE_CLIP": "particle_grad_clip",
    "SVIDAG_KL_THETA": "kl_theta_weight",
    "SVIDAG_WARMSTART_FRAC": "particle_warmstart_frac",
    "SVIDAG_WARMSTART_JITTER": "particle_warmstart_jitter",
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
            setattr(config, attr, os.environ[env_key])
            applied[env_key] = os.environ[env_key]
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
    if "SVIDAG_PARTICLE_CLIP_MODE" in os.environ:
        config.particle_grad_clip_mode = os.environ["SVIDAG_PARTICLE_CLIP_MODE"]
        applied["SVIDAG_PARTICLE_CLIP_MODE"] = config.particle_grad_clip_mode
    if "SVIDAG_LEARN_NOISE" in os.environ:
        config.learn_likelihood_noise = bool(int(os.environ["SVIDAG_LEARN_NOISE"]))
        applied["SVIDAG_LEARN_NOISE"] = config.learn_likelihood_noise
    # These Case-2-only values affect posterior sampling rather than a
    # ``config`` field, but include them in the reproducibility log.
    for env_key in (
        "SVIDAG_POSTERIOR_BIAS_INTERCEPT",
        "SVIDAG_POSTERIOR_BIAS_LOG10_SLOPE",
        "SVIDAG_POSTERIOR_BIAS_REFERENCE_N",
        "SVIDAG_POSTERIOR_BIAS_FLOOR",
        "SVIDAG_POSTERIOR_BIAS_CEILING",
        "SVIDAG_POSTERIOR_SCALE_INTERCEPT",
        "SVIDAG_POSTERIOR_SCALE_LOG10_SLOPE",
        "SVIDAG_POSTERIOR_SCALE_REFERENCE_N",
        "SVIDAG_POSTERIOR_SCALE_FLOOR",
        "SVIDAG_POSTERIOR_SCALE_CEILING",
        "SVIDAG_POSTERIOR_Z_SCALE_INTERCEPT",
        "SVIDAG_POSTERIOR_Z_SCALE_LOG10_SLOPE",
        "SVIDAG_POSTERIOR_Z_SCALE_REFERENCE_N",
        "SVIDAG_POSTERIOR_Z_SCALE_FLOOR",
        "SVIDAG_POSTERIOR_Z_SCALE_CEILING",
        "SVIDAG_POSTERIOR_PARTICLE_TEMP",
    ):
        if env_key in os.environ:
            applied[env_key] = float(os.environ[env_key])
    if verbose and applied:
        print(f"      [svidag] env profile: {applied}", flush=True)


def _build_prior(scenario: str, dataset: Dataset):
    """Prior edge-probability matrix -> (alpha, beta) for the given scenario.

    ``SVIDAG_PRIOR_P0`` (optionally with ``SVIDAG_PRIOR_NU``) replaces the
    scenario matrix with the SAME probability on every ordered pair. That is a
    sparsity prior, not domain knowledge: it says how many edges to expect,
    never which ones, and it never reads ``true_adj``. It is the direct
    analogue of the Erdos-Renyi prior DiBS and JSP-GFN use and of BCD Nets'
    horseshoe. Only applied to the noninformative row.
    """
    num_nodes = dataset.num_nodes
    p0 = os.environ.get("SVIDAG_PRIOR_P0")
    if p0 is not None and scenario == "noninformative":
        eye = jnp.eye(num_nodes, dtype=jnp.float32)
        p_prior = jnp.full((num_nodes, num_nodes), float(p0), dtype=jnp.float32) * (1.0 - eye)
        nu_env = os.environ.get("SVIDAG_PRIOR_NU")
        if nu_env is not None:
            nu = float(nu_env) * (1.0 - eye)
            alpha_mat, beta_mat = nu * p_prior + 1.0, nu * (1.0 - p_prior) + 1.0
        else:
            alpha_mat, beta_mat = compute_alpha_beta_from_prior(p_prior)
    else:
        p_prior = get_prior_matrix(
            scenario, dataset.node_names, dataset.true_adj_np, num_nodes
        )
        alpha_mat, beta_mat = compute_alpha_beta_from_prior(p_prior)
    return to_device(p_prior), to_device(alpha_mat), to_device(beta_mat)


def _build_svidag_dataset(
    X_train_scaled: np.ndarray,
    true_adj_np: np.ndarray,
    node_names,
    dataset_name: str,
) -> Dataset:
    """Wrap pre-standardised data into the SVIDAG ``Dataset`` container."""
    num_nodes = true_adj_np.shape[0]
    assert X_train_scaled.shape[1] == num_nodes
    n = X_train_scaled.shape[0]

    # SVIDAG's Dataset normally also carries a held-out test slice for
    # predictive evaluation.  We are reporting only structural CPDAG metrics
    # in case_2, so we feed a single-row placeholder that keeps the dataclass
    # well-formed without affecting training.
    X_test_placeholder = X_train_scaled[:1]

    train_data = to_device(jnp.asarray(X_train_scaled, dtype=jnp.float32))
    test_data_scaled = to_device(jnp.asarray(X_test_placeholder, dtype=jnp.float32))
    noise_scales = to_device(
        jnp.ones((num_nodes,), dtype=jnp.float32) * config.obs_noise_scale
    )
    true_adj_np_f = true_adj_np.astype(np.float32)
    true_DAG = jnp.array(true_adj_np_f)

    # The SVIDAG ``Dataset`` dataclass requires a ``scaler`` slot for
    # ``predict_x_given_parents`` etc.  We attach a degenerate StandardScaler
    # already fitted on the standardised training data so any downstream
    # call that expects ``dataset.scaler.transform(...)`` is a no-op.
    sc = StandardScaler()
    sc.fit(np.asarray(X_train_scaled))

    return Dataset(
        train_data=train_data,
        test_data_scaled=test_data_scaled,
        dataset_size=n,
        noise_scales=noise_scales,
        obs_train_orig=np.asarray(X_train_scaled, dtype=np.float32),
        obs_test_orig=np.asarray(X_test_placeholder, dtype=np.float32),
        scaler=sc,
        true_adj_np=true_adj_np_f,
        true_DAG=true_DAG,
        node_names=list(node_names),
        num_nodes=num_nodes,
        dataset_name=dataset_name,
    )


@dataclass
class _TrainedSVIDAG:
    state: object
    alpha_mat: jnp.ndarray
    beta_mat: jnp.ndarray


def _train_svidag(
    scenario: str,
    dataset: Dataset,
    seed: int,
    num_iters: int,
    verbose: bool = True,
) -> _TrainedSVIDAG:
    """Train SVIDAG with ELBO-based early stopping.  Mini-batch size is
    clamped to ``min(config.batch_size, dataset.dataset_size)`` to support
    the small-n end of the case_2 grid (n=10)."""
    key = jrand.PRNGKey(seed)

    p_prior, alpha_mat, beta_mat = _build_prior(scenario, dataset)

    key, init_key = jrand.split(key)
    _model, state = make_model_and_state(
        init_key, dataset.train_data, p_prior, dataset.num_nodes,
        fixed_noise_scales=dataset.noise_scales,
    )

    eff_batch = int(min(config.batch_size, dataset.dataset_size))

    best_elbo = -np.inf
    best_state = _clone_train_state(state)
    no_improve = 0
    stopped_early = False
    # Early stopping must skip the relaxation warm-up: while ``st_weight`` < 1
    # the likelihood is evaluated on the SOFT adjacency, which fits strictly
    # better than the hard DAG, so every warm-up iterate outscores everything
    # that follows and "best ELBO" would always land inside the warm-up.
    warm_end = int(float(config.st_warmup_frac) * num_iters)
    t0 = time.time()

    for it in range(1, num_iters + 1):
        key, kb, ks = jrand.split(key, 3)
        idx = jrand.randint(kb, (eff_batch,), 0, dataset.dataset_size)
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

    # The single-batch ELBO is noisy enough that its running maximum is close
    # to an arbitrary iterate; keep the final state unless early stopping
    # actually fired and restored the best one.
    final = best_state if stopped_early else state
    return _TrainedSVIDAG(state=final, alpha_mat=alpha_mat, beta_mat=beta_mat)


def _sample_A_hard_with_bias(model, r, rng, T_B, logit_bias, logit_scale,
                             z_scale):
    """Case-2 hard-DAG draw with the predeclared posterior tempering family.

    ``logit_scale`` is a posterior sampling temperature: because the hard draw
    is ``1{gamma' + L > 0}`` with logistic noise ``L``, scaling the logits by
    ``a`` is the same as sampling at noise temperature ``1/a`` -- it sharpens
    (a > 1) or flattens (a < 1) every per-draw edge marginal without changing
    their ranking.

    ``z_scale`` is the flow's latent temperature: ``z ~ z_scale * N(0, I)``.
    ``1`` is the untempered posterior; ``0`` collapses each draw's edge logits
    to the flow's central tendency for its ordering (per-draw ranking noise
    from the latent is removed while the Bernoulli edge draw stays stochastic).
    """
    rng, k_z, k_B, _k_noise = jrand.split(rng, 4)
    r_centered = center_order_potentials(r)
    d = model.num_nodes * (model.num_nodes - 1)
    z = z_scale * jrand.normal(k_z, (d,))
    gamma_flat, _log_det = model.flow(z, cond=r_centered)
    gamma_flat = jnp.clip(logit_scale * gamma_flat + logit_bias, -15.0, 15.0)
    B_tilde = vec_to_offdiag_matrix(
        logistic_concrete(k_B, gamma_flat, T_B), model.num_nodes
    )
    B_hard = jnp.where(B_tilde >= 0.5, 1.0, 0.0)
    return B_hard * build_mask_M_hard(r_centered)


@partial(jax.jit, static_argnums=(0,))
def _sample_hard_adj_biased_jit(
    apply_fn, params, selected_r, rng, T_B, logit_bias, logit_scale, z_scale
):
    """Compiled calibrated sampler; the tempering values remain dynamic.

    ``selected_r`` is the [S, m] matrix of order-potential draws -- particle
    selection (uniform or ELBO-tempered) happens on the host in
    ``_sample_posterior``.
    """
    num_samples = selected_r.shape[0]
    keys = jrand.split(rng, num_samples)

    def single_eval(r_val, sample_key):
        return apply_fn(
            {"params": params},
            r_val,
            sample_key,
            T_B,
            logit_bias,
            logit_scale,
            z_scale,
            method=_sample_A_hard_with_bias,
        )

    return jax.vmap(single_eval)(selected_r, keys)


def _scheduled_log10_value(prefix: str, dataset_size: int, intercept_default: float) -> float:
    """Shared form of the predeclared log-sample-size calibration schedules:
    ``clip(intercept + slope * log10(n / reference_n), floor, ceiling)``.
    """
    intercept = float(os.environ.get(f"{prefix}_INTERCEPT", str(intercept_default)))
    slope = float(os.environ.get(f"{prefix}_LOG10_SLOPE", "0"))
    reference_n = float(os.environ.get(f"{prefix}_REFERENCE_N", "100"))
    floor = float(os.environ.get(f"{prefix}_FLOOR", "-inf"))
    ceiling = float(os.environ.get(f"{prefix}_CEILING", "inf"))
    if reference_n <= 0:
        raise ValueError(f"{prefix}_REFERENCE_N must be positive")
    if floor > ceiling:
        raise ValueError(f"{prefix} floor must not exceed its ceiling")
    scheduled = intercept + slope * np.log10(float(dataset_size) / reference_n)
    return float(np.clip(scheduled, floor, ceiling))


def _posterior_logit_bias(dataset_size: int) -> float:
    """Predeclared log-sample-size shift calibration used only by Case 2.

    NOTE: the historical case-2 convention is that a *positive*
    ``SVIDAG_POSTERIOR_BIAS_LOG10_SLOPE`` shifts logits *down* as n grows, so
    the slope is negated here (case 3 uses the non-negated form).
    """
    intercept = float(os.environ.get("SVIDAG_POSTERIOR_BIAS_INTERCEPT", "0"))
    slope = float(os.environ.get("SVIDAG_POSTERIOR_BIAS_LOG10_SLOPE", "0"))
    reference_n = float(os.environ.get("SVIDAG_POSTERIOR_BIAS_REFERENCE_N", "100"))
    floor = float(os.environ.get("SVIDAG_POSTERIOR_BIAS_FLOOR", "-inf"))
    ceiling = float(os.environ.get("SVIDAG_POSTERIOR_BIAS_CEILING", "inf"))
    if reference_n <= 0:
        raise ValueError("SVIDAG_POSTERIOR_BIAS_REFERENCE_N must be positive")
    if floor > ceiling:
        raise ValueError("SVIDAG posterior-bias floor must not exceed its ceiling")
    scheduled = intercept - slope * np.log10(float(dataset_size) / reference_n)
    return float(np.clip(scheduled, floor, ceiling))


def _posterior_logit_scale(dataset_size: int) -> float:
    """Predeclared log-sample-size temperature calibration used only by Case 2.

    Returns the multiplicative logit scale ``a`` (posterior sampling
    temperature ``1/a``); ``a = 1`` reproduces the uncalibrated sampler.
    """
    return _scheduled_log10_value("SVIDAG_POSTERIOR_SCALE", dataset_size, 1.0)


def _posterior_z_scale(dataset_size: int) -> float:
    """Predeclared latent-temperature schedule for the flow's base noise.

    ``1`` reproduces the untempered sampler; ``0`` collapses each ordering's
    edge logits to the flow's conditional central tendency.
    """
    return _scheduled_log10_value("SVIDAG_POSTERIOR_Z_SCALE", dataset_size, 1.0)


def _select_particles_tempered(trained, dataset, key, num_samples):
    """Particle draw for posterior sampling.

    Uniform-with-replacement by default (the historical sampler).  With
    ``SVIDAG_POSTERIOR_PARTICLE_TEMP`` set, particles are drawn from softmax
    weights of the per-particle SVGD objective (ELBO(r) + log p(r)) at
    ``temp`` times the objective's spread across the cloud -- the same
    weighting ``svidag.train.resample_particles`` uses during training, here
    applied only at sampling time (a cold posterior over orderings).
    """
    temp_env = os.environ.get("SVIDAG_POSTERIOR_PARTICLE_TEMP")
    K = trained.state.particles.shape[0]
    if temp_env is None:
        idx = jrand.randint(key, (num_samples,), 0, config.n_particles)
        return trained.state.particles[idx]
    from svidag.train import _particle_objectives
    temp = float(temp_env)
    wb = dataset.train_data[: min(1024, dataset.dataset_size)]
    ell_scale = dataset.dataset_size / wb.shape[0]
    obj = _particle_objectives(
        trained.state.apply_fn, trained.state.params, trained.state.particles,
        wb, jrand.PRNGKey(0), trained.alpha_mat, trained.beta_mat, 4,
        ell_scale, config.T_B_end, config.tau_sink_end, 1.0)
    obj = jnp.nan_to_num(obj, nan=-jnp.inf)
    scale = jnp.std(obj) * temp + 1e-8
    w = jax.nn.softmax((obj - jnp.max(obj)) / scale)
    idx = jrand.choice(key, K, shape=(num_samples,), p=w)
    return trained.state.particles[idx]


def _sample_posterior(
    trained: _TrainedSVIDAG,
    dataset: Dataset,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    """Draw ``num_samples`` hard (binary) posterior DAG samples A = B ⊙ M(r).
    Binary and acyclic by construction; no thresholding. Convention:
    SVIDAG-native (j -> i).
    """
    key = jrand.PRNGKey(seed + 10_007)
    # A = B ⊙ M(r) depends only on the flow and the order potentials, so the
    # per-node Bayesian MLPs / noise posterior / Sinkhorn that the full
    # forward pass computes are all dead work here.  sample_hard_adj_fast
    # returns bit-identical draws without them.
    logit_bias = _posterior_logit_bias(dataset.dataset_size)
    logit_scale = _posterior_logit_scale(dataset.dataset_size)
    z_scale = _posterior_z_scale(dataset.dataset_size)
    untempered = (logit_bias == 0.0 and logit_scale == 1.0 and z_scale == 1.0
                  and os.environ.get("SVIDAG_POSTERIOR_PARTICLE_TEMP") is None)
    if untempered:
        A_samples = sample_hard_adj_fast(
            trained.state.apply_fn,
            trained.state.params,
            trained.state.particles,
            key,
            config.T_B_end,
            num_samples=num_samples,
            distinct_particles=True,
        )
    else:
        selected_r = _select_particles_tempered(trained, dataset, key, num_samples)
        A_samples = _sample_hard_adj_biased_jit(
            trained.state.apply_fn,
            trained.state.params,
            selected_r,
            key,
            jnp.asarray(config.T_B_end, dtype=jnp.float32),
            jnp.asarray(logit_bias, dtype=jnp.float32),
            jnp.asarray(logit_scale, dtype=jnp.float32),
            jnp.asarray(z_scale, dtype=jnp.float32),
        )
    return np.asarray(A_samples).astype(np.float32, copy=False)


def run_svidag_synthetic(
    X_train_scaled: np.ndarray,
    true_adj: np.ndarray,
    node_names,
    scenario: str,
    cell_index: int,
    num_posterior_samples: int,
    num_iters: int,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[np.ndarray, str]:
    """Train SVIDAG for one (scenario, cell) and return relaxed posterior samples.

    Parameters
    ----------
    X_train_scaled : pre-standardised observational matrix.
    true_adj       : ground-truth DAG (SVIDAG j->i convention).
    node_names     : list of node names (strings).
    scenario       : one of "strong_correct" / "noninformative" / "strong_incorrect".
    cell_index     : a deterministic index of the (scenario_graph, n, replicate)
                     cell -- folded into the training and sampling seed so each
                     cell of the grid trains from a different RNG state.
    num_posterior_samples : S, posterior draws per fit.
    num_iters             : SGD iters (ELBO-early-stopping caps this).
    seed                  : base seed.
    """
    _apply_env_overrides(verbose=verbose)
    dataset = _build_svidag_dataset(
        X_train_scaled, true_adj, node_names,
        dataset_name=f"case2_cell{cell_index}_{scenario}",
    )
    trained = _train_svidag(
        scenario=scenario, dataset=dataset,
        seed=seed + cell_index * 997,
        num_iters=num_iters, verbose=verbose,
    )
    A_relaxed = _sample_posterior(
        trained, dataset,
        num_samples=num_posterior_samples,
        seed=seed + cell_index * 997,
    )
    return A_relaxed, "j_to_i"
