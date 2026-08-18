"""
Variant implementations for the SVI-DAG component ablation.

Everything that is NOT being ablated is imported from ``case_3/`` (dataset
generation, priors, biased posterior sampler, metrics, the SVGD trainer), so a
variant differs from the benchmark pipeline in exactly one seam:

* ``MeanFieldGamma``      -- flow ablation: a mean-field Gaussian over the edge
                             logits behind the flow interface, independent of r.
* ``train_gaussian_r``    -- SVGD ablation: a reparameterized Gaussian guide on
                             the order potentials, trained jointly by Adam.
* the prior matrix        -- prior ablation: handled by env vars consumed by
                             case_3's ``_build_prior`` (see run_ablation.py).

Import this module BEFORE any model is built: importing installs the
``create_flow_stack`` shim that makes ``flow_type='meanfield'`` resolvable.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PRR = _THIS_DIR.parent
_REPO = _PRR.parent
# case_3 ships the benchmark ``common`` + ``svidag_runner`` this ablation
# reuses.  It must be FIRST so ``import common`` resolves to case_3's module
# (case_4 has a different ``common``; do not add case_4 to the path here).
for _p in (str(_REPO / "src"), str(_PRR / "case_3")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
import jax.numpy as jnp
import jax.random as jrand
from flax import linen as nn
from flax.training import train_state as _flax_train_state

import svidag.model as _svidag_model
from svidag import config
from svidag.train import make_model_and_state, make_optimizer
from svidag.utils import gaussian_logpdf, linear_anneal, std_normal_logpdf

import svidag_runner as SR  # case_3's runner (path injected above)


# ---------------------------------------------------------------------------
# Flow ablation: mean-field Gaussian over edge logits, independent of r
# ---------------------------------------------------------------------------
class MeanFieldGamma(nn.Module):
    """
    Drop-in replacement for the conditional flow stack.

    Interface-compatible with ``NSFCouplingStack``: ``(z, cond) -> (gamma,
    log_det)`` with ``gamma = mu + exp(log_sigma) * z`` and ``log_det =
    sum(log_sigma)``, so the trainer's change-of-variables algebra

        log q(gamma) = log N(z) - log_det

    is exactly the mean-field Gaussian log-density.  ``cond`` (the order
    potentials r) is deliberately ignored -- that independence IS the
    ablation.  Initialised at mu=0, log_sigma=0, i.e. q(gamma) = N(0, I),
    matching the near-identity initialisation of the real flows.
    """
    latent_dim: int

    @nn.compact
    def __call__(self, z, cond=None):
        mu = self.param("mu", nn.initializers.zeros, (self.latent_dim,))
        log_sigma = self.param("log_sigma", nn.initializers.zeros, (self.latent_dim,))
        gamma = mu + jnp.exp(log_sigma) * z
        return gamma, jnp.sum(log_sigma)


_ORIG_CREATE_FLOW_STACK = _svidag_model.create_flow_stack


def _create_flow_stack_shim(flow_type, latent_dim, cond_dim, hidden_dims,
                            n_blocks, num_bins=8, tail_bound=5.0, name="flow"):
    if str(flow_type) == "meanfield":
        return MeanFieldGamma(latent_dim=latent_dim, name=name)
    return _ORIG_CREATE_FLOW_STACK(
        flow_type=flow_type, latent_dim=latent_dim, cond_dim=cond_dim,
        hidden_dims=hidden_dims, n_blocks=n_blocks, num_bins=num_bins,
        tail_bound=tail_bound, name=name,
    )


# ``svidag.model`` binds the factory by name at import; rebinding that name is
# the single injection point every model construction goes through.
_svidag_model.create_flow_stack = _create_flow_stack_shim


# ---------------------------------------------------------------------------
# SVGD ablation: reparameterized Gaussian guide on r
# ---------------------------------------------------------------------------
def _softplus_inv(y: float) -> float:
    return float(np.log(np.expm1(y)))


_GAUSS_STEP_CACHE = {}


def _make_gauss_step(apply_fn, mc_samples: int, num_nodes: int):
    """
    Build the jitted train step for the Gaussian-r variant.

    Mirrors ``svidag.train._train_step_impl`` term for term -- same annealing
    schedules, same ELBO scaling, same per-draw objective -- except that the
    K SVGD particles are replaced by K reparameterized draws from
    ``q(r) = N(mu_r, diag softplus(rho_r)^2)`` and the analytic
    ``KL(q(r) || p(r))`` joins the objective (SVGD realises that term through
    its score + repulsion dynamics; a parametric guide needs it explicitly).
    All parameters -- model AND guide -- share the stock optax chain
    (global-norm clip + Adam at config.lr).

    TWO GRADIENT PASSES, NOT ONE.  A single fused ``grad`` over
    {model params, guide params} builds one backward graph that
    differentiates through the Sinkhorn mask w.r.t. BOTH blocks at once;
    XLA:GPU spent 27+ minutes failing to compile it (0% GPU, 13 CPU cores
    pinned) while the CPU backend compiled it in seconds.  The stock trainer
    never builds that graph: its parameter loss treats the particles as
    constants, and its SVGD score pass treats the parameters as constants --
    and both of those half-graphs compile on GPU in about a minute.  This
    step reproduces exactly that split.  It is not an approximation: the two
    blocks are disjoint, so the joint gradient is the pair of block partials,
    and both passes see the SAME draws (same eps, same MC keys), so each pass
    computes the exact partial of the same scalar objective.
    """
    K = int(config.n_particles)
    prior_sigma = float(config.prior_r_sigma)

    def _schedules(it, total_iters):
        T_B = linear_anneal(it, total_iters, config.T_B_start, config.T_B_end)
        tau_sn = linear_anneal(
            it, jnp.maximum(1.0, float(config.tau_anneal_frac) * total_iters),
            config.tau_sink_start, config.tau_sink_end)
        warm = float(config.st_warmup_frac)
        if warm > 0.0:
            st_weight = jnp.clip(it / jnp.maximum(warm * total_iters, 1.0), 0.0, 1.0)
        else:
            st_weight = 1.0
        return T_B, tau_sn, st_weight

    def _scales(batch_size, N_total):
        if config.elbo_per_datum:
            return 1.0 / batch_size, 1.0 / N_total
        return N_total / batch_size, 1.0

    def _data_elbo(params, r_draws, mc_keys, batch, T_B, tau_sn, st_weight,
                   alpha_mat, beta_mat, ell_scale, kl_weight):
        """mean_k ELBO_terms(params, r_k): the r-dependent part of the objective."""

        def draw_elbo(r_val, keys_mc):
            def single_mc(k):
                _, terms = apply_fn(
                    {"params": params}, batch, r_val, k, T_B, tau_sn,
                    alpha_mat, beta_mat, st_weight)
                log_q_gamma = std_normal_logpdf(terms["z"]) - terms["log_det"]
                kl_gamma = log_q_gamma - terms["log_p_gamma"]
                ell = ell_scale * terms["loglik"]
                return ell - kl_weight * (config.kl_theta_weight * terms["kl_theta"] + kl_gamma)

            return jnp.mean(jax.vmap(single_mc)(keys_mc))

        return jnp.mean(jax.vmap(draw_elbo)(r_draws, mc_keys))

    def _kl_r(mu_r, sigma_r):
        return jnp.sum(
            jnp.log(prior_sigma / sigma_r)
            + (sigma_r ** 2 + mu_r ** 2) / (2.0 * prior_sigma ** 2)
            - 0.5
        )

    @jax.jit
    def step(state, batch, rng, it, total_iters, alpha_mat, beta_mat, N_total):
        T_B, tau_sn, st_weight = _schedules(it, total_iters)
        ell_scale, kl_weight = _scales(batch.shape[0], N_total)

        mu_r = state.params["r_mu"]
        sigma_r = jax.nn.softplus(state.params["r_rho"]) + 1e-6

        k_eps, k_mc = jrand.split(rng)
        eps = jrand.normal(k_eps, (K, num_nodes))
        # Nested splits rather than split-and-reshape: shape-stable under both
        # the legacy uint32[2] and the typed key representations.
        mc_keys = jax.vmap(lambda kk: jrand.split(kk, mc_samples))(
            jrand.split(k_mc, K))

        # --- pass A: model-parameter gradient, r draws held constant --------
        r_const = jax.lax.stop_gradient(mu_r + sigma_r * eps)

        def loss_model(params):
            elbo_data = _data_elbo(params, r_const, mc_keys, batch, T_B, tau_sn,
                                   st_weight, alpha_mat, beta_mat,
                                   ell_scale, kl_weight)
            return -elbo_data, elbo_data

        (_, elbo_data), g_model = jax.value_and_grad(loss_model, has_aux=True)(
            state.params["model"])

        # --- pass B: guide gradient, model parameters held constant ---------
        params_const = jax.lax.stop_gradient(state.params["model"])

        def loss_guide(guide):
            mu = guide["r_mu"]
            sig = jax.nn.softplus(guide["r_rho"]) + 1e-6
            r_draws = mu[None, :] + sig[None, :] * eps      # same eps: same draws
            elbo_g = _data_elbo(params_const, r_draws, mc_keys, batch, T_B,
                                tau_sn, st_weight, alpha_mat, beta_mat,
                                ell_scale, kl_weight)
            return -(elbo_g - kl_weight * _kl_r(mu, sig)), _kl_r(mu, sig)

        (_, kl_r_val), g_guide = jax.value_and_grad(loss_guide, has_aux=True)(
            {"r_mu": state.params["r_mu"], "r_rho": state.params["r_rho"]})

        grads = {"model": g_model,
                 "r_mu": g_guide["r_mu"], "r_rho": g_guide["r_rho"]}
        elbo = elbo_data - kl_weight * kl_r_val
        aux = {"elbo": elbo, "kl_r": kl_r_val, "T_B": T_B, "tau_sn": tau_sn}
        return state.apply_gradients(grads=grads), -elbo, aux

    return step


def train_gaussian_r(scenario: str, dataset, seed: int, num_iters: int,
                     verbose: bool = True):
    """
    Train the no-SVGD variant; drop-in for case_3's ``_train_svidag``.

    Returns ``(trained_namespace, draw_r_fn)`` where ``draw_r_fn(key, n)``
    samples n order-potential vectors from the learned guide.
    """
    key = jrand.PRNGKey(seed)
    p_prior, alpha_mat, beta_mat = SR._build_prior(scenario, dataset)

    key, init_key = jrand.split(key)
    model, base_state = make_model_and_state(
        init_key, dataset.train_data, p_prior, dataset.num_nodes,
        fixed_noise_scales=dataset.noise_scales,
    )

    m = int(dataset.num_nodes)
    all_params = {
        "model": base_state.params,
        # Guide initialised AT the prior N(0, prior_r_sigma^2), matching how
        # the SVGD cloud is initialised (particles ~ that same prior).
        "r_mu": jnp.zeros((m,), dtype=jnp.float32),
        "r_rho": jnp.full((m,), _softplus_inv(config.prior_r_sigma), dtype=jnp.float32),
    }
    tx = make_optimizer(config.lr, config.grad_clip)
    state = _flax_train_state.TrainState.create(
        apply_fn=model.apply, params=all_params, tx=tx)

    mc_samples = int(getattr(config, "ELBO_MC_SAMPLES", config.elbo_mc_samples))
    # One jitted step per (model instance, mc, m) -- svidag's _MODEL_CACHE
    # returns the same model object for every seed of a job, so without this
    # cache each seed builds a fresh closure and pays the full XLA compile
    # again (measured: compile dominates the p=10 fit).  Params are explicit
    # arguments, so reusing the first seed's bound apply is semantically
    # identical.
    cache_key = (id(model), mc_samples, m)
    step = _GAUSS_STEP_CACHE.get(cache_key)
    if step is None:
        step = _make_gauss_step(model.apply, mc_samples, m)
        _GAUSS_STEP_CACHE[cache_key] = step

    # Early stopping identical to case_3's trainer (same knobs, same warm-up
    # exclusion).  With the case-3 profile PATIENCE is effectively infinite.
    best_elbo = -np.inf
    best_params = jax.tree_util.tree_map(lambda x: x, state.params)
    no_improve = 0
    stopped_early = False
    warm_end = int(float(config.st_warmup_frac) * num_iters)

    for it in range(1, num_iters + 1):
        key, kb, ks = jrand.split(key, 3)
        idx = jrand.randint(kb, (config.batch_size,), 0, dataset.dataset_size)
        batch = dataset.train_data[idx]
        state, _, aux = step(state, batch, ks, it, num_iters,
                             alpha_mat, beta_mat, dataset.dataset_size)

        if it % SR.EVAL_EVERY == 0 and it > warm_end:
            cur = float(aux["elbo"])
            if cur > best_elbo:
                best_elbo = cur
                best_params = jax.tree_util.tree_map(lambda x: x, state.params)
                no_improve = 0
            else:
                no_improve += SR.EVAL_EVERY
                if no_improve >= SR.PATIENCE:
                    if verbose:
                        print(f"      [gauss-r] early-stop @ iter {it}")
                    stopped_early = True
                    break
        if verbose and it % config.print_every == 0:
            print(f"      [gauss-r] iter {it:6d}/{num_iters} | ELBO {float(aux['elbo']):.3f}"
                  f" | KL_r {float(aux['kl_r']):.3f}")

    final_params = best_params if stopped_early else state.params
    mu_r = final_params["r_mu"]
    sigma_r = jax.nn.softplus(final_params["r_rho"]) + 1e-6

    def draw_r(key_draw, n_draws: int):
        eps = jrand.normal(key_draw, (n_draws, m))
        return mu_r[None, :] + sigma_r[None, :] * eps

    trained = SimpleNamespace(
        apply_fn=model.apply,
        params=final_params["model"],
        alpha_mat=alpha_mat, beta_mat=beta_mat,
        r_mu=mu_r, r_sigma=sigma_r,
    )
    return trained, draw_r


def sample_posterior_gaussian_r(trained, draw_r, dataset, num_samples: int, seed: int):
    """
    Hard posterior DAG draws for the Gaussian-r variants, through case_3's own
    (bias-aware) ``_sample_posterior``.

    The guide is continuous, so its faithful use is S independent r draws --
    not a K-point cloud.  The draws are handed to the stock sampler as the
    "particles" array; the sampler's internal random particle indexing then
    resamples them with replacement, which leaves their distribution q(r)
    unchanged.
    """
    key = jrand.PRNGKey(seed + 20_011)
    r_draws = draw_r(key, num_samples)
    ns = SimpleNamespace(apply_fn=trained.apply_fn, params=trained.params,
                         particles=r_draws)
    wrapped = SR._TrainedSVIDAG(state=ns, alpha_mat=trained.alpha_mat,
                                beta_mat=trained.beta_mat)
    return SR._sample_posterior(wrapped, dataset, num_samples=num_samples, seed=seed)


# ---------------------------------------------------------------------------
# Joint-posterior metric: log P(G* | posterior)
# ---------------------------------------------------------------------------
def true_dag_logprob(apply_fn, params, r_draws, true_adj_j2i, key,
                     n_draws: int = 1000) -> float:
    """
    Analytic estimate of the variational posterior's log-probability of the
    TRUE DAG G*.

    For a given ordering r and edge logits gamma, SVI-DAG's hard graph is
    A = B(gamma) o M(r) with independent per-slot edges B_ij ~ Bern(sigma(
    gamma_ij)).  Hence

        P(G* | r, gamma) = 1{G* consistent with M(r)}
                           * prod_{slots M=1} Bern(sigma(gamma); G*)

    and P(G*) is the expectation over q(r) q(gamma | r), estimated here by
    log-mean-exp over ``n_draws`` (r, gamma) draws.  Unlike counting exact
    sample hits (which is ~0 whenever the posterior is diffuse), this uses
    the model's own densities, so it is finite and discriminative -- and it
    is a JOINT-structure metric: mean-field q(gamma) pays the independence
    penalty on every slot simultaneously, which marginal metrics never see.

    ``r_draws``: [K, m] SVGD particles or guide samples; drawn from with
    replacement.  Convention: ``true_adj_j2i[i, j]=1 => j -> i`` matches the
    model's own A convention, so no transpose.
    """
    from svidag.utils import (build_mask_M_tau, center_order_potentials,
                              vec_to_offdiag_matrix)

    A_true = jnp.asarray((np.asarray(true_adj_j2i) != 0).astype(np.float32))
    m = A_true.shape[0]
    d = m * (m - 1)
    offdiag = 1.0 - jnp.eye(m)
    tau = float(config.tau_sink_end)

    # RELAXED mask, not the hard one.  Under the hard mask P(G* | r) is zero
    # whenever a single true edge violates the sampled ordering, and measured
    # on trained models 0/20 SVGD particles admitted G* -- the metric
    # degenerates to -inf for every variant (the same brittleness that made
    # exact-sample MEC coverage read 0.00).  Scoring against the soft
    # Sinkhorn mask at the profile's own temperature keeps every draw finite
    # -- orderings that misplace true edges pay log(eps) per edge instead of
    # vetoing the draw -- while remaining a JOINT metric: it rewards mass on
    # (ordering, edge-logit) pairs that jointly place G*'s edges, which is
    # precisely the coupling q(gamma|r) exists to model and a mean-field
    # q(gamma) cannot.  It converges to the hard-mask log P(G*) as tau -> 0.
    k_idx, k_z, k_r = jrand.split(key, 3)
    idx = jrand.randint(k_idx, (n_draws,), 0, r_draws.shape[0])
    selected_r = jnp.asarray(r_draws)[idx]
    keys = jrand.split(k_z, n_draws)

    def one(r_val, kz):
        r_c = center_order_potentials(r_val)
        z = jrand.normal(kz, (d,))
        gamma_flat, _ = apply_fn({"params": params}, z, r_c,
                                 method=lambda mod, z_, c_: mod.flow(z_, cond=c_))
        gamma = vec_to_offdiag_matrix(jnp.clip(gamma_flat, -15.0, 15.0), m)
        M_soft = build_mask_M_tau(r_c, tau)
        p = jnp.clip(jax.nn.sigmoid(gamma) * M_soft * offdiag, 1e-6, 1.0 - 1e-6)
        return jnp.sum((A_true * jnp.log(p) + (1.0 - A_true) * jnp.log1p(-p))
                       * offdiag)

    lps = jax.vmap(one)(selected_r, keys)
    return float(jax.scipy.special.logsumexp(lps) - jnp.log(n_draws))


# ---------------------------------------------------------------------------
# Held-out predictive log-likelihood
# ---------------------------------------------------------------------------
def predictive_loglik(apply_fn, params, r_draws, X_test, alpha_mat, beta_mat,
                      key, n_samples: int = 100) -> float:
    """
    Mean per-row held-out predictive log-density under the posterior:

        (1/N) sum_rows log (1/S) sum_s N(x_row ; f_s(x_row), sigma^2 I)

    where each s draws (r, A, theta) from the posterior and f_s is the
    model's forward prediction under that draw (same forward
    ``eval.posterior_predict`` uses; this keeps the per-draw predictions and
    scores them instead of averaging them away).  sigma is the model's own
    fixed observation scale, so every variant is scored under the likelihood
    it trained with.

    This is a proper scoring rule over WHOLE sampled graphs inside a log, so
    unlike Brier/AUROC/E-SHD it is not a marginal functional: matching edge
    marginals does not match it, which is what makes it flow-sensitive.
    Higher is better.
    """
    from svidag.train import apply_model

    X = jnp.asarray(X_test)
    sigma = float(config.obs_noise_scale)

    k_idx, k_draw = jrand.split(key)
    idx = jrand.randint(k_idx, (n_samples,), 0, r_draws.shape[0])
    selected_r = jnp.asarray(r_draws)[idx]
    keys = jrand.split(k_draw, n_samples)

    def single(r_val, k):
        preds, _ = apply_model(apply_fn, params, X, r_val, k,
                               config.T_B_end, config.tau_sink_end,
                               alpha_mat, beta_mat)
        return preds                                   # [N, m]

    P = jax.vmap(single)(selected_r, keys)             # [S, N, m]
    const = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(sigma)
    row_lp = jnp.sum(const - 0.5 * ((X[None] - P) / sigma) ** 2, axis=-1)  # [S, N]
    per_row = jax.scipy.special.logsumexp(row_lp, axis=0) - jnp.log(n_samples)
    return float(jnp.mean(per_row))


# ---------------------------------------------------------------------------
# One entry point per variant
# ---------------------------------------------------------------------------
VARIANTS = ("full", "no_flow", "no_svgd", "no_prior", "no_flow_no_svgd")


def fit_and_sample(variant: str, X_scaled: np.ndarray, true_adj: np.ndarray,
                   node_names, scenario: str, cell_index: int,
                   num_posterior_samples: int, num_iters: int, seed: int,
                   verbose: bool, X_test: np.ndarray = None):
    """
    Fit one variant on one replicate.  Returns ``(A, extras)`` where ``A`` is
    [S, p, p] binary hard-DAG posterior samples in SVIDAG (j -> i) convention
    and ``extras`` carries scalar metrics that need the trained state
    (currently ``logp_true``: log posterior probability of the true DAG).

    ``scenario`` and the flow type are decided by the caller (run_ablation.py
    sets SVIDAG_FLOW_TYPE / SVIDAG_PRIOR_P0 in the environment first); this
    function only routes to the right trainer.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")

    cell_seed = seed + cell_index * 997
    lp_key = jrand.PRNGKey(cell_seed + 40_009)

    if variant in ("no_svgd", "no_flow_no_svgd"):
        SR._apply_env_overrides(verbose=verbose)
        dataset = SR._build_svidag_dataset(
            X_scaled, true_adj, node_names,
            dataset_name=f"ablation_cell{cell_index}_{variant}",
        )
        trained, draw_r = train_gaussian_r(
            scenario=scenario, dataset=dataset,
            seed=cell_seed, num_iters=num_iters, verbose=verbose)
        A = sample_posterior_gaussian_r(
            trained, draw_r, dataset,
            num_samples=num_posterior_samples, seed=cell_seed)
        lp_key, k_r, k_p = jrand.split(lp_key, 3)
        r_eval = draw_r(k_r, 1000)
        logp = true_dag_logprob(trained.apply_fn, trained.params,
                                r_eval, true_adj, lp_key)
        extras = {"logp_true": logp}
        if X_test is not None:
            extras["pred_ll"] = predictive_loglik(
                trained.apply_fn, trained.params, r_eval, X_test,
                trained.alpha_mat, trained.beta_mat, k_p)
        return np.asarray(A), extras

    # full / no_flow / no_prior: the stock case_3 pipeline, opened up so the
    # trained state is available for the joint metric (run_svidag_synthetic
    # returns samples only).
    SR._apply_env_overrides(verbose=verbose)
    dataset = SR._build_svidag_dataset(
        X_scaled, true_adj, node_names,
        dataset_name=f"ablation_cell{cell_index}_{variant}",
    )
    trained = SR._train_svidag(scenario=scenario, dataset=dataset,
                               seed=cell_seed, num_iters=num_iters,
                               verbose=verbose)
    A = SR._sample_posterior(trained, dataset,
                             num_samples=num_posterior_samples, seed=cell_seed)
    lp_key, k_p = jrand.split(lp_key)
    logp = true_dag_logprob(trained.state.apply_fn, trained.state.params,
                            trained.state.particles, true_adj, lp_key)
    extras = {"logp_true": logp}
    if X_test is not None:
        extras["pred_ll"] = predictive_loglik(
            trained.state.apply_fn, trained.state.params,
            trained.state.particles, X_test,
            trained.alpha_mat, trained.beta_mat, k_p)
    return np.asarray(A), extras
