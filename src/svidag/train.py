'''
Training module for the SVIDAG.

Author: Shrenik Zinage
'''


from functools import lru_cache, partial

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
from flax.training import train_state
import optax

from . import config
from .data import generate_synthetic_dataset
from .model import SVIDAGModel
from .utils import (
    compute_alpha_beta_from_prior,
    linear_anneal,
    std_normal_logpdf,
    compute_svgd_update,
    gaussian_logpdf,
    to_device,
    build_mask_M_tau,
)


class TrainState(train_state.TrainState):
    """Extended Flax TrainState that includes SVGD particles for order potentials."""
    particles: jnp.ndarray  # [K, m] particles representing samples of r


@lru_cache(maxsize=None)
def _make_optimizer_cached(lr, clip):
    return optax.chain(optax.clip_by_global_norm(clip), optax.adam(lr))


def make_optimizer(lr, clip):
    """
    Creates Adam optimizer with gradient clipping.
    Clipping prevents gradient explosions common in flow-based models.

    Memoised on (lr, clip). ``tx`` is static pytree metadata on TrainState, so
    a fresh optax object per fit gives every cell a new static signature and
    forces a full retrace of ``train_step`` -- ~5 min of XLA compile per cell
    on the case-2 grid. optax transformations are stateless (the state lives
    in the TrainState), so sharing one instance across fits is safe and makes
    the compile a once-per-process cost.
    """
    return _make_optimizer_cached(float(lr), float(clip))


def _make_model_cached(*, num_nodes, flow_hidden, flow_n_blocks, flow_type,
                       nsf_num_bins, nsf_tail_bound, noise_scales_key,
                       hidden_dim, learn_likelihood_noise, _noise_scales):
    """
    Return a SVIDAGModel, reusing the instance for an identical configuration.

    ``TrainState.apply_fn`` is ``model.apply``, and Flax stores ``apply_fn`` as
    *static* pytree metadata. A Flax Module holding an array attribute
    (``fixed_noise_scales``) is unhashable, so two structurally identical
    models compare unequal and their bound ``.apply`` methods are distinct
    objects. Building a fresh model per fit therefore gives ``train_step`` a
    new static signature every time and forces a full XLA retrace -- measured
    at ~540 s per cell on the case-2 grid (cell 1: 675.2 s, cell 2: 675.9 s,
    with a jit cache growing 1 -> 2 -> 3).

    Caching on the full construction signature (including the config values
    read inside ``setup``) keeps ``apply_fn`` identical across fits, so the
    compile is paid once per batch shape instead of once per cell. Parameters
    are still initialised freshly per fit from the caller's PRNG key, so
    results are unchanged.
    """
    key = (num_nodes, flow_hidden, flow_n_blocks, flow_type, nsf_num_bins,
           nsf_tail_bound, noise_scales_key, hidden_dim, learn_likelihood_noise,
           bool(config.node_cond_row_only))
    cached = _MODEL_CACHE.get(key)
    if cached is None:
        cached = SVIDAGModel(
            num_nodes=num_nodes,
            flow_hidden=list(flow_hidden),
            flow_n_blocks=flow_n_blocks,
            fixed_noise_scales=_noise_scales,
            flow_type=flow_type,
            nsf_num_bins=nsf_num_bins,
            nsf_tail_bound=nsf_tail_bound,
        )
        _MODEL_CACHE[key] = cached
    return cached


_MODEL_CACHE = {}


def make_model_and_state(rng, train_data, p_prior, num_nodes: int, fixed_noise_scales=None):
    """
    Initializes the SVIDAG model and training state.

    Creates:
    - SVIDAGModel with flow and Bayesian node models
    - SVGD particles sampled from N(0, I) for order potentials
    - TrainState with optimizer and particle storage

    Args:
        rng: JAX PRNG key for initialization
        train_data: Training data for shape inference [N, m]
        p_prior: Prior edge probability matrix [m, m]
        num_nodes: Number of nodes m in the graph
        fixed_noise_scales: Per-node observation noise (defaults to config value)

    Returns:
        model: Initialized SVIDAGModel
        state: TrainState with params, optimizer state, and particles
    """
    # Initialize SVGD particles from prior: r ~ N(0, σ_r² I)
    rng, r_key = jrand.split(rng)
    init_particles = config.prior_r_sigma * jrand.normal(r_key, (config.n_particles, num_nodes))
    
    if fixed_noise_scales is None:
        fixed_noise_scales = jnp.ones((num_nodes,)) * config.obs_noise_scale
        fixed_noise_scales = to_device(fixed_noise_scales)

    # Flow conditioner widths.  ``config.flow_hidden`` wins when set to an
    # explicit list; otherwise fall back to the fixed ``[5, 5]``.  The flow
    # maps z, gamma in R^{m(m-1)} (600 dims at m=25) conditioned on r in R^m,
    # so a width-5 conditioner is a hard capacity ceiling on exactly the
    # edge-dependency structure the flow exists to model.
    if isinstance(config.flow_hidden, (list, tuple)) and len(config.flow_hidden) > 0:
        flow_hidden = [int(w) for w in config.flow_hidden]
    else:
        flow_hidden = [5, 5]

    model = _make_model_cached(
        num_nodes=num_nodes,
        flow_hidden=tuple(flow_hidden),
        flow_n_blocks=int(config.flow_n_blocks),
        flow_type=str(config.flow_type),
        nsf_num_bins=int(config.nsf_num_bins),
        nsf_tail_bound=float(config.nsf_tail_bound),
        noise_scales_key=tuple(np.asarray(fixed_noise_scales).ravel().tolist()),
        hidden_dim=int(config.hidden_dim),
        learn_likelihood_noise=bool(config.learn_likelihood_noise),
        _noise_scales=fixed_noise_scales,
    )

    # Flax requires dummy forward pass for parameter initialization
    dummy_x = train_data[: config.batch_size]
    dummy_r = init_particles[0]
    
    variables = model.init(
        {"params": rng},
        dummy_x,
        dummy_r,
        rng,
        config.T_B_start,
        config.tau_sink_start,
        *compute_alpha_beta_from_prior(p_prior),
    )
    params = variables["params"]
    tx = make_optimizer(config.lr, config.grad_clip)
    
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx, particles=init_particles)
    return model, state


@partial(jax.jit, static_argnums=(0,))
def apply_model(apply_fn, params, batch, r, rng, T_B, tau_sn, alpha_mat, beta_mat):
    """
    JIT-compiled model forward pass for a fixed order potential r.
    Static arg (apply_fn) is traced once and reused.
    """
    return apply_fn({"params": params}, batch, r, rng, T_B, tau_sn, alpha_mat, beta_mat)


def _bound_particle_grad(g):
    """
    Bound a [K, m] particle gradient / update field.

    ``config.particle_grad_clip_mode``:
        "elementwise" -- per-coordinate clip to +/- ``config.particle_grad_clip``
                         (the original behaviour).
        "norm"        -- per-particle L2 rescale to at most
                         ``config.particle_grad_clip``, preserving direction.
    """
    c = float(config.particle_grad_clip)
    if config.particle_grad_clip_mode == "norm":
        nrm = jnp.linalg.norm(g, axis=-1, keepdims=True)
        return g * jnp.minimum(1.0, c / (nrm + 1e-12))
    return jnp.clip(g, -c, c)


def _train_step_impl(state, batch, rng, it, total_iters, alpha_mat, beta_mat, N_total, mc_samples):
    """
    Single training step: updates both model parameters and SVGD particles.
    
    The optimization has two interleaved components:
    1. SVGD update for particles r: moves particles toward high-ELBO regions
    2. Gradient descent on params: maximizes ELBO averaged over particles
    
    ELBO = E_q[log p(x|A,θ)] - KL(q(θ)||p(θ)) - KL(q(γ|r)||p(γ))
    
    Args:
        state: TrainState with params and particles
        batch: Mini-batch of observations [batch_size, m]
        rng: JAX PRNG key
        it: Current iteration (for temperature annealing)
        total_iters: Total iterations (for annealing schedule)
        alpha_mat, beta_mat: Beta prior parameters for edge probabilities
        N_total: Total dataset size (for likelihood scaling)
        mc_samples: Number of Monte Carlo samples per particle
    
    Returns:
        state: Updated TrainState
        loss_val: Negative ELBO (loss being minimized)
        aux: Dict of metrics for logging
    """
    # Anneal temperatures from start to end values over training
    T_B = linear_anneal(it, total_iters, config.T_B_start, config.T_B_end)
    # linear_anneal clips t to [0, 1], so shrinking the window makes tau reach
    # tau_sink_end early and hold there (config.tau_anneal_frac = 1.0 is the
    # original full-length schedule).
    tau_sn = linear_anneal(it,
                           jnp.maximum(1.0, float(config.tau_anneal_frac) * total_iters),
                           config.tau_sink_start, config.tau_sink_end)
    # Straight-through warm-up: the forward value moves from the relaxed
    # surrogate to the hard DAG over the first ``st_warmup_frac`` of training.
    # Gradients are unaffected (see model.forward_sample).
    warm = float(config.st_warmup_frac)
    if warm > 0.0:
        st_weight = jnp.clip(it / jnp.maximum(warm * total_iters, 1.0), 0.0, 1.0)
    else:
        st_weight = 1.0
    batch_size = batch.shape[0]
    # Scale likelihood by N/batch_size to get unbiased gradient estimate
    # ELBO scale.
    #
    # The sum-form objective (ell = N/|B| * sum_batch loglik, KL unscaled) has
    # a gradient whose magnitude grows linearly with N. Combined with the two
    # fixed-threshold clips in this function -- optax.clip_by_global_norm on
    # the parameter gradient and the +/-10 elementwise clip on the particle
    # score below -- that makes BOTH updates n-independent: at n=1000 the raw
    # gradient is ~100x the n=10 one, every component saturates its clip, and
    # the surviving update is just a direction. That is why the case-2 metrics
    # were flat in n.
    #
    # The per-datum form divides through by N:
    #     ELBO/N = mean_batch loglik - (KL_theta + KL_gamma)/N
    # which (a) keeps the gradient O(1) at every n so the clips no longer
    # erase the scale, and (b) restores the correct Bayesian n-dependence --
    # the prior's weight per observation decays as 1/N, so more data means a
    # sharper posterior over structures instead of an identical one.
    if config.elbo_per_datum:
        ell_scale = 1.0 / batch_size
        kl_scale = 1.0 / N_total
    else:
        ell_scale = N_total / batch_size
        kl_scale = 1.0
    kl_weight = kl_scale  # Could be annealed for KL warmup

    particles = state.particles
    rng, rng_score, rng_elbo = jrand.split(rng, 3)

    # ============ SVGD Update for Order Potential Particles ============
    # Compute ∇_r log p(r | data) ≈ ∇_r [ELBO(r) + log p(r)]
    
    keys_particles = jrand.split(rng_score, config.n_particles)

    def particle_target_sum(particles_val):
        """Sum particle objectives so one reverse pass yields all score gradients."""

        def particle_objective(r_val, key):
            keys_mc = jrand.split(key, mc_samples)

            def single_mc_eval(k):
                _, terms = state.apply_fn(
                    {'params': state.params}, batch, r_val, k, T_B, tau_sn, alpha_mat,
                    beta_mat, st_weight
                )
                log_q_gamma = std_normal_logpdf(terms["z"]) - terms["log_det"]
                kl_gamma_term = terms["log_p_gamma"] - log_q_gamma
                ell = ell_scale * terms["loglik"]
                return ell - kl_weight * config.kl_theta_weight * terms["kl_theta"] + kl_weight * kl_gamma_term

            avg_obj = jnp.mean(jax.vmap(single_mc_eval)(keys_mc))
            log_prior = jnp.sum(gaussian_logpdf(r_val, 0.0, config.prior_r_sigma))
            return avg_obj + log_prior

        return jnp.sum(jax.vmap(particle_objective)(particles_val, keys_particles))

    grads_logp = jax.grad(particle_target_sum)(particles)

    # Sanitize gradients: replace NaN and bound the magnitude for stability.
    #
    # The elementwise clip below is the original behaviour, but the sum-form
    # ELBO multiplies the likelihood by N/|B| (105x on the Sachs kfold split),
    # so every coordinate lands far outside +/-10 and the clip returns a pure
    # sign vector: the *relative* magnitudes across coordinates -- precisely
    # the signal that says which node should move where in the ordering -- are
    # discarded. Norm clipping bounds the step while preserving direction.
    grads_logp = jnp.nan_to_num(grads_logp)
    grads_logp = _bound_particle_grad(grads_logp)

    # Repel particles in the induced DAG-mask geometry rather than raw r-space.
    kernel_feature_fn = lambda r_val: build_mask_M_tau(r_val, tau_sn).reshape(-1)
    # Anneal the repulsion away so the cloud may reach consensus once the
    # warm-up has made the ordering landscape informative (see config).
    ra = config.svgd_repulsion_anneal_frac
    if ra is None:
        repel_scale = 1.0
    else:
        repel_scale = jnp.clip(1.0 - it / jnp.maximum(float(ra) * total_iters, 1.0), 0.0, 1.0)
    phi_svgd = compute_svgd_update(particles, grads_logp,
                                   feature_fn=kernel_feature_fn,
                                   repel_scale=repel_scale)

    phi_svgd = jnp.nan_to_num(phi_svgd)
    phi_svgd = _bound_particle_grad(phi_svgd)
    
    # Gradient ascent step for particles.  Held at zero for the first
    # ``eta_r_warmup_frac`` of training: the ordering score is meaningless while
    # the structural equations are untrained (see config).
    ew = float(config.eta_r_warmup_frac)
    if ew > 0.0:
        eta_gate = jnp.where(it >= ew * total_iters, 1.0, 0.0)
    else:
        eta_gate = 1.0
    new_particles = particles + eta_gate * config.eta_r * phi_svgd

    # Clamp to prevent particles from drifting too far
    new_particles = jnp.clip(new_particles, -5.0, 5.0)

    # ============ Parameter Update via ELBO Maximization ============
    
    def loss_fn(params):
        """Negative ELBO averaged over particles and MC samples."""
        keys_elbo = jrand.split(rng_elbo, config.n_particles)

        def particle_elbo(r_val, key_p):
            """ELBO for a single particle r, averaged over MC samples."""
            keys_mc = jrand.split(key_p, mc_samples)
            
            def single_mc(k):
                _, terms = state.apply_fn(
                    {'params': params}, batch, r_val, k, T_B, tau_sn, alpha_mat,
                    beta_mat, st_weight
                )
                # Flow KL: log q(γ|r) - log p(γ)
                log_q_gamma = std_normal_logpdf(terms["z"]) - terms["log_det"]
                kl_gamma = log_q_gamma - terms["log_p_gamma"]
                ell = ell_scale * terms["loglik"]
                
                # Full ELBO = E[log p(x|A,θ)] - KL(θ) - KL(γ)
                elbo_term = ell - kl_weight * (config.kl_theta_weight * terms["kl_theta"] + kl_gamma)
                
                return elbo_term, (
                    terms["ll_mse"],
                    terms["ll_const"],
                    terms["kl_theta"],
                    terms["kl_noise"],
                    kl_gamma,
                    terms["A_relaxed"],
                    terms["noise_scales"],
                )

            elbo_vals, auxs = jax.vmap(single_mc)(keys_mc)
            return jnp.mean(elbo_vals), jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), auxs)

        # Average ELBO over all particles
        elbos, aux_data = jax.vmap(particle_elbo)(particles, keys_elbo)
        total_elbo = jnp.mean(elbos)
        
        loss = -total_elbo  # Minimize negative ELBO
        
        # Aggregate auxiliary metrics for logging
        mean_ell = jnp.mean(aux_data[0])
        mean_ll_const = jnp.mean(aux_data[1])
        mean_kl_theta = jnp.mean(aux_data[2])
        mean_kl_noise = jnp.mean(aux_data[3])
        mean_kl_gamma = jnp.mean(aux_data[4])
        mean_A_relaxed = jnp.mean(aux_data[5], axis=0)
        mean_noise_scales = jnp.mean(aux_data[6], axis=0)

        aux = {
            "elbo": total_elbo,
            "ell": mean_ell,
            "ll_const": mean_ll_const,
            "kl_theta": mean_kl_theta,
            "kl_noise": mean_kl_noise,
            "kl_gamma": mean_kl_gamma,
            "A_relaxed": mean_A_relaxed,
            "noise_scales": mean_noise_scales,
            "T_B": T_B,
            "tau_sn": tau_sn
        }
        return loss, aux

    # Compute loss and gradients w.r.t. params
    (loss_val, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    # Apply parameter gradients via optimizer
    state = state.apply_gradients(grads=grads)
    # Update particles (SVGD step computed earlier)
    state = state.replace(particles=new_particles)
    
    return state, loss_val, aux


@partial(jax.jit, static_argnums=8)
def train_step(state, batch, rng, it, total_iters, alpha_mat, beta_mat, N_total, mc_samples):
    """Public training step without buffer donation for general callers and tests."""
    return _train_step_impl(state, batch, rng, it, total_iters, alpha_mat, beta_mat, N_total, mc_samples)


@partial(jax.jit, static_argnums=8, donate_argnums=(0,))
def train_step_donated(state, batch, rng, it, total_iters, alpha_mat, beta_mat, N_total, mc_samples):
    """Training step with state donation for throughput-oriented loops."""
    return _train_step_impl(state, batch, rng, it, total_iters, alpha_mat, beta_mat, N_total, mc_samples)


@partial(jax.jit, static_argnums=(0, 7))
def _particle_objectives(apply_fn, params, particles, batch, rng, alpha_mat,
                         beta_mat, mc_samples, ell_scale, T_B, tau_sn, st_weight):
    """
    Per-particle value of the objective SVGD ascends: ELBO(r) + log p(r).

    Same expression as ``particle_target_sum`` in ``_train_step_impl``, but it
    returns the [K] vector instead of the sum, so the particles can be ranked
    against one another.
    """
    keys = jrand.split(rng, particles.shape[0])

    def one(r_val, key):
        keys_mc = jrand.split(key, mc_samples)

        def single(k):
            _, terms = apply_fn({"params": params}, batch, r_val, k, T_B, tau_sn,
                                alpha_mat, beta_mat, st_weight)
            log_q_gamma = std_normal_logpdf(terms["z"]) - terms["log_det"]
            return (ell_scale * terms["loglik"]
                    - config.kl_theta_weight * terms["kl_theta"]
                    + (terms["log_p_gamma"] - log_q_gamma))

        return (jnp.mean(jax.vmap(single)(keys_mc))
                + jnp.sum(gaussian_logpdf(r_val, 0.0, config.prior_r_sigma)))

    return jax.vmap(one)(particles, keys)


def resample_particles(state, batch, rng, alpha_mat, beta_mat, N_total,
                       mc_samples, T_B, tau_sn):
    """
    Replace the particle cloud by an ELBO-weighted resample of itself.

    Scores every particle with ``_particle_objectives``, forms softmax weights
    at ``config.particle_resample_temp`` times the objective's own spread across
    the cloud (so the temperature is dataset-independent), draws K indices with
    replacement, and adds ``config.particle_resample_jitter * prior_r_sigma`` of
    Gaussian noise so duplicated particles separate again.

    This is a selection move, not a proposal: every surviving r was already in
    the cloud, so the flow's conditioning input never leaves the support it was
    trained on. Returns ``(state, ess)`` where ess is the effective sample size,
    useful for logging how much the cloud actually collapsed.
    """
    k_score, k_pick, k_jit = jrand.split(rng, 3)
    ell_scale = 1.0 / batch.shape[0] if config.elbo_per_datum else N_total / batch.shape[0]
    obj = _particle_objectives(state.apply_fn, state.params, state.particles, batch,
                               k_score, alpha_mat, beta_mat, int(mc_samples),
                               ell_scale, T_B, tau_sn, 1.0)
    obj = jnp.nan_to_num(obj, nan=-jnp.inf)
    scale = jnp.std(obj) * float(config.particle_resample_temp) + 1e-8
    logw = (obj - jnp.max(obj)) / scale
    w = jax.nn.softmax(logw)
    K = state.particles.shape[0]
    idx = jrand.choice(k_pick, K, shape=(K,), p=w)
    jitter = float(config.particle_resample_jitter) * float(config.prior_r_sigma)
    new_particles = state.particles[idx] + jitter * jrand.normal(k_jit, state.particles.shape)
    ess = 1.0 / jnp.sum(w ** 2)
    return state.replace(particles=new_particles), ess


def maybe_resample(state, batch, rng, it, total_iters, alpha_mat, beta_mat,
                   N_total, mc_samples, T_B, tau_sn):
    """
    Apply ``resample_particles`` on the schedule set by
    ``config.particle_resample_every`` (None disables it), skipping the
    relaxation warm-up -- during the warm-up the mask is soft, so the objective
    barely depends on the ordering and the weights would be noise.

    Returns ``(state, ess_or_None)``.
    """
    every = config.particle_resample_every
    if not every:
        return state, None
    warm_end = int(float(config.st_warmup_frac) * total_iters)
    if it <= warm_end or it % int(every) != 0:
        return state, None
    return resample_particles(state, batch, rng, alpha_mat, beta_mat, N_total,
                              mc_samples, T_B, tau_sn)


def warmstart_particles(state, rng, T_B, num_samples: int = 256):
    """
    Re-seed the SVGD particles from the edge structure the flow has learned.

    Draws ``num_samples`` unmasked edge matrices B (``model.sample_B_hard``),
    averages them into marginals ``P[i, j] = P(j -> i)``, and scores each node
    by out-mass minus in-mass::

        r_k  ~  sum_i P[i, k]  -  sum_j P[k, j]

    A node the flow wants to point *out* of is a source and gets a high
    potential, so the descending sort of r is a topological order for the
    learned edge set. Particles are placed around that ranking with
    ``config.particle_warmstart_jitter`` of Gaussian noise, scaled to keep
    ``|r|`` comparable to the prior, so SVGD still starts with a spread of
    orderings rather than a single point.

    Returns the state with new particles; the model parameters are untouched.
    """
    K, m = state.particles.shape
    k_draw, k_noise = jrand.split(rng)
    idx = jrand.randint(k_draw, (num_samples,), 0, K)
    keys = jrand.split(k_draw, num_samples)

    def one(r_val, key):
        return state.apply_fn({"params": state.params}, r_val, key, T_B,
                              method=lambda mod, *a: mod.sample_B_hard(*a))

    B = jax.vmap(one)(state.particles[idx], keys)          # [S, m, m]
    P = jnp.mean(B, axis=0)
    score = jnp.sum(P, axis=0) - jnp.sum(P, axis=1)        # out-mass - in-mass
    score = score - jnp.mean(score)
    scale = float(config.prior_r_sigma) / (jnp.std(score) + 1e-6)
    base = score * scale
    noise = float(config.particle_warmstart_jitter) * float(config.prior_r_sigma)
    new_particles = base[None, :] + noise * jrand.normal(k_noise, (K, m))
    return state.replace(particles=new_particles)


def maybe_warmstart(state, rng, it: int, total_iters: int, T_B) -> tuple:
    """
    Apply ``warmstart_particles`` on the single iteration selected by
    ``config.particle_warmstart_frac`` (None disables it entirely).

    Returns ``(state, fired)`` so callers can log it.
    """
    frac = config.particle_warmstart_frac
    if frac is None:
        return state, False
    target = max(1, int(float(frac) * total_iters))
    if it != target:
        return state, False
    return warmstart_particles(state, rng, T_B), True


def build_benchmark_inputs():
    """Build stable benchmark inputs for the generated train_step harness."""
    from .data import two_node_generator_default
    dataset = generate_synthetic_dataset(generator_fn=two_node_generator_default, normalize=True)
    p_prior = to_device(jnp.full((dataset.num_nodes, dataset.num_nodes), 0.5, dtype=jnp.float32) * (1.0 - jnp.eye(dataset.num_nodes, dtype=jnp.float32)))
    alpha_mat, beta_mat = compute_alpha_beta_from_prior(p_prior)
    rng = jrand.PRNGKey(config.seed)
    rng, init_key, batch_key, step_key = jrand.split(rng, 4)
    _, state = make_model_and_state(init_key, dataset.train_data, p_prior, dataset.num_nodes, fixed_noise_scales=dataset.noise_scales)
    idx = jrand.randint(batch_key, (config.batch_size,), 0, dataset.dataset_size)
    batch = dataset.train_data[idx]
    return {
        "args": [
            state,
            batch,
            step_key,
            1,
            max(2, config.num_iters),
            alpha_mat,
            beta_mat,
            dataset.dataset_size,
            min(2, config.ELBO_MC_SAMPLES),
        ],
        "metadata": {
            "dataset_name": dataset.dataset_name,
            "num_nodes": dataset.num_nodes,
            "batch_size": int(config.batch_size),
            "n_particles": int(config.n_particles),
        },
    }
