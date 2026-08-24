import math
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import jax
import jax.nn as jnn
import jax.numpy as jnp
import networkx as nx
import numpy as np


def _next_key() -> jax.Array:
    seed = int(np.random.randint(0, np.iinfo(np.int32).max))
    return jax.random.PRNGKey(seed)


@jax.tree_util.register_pytree_node_class
@dataclass
class AdamState:
    step: jax.Array
    m: object
    v: object

    def tree_flatten(self):
        return (self.step, self.m, self.v), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        step, m, v = children
        return cls(step=step, m=m, v=v)


def _adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return AdamState(step=jnp.asarray(0, dtype=jnp.int32), m=zeros, v=zeros)


def _adam_update(params, grads, state: AdamState, lr: float):
    step = state.step + 1
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    m = jax.tree_util.tree_map(lambda m_i, g_i: beta1 * m_i + (1.0 - beta1) * g_i, state.m, grads)
    v = jax.tree_util.tree_map(lambda v_i, g_i: beta2 * v_i + (1.0 - beta2) * (g_i * g_i), state.v, grads)

    bc1 = 1.0 - beta1 ** step.astype(jnp.float32)
    bc2 = 1.0 - beta2 ** step.astype(jnp.float32)
    m_hat = jax.tree_util.tree_map(lambda x: x / bc1, m)
    v_hat = jax.tree_util.tree_map(lambda x: x / bc2, v)
    params = jax.tree_util.tree_map(lambda p, m_i, v_i: p - lr * m_i / (jnp.sqrt(v_i) + eps), params, m_hat, v_hat)
    return params, AdamState(step=step, m=m, v=v)


def early_stopping(score: float, best_score: float, stagnant_epochs: int, patience: int, min_dist: float = 0.0):
    delta = best_score - score
    improved = delta > min_dist
    if improved:
        return score, 0, False
    stagnant_epochs += 1
    return best_score, stagnant_epochs, stagnant_epochs >= patience


def xw_mult(w, *, x):
    return jnp.einsum("np,pqs->nqs", x, w)


def _batch_inverse(mats):
    return jax.vmap(jnp.linalg.inv)(mats)


# ---------------------------------------------------------------------------
# Column-major (Julia) reshapes between a flat p*p axis and a (p, p) block.
#
# ProDAG.jl uses `reshape(v, p, p, n)`, and Julia reshapes in COLUMN-major
# order: element (a, b, k) reads flat index `a + (b-1)*p`.  NumPy/JAX reshape
# in ROW-major order, which reads `a*p + b` instead -- i.e. the transpose.
#
# This matters because the flat axis carries the first-hidden-layer weight
# GROUPS of the MLP, ordered subnetwork-outer / input-inner.  Under Julia's
# ordering the block comes out as w[parent, child]; under a plain NumPy
# reshape it comes out as w[child, parent], so every recovered edge would be
# reversed relative to the reference implementation.
# ---------------------------------------------------------------------------
def _reshape_pp(flat, p, n_sample):
    """Julia's `reshape(flat, p, p, n_sample)` for `flat` of shape (p*p, n)."""
    return jnp.swapaxes(jnp.reshape(flat, (p, p, n_sample)), 0, 1)


def _flatten_pp(block, p, n_sample):
    """Julia's `reshape(block, p*p, n_sample)`; inverse of `_reshape_pp`."""
    return jnp.reshape(jnp.swapaxes(block, 0, 1), (p * p, n_sample))


def _reshape_pp_vec(vec, p):
    """Julia's `reshape(vec, p, p)` for a flat length-p*p vector."""
    return jnp.reshape(vec, (p, p)).T


def project_dag(w_tilde, params):
    s, mu_path, c, tol, max_step, max_iter, threshold, lr = params
    max_ = jnp.max(jnp.abs(w_tilde), axis=(0, 1), keepdims=True)
    max_ = jnp.where(max_ > 0.0, max_, 1.0)

    w = jnp.zeros_like(w_tilde)
    identity = jnp.broadcast_to(jnp.eye(w_tilde.shape[0], dtype=w_tilde.dtype)[..., None], w_tilde.shape)
    mu_path = jnp.asarray(mu_path, dtype=w_tilde.dtype)

    def outer_body(_step, carry):
        w_inner, mu_inner = carry

        def cond_fun(state):
            iter_idx, _w_state, grad_max = state
            return jnp.logical_and(iter_idx < int(max_iter), grad_max > tol)

        def body_fun(state):
            iter_idx, w_state, _grad_max = state
            # ProDAG.jl:
            #     sIw2 .= permutedims(s * I - w .^ 2, (2, 1, 3))
            #     matinv_batched!(sIw2_batch, sIw2_inv_batch)
            #     grad = 2 * sIw2_inv .* w + mu * (w - w̃ ./ max_)
            # so the elementwise factor is ((sI - w^2)^T)^-1, which is exactly
            # d/dW log det(sI - W.W) = -2 ((sI - W.W)^-1)^T . W.
            #
            # `(2, 1, 0)` both moves the batch axis AND transposes the matrix,
            # so it is right on the way IN (it produces the transposed slices
            # Julia inverts) but wrong on the way OUT -- applying it twice
            # cancels the transpose and leaves the plain inverse.  `(1, 2, 0)`
            # moves the batch axis back without touching the matrix.
            mats = jnp.transpose(s * identity - w_state**2, (2, 1, 0))
            inv = _batch_inverse(mats)
            inv = jnp.transpose(inv, (1, 2, 0))
            grad = 2.0 * inv * w_state + mu_inner * (w_state - w_tilde / max_)
            w_state = w_state - lr * grad
            grad_max = jnp.max(jnp.abs(grad))
            return iter_idx + 1, w_state, grad_max

        init_state = (jnp.asarray(0, dtype=jnp.int32), w_inner, jnp.asarray(jnp.inf, dtype=w_tilde.dtype))
        _iter_idx, w_inner, _grad_max = jax.lax.while_loop(cond_fun, body_fun, init_state)
        return w_inner, mu_inner * c

    w, _mu_path = jax.lax.fori_loop(0, int(max_step), outer_body, (w, mu_path))

    w = w * max_
    mask = jnp.abs(w) > threshold
    return jnp.where(mask, w_tilde, 0.0)


def project_l1(w, lam):
    dims = w.shape
    lam = jnp.asarray(lam, dtype=w.dtype).reshape(-1)
    current_norms = jnp.sum(jnp.abs(w), axis=(0, 1))
    all_in_ball = jnp.all(current_norms <= lam)
    all_zero = jnp.all(lam == 0)

    def do_project(_):
        flat = jnp.reshape(w, (dims[0] * dims[1], dims[2]))
        w_abs = jnp.abs(flat)
        w_sort = jnp.sort(w_abs, axis=0)[::-1, :]
        csum = jnp.cumsum(w_sort, axis=0)
        indices = jnp.arange(1, dims[0] * dims[1] + 1, dtype=w.dtype)[:, None]
        indicator = (w_sort * indices) > (csum - lam[None, :])
        max_j = jnp.max(indicator * indices, axis=0).astype(jnp.int32)
        # j = 1 always satisfies the condition when lam > 0 (it reduces to
        # lam > 0), and lam == 0 is handled by the early exit above, so max_j
        # >= 1 in every reachable case.  Clamp anyway: max_j = 0 would index
        # csum[-1] -- the LAST row -- silently, rather than erroring the way
        # Julia's 1-based indexing would.
        max_j = jnp.maximum(max_j, 1)
        theta = jnp.maximum((csum[max_j - 1, jnp.arange(dims[2])] - lam) / max_j, 0.0)
        projected = jnp.maximum(w_abs - theta[None, :], 0.0) * jnp.sign(flat)
        return jnp.reshape(projected, dims)

    def zero_case(_):
        return jnp.zeros_like(w)

    def not_in_ball_case(_):
        return jax.lax.cond(all_zero, zero_case, do_project, operand=None)

    return jax.lax.cond(all_in_ball, lambda _: w, not_in_ball_case, operand=None)


def _make_project(params):
    def _project_impl(w_tilde, lam):
        w = project_dag(w_tilde, params)
        return project_l1(w, lam)

    @jax.custom_vjp
    def _project(w_tilde, lam):
        return _project_impl(w_tilde, lam)

    def fwd(w_tilde, lam):
        w = _project_impl(w_tilde, lam)
        return w, (w, lam)

    def bwd(res, dw):
        w, lam = res
        zero_lambda = jnp.all(lam == 0)

        def zero_case(_):
            return jnp.zeros_like(w), jnp.zeros_like(lam)

        def nonzero_case(_):
            support = w != 0
            support_count = jnp.sum(support, axis=(0, 1))
            dlam = jnp.sum(jnp.sign(w) * dw, axis=(0, 1)) / support_count
            dw_tilde = support * dw - dlam[None, None, :] * jnp.sign(w)

            non_binding = ~jnp.isclose(jnp.sum(jnp.abs(w), axis=(0, 1)), lam)
            dw_tilde = jnp.where(non_binding[None, None, :], support * dw, dw_tilde)
            dlam = jnp.where(non_binding, 0.0, dlam)
            return dw_tilde, dlam

        return jax.lax.cond(zero_lambda, zero_case, nonzero_case, operand=None)

    _project.defvjp(fwd, bwd)
    return _project


def kl_norm(mu_q, sigma_q, mu_p, sigma_p):
    return jnp.log(sigma_p / sigma_q) + (sigma_q**2 + (mu_q - mu_p) ** 2) / (2.0 * sigma_p**2) - 0.5


def kl_exp(alpha_q, alpha_p):
    return jnp.log(alpha_p / alpha_q) + alpha_q / alpha_p - 1.0


def _sample_lambda(alpha, dirac_lambda: bool, n_sample: int, key):
    if dirac_lambda:
        return alpha * jnp.ones((n_sample,), dtype=jnp.float32)
    u = jax.random.uniform(key, (n_sample,), minval=1e-20, maxval=1.0, dtype=jnp.float32)
    return -alpha * jnp.log(u)


def _train_linear(
    x,
    mu,
    cont_sigma,
    cont_alpha,
    prior_mu,
    prior_sigma,
    prior_alpha,
    dirac_lambda,
    noise_var,
    epoch_max,
    patience,
    optimiser_args,
    params,
    n_sample,
    verbose,
):
    lr = float(optimiser_args[0]) if optimiser_args else 0.1
    state = _adam_init((mu, cont_sigma, cont_alpha))
    best_score = float("inf")
    stagnant_epochs = 0
    project = _make_project(params)
    p = x.shape[1]
    n = x.shape[0]
    key = _next_key()

    def objective(mu_param, cont_sigma_param, cont_alpha_param, subkey):
        sigma = jnn.softplus(cont_sigma_param)
        kl = jnp.sum(kl_norm(mu_param, sigma, prior_mu, prior_sigma))

        key_lambda, key_eps = jax.random.split(subkey)
        if dirac_lambda:
            lam = prior_alpha * jnp.ones((n_sample,), dtype=jnp.float32)
        else:
            alpha = jnn.softplus(cont_alpha_param)
            lam = _sample_lambda(alpha[0], False, n_sample, key_lambda)
            kl = kl + jnp.sum(kl_exp(alpha, prior_alpha))

        mu_mat = _reshape_pp_vec(mu_param, p)
        sigma_mat = _reshape_pp_vec(sigma, p)
        eps = jax.random.normal(key_eps, (p, p, n_sample), dtype=jnp.float32)
        w_tilde = mu_mat[:, :, None] + sigma_mat[:, :, None] * eps
        w = project(w_tilde, lam)
        x_hat = xw_mult(w, x=x)
        ell = (
            -0.5 * jnp.sum(((x_hat - x[:, :, None]) ** 2) / noise_var[None, :, None]) / n_sample
            - 0.5 * n * p * math.log(2 * math.pi)
            - 0.5 * n * jnp.sum(jnp.log(noise_var))
        )
        return (-ell + kl) / n

    # Compile the ELBO + gradient. `project` contains lax.while_loop/fori_loop
    # and is wrapped in a custom_vjp, so autodiff never traverses the loop --
    # jit only removes the per-primitive Python dispatch.
    grad_fn = jax.jit(jax.value_and_grad(objective, argnums=(0, 1, 2)))
    update_fn = jax.jit(_adam_update, static_argnums=(3,))

    for epoch in range(1, int(epoch_max) + 1):
        key, subkey = jax.random.split(key)
        (neg_elbo, grads) = grad_fn(mu, cont_sigma, cont_alpha, subkey)

        # ProDAG.jl checks convergence BEFORE `Flux.update!`, so the epoch that
        # triggers the stop applies no update.
        best_score, stagnant_epochs, should_stop = early_stopping(
            float(neg_elbo), best_score, stagnant_epochs, int(patience)
        )
        if verbose:
            print(f"\rEpoch: {epoch}, Neg. ELBO: {float(neg_elbo):.4f}", end="")
        if should_stop:
            break

        (mu, cont_sigma, cont_alpha), state = update_fn((mu, cont_sigma, cont_alpha), grads, state, lr)
    if verbose:
        print()

    return mu, cont_sigma, cont_alpha


def create_mlp(hidden_layers, p, activation_fun, bias):
    layers = [p, *hidden_layers, 1]
    ind = np.repeat(np.arange(p * p), hidden_layers[0])
    ind_mat = np.zeros((p * p, p * p * hidden_layers[0]), dtype=np.float32)
    ind_mat[ind, np.arange(p * p * hidden_layers[0])] = 1.0

    non_fhl_ind = []
    current = 0
    fhl_ind = []
    for _ in range(p):
        first = True
        for in_dim, out_dim in zip(layers[:-1], layers[1:]):
            weight_size = in_dim * out_dim
            if first:
                fhl_ind.extend(range(current, current + weight_size))
            else:
                non_fhl_ind.extend(range(current, current + weight_size))
            current += weight_size
            if bias:
                non_fhl_ind.extend(range(current, current + out_dim))
                current += out_dim
            first = False
    total_weights = current

    fhl_ind = np.asarray(fhl_ind, dtype=np.int32)
    non_fhl_ind = np.asarray(non_fhl_ind, dtype=np.int32)
    ind_order = np.argsort(np.concatenate([fhl_ind, non_fhl_ind]))

    # Every subnetwork has the same architecture and its parameters occupy a
    # contiguous, equal-length block of the flat vector, so the p subnetworks
    # can be sliced as one stacked tensor per layer rather than one Python
    # slice per (subnetwork, layer).  This is what makes the model cheap to
    # trace: the graph is O(#layers) ops instead of O(p * #layers), and with
    # p=25 subnetworks x 1000 ELBO draws the latter built a ~25k-op jaxpr whose
    # XLA compilation dominated the entire fit.
    per_sub = total_weights // p

    def construct(flat_params):
        flat_params = jnp.asarray(flat_params, dtype=jnp.float32)
        blocks = jnp.reshape(flat_params, (p, per_sub))     # [subnetwork, param]
        stacked = []
        offset = 0
        for in_dim, out_dim in zip(layers[:-1], layers[1:]):
            size = in_dim * out_dim
            # (p, in, out) -> (p, out, in), matching Flux's Dense(out, in).
            w = jnp.transpose(
                jnp.reshape(blocks[:, offset:offset + size], (p, in_dim, out_dim)), (0, 2, 1)
            )
            offset += size
            if bias:
                b = blocks[:, offset:offset + out_dim]
                offset += out_dim
            else:
                b = None
            stacked.append((w, b))

        def model(x):
            # x: [p_in, n] -- every subnetwork sees all p variables.
            out = jnp.broadcast_to(x[None, :, :], (p,) + x.shape)
            for layer_index, (w, b) in enumerate(stacked):
                out = jnp.einsum("soi,sin->son", w, out)
                if b is not None:
                    out = out + b[:, :, None]
                if layer_index < len(stacked) - 1:
                    out = activation_fun(out)
            # Final out_dim is 1; Flux.Parallel(vcat, ...) stacks the p
            # subnetwork outputs into [p, n].
            return out[:, 0, :]

        return model

    return construct, non_fhl_ind, fhl_ind, ind_order, jnp.asarray(ind_mat), total_weights


def _train_mlp(
    x,
    construct,
    mu,
    cont_sigma,
    cont_alpha,
    prior_mu,
    prior_sigma,
    prior_alpha,
    dirac_lambda,
    noise_var,
    epoch_max,
    patience,
    optimiser_args,
    params,
    n_sample,
    verbose,
    non_fhl_ind,
    fhl_ind,
    ind_order,
    ind_mat,
):
    lr = float(optimiser_args[0]) if optimiser_args else 0.1
    state = _adam_init((mu, cont_sigma, cont_alpha))
    best_score = float("inf")
    stagnant_epochs = 0
    project = _make_project(params)
    p = x.shape[0]
    n = x.shape[1]
    key = _next_key()

    fhl_ind = jnp.asarray(fhl_ind, dtype=jnp.int32)
    non_fhl_ind = jnp.asarray(non_fhl_ind, dtype=jnp.int32)
    ind_order = jnp.asarray(ind_order, dtype=jnp.int32)

    def _forward_all(omega, x_in):
        """Batched analogue of Julia's per-sample `construct(ω[:, i])(x)` loop."""
        return jax.vmap(lambda o: construct(o)(x_in), in_axes=1, out_axes=2)(omega)

    def objective(mu_param, cont_sigma_param, cont_alpha_param, subkey):
        sigma = jnn.softplus(cont_sigma_param)
        kl = jnp.sum(kl_norm(mu_param, sigma, prior_mu, prior_sigma))

        key_lambda, key_eps = jax.random.split(subkey)
        if dirac_lambda:
            lam = prior_alpha * jnp.ones((n_sample,), dtype=jnp.float32)
        else:
            alpha = jnn.softplus(cont_alpha_param)
            lam = _sample_lambda(alpha[0], False, n_sample, key_lambda)
            kl = kl + jnp.sum(kl_exp(alpha, prior_alpha))

        eps = jax.random.normal(key_eps, (mu_param.shape[0], n_sample), dtype=jnp.float32)
        omega_tilde = mu_param[:, None] + sigma[:, None] * eps
        omega_tilde_fhl = omega_tilde[fhl_ind, :]
        omega_non_fhl = omega_tilde[non_fhl_ind, :]
        w_tilde = _reshape_pp(jnp.sqrt(ind_mat @ (omega_tilde_fhl**2)), p, n_sample)
        w = project(w_tilde, lam)
        scale_factor = ind_mat.T @ _flatten_pp(w / w_tilde, p, n_sample)
        omega_fhl = omega_tilde_fhl * scale_factor
        omega = jnp.concatenate([omega_fhl, omega_non_fhl], axis=0)[ind_order, :]

        # ProDAG.jl loops `for i in 1:n_sample` building one network per sample.
        # vmap is the same computation with one dispatch instead of n_sample
        # (x100 of them, each constructing p subnetworks, dominated the cost).
        x_hat = _forward_all(omega, x)                       # [p, n, n_sample]
        ell = -0.5 * jnp.sum(((x_hat - x[:, :, None]) ** 2) / noise_var[:, None, None]) / n_sample
        # Verbatim from ProDAG.jl's train_mlp!:
        #     ell -= 0.5 * n * p * log(2 * pi) - 0.5 * n * sum(log.(noise_var))
        # Note this subtracts the whole expression, so the noise term enters
        # with the OPPOSITE sign to train_linear!.  Preserved for alignment;
        # both terms are constant in the variational parameters, so neither the
        # gradients nor the early-stopping comparisons are affected.
        ell = ell - (0.5 * n * p * math.log(2 * math.pi) - 0.5 * n * jnp.sum(jnp.log(noise_var)))
        return (-ell + kl) / n

    grad_fn = jax.jit(jax.value_and_grad(objective, argnums=(0, 1, 2)))
    update_fn = jax.jit(_adam_update, static_argnums=(3,))

    for epoch in range(1, int(epoch_max) + 1):
        key, subkey = jax.random.split(key)
        neg_elbo, grads = grad_fn(mu, cont_sigma, cont_alpha, subkey)

        # Convergence is checked BEFORE the update, as in ProDAG.jl.
        best_score, stagnant_epochs, should_stop = early_stopping(
            float(neg_elbo), best_score, stagnant_epochs, int(patience)
        )
        if verbose:
            print(f"\rEpoch: {epoch}, Neg. ELBO: {float(neg_elbo):.4f}", end="")
        if should_stop:
            break

        (mu, cont_sigma, cont_alpha), state = update_fn((mu, cont_sigma, cont_alpha), grads, state, lr)
    if verbose:
        print()

    return mu, cont_sigma, cont_alpha


@dataclass
class ProDAGLinearFit:
    mu: np.ndarray
    sigma: np.ndarray
    alpha: np.ndarray
    dirac_lambda: bool
    p: int


@dataclass
class ProDAGMLPFit:
    mu: np.ndarray
    sigma: np.ndarray
    alpha: np.ndarray
    dirac_lambda: bool
    p: int
    construct: Callable
    non_fhl_ind: np.ndarray
    fhl_ind: np.ndarray
    ind_order: np.ndarray
    ind_mat: np.ndarray


def fit_linear(
    x,
    prior_mu=0.0,
    prior_sigma=1.0,
    prior_alpha=np.inf,
    init_mu=None,
    init_sigma=None,
    init_alpha=None,
    dirac_lambda=True,
    noise_var=None,
    epoch_max=1000,
    patience=5,
    optimiser="adam",
    optimiser_args=(0.1,),
    params=(1, 1, 0.5, 1e-2, 10, 10000, 0.1, None),
    n_sample=100,
    verbose=True,
):
    del optimiser
    x = jnp.asarray(x, dtype=jnp.float32)
    p = x.shape[1]
    n_weights = p**2
    if init_mu is None:
        init_mu = prior_mu
    if init_sigma is None:
        init_sigma = prior_sigma
    if init_alpha is None:
        init_alpha = prior_alpha
    if noise_var is None:
        noise_var = jnp.ones((p,), dtype=jnp.float32)
    else:
        noise_var = jnp.asarray(noise_var, dtype=jnp.float32)
    if params[-1] is None:
        params = (*params[:-1], 1.0 / p)

    prior_mu = jnp.asarray(prior_mu * np.ones(n_weights), dtype=jnp.float32)
    prior_sigma = jnp.asarray(prior_sigma * np.ones(n_weights), dtype=jnp.float32)
    prior_alpha = jnp.asarray(prior_alpha * np.ones(1), dtype=jnp.float32)

    mu = jnp.asarray(np.array(init_mu) * np.ones(n_weights), dtype=jnp.float32)
    sigma = jnp.asarray(np.array(init_sigma) * np.ones(n_weights), dtype=jnp.float32)
    alpha = jnp.asarray(np.array(init_alpha) * np.ones(1), dtype=jnp.float32)
    cont_sigma = jnp.log(jnp.exp(sigma) - 1.0)
    cont_alpha = jnp.log(jnp.exp(alpha) - 1.0)

    mu, cont_sigma, cont_alpha = _train_linear(
        x,
        mu,
        cont_sigma,
        cont_alpha,
        prior_mu,
        prior_sigma,
        prior_alpha,
        dirac_lambda,
        noise_var,
        epoch_max,
        patience,
        optimiser_args,
        params,
        n_sample,
        verbose,
    )

    sigma = jnn.softplus(cont_sigma)
    alpha = jnn.softplus(cont_alpha)
    return ProDAGLinearFit(np.asarray(mu), np.asarray(sigma), np.asarray(alpha), dirac_lambda, p)


def fit_mlp(
    x,
    hidden_layers=(10,),
    activation_fun=jnn.relu,
    bias=True,
    prior_mu=0.0,
    prior_sigma=1.0,
    prior_alpha=np.inf,
    init_mu=None,
    init_sigma=None,
    init_alpha=None,
    dirac_lambda=True,
    noise_var=None,
    epoch_max=1000,
    patience=5,
    optimiser="adam",
    optimiser_args=(0.1,),
    params=(1, 1, 0.5, 1e-2, 10, 10000, 0.1, None),
    n_sample=100,
    verbose=True,
):
    del optimiser
    x = jnp.asarray(x, dtype=jnp.float32)
    p = x.shape[1]
    if init_mu is None:
        init_mu = prior_mu
    if init_sigma is None:
        init_sigma = prior_sigma
    if init_alpha is None:
        init_alpha = prior_alpha
    if noise_var is None:
        noise_var = jnp.ones((p,), dtype=jnp.float32)
    else:
        noise_var = jnp.asarray(noise_var, dtype=jnp.float32)
    if params[-1] is None:
        params = (*params[:-1], 0.25 / p)

    construct, non_fhl_ind, fhl_ind, ind_order, ind_mat, n_weights = create_mlp(hidden_layers, p, activation_fun, bias)

    prior_mu = jnp.asarray(prior_mu * np.ones(n_weights), dtype=jnp.float32)
    prior_sigma = jnp.asarray(prior_sigma * np.ones(n_weights), dtype=jnp.float32)
    prior_alpha = jnp.asarray(prior_alpha * np.ones(1), dtype=jnp.float32)

    mu = jnp.asarray(np.array(init_mu) * np.ones(n_weights), dtype=jnp.float32)
    sigma = jnp.asarray(np.array(init_sigma) * np.ones(n_weights), dtype=jnp.float32)
    alpha = jnp.asarray(np.array(init_alpha) * np.ones(1), dtype=jnp.float32)
    cont_sigma = jnp.log(jnp.exp(sigma) - 1.0)
    cont_alpha = jnp.log(jnp.exp(alpha) - 1.0)

    x_t = x.T
    mu, cont_sigma, cont_alpha = _train_mlp(
        x_t,
        construct,
        mu,
        cont_sigma,
        cont_alpha,
        prior_mu,
        prior_sigma,
        prior_alpha,
        dirac_lambda,
        noise_var,
        epoch_max,
        patience,
        optimiser_args,
        params,
        n_sample,
        verbose,
        non_fhl_ind,
        fhl_ind,
        ind_order,
        ind_mat,
    )

    sigma = jnn.softplus(cont_sigma)
    alpha = jnn.softplus(cont_alpha)
    return ProDAGMLPFit(
        np.asarray(mu),
        np.asarray(sigma),
        np.asarray(alpha),
        dirac_lambda,
        p,
        construct,
        np.asarray(non_fhl_ind),
        np.asarray(fhl_ind),
        np.asarray(ind_order),
        np.asarray(ind_mat),
    )


def guarantee_dag_(w):
    w = np.array(w, copy=True)
    for i in range(w.shape[2]):
        while not nx.is_directed_acyclic_graph(nx.DiGraph(w[:, :, i])):
            nonzero_ind = np.argwhere(w[:, :, i] != 0)
            if nonzero_ind.size == 0:
                break
            nonzero_values = np.array([w[ind[0], ind[1], i] for ind in nonzero_ind])
            min_ind = np.argmin(np.abs(nonzero_values))
            src, dst = nonzero_ind[min_ind]
            w[src, dst, i] = 0.0
    return w


def sample(fit, n_sample=100, guarantee_dag=True, params=None):
    if isinstance(fit, ProDAGLinearFit):
        if params is None:
            params = (1, 1, 0.5, 1e-4, 10, 10000, 0.1, 1 / fit.p)

        mu = jnp.asarray(fit.mu, dtype=jnp.float32)
        sigma = jnp.asarray(fit.sigma, dtype=jnp.float32)
        alpha = jnp.asarray(fit.alpha, dtype=jnp.float32)

        key = _next_key()
        key_lambda, key_eps = jax.random.split(key)
        if fit.dirac_lambda:
            lam = alpha * jnp.ones((n_sample,), dtype=jnp.float32)
        else:
            lam = _sample_lambda(alpha[0], False, n_sample, key_lambda)

        mu = _reshape_pp_vec(mu, fit.p)
        sigma = _reshape_pp_vec(sigma, fit.p)
        eps = jax.random.normal(key_eps, (fit.p, fit.p, n_sample), dtype=jnp.float32)
        w_tilde = mu[:, :, None] + sigma[:, :, None] * eps
        project = _make_project(params)
        w = project(w_tilde, lam)
        w_np = np.asarray(w)
        if guarantee_dag:
            w_np = guarantee_dag_(w_np)
        return jnp.asarray(w_np)

    if isinstance(fit, ProDAGMLPFit):
        if params is None:
            params = (1, 1, 0.5, 1e-4, 10, 10000, 0.1, 0.25 / fit.p)

        mu = jnp.asarray(fit.mu, dtype=jnp.float32)
        sigma = jnp.asarray(fit.sigma, dtype=jnp.float32)
        alpha = jnp.asarray(fit.alpha, dtype=jnp.float32)
        ind_mat = jnp.asarray(fit.ind_mat, dtype=jnp.float32)
        fhl_ind = jnp.asarray(fit.fhl_ind, dtype=jnp.int32)
        non_fhl_ind = jnp.asarray(fit.non_fhl_ind, dtype=jnp.int32)
        ind_order = jnp.asarray(fit.ind_order, dtype=jnp.int32)

        key = _next_key()
        key_lambda, key_eps = jax.random.split(key)
        if fit.dirac_lambda:
            lam = alpha * jnp.ones((n_sample,), dtype=jnp.float32)
        else:
            lam = _sample_lambda(alpha[0], False, n_sample, key_lambda)

        eps = jax.random.normal(key_eps, (mu.shape[0], n_sample), dtype=jnp.float32)
        omega_tilde = mu[:, None] + sigma[:, None] * eps
        omega_tilde_fhl = omega_tilde[fhl_ind, :]
        omega_non_fhl = omega_tilde[non_fhl_ind, :]
        w_tilde = _reshape_pp(jnp.sqrt(ind_mat @ (omega_tilde_fhl**2)), fit.p, n_sample)
        project = _make_project(params)
        w = project(w_tilde, lam)

        if guarantee_dag:
            w = jnp.asarray(guarantee_dag_(np.asarray(w)), dtype=jnp.float32)

        scale_factor = ind_mat.T @ _flatten_pp(w / w_tilde, fit.p, n_sample)
        omega_fhl = omega_tilde_fhl * scale_factor
        omega = jnp.concatenate([omega_fhl, omega_non_fhl], axis=0)[ind_order, :]

        models = [fit.construct(omega[:, i]) for i in range(n_sample)]
        return jnp.asarray(w), models

    raise TypeError(f"Unsupported fit type: {type(fit)!r}")
