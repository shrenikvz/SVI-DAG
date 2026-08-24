"""
JAX implementation of BayesDAG (linear + nonlinear), GPU-optimized.

Algorithm is IDENTICAL to the previous eager implementation in this file
(itself a port of microsoft/Project-BayesDAG): same SG-MCMC updates, same
Sinkhorn+Hungarian straight-through permutation, same relaxed-Bernoulli W
sampling, same priors/likelihood, same training schedule, same buffer
semantics, and the same JAX PRNG key sequence (the SG-MCMC noise-key
splitting emulates the old pytree leaf enumeration, so results match the
previous implementation bit-for-bit up to XLA fusion rounding).

What changed (performance only):
  * Every hot-path computation now runs inside `jax.jit`; the previous
    implementation called `jax.value_and_grad` eagerly, which re-traced all
    three loss functions (including the 500-iteration Sinkhorn scan) on
    every batch step. That tracing overhead dominated runtime.
  * All batches of an epoch execute in a single `lax.scan`, so one dispatch
    per epoch instead of ~10 host round-trips per batch.
  * Per-step host syncs are gone: losses and the p/theta sample histories
    are accumulated on device inside the scan and transferred to the host
    deques once per epoch (same deque contents, same order).
  * Jitted functions are module-level and keyed by a hashable static config,
    so repeated fits in one process (e.g. the wrapper's lambda grid search)
    reuse compiled executables. `lambda_sparse`, `dataset_size` and the two
    SG-MCMC noise scales are traced scalars, so changing them does NOT
    trigger recompilation.
  * Fixed a fatal API bug: `jnp.logsumexp` does not exist in current JAX;
    the Sinkhorn normalization now uses `jax.scipy.special.logsumexp`.
  * Layer config flags (normalize/residual/activation) are static metadata
    instead of pytree leaves, which removes the `allow_int=True` float0
    workarounds from the differentiated code paths.

The Hungarian assignment stays an exact SciPy `linear_sum_assignment` via
`jax.pure_callback` (batched: one host round-trip per graph transform), so
the hard-permutation semantics of the original are preserved exactly.

One deliberate deviation (stability only): both optimizer updates
(`_sgmcmc_update`, `_adam_update`) reject a step leaf-wise if it would make
the parameter / momentum / variance state non-finite. At large dataset_size
the N-scaled SG-MCMC gradients can overflow float32 ((grad*N)**2 -> inf,
G = 0, 0*inf = NaN), after which the NaN reaches SciPy's Hungarian solver
and the run aborts ("matrix contains invalid numeric entries") -- the
original PyTorch implementation crashes identically (observed on every
case-2/3 fit with n >= 316 and every case-4 Sachs split). A rejected step
keeps the previous state; steps whose results are all finite are selected
unchanged, so any run that never produces a non-finite value is
bit-identical with or without the guard.

Note: parameter-state donation is deliberately NOT used — `best_state`
snapshots alias the live state arrays (JAX arrays are immutable), and
donation would invalidate those aliases.
"""

import dataclasses
import json
import math
import os
import pickle
from collections import deque
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment

# Optional persistent compilation cache (helps across cluster job restarts).
_cache_dir = os.environ.get("BAYESDAG_JAX_CACHE_DIR")
if _cache_dir:
    try:  # pragma: no cover
        jax.config.update("jax_compilation_cache_dir", _cache_dir)
    except Exception:
        pass


def _next_key(seed: Optional[int] = None) -> jax.Array:
    if seed is None:
        seed = int(np.random.randint(0, np.iinfo(np.int32).max))
    return jax.random.PRNGKey(int(seed))


def _torch_linear_weight_init(key, out_dim: int, in_dim: int, dtype=jnp.float32):
    gain = math.sqrt(2.0 / (1.0 + 5.0))
    std = gain / math.sqrt(float(in_dim))
    bound = math.sqrt(3.0) * std
    return jax.random.uniform(key, (out_dim, in_dim), minval=-bound, maxval=bound, dtype=dtype)


def _torch_linear_bias_init(key, out_dim: int, in_dim: int, dtype=jnp.float32):
    bound = 1.0 / math.sqrt(float(in_dim))
    return jax.random.uniform(key, (out_dim,), minval=-bound, maxval=bound, dtype=dtype)


# ---------------------------------------------------------------------------
# Optimizer states
# ---------------------------------------------------------------------------

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


@jax.tree_util.register_pytree_node_class
@dataclass
class SGMCMCState:
    momentum: object
    variance: object

    def tree_flatten(self):
        return (self.momentum, self.variance), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        momentum, variance = children
        return cls(momentum=momentum, variance=variance)


def _adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return AdamState(step=jnp.asarray(0, dtype=jnp.int32), m=zeros, v=zeros)


def _adam_update(params, grads, state: AdamState, lr: float):
    """Torch-default Adam (betas 0.9/0.999, eps 1e-8, bias correction).

    Identical math to the previous implementation; parameter trees contain
    only float arrays now, so no float0/bool special-casing is needed.
    """
    step = state.step + 1
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    bc1 = 1.0 - beta1 ** step.astype(jnp.float32)
    bc2 = 1.0 - beta2 ** step.astype(jnp.float32)

    leaves, treedef = jax.tree_util.tree_flatten(params)
    grad_leaves = treedef.flatten_up_to(grads)
    m_leaves = treedef.flatten_up_to(state.m)
    v_leaves = treedef.flatten_up_to(state.v)

    new_p, new_m, new_v = [], [], []
    for p, g, m_i, v_i in zip(leaves, grad_leaves, m_leaves, v_leaves):
        m_new = beta1 * m_i + (1.0 - beta1) * g
        v_new = beta2 * v_i + (1.0 - beta2) * (g * g)
        p_new = p - lr * (m_new / bc1) / (jnp.sqrt(v_new / bc2) + eps)
        # Divergence guard (see _sgmcmc_update for the mechanism): reject the
        # step for this leaf if it would leave any of (param, m, v) non-finite
        # (NaN gradients propagate here when the likelihood forward pass
        # overflows float32). All-finite steps are selected unchanged, so
        # trajectories that never produce a non-finite value are bit-identical
        # with or without the guard.
        ok = (
            jnp.all(jnp.isfinite(p_new))
            & jnp.all(jnp.isfinite(m_new))
            & jnp.all(jnp.isfinite(v_new))
        )
        new_p.append(jnp.where(ok, p_new, p))
        new_m.append(jnp.where(ok, m_new, m_i))
        new_v.append(jnp.where(ok, v_new, v_i))

    return (
        jax.tree_util.tree_unflatten(treedef, new_p),
        AdamState(
            step=step,
            m=jax.tree_util.tree_unflatten(treedef, new_m),
            v=jax.tree_util.tree_unflatten(treedef, new_v),
        ),
    )


def _sgmcmc_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return SGMCMCState(momentum=zeros, variance=zeros)


def _sgmcmc_update(
    params,
    grads,
    state: SGMCMCState,
    *,
    lr: float,
    dataset_size,
    betas=(0.9, 0.99),
    scale_noise,
    key: jax.Array,
    key_positions: Tuple[int, ...],
    total_keys: int,
):
    """Adam-SGMCMC step, identical math to the previous implementation.

    RNG parity note: the old code split `key` into one subkey per pytree
    leaf, where the leaf enumeration *included* Python-bool config flags
    (skipped in the update). The flags are no longer pytree leaves, so to
    reproduce the exact same noise stream we split into `total_keys`
    (= old leaf count) and index with `key_positions` (= old positions of
    the float leaves).
    """
    lr_sqrt = math.sqrt(lr)
    beta1, beta2 = betas
    beta = (1.0 - beta1) / lr_sqrt
    noise_const = math.sqrt(2 * lr_sqrt * (beta - 0.5 * lr_sqrt))

    leaves, treedef = jax.tree_util.tree_flatten(params)
    grad_leaves = treedef.flatten_up_to(grads)
    mom_leaves = treedef.flatten_up_to(state.momentum)
    var_leaves = treedef.flatten_up_to(state.variance)
    noise_keys = jax.random.split(key, total_keys)

    new_params, new_momentum, new_variance = [], [], []
    for i, (param, grad, momentum, variance) in enumerate(
        zip(leaves, grad_leaves, mom_leaves, var_leaves)
    ):
        grad_scaled = grad * dataset_size
        variance_new = beta2 * variance + (1.0 - beta2) * (grad_scaled**2)
        G = 1.0 / jnp.sqrt(1e-8 + jnp.sqrt(1e-8 + variance_new))
        noise = scale_noise * jax.random.normal(
            noise_keys[key_positions[i]], param.shape, dtype=param.dtype
        )
        momentum_new = beta1 * momentum + lr_sqrt * G * grad_scaled + noise * noise_const
        param_new = param - lr_sqrt * G * momentum_new
        # Divergence guard: with dataset_size-scaled gradients a likelihood
        # spike can overflow float32 -- (grad*N)**2 -> inf makes G = 0 and
        # G*grad_scaled = 0*inf = NaN, which then reaches the Sinkhorn /
        # Hungarian permutation where SciPy's assignment solver aborts with
        # "matrix contains invalid numeric entries" (the original PyTorch
        # implementation crashes identically at large N). Reject such steps
        # leaf-wise: keep the previous (param, momentum, variance) as if the
        # step had not happened. All-finite steps are selected unchanged, so
        # runs that never spike are bit-identical with or without the guard.
        ok = (
            jnp.all(jnp.isfinite(param_new))
            & jnp.all(jnp.isfinite(momentum_new))
            & jnp.all(jnp.isfinite(variance_new))
        )
        new_params.append(jnp.where(ok, param_new, param))
        new_momentum.append(jnp.where(ok, momentum_new, momentum))
        new_variance.append(jnp.where(ok, variance_new, variance))

    return (
        jax.tree_util.tree_unflatten(treedef, new_params),
        SGMCMCState(
            momentum=jax.tree_util.tree_unflatten(treedef, new_momentum),
            variance=jax.tree_util.tree_unflatten(treedef, new_variance),
        ),
    )


# ---------------------------------------------------------------------------
# Hard permutation via exact Hungarian assignment (host callback)
# ---------------------------------------------------------------------------

def _host_callback(fn, argument, result_shape):
    if hasattr(jax, "pure_callback"):
        return jax.pure_callback(fn, result_shape, argument)
    from jax.experimental import host_callback  # pragma: no cover

    return host_callback.call(fn, argument, result_shape=result_shape)


@jax.custom_jvp
def _batched_hungarian(prob_mats: jax.Array) -> jax.Array:
    def solve(x):
        x = np.asarray(x)
        out = np.zeros((x.shape[0], x.shape[1], x.shape[2]), dtype=np.float32)
        for i in range(x.shape[0]):
            row_ind, col_ind = linear_sum_assignment(-x[i])
            out[i, row_ind, col_ind] = 1.0
        return out

    return _host_callback(solve, prob_mats, jax.ShapeDtypeStruct(prob_mats.shape, jnp.float32))


# Straight-through estimator: forward pass uses the hard (non-differentiable)
# Hungarian assignment; gradients pass through as identity so AD can
# propagate through pure_callback (which has no native JVP).
@_batched_hungarian.defjvp
def _batched_hungarian_jvp(primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    return _batched_hungarian(x), x_dot


def _layer_norm(x, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)


# ---------------------------------------------------------------------------
# MLPs. Parameters are pytrees of float arrays ONLY; the per-layer config
# (normalize / activation / residual) lives in static tuples so that jitted
# functions can branch on it at trace time.
# ---------------------------------------------------------------------------

def _mlp_layer_flags(dims: Sequence[int], normalization: bool, res_connection: bool):
    """Per-layer (normalize, activation, residual) flags; layout identical to
    the previous implementation (norm before every layer except the first,
    activation on hidden layers, residual on width-preserving hidden layers)."""
    num_hidden = len(dims) - 2
    flags = []
    for idx, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        flags.append(
            (
                bool(normalization and idx > 0),
                bool(idx < num_hidden),
                bool(res_connection and idx < num_hidden and in_dim == out_dim),
            )
        )
    return tuple(flags)


def _build_shared_mlp(input_dim, output_dim, hidden_dims, *, key, normalization, res_connection):
    params = []
    dims = [input_dim, *hidden_dims, output_dim]
    keys = jax.random.split(key, len(dims) - 1)
    for idx, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        weight_key, bias_key = jax.random.split(keys[idx])
        params.append(
            {
                "weight": _torch_linear_weight_init(weight_key, out_dim, in_dim),
                "bias": _torch_linear_bias_init(bias_key, out_dim, in_dim),
            }
        )
    return params, _mlp_layer_flags(dims, normalization, res_connection)


def _apply_shared_mlp(params, flags, x, *, nonlinearity="relu"):
    out = x
    for layer, (normalize, activation, residual) in zip(params, flags):
        res = out
        if normalize:
            out = _layer_norm(out)
        out = out @ layer["weight"].T + layer["bias"]
        if activation:
            out = jnn.relu(out) if nonlinearity == "relu" else jnn.leaky_relu(out)
        if residual:
            out = out + res
    return out


def _build_chain_mlp(num_chains, input_dim, output_dim, hidden_dims, *, key, normalization, res_connection):
    dims = [input_dim, *hidden_dims, output_dim]
    params = []
    keys = jax.random.split(key, len(dims) - 1)
    for idx, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        weight_keys = jax.random.split(keys[idx], num_chains + 1)
        params.append(
            {
                "weight": jax.vmap(lambda subkey: _torch_linear_weight_init(subkey, out_dim, in_dim))(weight_keys[:-1]),
                "bias": jax.vmap(lambda subkey: _torch_linear_bias_init(subkey, out_dim, in_dim))(weight_keys[1:]),
            }
        )
    return params, _mlp_layer_flags(dims, normalization, res_connection)


def _apply_chain_mlp(params, flags, x, *, nonlinearity="leaky_relu"):
    out = x
    for layer, (normalize, activation, residual) in zip(params, flags):
        res = out
        if normalize:
            out = _layer_norm(out)
        bias = jnp.reshape(
            layer["bias"], (layer["bias"].shape[0],) + (1,) * (out.ndim - 2) + (layer["bias"].shape[1],)
        )
        out = jnp.einsum("coi,c...i->c...o", layer["weight"], out) + bias
        if activation:
            out = jnn.leaky_relu(out) if nonlinearity == "leaky_relu" else jnn.relu(out)
        if residual:
            out = out + res
    return out


def _gaussian_log_prob(z: jax.Array, logscale: jax.Array) -> jax.Array:
    logvar = 2.0 * logscale[:, None, :]
    return -0.5 * (math.log(2 * math.pi) + logvar + (z**2) / (jnp.exp(logvar) + 1e-7))


def _gaussian_sample(key: jax.Array, logscale: jax.Array, n_samples: int) -> jax.Array:
    eps = jax.random.normal(key, (n_samples, *logscale.shape), dtype=jnp.float32)
    return eps * jnp.exp(logscale)[None, :, :]


@dataclass
class SimpleDataset:
    train_data: np.ndarray
    test_data: Optional[np.ndarray] = None
    adjacency: Optional[np.ndarray] = None


@dataclass
class ContinuousVariables:
    num_groups: int
    num_processed_non_aux_cols: int
    group_mask: np.ndarray
    processed_cols_by_type: Dict[str, List[List[int]]]

    @classmethod
    def from_num_nodes(cls, num_nodes: int):
        group_mask = np.eye(num_nodes, dtype=np.float32)
        return cls(
            num_groups=num_nodes,
            num_processed_non_aux_cols=num_nodes,
            group_mask=group_mask,
            processed_cols_by_type={"continuous": [[i] for i in range(num_nodes)], "binary": [], "categorical": []},
        )


def _slice_chain_tree(tree, index: int):
    def _take(x):
        if not hasattr(x, "ndim") or x.ndim == 0:
            return x
        return np.asarray(x[index])

    return jax.tree_util.tree_map(_take, tree)


def _stack_chain_trees(trees: Sequence[Any]):
    if not trees:
        raise ValueError("Cannot stack an empty sequence of parameter trees.")

    def _stack(*xs):
        first = xs[0]
        if not hasattr(first, "ndim") or first.ndim == 0:
            return first
        return jnp.stack([jnp.asarray(x) for x in xs], axis=0)

    return jax.tree_util.tree_map(_stack, *trees)


def _normalize_seed_value(seed_value: Any, fallback: int) -> int:
    if isinstance(seed_value, (list, tuple)):
        if not seed_value:
            return int(fallback)
        seed_value = seed_value[0]
    if seed_value is None:
        return int(fallback)
    return int(seed_value)


# ---------------------------------------------------------------------------
# Static configuration + train state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _StaticConfig:
    """Hashable, trace-time-only model description. Used as a `static_argnames`
    jit key, so all fits with the same architecture share compiled code."""

    num_nodes: int
    processed_dim_all: int
    num_chains: int
    sinkhorn_n_iter: int
    model_type: str
    input_perm: bool
    vi_norm: bool
    logit_const: float
    o_scale: float
    activation: str
    helper_flags: tuple
    g_flags: Optional[tuple]
    f_flags: Optional[tuple]
    # RNG-parity emulation of the old (bool-leaf-bearing) pytree layout:
    icgnn_key_positions: tuple
    icgnn_total_leaves: int


@jax.tree_util.register_pytree_node_class
@dataclass
class _TrainState:
    key: jax.Array
    p: jax.Array
    helper: Any
    logscale: jax.Array
    icgnn: Any
    p_opt: SGMCMCState
    w_opt: SGMCMCState
    h_opt: AdamState

    def tree_flatten(self):
        return (
            (self.key, self.p, self.helper, self.logscale, self.icgnn, self.p_opt, self.w_opt, self.h_opt),
            None,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


# Traced hyperparameters: (lambda_sparse, dataset_size, scale_noise_p, scale_noise).
# Passed as device scalars so sweeping them (e.g. the lambda grid search)
# does not recompile.
def _make_hypers(lambda_sparse, dataset_size, scale_noise_p, scale_noise):
    return (
        jnp.asarray(lambda_sparse, dtype=jnp.float32),
        jnp.asarray(dataset_size, dtype=jnp.float32),
        jnp.asarray(scale_noise_p, dtype=jnp.float32),
        jnp.asarray(scale_noise, dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# Pure model functions (everything below is jit-safe)
# ---------------------------------------------------------------------------

def _compute_perm_hard(cfg: _StaticConfig, p: jax.Array):
    O = cfg.o_scale * jnp.arange(1, cfg.num_nodes + 1, dtype=p.dtype)[None, :]
    X = p[:, :, None] * O[:, None, :]

    # Fixed-length scan (reverse-mode differentiable); converged Sinkhorn
    # iterates are fixed points of the row/column normalization, so extra
    # iterations are a no-op. unroll=8 amortizes loop overhead on GPU
    # without changing the operation sequence.
    def sinkhorn_one(log_alpha):
        def body(alpha, _):
            alpha = alpha - logsumexp(alpha, axis=-1, keepdims=True)
            alpha = alpha - logsumexp(alpha, axis=-2, keepdims=True)
            return alpha, None

        alpha, _ = jax.lax.scan(body, log_alpha, xs=None, length=int(cfg.sinkhorn_n_iter), unroll=8)
        return jnp.exp(alpha)

    perm = jax.vmap(sinkhorn_one)(X / 0.2)
    perm_matrix = _batched_hungarian(perm)
    perm_matrix_hard = jax.lax.stop_gradient(perm_matrix - perm) + perm
    return perm_matrix_hard, perm


def _transform_adj(cfg: _StaticConfig, helper, p: jax.Array, key: jax.Array):
    perm_matrix_hard, perm = _compute_perm_hard(cfg, p)
    if cfg.input_perm:
        helper_input = jnp.reshape(perm_matrix_hard, (perm_matrix_hard.shape[0], -1))
    else:
        helper_input = _layer_norm(p) if cfg.vi_norm else p

    logits = _apply_shared_mlp(helper, cfg.helper_flags, helper_input, nonlinearity="relu") + cfg.logit_const
    u = jax.random.uniform(key, logits.shape, minval=1e-20, maxval=1.0)
    logistic = jnp.log(u) - jnp.log(1.0 - u)
    w_soft = jnn.sigmoid((logits + logistic) / 0.2)
    w_hard = jnp.round(w_soft)
    w_vec = jax.lax.stop_gradient(w_hard - w_soft) + w_soft
    W = jnp.reshape(w_vec, (perm.shape[0], cfg.num_nodes, cfg.num_nodes))
    # Constant strict-lower-triangular ones matrix; identical to the previous
    # fill_triangular(ones) construction, without materializing per-chain copies.
    lower = jnp.tril(jnp.ones((cfg.num_nodes, cfg.num_nodes), dtype=p.dtype), k=-1)
    return W * (perm_matrix_hard @ lower @ jnp.swapaxes(perm_matrix_hard, -1, -2))


def _weighted_adjacency(cfg: _StaticConfig, icgnn):
    return icgnn["W"] * (1.0 - jnp.eye(cfg.num_nodes, dtype=jnp.float32))[None, :, :]


def _predict(cfg: _StaticConfig, icgnn, x: jax.Array, w_adj: jax.Array) -> jax.Array:
    if cfg.model_type == "linear":
        return jnp.einsum("bd,cde->cbe", x, w_adj)

    num_chains = icgnn["embeddings"].shape[0]
    group_mask = jnp.eye(cfg.num_nodes, dtype=jnp.float32)
    X = x[:, None, :]
    X_masked = X * group_mask[None, :, :]
    X_masked = jnp.broadcast_to(
        X_masked[None, :, :, :], (num_chains, x.shape[0], cfg.num_nodes, cfg.processed_dim_all)
    )
    E = jnp.broadcast_to(
        icgnn["embeddings"][:, None, :, :],
        (num_chains, x.shape[0], cfg.num_nodes, icgnn["embeddings"].shape[-1]),
    )
    X_in_g = jnp.concatenate([X_masked, E], axis=-1)
    X_emb = _apply_chain_mlp(icgnn["g"], cfg.g_flags, X_in_g, nonlinearity=cfg.activation)
    X_aggr_sum = jnp.einsum("cij,cbjk->cbik", jnp.swapaxes(w_adj, -1, -2), X_emb)
    X_in_f = jnp.concatenate([X_aggr_sum, E], axis=-1)
    X_rec = _apply_chain_mlp(icgnn["f"], cfg.f_flags, X_in_f, nonlinearity=cfg.activation)
    X_rec = X_rec * group_mask[None, None, :, :]
    return jnp.sum(X_rec, axis=-2)


def _params_flat(cfg: _StaticConfig, icgnn):
    return jnp.concatenate(
        [jnp.reshape(leaf, (cfg.num_chains, -1)) for leaf in jax.tree_util.tree_leaves(icgnn)],
        axis=-1,
    )


def _data_likelihood(cfg, hyp, *, p_value, icgnn, logscale, x, A_samples, return_prior):
    lam, dataset_size, _, _ = hyp
    theta_prior = p_prior = sparse_loss = None
    if return_prior:
        pf = _params_flat(cfg, icgnn)
        theta_prior = (1.0 / dataset_size) * jnp.sum(-0.5 * (math.log(2 * math.pi) + pf**2), axis=-1)
        p_prior = (1.0 / dataset_size) * jnp.sum(
            -0.5 * (math.log(2 * math.pi * (0.1**2)) + (p_value**2) / (0.1**2)), axis=-1
        )
        sparse_loss = -(1.0 / dataset_size) * lam * jnp.sum(jnp.abs(A_samples), axis=(1, 2))

    weighted_adj = A_samples * _weighted_adjacency(cfg, icgnn)
    predict = _predict(cfg, icgnn, x, weighted_adj)
    log_p_base = _gaussian_log_prob(x[None, :, :] - predict, logscale).sum(-1).T
    if return_prior:
        return log_p_base, theta_prior, p_prior, sparse_loss
    return log_p_base


def _w_prior_entropy(cfg: _StaticConfig, hyp, helper, p: jax.Array, key: jax.Array):
    _, dataset_size, _, _ = hyp
    if cfg.input_perm:
        perm_matrix_hard, _perm = _compute_perm_hard(cfg, p)
        helper_input = jnp.reshape(perm_matrix_hard, (perm_matrix_hard.shape[0], -1))
    else:
        helper_input = _layer_norm(p) if cfg.vi_norm else p

    logits = _apply_shared_mlp(helper, cfg.helper_flags, helper_input, nonlinearity="relu") + cfg.logit_const
    del key  # The sampled W was never used by the previous implementation
    # (the prior is a constant in W); the summation form is kept identical.
    prior = (1.0 / dataset_size) * jnp.sum(
        jnp.log(0.5) * jnp.ones((p.shape[0], cfg.num_nodes, cfg.num_nodes), dtype=p.dtype),
        axis=(1, 2),
    )
    probs = jnn.sigmoid(logits)
    entropy = (1.0 / dataset_size) * jnp.sum(
        -(probs * jnp.log(probs + 1e-8) + (1.0 - probs) * jnp.log(1.0 - probs + 1e-8)),
        axis=-1,
    )
    return prior, entropy


# --- The three per-batch phases (same order and same key splits as before) ---

def _p_phase(cfg, hyp, st: _TrainState, x):
    _, dataset_size, scale_noise_p, _ = hyp
    key, sample_key = jax.random.split(st.key)

    def loss_fn(p_value):
        A_samples = _transform_adj(cfg, st.helper, p_value, sample_key)
        ll, _tp, p_prior, sparse = _data_likelihood(
            cfg, hyp, p_value=p_value, icgnn=st.icgnn, logscale=st.logscale,
            x=x, A_samples=A_samples, return_prior=True,
        )
        return -(ll + p_prior[None, :] + sparse[None, :]).mean()

    loss, grad = jax.value_and_grad(loss_fn)(st.p)
    key, update_key = jax.random.split(key)
    p_new, p_opt = _sgmcmc_update(
        st.p, grad, st.p_opt,
        lr=0.0003, dataset_size=dataset_size, betas=(0.9, 0.99),
        scale_noise=scale_noise_p, key=update_key,
        key_positions=(0,), total_keys=1,
    )
    return st.replace(key=key, p=p_new, p_opt=p_opt), loss


def _helper_phase(cfg, hyp, st: _TrainState, x):
    key, adj_key, prior_key = jax.random.split(st.key, 3)

    def loss_fn(helper_and_logscale):
        helper, logscale = helper_and_logscale
        A_samples = _transform_adj(cfg, helper, st.p, adj_key)
        ll, _tp, _pp, sparse = _data_likelihood(
            cfg, hyp, p_value=st.p, icgnn=st.icgnn, logscale=logscale,
            x=x, A_samples=A_samples, return_prior=True,
        )
        prior, entropy = _w_prior_entropy(cfg, hyp, helper, st.p, prior_key)
        return -(ll + prior[None, :] + entropy[None, :] + sparse[None, :]).mean()

    loss, grads = jax.value_and_grad(loss_fn)((st.helper, st.logscale))
    (helper, logscale), h_opt = _adam_update((st.helper, st.logscale), grads, st.h_opt, lr=0.005)
    return st.replace(key=key, helper=helper, logscale=logscale, h_opt=h_opt), loss


def _w_phase(cfg, hyp, st: _TrainState, x):
    _, dataset_size, _, scale_noise = hyp
    key, sample_key = jax.random.split(st.key)

    def loss_fn(icgnn):
        A_samples = _transform_adj(cfg, st.helper, st.p, sample_key)
        ll, theta_prior, _pp, sparse = _data_likelihood(
            cfg, hyp, p_value=st.p, icgnn=icgnn, logscale=st.logscale,
            x=x, A_samples=A_samples, return_prior=True,
        )
        return -(ll + theta_prior[None, :] + sparse[None, :]).mean()

    loss, grad = jax.value_and_grad(loss_fn)(st.icgnn)
    key, update_key = jax.random.split(key)
    icgnn, w_opt = _sgmcmc_update(
        st.icgnn, grad, st.w_opt,
        lr=0.0003, dataset_size=dataset_size, betas=(0.9, 0.99),
        scale_noise=scale_noise, key=update_key,
        key_positions=cfg.icgnn_key_positions, total_keys=cfg.icgnn_total_leaves,
    )
    return st.replace(key=key, icgnn=icgnn, w_opt=w_opt), loss


def _train_batch(cfg, hyp, st: _TrainState, x):
    """One regular batch: p SG-MCMC step, helper VI step, theta SG-MCMC step.
    Emits the post-update p and theta samples (the per-step buffer appends
    of the previous implementation), with a leading history axis of 1."""
    st, p_loss = _p_phase(cfg, hyp, st, x)
    p_hist = st.p[None]
    st, w_loss = _helper_phase(cfg, hyp, st, x)
    st, theta_loss = _w_phase(cfg, hyp, st, x)
    ic_hist = jax.tree_util.tree_map(lambda a: a[None], st.icgnn)
    return st, (jnp.stack([p_loss, w_loss, theta_loss]), p_hist, ic_hist)


def _train_batch_first(cfg, hyp, st: _TrainState, x):
    """The very first batch ever: with num_burnin_steps=1 the p and theta
    phases each run TWO steps (burn-in + sample) and both post-update values
    are appended to the buffers; reported losses are per-phase averages.
    This mirrors `num_steps = num_burnin_steps - steps + num_samples`."""
    st, p_loss_0 = _p_phase(cfg, hyp, st, x)
    p_0 = st.p
    st, p_loss_1 = _p_phase(cfg, hyp, st, x)
    p_hist = jnp.stack([p_0, st.p])
    p_loss = (p_loss_0 + p_loss_1) / 2.0

    st, w_loss = _helper_phase(cfg, hyp, st, x)

    st, t_loss_0 = _w_phase(cfg, hyp, st, x)
    ic_0 = st.icgnn
    st, t_loss_1 = _w_phase(cfg, hyp, st, x)
    ic_hist = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), ic_0, st.icgnn)
    theta_loss = (t_loss_0 + t_loss_1) / 2.0

    return st, (jnp.stack([p_loss, w_loss, theta_loss]), p_hist, ic_hist)


@partial(jax.jit, static_argnames=("cfg",))
def _train_batch_jit(cfg, hyp, st, x):
    return _train_batch(cfg, hyp, st, x)


@partial(jax.jit, static_argnames=("cfg",))
def _train_batch_first_jit(cfg, hyp, st, x):
    return _train_batch_first(cfg, hyp, st, x)


@partial(jax.jit, static_argnames=("cfg",))
def _run_epoch_jit(cfg, hyp, st, data, idx):
    """All full batches of an epoch in a single lax.scan (one dispatch)."""

    def body(carry, batch_idx):
        x = data[batch_idx]
        carry, (losses, p_hist, ic_hist) = _train_batch(cfg, hyp, carry, x)
        return carry, (losses, p_hist[0], jax.tree_util.tree_map(lambda a: a[0], ic_hist))

    st, (losses, p_hist, ic_hist) = jax.lax.scan(body, st, idx)
    return st, (losses, p_hist, ic_hist)


@partial(jax.jit, static_argnames=("cfg",))
def _sample_scan_jit(cfg, hyp, st, data, idx):
    """MCMC steps performed inside get_adj_matrix_tensor: per step one
    p phase and one theta phase (no helper VI), same as before."""

    def body(carry, batch_idx):
        x = data[batch_idx]
        carry, _ = _p_phase(cfg, hyp, carry, x)
        p_out = carry.p
        carry, _ = _w_phase(cfg, hyp, carry, x)
        return carry, (p_out, carry.icgnn)

    st, (p_hist, ic_hist) = jax.lax.scan(body, st, idx)
    return st, (p_hist, ic_hist)


@partial(jax.jit, static_argnames=("cfg",))
def _transform_adj_jit(cfg, helper, p, key):
    return _transform_adj(cfg, helper, p, key)


# ---------------------------------------------------------------------------
# RNG-parity bookkeeping: old pytree leaf layout of the ICGNN parameters
# ---------------------------------------------------------------------------

def _icgnn_old_leaf_layout(icgnn_arrays, g_flags, f_flags, model_type: str):
    """Reconstruct the previous implementation's ICGNN pytree (which carried
    Python-bool layer flags as leaves) and return (positions, total): the
    flatten positions of the float-array leaves and the total leaf count.
    Used to reproduce the exact SG-MCMC noise key assignment."""

    def relayer(layers, flags):
        out = []
        for layer, (normalize, activation, residual) in zip(layers, flags):
            out.append(
                {
                    "weight": layer["weight"],
                    "bias": layer["bias"],
                    "normalize": normalize,
                    "residual": residual,
                    "activation": activation,
                }
            )
        return out

    if model_type == "linear":
        old_tree = {"W": icgnn_arrays["W"]}
    else:
        old_tree = {
            "W": icgnn_arrays["W"],
            "embeddings": icgnn_arrays["embeddings"],
            "g": relayer(icgnn_arrays["g"], g_flags),
            "f": relayer(icgnn_arrays["f"], f_flags),
        }

    old_leaves = jax.tree_util.tree_leaves(old_tree)
    positions = tuple(
        i for i, leaf in enumerate(old_leaves)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)
    )
    new_leaves = jax.tree_util.tree_leaves(icgnn_arrays)
    if len(positions) != len(new_leaves):
        raise AssertionError("ICGNN leaf-position bookkeeping is inconsistent.")
    for pos, new_leaf in zip(positions, new_leaves):
        if old_leaves[pos].shape != new_leaf.shape:
            raise AssertionError("ICGNN leaf-order bookkeeping is inconsistent.")
    return positions, len(old_leaves)


def _strip_layer_flags(tree):
    """Sanitize state dicts saved by the previous implementation (which stored
    bool layer flags inside the parameter trees) down to arrays only."""
    if isinstance(tree, dict):
        out = {}
        for k, v in tree.items():
            if k in ("normalize", "residual", "activation", "is_final"):
                continue
            out[k] = _strip_layer_flags(v)
        return out
    if isinstance(tree, (list, tuple)):
        return type(tree)(_strip_layer_flags(v) for v in tree)
    return tree


# ---------------------------------------------------------------------------
# Model classes (public API unchanged)
# ---------------------------------------------------------------------------

class BayesDAGBase:
    def __init__(
        self,
        model_id: str,
        variables: ContinuousVariables,
        save_dir: str,
        *,
        seed: int = 0,
        lambda_sparse: float = 1.0,
        num_chains: int = 10,
        sinkhorn_n_iter: int = 3000,
        scale_noise: float = 0.1,
        scale_noise_p: float = 1.0,
        norm_layers: bool = False,
        res_connection: bool = False,
        model_type: str = "nonlinear",
        sparse_init: bool = False,
        input_perm: bool = False,
        VI_norm: bool = False,
        # Paper-spec ICGNN architecture (BayesDAG nonlinear); see the previous
        # implementation's notes. Overridable via the JSON config.
        hidden_size: int = 128,
        num_hidden_layers: int = 2,
        activation: str = "relu",
    ):
        self.model_id = model_id
        self.variables = variables
        self.save_dir = save_dir
        self.num_nodes = variables.num_groups
        self.processed_dim_all = variables.num_processed_non_aux_cols
        self.lambda_sparse = float(lambda_sparse)
        self.num_chains = int(num_chains)
        self.sinkhorn_n_iter = int(sinkhorn_n_iter)
        self.scale_noise = float(scale_noise)
        self.scale_noise_p = float(scale_noise_p)
        self.norm_layers = bool(norm_layers)
        self.res_connection = bool(res_connection)
        self.model_type = model_type
        self.input_perm = bool(input_perm)
        self.sparse_init = bool(sparse_init)
        self.VI_norm = bool(VI_norm)
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        if str(activation) not in ("relu", "leaky_relu"):
            raise ValueError(f"activation must be 'relu' or 'leaky_relu'; got {activation!r}")
        self.activation = str(activation)
        self.logit_const = -1.0 if self.sparse_init else 0.0
        self.o_scale = 10.0
        self.p_scale = 0.01
        self.num_burnin_steps = 1
        self.dataset_size = 5000
        self.train_data = None

        os.makedirs(save_dir, exist_ok=True)

        # --- Initialization: identical key sequence to the previous version ---
        self.key = _next_key(seed)
        self.key, p_key, helper_key, icgnn_key = jax.random.split(self.key, 4)
        p_init = self.p_scale * jax.random.normal(p_key, (self.num_chains, self.num_nodes), dtype=jnp.float32)
        self.p_buffer = deque(maxlen=5000)
        self.weights_buffer = deque(maxlen=5000)
        self.p_steps = 0
        self.weights_steps = 0

        helper_params, self._helper_flags = self._init_helper_network(helper_key)
        icgnn_params, self._g_flags, self._f_flags = self._init_icgnn(icgnn_key)

        positions, total = _icgnn_old_leaf_layout(
            icgnn_params, self._g_flags, self._f_flags, self._cfg_model_type()
        )

        self._cfg = _StaticConfig(
            num_nodes=self.num_nodes,
            processed_dim_all=self.processed_dim_all,
            num_chains=self.num_chains,
            sinkhorn_n_iter=self.sinkhorn_n_iter,
            model_type=self._cfg_model_type(),
            input_perm=self.input_perm,
            vi_norm=self.VI_norm,
            logit_const=float(self.logit_const),
            o_scale=float(self.o_scale),
            activation=self.activation,
            helper_flags=self._helper_flags,
            g_flags=self._g_flags,
            f_flags=self._f_flags,
            icgnn_key_positions=positions,
            icgnn_total_leaves=total,
        )

        self._state = _TrainState(
            key=self.key,
            p=p_init,
            helper=helper_params,
            logscale=self.logscale_base,
            icgnn=icgnn_params,
            p_opt=_sgmcmc_init(p_init),
            w_opt=_sgmcmc_init(icgnn_params),
            h_opt=_adam_init((helper_params, self.logscale_base)),
        )
        del self.key  # live key is carried inside _state from here on

        self.best_state = None

    # -- attribute compatibility with the previous implementation ------------

    @property
    def key(self):
        return self._state.key if hasattr(self, "_state") else self._key_tmp

    @key.setter
    def key(self, value):
        if hasattr(self, "_state"):
            self._state = self._state.replace(key=value)
        else:
            self._key_tmp = value

    @key.deleter
    def key(self):
        if hasattr(self, "_key_tmp"):
            del self._key_tmp

    @property
    def p(self):
        return self._state.p

    @property
    def helper_params(self):
        return self._state.helper

    @property
    def icgnn_params(self):
        return self._state.icgnn

    @property
    def logscale_base(self):
        return self._state.logscale if hasattr(self, "_state") else self._logscale_tmp

    @logscale_base.setter
    def logscale_base(self, value):
        if hasattr(self, "_state"):
            self._state = self._state.replace(logscale=value)
        else:
            self._logscale_tmp = value

    def _cfg_model_type(self) -> str:
        raise NotImplementedError

    def _init_helper_network(self, key):
        if self.input_perm:
            hidden_size = 128
            input_dim = self.num_nodes * self.num_nodes
            normalization = True
        else:
            hidden_size = 48
            input_dim = self.num_nodes
            normalization = self.VI_norm
        # The previous implementation split self.key here for an (unused)
        # logscale key; reproduce the split to keep the key sequence intact.
        self.key, _logscale_key = jax.random.split(self.key)
        self.logscale_base = jnp.zeros((self.num_chains, self.processed_dim_all), dtype=jnp.float32)
        return _build_shared_mlp(
            input_dim=input_dim,
            output_dim=self.num_nodes * self.num_nodes,
            hidden_dims=[hidden_size, hidden_size],
            key=key,
            normalization=normalization,
            res_connection=True,
        )

    def _init_icgnn(self, key):
        raise NotImplementedError

    def _hypers(self):
        return _make_hypers(self.lambda_sparse, self.dataset_size, self.scale_noise_p, self.scale_noise)

    # -- public computational API (same signatures as before) ----------------

    def extract_icgnn_weights(self, use_param_weights: bool = False, num_particles: int = 1):
        if use_param_weights or len(self.weights_buffer) == 0:
            return self._state.icgnn
        params = [self.weights_buffer.pop() for _ in range(num_particles)]
        return _stack_chain_trees(params)

    def _layer_norm_p(self, p):
        return _layer_norm(p) if self.VI_norm else p

    def transform_adj(self, p: jax.Array):
        key, w_key = jax.random.split(self._state.key)
        self._state = self._state.replace(key=key)
        return _transform_adj_jit(self._cfg, self._state.helper, jnp.asarray(p, dtype=jnp.float32), w_key)

    def data_likelihood(self, X: jax.Array, A_samples: jax.Array, *, return_prior: bool = False):
        return _data_likelihood(
            self._cfg, self._hypers(),
            p_value=self._state.p, icgnn=self._state.icgnn, logscale=self._state.logscale,
            x=jnp.asarray(X, dtype=jnp.float32), A_samples=A_samples, return_prior=return_prior,
        )

    def compute_W_prior_entropy(self, p: jax.Array):
        key, w_key = jax.random.split(self._state.key)
        self._state = self._state.replace(key=key)
        return _w_prior_entropy(self._cfg, self._hypers(), self._state.helper, jnp.asarray(p, dtype=jnp.float32), w_key)

    # -- training -------------------------------------------------------------

    def _append_histories(self, p_hist: np.ndarray, ic_leaves: List[np.ndarray], ic_treedef):
        """Bulk-append per-step (per-chain) samples to the host deques.
        Same entries in the same order as the previous per-step appends."""
        num_steps = p_hist.shape[0]
        for s in range(num_steps):
            for i in range(self.num_chains):
                self.p_buffer.append(p_hist[s, i].copy())
        for s in range(num_steps):
            for i in range(self.num_chains):
                entry = jax.tree_util.tree_unflatten(
                    ic_treedef, [leaf[s, i].copy() for leaf in ic_leaves]
                )
                self.weights_buffer.append(entry)
        self.p_steps += num_steps
        self.weights_steps += num_steps

    def run_train(self, dataset: SimpleDataset, train_config_dict: Optional[Dict[str, Any]] = None):
        if train_config_dict is None:
            train_config_dict = {}
        data = np.asarray(dataset.train_data, dtype=np.float32)
        if train_config_dict.get("standardize_data_mean", False) or train_config_dict.get("standardize_data_std", False):
            mean = data.mean(axis=0, keepdims=True) if train_config_dict.get("standardize_data_mean", False) else 0.0
            std = data.std(axis=0, keepdims=True) if train_config_dict.get("standardize_data_std", False) else 1.0
            data = (data - mean) / np.where(std == 0, 1.0, std)
        self.train_data = jnp.asarray(data, dtype=jnp.float32)
        self.dataset_size = int(self.train_data.shape[0])

        batch_size = int(train_config_dict.get("batch_size", 128))
        max_epochs = int(train_config_dict.get("max_epochs", 100))
        best_loss = float("inf")

        cfg = self._cfg
        hyp = self._hypers()
        N = self.dataset_size
        n_full = N // batch_size
        ic_treedef = jax.tree_util.tree_structure(self._state.icgnn)

        for _step in range(max_epochs):
            indices = np.random.permutation(N)
            state = self._state
            loss_rows: List[np.ndarray] = []
            p_chunks: List[np.ndarray] = []
            ic_chunks: List[Any] = []
            first_special = self.p_steps < self.num_burnin_steps

            full_idx = indices[: n_full * batch_size].reshape(n_full, batch_size).astype(np.int32)
            start_row = 0
            if first_special and n_full > 0:
                # Only the very first batch ever runs the burn-in variant;
                # the epoch scan below therefore compiles for two lengths
                # (n_full-1 once, then n_full) — a one-time cost. Peeling a
                # batch every epoch to avoid it costs more in steady state.
                x0 = self.train_data[jnp.asarray(full_idx[0])]
                state, (l0, p0, ic0) = _train_batch_first_jit(cfg, hyp, state, x0)
                loss_rows.append(np.asarray(l0))
                p_chunks.append(np.asarray(p0))
                ic_chunks.append(ic0)
                start_row = 1
                first_special = False

            if n_full - start_row > 0:
                rest = jnp.asarray(full_idx[start_row:])
                state, (ls, ps, ics) = _run_epoch_jit(cfg, hyp, state, self.train_data, rest)
                loss_rows.extend(list(np.asarray(ls)))
                p_chunks.append(np.asarray(ps))
                ic_chunks.append(ics)

            rem = N - n_full * batch_size
            if rem > 0:
                x_r = self.train_data[jnp.asarray(indices[n_full * batch_size:].astype(np.int32))]
                fn = _train_batch_first_jit if first_special else _train_batch_jit
                state, (lr_, pr, icr) = fn(cfg, hyp, state, x_r)
                loss_rows.append(np.asarray(lr_))
                p_chunks.append(np.asarray(pr))
                ic_chunks.append(icr)

            self._state = state

            # Host-side history sync: one device->host transfer per epoch.
            p_hist = np.concatenate(p_chunks, axis=0)
            ic_leaves_per_chunk = [
                [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(chunk)] for chunk in ic_chunks
            ]
            ic_leaves = [
                np.concatenate([chunk[j] for chunk in ic_leaves_per_chunk], axis=0)
                for j in range(len(ic_leaves_per_chunk[0]))
            ]
            self._append_histories(p_hist, ic_leaves, ic_treedef)

            # Same float accumulation as the previous per-batch Python loop.
            loss_epoch = 0.0
            for row in loss_rows:
                loss_epoch += (float(row[0]) + float(row[1]) + float(row[2])) / 3.0

            if loss_epoch < best_loss:
                best_loss = loss_epoch
                self.best_state = self.state_dict()

    # -- posterior sampling ----------------------------------------------------

    def get_adj_matrix_tensor(self, samples: int = 5):
        batch_size = min(500, self.train_data.shape[0])
        num_steps = int(np.ceil(samples / self.num_chains))
        N = int(self.train_data.shape[0])
        idx = np.stack(
            [np.random.permutation(N)[:batch_size] for _ in range(num_steps)]
        ).astype(np.int32)

        state, (p_hist, ic_hist) = _sample_scan_jit(
            self._cfg, self._hypers(), self._state, self.train_data, jnp.asarray(idx)
        )
        self._state = state

        ic_treedef = jax.tree_util.tree_structure(self._state.icgnn)
        ic_leaves = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(ic_hist)]
        self._append_histories(np.asarray(p_hist), ic_leaves, ic_treedef)

        p_vec = [self.p_buffer.pop() for _ in range(samples)]
        p_eval = jnp.asarray(np.stack(p_vec), dtype=jnp.float32)
        adj_matrix = self.transform_adj(p_eval) != 0.0
        return adj_matrix, jnp.ones((samples,), dtype=bool)

    def get_adj_matrix(self, samples: int = 100, squeeze: bool = False):
        adj_matrix, is_dag = self.get_adj_matrix_tensor(samples=samples)
        if squeeze and samples == 1:
            adj_matrix = adj_matrix.squeeze(0)
        return np.asarray(adj_matrix).astype(np.float64), np.asarray(is_dag).astype(bool)

    def get_weighted_adj_matrix(self, samples: int = 100, squeeze: bool = False):
        adj_matrix, _is_dag = self.get_adj_matrix_tensor(samples=samples)
        params = self.extract_icgnn_weights(use_param_weights=False, num_particles=adj_matrix.shape[0])
        params = jax.tree_util.tree_map(jnp.asarray, params)
        weighted_adj = adj_matrix.astype(jnp.float32) * _weighted_adjacency(self._cfg, params)
        if squeeze and samples == 1:
            weighted_adj = weighted_adj.squeeze(0)
        return np.asarray(weighted_adj).astype(np.float64), params, None

    # -- persistence -----------------------------------------------------------

    def state_dict(self):
        return {
            "p": np.asarray(self._state.p),
            "helper_params": jax.tree_util.tree_map(np.asarray, self._state.helper),
            "logscale_base": np.asarray(self._state.logscale),
            "icgnn_params": jax.tree_util.tree_map(np.asarray, self._state.icgnn),
            "dataset_size": self.dataset_size,
        }

    def load_state_dict(self, state_dict):
        helper = _strip_layer_flags(state_dict["helper_params"])
        icgnn = _strip_layer_flags(state_dict["icgnn_params"])
        self._state = self._state.replace(
            p=jnp.asarray(state_dict["p"]),
            helper=jax.tree_util.tree_map(jnp.asarray, helper),
            logscale=jnp.asarray(state_dict["logscale_base"]),
            icgnn=jax.tree_util.tree_map(jnp.asarray, icgnn),
        )
        self.dataset_size = int(state_dict.get("dataset_size", self.dataset_size))

    def save(self, best: bool = False):
        filename = "best_model.pkl" if best else "model.pkl"
        state = self.best_state if best and self.best_state is not None else self.state_dict()
        with open(os.path.join(self.save_dir, filename), "wb") as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)


class BayesDAGLinear(BayesDAGBase):
    def __init__(self, model_id: str, variables: ContinuousVariables, save_dir: str, **kwargs):
        super().__init__(model_id, variables, save_dir, model_type="linear", **kwargs)

    @classmethod
    def name(cls):
        return "bayesdag_linear"

    def _cfg_model_type(self):
        return "linear"

    def _init_icgnn(self, key):
        weight_keys = jax.random.split(key, self.num_chains)
        W = jax.vmap(
            lambda subkey: 0.1 * jax.random.normal(subkey, (self.num_nodes, self.num_nodes), dtype=jnp.float32)
        )(weight_keys)
        return {"W": W}, None, None


class BayesDAGNonLinear(BayesDAGBase):
    def __init__(self, model_id: str, variables: ContinuousVariables, save_dir: str, **kwargs):
        super().__init__(model_id, variables, save_dir, model_type="nonlinear", **kwargs)

    @classmethod
    def name(cls):
        return "bayesdag_nonlinear"

    def _cfg_model_type(self):
        return "nonlinear"

    def _init_icgnn(self, key):
        # Mirrors the previous implementation exactly, including the quirk of
        # re-seating self.key from the passed icgnn key.
        self.key, emb_key, g_key, f_key, W_key = jax.random.split(key, 5)
        embedding_size = self.processed_dim_all
        hidden_dims = [self.hidden_size] * self.num_hidden_layers
        W = jax.vmap(
            lambda subkey: 0.1 * jax.random.normal(subkey, (self.num_nodes, self.num_nodes), dtype=jnp.float32)
        )(jax.random.split(W_key, self.num_chains))
        embeddings = 0.01 * jax.random.normal(
            emb_key, (self.num_chains, self.num_nodes, embedding_size), dtype=jnp.float32
        )
        g_params, g_flags = _build_chain_mlp(
            self.num_chains,
            input_dim=embedding_size + self.processed_dim_all,
            output_dim=embedding_size,
            hidden_dims=hidden_dims,
            key=g_key,
            normalization=self.norm_layers,
            res_connection=self.res_connection,
        )
        f_params, f_flags = _build_chain_mlp(
            self.num_chains,
            input_dim=embedding_size + embedding_size,
            output_dim=self.processed_dim_all,
            hidden_dims=hidden_dims,
            key=f_key,
            normalization=self.norm_layers,
            res_connection=self.res_connection,
        )
        return {"W": W, "embeddings": embeddings, "g": g_params, "f": f_params}, g_flags, f_flags


# ---------------------------------------------------------------------------
# Config-driven entry points (unchanged)
# ---------------------------------------------------------------------------

def _load_model_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _flatten_training_config(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model_hyperparams = dict(config.get("model_hyperparams", {}))
    training_hyperparams = dict(config.get("training_hyperparams", {}))
    if "batch_size" in config:
        training_hyperparams = {
            "batch_size": config.get("batch_size"),
            "max_epochs": config.get("max_epochs"),
            "standardize_data_mean": config.get("stardardize_data_mean", False),
            "standardize_data_std": config.get("stardardize_data_std", False),
        }
    return model_hyperparams, training_hyperparams


def train_from_config_dict(
    train_data: np.ndarray,
    *,
    model_type: str,
    model_config_dict: Dict[str, Any],
    save_dir: str,
    adjacency: Optional[np.ndarray] = None,
    seed: int = 0,
):
    config = json.loads(json.dumps(model_config_dict))
    model_hyperparams, training_hyperparams = _flatten_training_config(config)
    model_seed = _normalize_seed_value(model_hyperparams.pop("random_seed", None), seed)
    variables = ContinuousVariables.from_num_nodes(train_data.shape[1])
    dataset = SimpleDataset(train_data=np.asarray(train_data, dtype=np.float32), adjacency=adjacency)

    if model_type == "bayesdag_linear":
        model = BayesDAGLinear("bayesdag_linear", variables, save_dir, seed=model_seed, **model_hyperparams)
    elif model_type == "bayesdag_nonlinear":
        model = BayesDAGNonLinear("bayesdag_nonlinear", variables, save_dir, seed=model_seed, **model_hyperparams)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.run_train(dataset, training_hyperparams)
    model.save(best=True)
    return model


def train_from_config(
    train_data: np.ndarray,
    *,
    model_type: str,
    model_config_path: str,
    save_dir: str,
    adjacency: Optional[np.ndarray] = None,
    seed: int = 0,
):
    config = _load_model_config(model_config_path)
    return train_from_config_dict(
        train_data,
        model_type=model_type,
        model_config_dict=config,
        save_dir=save_dir,
        adjacency=adjacency,
        seed=seed,
    )
