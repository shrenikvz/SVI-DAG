import numpy as np
import jax
import jax.nn as jnn
import jax.numpy as jnp
from scipy.optimize import linear_sum_assignment


def _host_callback(fn, argument, result_shape):
    if hasattr(jax, "pure_callback"):
        return jax.pure_callback(fn, result_shape, argument)

    from jax.experimental import host_callback  # pragma: no cover

    return host_callback.call(fn, argument, result_shape=result_shape)


class SoftSort:
    def __init__(self, tau=1.0, hard=False, pow=1.0):
        self.hard = hard
        self.tau = tau
        self.pow = pow

    def forward(self, scores):
        scores = jnp.asarray(scores, dtype=jnp.float32)[..., None]
        sorted_scores = jnp.flip(jnp.sort(scores, axis=1), axis=1)
        pairwise_diff = -jnp.abs(jnp.swapaxes(scores, 1, 2) - sorted_scores) ** self.pow / self.tau
        p_hat = jnn.softmax(pairwise_diff, axis=-1)
        if self.hard:
            indices = jnp.argmax(p_hat, axis=-1)
            p_hard = jnn.one_hot(indices, p_hat.shape[-1], dtype=p_hat.dtype)
            p_hat = jax.lax.stop_gradient(p_hard - p_hat) + p_hat
        return p_hat

    __call__ = forward


class SoftSort_p1(SoftSort):
    def __init__(self, tau=1.0, hard=False):
        super().__init__(tau=tau, hard=hard, pow=1.0)


class SoftSort_p2(SoftSort):
    def __init__(self, tau=1.0, hard=False):
        super().__init__(tau=tau, hard=hard, pow=2.0)


def sample_gumbel(key, shape, eps=1e-20):
    u = jax.random.uniform(key, shape=shape, minval=0.0, maxval=1.0, dtype=jnp.float32)
    return -jnp.log(-jnp.log(u + eps) + eps)


@jax.custom_jvp
def _matching_impl(matrix_batch):
    def hungarian(x):
        x = np.asarray(x)
        if x.ndim == 2:
            x = np.reshape(x, (1, x.shape[0], x.shape[1]))
        sol = np.zeros((x.shape[0], x.shape[1]), dtype=np.int32)
        for i in range(x.shape[0]):
            sol[i, :] = linear_sum_assignment(-x[i, :])[1].astype(np.int32)
        return sol

    result_shape = jax.ShapeDtypeStruct(matrix_batch.shape[:-1], jnp.int32)
    return _host_callback(hungarian, matrix_batch, result_shape)


# Straight-through estimator: the Hungarian assignment produces int32 indices
# and is non-differentiable. Downstream code uses
# `stop_gradient(sink_hard - sink) + sink`, so the gradient signal never
# genuinely flows through `matching`. However, `jax.value_and_grad` still
# dispatches a JVP rule before `stop_gradient` masks the contribution, and
# `jax.pure_callback` has no JVP rule. Providing a zero-tangent JVP lets the
# gradient machinery pass through without actually propagating signal.
@_matching_impl.defjvp
def _matching_impl_jvp(primals, tangents):
    (x,) = primals
    del tangents  # integer output has float0 tangents; zero is the only valid value
    out = _matching_impl(x)
    # Tangent of an int32 primal must be dtype jax.dtypes.float0, not int32.
    zero_tangent = jnp.zeros(out.shape, dtype=jax.dtypes.float0)
    return out, zero_tangent


def matching(matrix_batch):
    matrix_batch = jnp.asarray(matrix_batch, dtype=jnp.float32)
    if matrix_batch.ndim == 2:
        matrix_batch = matrix_batch[None, ...]
    return _matching_impl(matrix_batch)


def sinkhorn(log_alpha, n_iters=20):
    n = log_alpha.shape[1]
    log_alpha = log_alpha.reshape(-1, n, n)
    for _ in range(n_iters):
        log_alpha = log_alpha - jnp.logsumexp(log_alpha, axis=2, keepdims=True)
        log_alpha = log_alpha - jnp.logsumexp(log_alpha, axis=1, keepdims=True)
    return jnp.exp(log_alpha)


def gumbel_sinkhorn(
    log_alpha,
    *,
    key=None,
    temp=1.0,
    n_samples=1,
    noise_factor=1.0,
    n_iters=20,
    squeeze=True,
    hard=False,
):
    n = log_alpha.shape[1]
    log_alpha = jnp.asarray(log_alpha, dtype=jnp.float32).reshape(-1, n, n)
    batch_size = log_alpha.shape[0]
    log_alpha_w_noise = jnp.tile(log_alpha, (n_samples, 1, 1))

    if noise_factor == 0:
        noise = 0.0
    else:
        if key is None:
            raise ValueError("A PRNG key is required when Sinkhorn noise is enabled.")
        noise = sample_gumbel(key, (n_samples * batch_size, n, n)) * noise_factor

    log_alpha_w_noise = (log_alpha_w_noise + noise) / temp
    sink = sinkhorn(log_alpha_w_noise, n_iters)

    if n_samples > 1 or squeeze is False:
        sink = jnp.swapaxes(sink.reshape(n_samples, batch_size, n, n), 0, 1)
        log_alpha_w_noise = jnp.swapaxes(log_alpha_w_noise.reshape(n_samples, batch_size, n, n), 0, 1)

    result = (sink, log_alpha_w_noise)

    if hard:
        if log_alpha_w_noise.ndim == 4:
            flat = jnp.swapaxes(log_alpha_w_noise, 0, 1).reshape(-1, n, n)
        else:
            flat = log_alpha_w_noise.reshape(-1, n, n)
        hard_perms = matching(flat)
        sink_hard = listperm2matperm(hard_perms).astype(sink.dtype)
        if log_alpha_w_noise.ndim == 4:
            sink_hard = jnp.swapaxes(sink_hard.reshape(n_samples, batch_size, n, n), 0, 1)
        result = (jax.lax.stop_gradient(sink_hard - sink) + sink, log_alpha_w_noise)

    return result


def listperm2matperm(listperm):
    listperm = jnp.asarray(listperm, dtype=jnp.int32)
    return jnn.one_hot(listperm, listperm.shape[1], dtype=jnp.int32)


def matperm2listperm(matperm):
    matperm = jnp.asarray(matperm)
    if matperm.ndim == 2:
        matperm = matperm[None, ...]
    return jnp.argmax(matperm, axis=2)


def invert_listperm(listperm):
    return matperm2listperm(jnp.swapaxes(listperm2matperm(listperm), 1, 2))


def permute_batch_split(batch_split, permutations):
    batch_split = jnp.asarray(batch_split)
    permutations = jnp.asarray(permutations, dtype=jnp.int32)
    gather_indices = permutations[..., None]
    return jnp.take_along_axis(batch_split, gather_indices, axis=1)

