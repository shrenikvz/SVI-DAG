from scipy.optimize import linear_sum_assignment
import numpy as np
import jax
from jax.nn import one_hot
from jax import vmap

# ---------------------------------------------------------------------------
# Callback compatibility shim.
#
# BCD originally used ``jax.experimental.host_callback.call`` to invoke scipy's
# Hungarian-assignment solver from inside a jitted computation.  That module
# was deprecated in March 2024 and removed from modern JAX (import now raises
# ``NotImplementedError``).  The drop-in replacement is ``jax.pure_callback``,
# which has almost identical semantics for this deterministic, side-effect-
# free use-case.  We expose ``_host_call(fn, x, result_shape)`` that matches
# the legacy interface so the rest of the file need not change.
# ---------------------------------------------------------------------------
def _host_call(fn, x, result_shape):
    try:
        return jax.pure_callback(fn, result_shape, x)
    except TypeError:
        # Older jax builds where pure_callback's signature flipped -- try both.
        return jax.pure_callback(fn, result_shape, x, vectorized=False)
# The optional C extension `c_modules.mine` (Cython) provides a parallel
# Hungarian solver. It is only used by `parlinear_assignment` below, which
# is not on the active code path (`hungarian_callback_loop` uses the
# pure-scipy `multiple_linear_assignment`). Make the import optional so the
# module still loads when the extension has not been compiled.
try:
    from c_modules.mine import compute_parallel  # type: ignore
except Exception:  # pragma: no cover - optional native dependency
    compute_parallel = None
from jax import numpy as jnp


def multiple_linear_assignment(X):
    # X has shape (batch_size, k, k)
    X_out = np.zeros(X.shape[:-1], dtype=int)
    for i, x in enumerate(X):
        X_out[i] = linear_sum_assignment(x)[1]
    return X_out


def parlinear_assignment(X):
    if compute_parallel is None:
        raise ImportError(
            "c_modules.mine is not built; fall back to multiple_linear_assignment."
        )
    return compute_parallel(X)


# def hungarian_callback_loop(X):
#     # This will send a batch of matrices of size (n, k, k)
#     # to the host and returns an (n, k) set of indices
#     # of the locations of assignments in the rows of the matrices
#     return host_callback.call(
#         parlinear_assignment,
#         X,
#         result_shape=jax.ShapeDtypeStruct(X.shape[:-1], jnp.int32),
#     )


def hungarian_callback_loop(X):
    # This will send a batch of matrices of size (n, k, k)
    # to the host and returns an (n, k) set of indices
    # of the locations of assignments in the rows of the matrices
    return _host_call(
        multiple_linear_assignment,
        # parlinear_assignment,
        X,
        result_shape=jax.ShapeDtypeStruct(X.shape[:-1], jnp.int64),
    )


# def batched_hungarian(X):
#     # Input is (n, k, k)
#     n = X.shape[-1]
#     indices = hungarian_callback_loop(X)
#     return vmap(one_hot, in_axes=(0, None))(indices, n)


def batched_hungarian(X):
    # Input is (n, k, k)
    n = X.shape[-1]
    # indices = hungarian_callback(X)
    indices = hungarian_callback_loop(X)
    return vmap(one_hot, in_axes=(0, None))(indices, n)


def linear_sum_assignment_wrapper(X):
    return linear_sum_assignment(X)[1]


def hungarian(X):
    # Probably more efficient to use the batched version of this generally
    indices = _host_call(
        linear_sum_assignment_wrapper,
        X,
        result_shape=jax.ShapeDtypeStruct(X.shape[:-1], int),
    )
    return one_hot(indices, X.shape[0])
