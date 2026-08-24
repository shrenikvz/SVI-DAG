import math
import pickle
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import jax
import jax.numpy as jnp
import numpy as np


def torch_linear_weight_init(key, out_dim: int, in_dim: int, *, dtype=jnp.float32) -> jnp.ndarray:
    """Match PyTorch's default Linear weight init (kaiming_uniform_ with a=sqrt(5))."""
    gain = math.sqrt(2.0 / (1.0 + 5.0))
    std = gain / math.sqrt(float(in_dim))
    bound = math.sqrt(3.0) * std
    return jax.random.uniform(
        key,
        (out_dim, in_dim),
        minval=-bound,
        maxval=bound,
        dtype=dtype,
    )


def torch_linear_bias_init(
    key,
    out_dim: int,
    in_dim: int,
    *,
    prefix_shape: Sequence[int] = (),
    dtype=jnp.float32,
) -> jnp.ndarray:
    bound = 1.0 / math.sqrt(float(in_dim))
    return jax.random.uniform(
        key,
        tuple(prefix_shape) + (out_dim,),
        minval=-bound,
        maxval=bound,
        dtype=dtype,
    )


@jax.tree_util.register_pytree_node_class
@dataclass
class AdamState:
    step: jnp.ndarray
    m: object
    v: object

    def tree_flatten(self):
        return (self.step, self.m, self.v), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        step, m, v = children
        return cls(step=step, m=m, v=v)


def adam_init(params) -> AdamState:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return AdamState(step=jnp.asarray(0, dtype=jnp.int32), m=zeros, v=zeros)


def adam_update(
    params,
    grads,
    state: AdamState,
    *,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
):
    step = state.step + 1
    m = jax.tree_util.tree_map(lambda m_i, g_i: beta1 * m_i + (1.0 - beta1) * g_i, state.m, grads)
    v = jax.tree_util.tree_map(lambda v_i, g_i: beta2 * v_i + (1.0 - beta2) * (g_i * g_i), state.v, grads)

    bias_correction1 = 1.0 - beta1 ** step.astype(jnp.float32)
    bias_correction2 = 1.0 - beta2 ** step.astype(jnp.float32)
    m_hat = jax.tree_util.tree_map(lambda x: x / bias_correction1, m)
    v_hat = jax.tree_util.tree_map(lambda x: x / bias_correction2, v)
    new_params = jax.tree_util.tree_map(
        lambda p, m_i, v_i: p - lr * m_i / (jnp.sqrt(v_i) + eps),
        params,
        m_hat,
        v_hat,
    )
    return new_params, AdamState(step=step, m=m, v=v)


class ArrayDataLoader:
    """Small JAX-friendly replacement for the original torch DataLoader usage."""

    def __init__(
        self,
        array,
        *,
        indices: Iterable[int],
        batch_size: int,
        shuffle: bool,
        seed: int,
    ):
        self.array = jnp.asarray(array, dtype=jnp.float32)
        self.indices = np.asarray(list(indices), dtype=np.int32)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0

    def __iter__(self) -> Iterator[jnp.ndarray]:
        indices = self.indices.copy()
        if self.shuffle:
            rng = np.random.RandomState(self.seed + self._epoch)
            rng.shuffle(indices)
        self._epoch += 1

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            yield self.array[jnp.asarray(batch_indices)]

    def __len__(self) -> int:
        if self.batch_size <= 0:
            return 0
        return int(math.ceil(len(self.indices) / float(self.batch_size)))


def tree_to_numpy(tree):
    def _convert(value):
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            return value
        return np.asarray(value)

    return jax.tree_util.tree_map(_convert, tree)


def save_checkpoint(path: str, payload) -> None:
    with open(path, "wb") as handle:
        pickle.dump(tree_to_numpy(payload), handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path: str):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    return payload
