import numpy as np
import jax
import jax.numpy as jnp

from src.architectures.linear_sequential import apply_linear_sequential, linear_sequential
from src.architectures.parallel_linear_sequential import (
    apply_parallel_linear_sequential,
    parallel_linear_sequential,
)
from src.jax_utils import adam_init, adam_update


class MaskedAutoencoder:
    def __init__(self, input_dim, output_dim, hidden_dims=(64, 64, 64), architecture="linear", lr=1e-3, seed=0):
        np.random.seed(seed)

        self.mask = None
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dims = list(hidden_dims)
        self.lr = float(lr)
        self.training = True
        self.rng_key = jax.random.PRNGKey(seed)

        if architecture != "linear":
            raise NotImplementedError

        layer_keys = jax.random.split(self.rng_key, self.input_dim + 1)
        self.rng_key = layer_keys[0]
        self.params = [
            linear_sequential(
                input_dims=self.input_dim,
                hidden_dims=self.hidden_dims,
                output_dim=self.output_dim,
                k_lipschitz=None,
                key=layer_keys[i + 1],
            )
            for i in range(self.input_dim)
        ]
        self.optimizer_state = adam_init(self.params)

    def to(self, *_args, **_kwargs):
        return self

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def state_dict(self):
        return {
            "params": self.params,
            "optimizer_state": self.optimizer_state,
            "mask": self.mask,
            "rng_key": self.rng_key,
        }

    def load_state_dict(self, state_dict):
        self.params = jax.tree_util.tree_map(jnp.asarray, state_dict["params"])
        if "optimizer_state" in state_dict:
            self.optimizer_state = jax.tree_util.tree_map(jnp.asarray, state_dict["optimizer_state"])
        self.mask = None if state_dict.get("mask") is None else jnp.asarray(state_dict["mask"])
        if "rng_key" in state_dict:
            self.rng_key = jnp.asarray(state_dict["rng_key"])
        return self

    def apply_gradients(self, grads):
        self.params, self.optimizer_state = adam_update(
            self.params,
            grads,
            self.optimizer_state,
            lr=self.lr,
        )

    def apply(self, X, *, params=None, mask=None):
        active_params = self.params if params is None else params
        active_mask = self.mask if mask is None else mask
        masked_duplicate_X = jnp.broadcast_to(X[:, None, :], (X.shape[0], self.input_dim, X.shape[1])) * active_mask.T[None, :, :]
        pred_X = []
        for i in range(self.input_dim):
            pred_X.append(apply_linear_sequential(active_params[i], masked_duplicate_X[:, i, :])[:, 0])
        return jnp.stack(pred_X, axis=1)

    def forward(self, X):
        return self.apply(jnp.asarray(X, dtype=jnp.float32))

    __call__ = forward


class MaskedAutoencoderNoise(MaskedAutoencoder):
    def __init__(self, input_dim, output_dim, hidden_dims=(64, 64, 64), architecture="linear", lr=1e-3, seed=0):
        np.random.seed(seed)

        self.mask = None
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dims = list(hidden_dims)
        self.lr = float(lr)
        self.training = True
        self.rng_key = jax.random.PRNGKey(seed)

        if architecture != "linear":
            raise NotImplementedError

        layer_keys = jax.random.split(self.rng_key, self.input_dim + 1)
        self.rng_key = layer_keys[0]
        self.params = [
            linear_sequential(
                input_dims=2 * self.input_dim,
                hidden_dims=self.hidden_dims,
                output_dim=self.output_dim,
                k_lipschitz=None,
                key=layer_keys[i + 1],
            )
            for i in range(self.input_dim)
        ]
        self.optimizer_state = adam_init(self.params)

    def apply(self, X, *, params=None, mask=None, Z=None):
        active_params = self.params if params is None else params
        active_mask = self.mask if mask is None else mask
        if Z is None:
            raise ValueError("Z is required for MaskedAutoencoderNoise.")
        masked_duplicate_X = jnp.broadcast_to(X[:, None, :], (X.shape[0], self.input_dim, X.shape[1])) * active_mask.T[None, :, :]
        masked_duplicate_Z = jnp.broadcast_to(Z[:, None, :], (Z.shape[0], self.input_dim, Z.shape[1])) * jnp.eye(self.input_dim)[None, :, :]
        pred_X = []
        for i in range(self.input_dim):
            inputs = jnp.concatenate((masked_duplicate_X[:, i, :], masked_duplicate_Z[:, i, :]), axis=1)
            pred_X.append(apply_linear_sequential(active_params[i], inputs)[:, 0])
        return jnp.stack(pred_X, axis=1)

    def forward(self, X, Z):
        return self.apply(jnp.asarray(X, dtype=jnp.float32), Z=jnp.asarray(Z, dtype=jnp.float32))

    __call__ = forward


class MaskedAutoencoderFast:
    def __init__(self, input_dim, output_dim, hidden_dims=(64, 64, 64), architecture="linear", lr=1e-3, seed=0):
        np.random.seed(seed)

        self.mask = None
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dims = list(hidden_dims)
        self.lr = float(lr)
        self.training = True
        self.rng_key = jax.random.PRNGKey(seed)

        if architecture != "linear":
            raise NotImplementedError

        self.rng_key, layer_key = jax.random.split(self.rng_key)
        self.params = parallel_linear_sequential(
            input_dims=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            parallel_dim=self.input_dim,
            key=layer_key,
        )
        self.optimizer_state = adam_init(self.params)

    def to(self, *_args, **_kwargs):
        return self

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def state_dict(self):
        return {
            "params": self.params,
            "optimizer_state": self.optimizer_state,
            "mask": self.mask,
            "rng_key": self.rng_key,
        }

    def load_state_dict(self, state_dict):
        self.params = jax.tree_util.tree_map(jnp.asarray, state_dict["params"])
        if "optimizer_state" in state_dict:
            self.optimizer_state = jax.tree_util.tree_map(jnp.asarray, state_dict["optimizer_state"])
        self.mask = None if state_dict.get("mask") is None else jnp.asarray(state_dict["mask"])
        if "rng_key" in state_dict:
            self.rng_key = jnp.asarray(state_dict["rng_key"])
        return self

    def apply_gradients(self, grads):
        self.params, self.optimizer_state = adam_update(
            self.params,
            grads,
            self.optimizer_state,
            lr=self.lr,
        )

    def apply(self, X, *, params=None, mask=None):
        active_params = self.params if params is None else params
        active_mask = self.mask if mask is None else mask
        masked_duplicate_X = jnp.broadcast_to(X[:, None, :], (X.shape[0], self.input_dim, X.shape[1])) * active_mask.T[None, :, :]
        masked_duplicate_X = jnp.swapaxes(masked_duplicate_X, 0, 1)
        pred_X = apply_parallel_linear_sequential(active_params, masked_duplicate_X)
        return jnp.swapaxes(pred_X, 0, 1)[..., 0]

    def forward(self, X):
        return self.apply(jnp.asarray(X, dtype=jnp.float32))

    __call__ = forward

