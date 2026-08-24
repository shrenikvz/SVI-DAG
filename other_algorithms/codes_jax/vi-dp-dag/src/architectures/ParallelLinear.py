from dataclasses import dataclass

import jax
import jax.numpy as jnp

from src.jax_utils import torch_linear_bias_init, torch_linear_weight_init


@dataclass
class ParallelLinear:
    input_dim: int
    output_dim: int
    parallel_dim: int
    params: dict

    @classmethod
    def create(cls, input_dim: int, output_dim: int, parallel_dim: int, *, key):
        weight_key, bias_key = jax.random.split(key)
        params = {
            "weight": jax.vmap(
                lambda subkey: torch_linear_weight_init(subkey, output_dim, input_dim)
            )(jax.random.split(weight_key, parallel_dim)),
            "bias": torch_linear_bias_init(
                bias_key,
                output_dim,
                input_dim,
                prefix_shape=(parallel_dim, 1),
            ),
        }
        return cls(input_dim=input_dim, output_dim=output_dim, parallel_dim=parallel_dim, params=params)

    def forward(self, inputs, params=None):
        active_params = self.params if params is None else params
        outputs = jnp.matmul(inputs, jnp.swapaxes(active_params["weight"], -1, -2))
        return outputs + active_params["bias"]

    __call__ = forward

