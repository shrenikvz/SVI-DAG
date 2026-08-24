import jax
import jax.nn as jnn
import numpy as np

from src.jax_utils import torch_linear_bias_init, torch_linear_weight_init


def parallel_linear_sequential(input_dims, hidden_dims, output_dim, parallel_dim, p_drop=None, *, key):
    if p_drop is not None:
        raise NotImplementedError("Dropout is not used in the original vi-dp-dag training path.")

    dims = [int(np.prod(input_dims))] + list(hidden_dims) + [int(output_dim)]
    params = []
    keys = jax.random.split(key, len(dims) - 1)
    for layer_index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        weight_key, bias_key = jax.random.split(keys[layer_index])
        weight_keys = jax.random.split(weight_key, parallel_dim)
        params.append(
            {
                "weight": jax.vmap(
                    lambda subkey: torch_linear_weight_init(subkey, out_dim, in_dim)
                )(weight_keys),
                "bias": torch_linear_bias_init(
                    bias_key,
                    out_dim,
                    in_dim,
                    prefix_shape=(parallel_dim, 1),
                ),
            }
        )
    return params


def apply_parallel_linear_sequential(params, inputs):
    outputs = inputs
    for layer_index, layer_params in enumerate(params):
        outputs = outputs @ jax.numpy.swapaxes(layer_params["weight"], -1, -2) + layer_params["bias"]
        if layer_index < len(params) - 1:
            outputs = jnn.relu(outputs)
    return outputs

