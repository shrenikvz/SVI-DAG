import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

from src.jax_utils import torch_linear_bias_init, torch_linear_weight_init


def linear_sequential(input_dims, hidden_dims, output_dim, k_lipschitz=None, p_drop=None, *, key):
    if k_lipschitz is not None:
        raise NotImplementedError("SpectralLinear is not part of the original vi-dp-dag training path.")
    if p_drop is not None:
        raise NotImplementedError("Dropout is not used in the original vi-dp-dag training path.")

    dims = [int(np.prod(input_dims))] + list(hidden_dims) + [int(output_dim)]
    params = []
    keys = jax.random.split(key, len(dims) - 1)
    for layer_index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        weight_key, bias_key = jax.random.split(keys[layer_index])
        params.append(
            {
                "weight": torch_linear_weight_init(weight_key, out_dim, in_dim),
                "bias": torch_linear_bias_init(bias_key, out_dim, in_dim),
            }
        )
    return params


def apply_linear_sequential(params, inputs):
    outputs = inputs.reshape(inputs.shape[0], -1)
    for layer_index, layer_params in enumerate(params):
        outputs = outputs @ layer_params["weight"].T + layer_params["bias"]
        if layer_index < len(params) - 1:
            outputs = jnn.relu(outputs)
    return outputs

