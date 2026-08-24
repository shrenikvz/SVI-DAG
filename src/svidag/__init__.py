"""
SVIDAG: A Structural Variational Inference Approach to Causal Discovery

Author: Shrenik Zinage

This module contains the main SVIDAG implementation.

It includes the following submodules:
- config: configuration module
- data: data module
- eval: evaluation module
- flows: normalizing flows module
- guides: order potential guide module
- model: model module
- plots: plotting module
- runner: runner module
- train: training module
- utils: utility module
"""

import jax

# Preserve original JAX configuration
jax.config.update("jax_debug_nans", False)

__all__ = [
    "config",
]

