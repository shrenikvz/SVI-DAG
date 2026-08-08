"""
Shared pytest fixtures for SVIDAG test suite.

This module provides common fixtures used across multiple test modules,
including random key generation, sample data, and model configurations.

Author: Test Suite for SVIDAG
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
import sys
from pathlib import Path

# Add source directory to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from svidag import config


# =============================================================================
# Random Key Fixtures
# =============================================================================

@pytest.fixture
def rng_key():
    """Provides a reproducible JAX PRNG key for tests."""
    return jrand.PRNGKey(42)


@pytest.fixture
def rng_key_alt():
    """Provides an alternative PRNG key for tests requiring multiple keys."""
    return jrand.PRNGKey(123)


@pytest.fixture
def numpy_rng():
    """Provides a reproducible NumPy random generator."""
    return np.random.default_rng(42)


# =============================================================================
# Data Fixtures
# =============================================================================

@pytest.fixture
def small_batch_2d():
    """Small 2D batch data for quick tests. Shape: (10, 2)"""
    return jnp.array(np.random.default_rng(42).normal(size=(10, 2)).astype(np.float32))


@pytest.fixture
def small_batch_3d():
    """Small 3D batch data for tests. Shape: (10, 3)"""
    return jnp.array(np.random.default_rng(42).normal(size=(10, 3)).astype(np.float32))


@pytest.fixture
def medium_batch_2d():
    """Medium 2D batch data. Shape: (100, 2)"""
    return jnp.array(np.random.default_rng(42).normal(size=(100, 2)).astype(np.float32))


@pytest.fixture
def sample_obs_data():
    """Sample observational data for 2-node DAG. Shape: (500, 2)"""
    rng = np.random.default_rng(42)
    x1 = rng.normal(0, 0.5, size=500)
    x2 = 2.0 * x1 + rng.normal(0, 0.1, size=500)
    return np.stack([x1, x2], axis=1).astype(np.float32)


@pytest.fixture
def sample_obs_data_3node():
    """Sample observational data for 3-node DAG. Shape: (500, 3)"""
    rng = np.random.default_rng(42)
    x1 = rng.normal(0, 0.5, size=500)
    x2 = 0.5 * x1 + rng.normal(0, 0.1, size=500)
    x3 = 0.8 * x2 + rng.normal(0, 0.1, size=500)
    return np.stack([x1, x2, x3], axis=1).astype(np.float32)


# =============================================================================
# Adjacency Matrix Fixtures
# =============================================================================

@pytest.fixture
def true_adj_2node():
    """True adjacency matrix for 2-node DAG (x1 -> x2).
    Convention: A[i,j]=1 means j->i (column causes row).
    """
    adj = np.zeros((2, 2), dtype=np.float32)
    adj[1, 0] = 1.0  # x1 -> x2
    return adj


@pytest.fixture
def true_adj_3node():
    """True adjacency matrix for 3-node chain DAG (x1 -> x2 -> x3).
    Convention: A[i,j]=1 means j->i.
    """
    adj = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=np.float32)
    return adj


@pytest.fixture
def true_adj_3node_vstruct():
    """True adjacency for v-structure: x1 -> x3 <- x2."""
    adj = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 0]
    ], dtype=np.float32)
    return adj


@pytest.fixture
def empty_adj_2node():
    """Empty adjacency matrix (no edges)."""
    return np.zeros((2, 2), dtype=np.float32)


@pytest.fixture
def full_adj_2node():
    """Full adjacency matrix (all off-diagonal edges)."""
    adj = np.ones((2, 2), dtype=np.float32)
    np.fill_diagonal(adj, 0)
    return adj


# =============================================================================
# Prior Matrix Fixtures
# =============================================================================

@pytest.fixture
def uniform_prior_2node():
    """Uniform (non-informative) prior matrix for 2 nodes."""
    prior = np.full((2, 2), 0.5, dtype=np.float32)
    np.fill_diagonal(prior, 0)
    return jnp.array(prior)


@pytest.fixture
def uniform_prior_3node():
    """Uniform (non-informative) prior matrix for 3 nodes."""
    prior = np.full((3, 3), 0.5, dtype=np.float32)
    np.fill_diagonal(prior, 0)
    return jnp.array(prior)


@pytest.fixture
def strong_correct_prior_2node(true_adj_2node):
    """Strong correct prior for 2-node DAG."""
    true = jnp.array(true_adj_2node)
    p_prior = true * 0.99 + (1 - true) * 0.01
    diag_mask = jnp.eye(2)
    return p_prior * (1 - diag_mask)


# =============================================================================
# Model Configuration Fixtures
# =============================================================================

@pytest.fixture
def model_config_2node():
    """Model configuration for 2-node tests."""
    return {
        'num_nodes': 2,
        'flow_hidden': [16, 16],
        'flow_n_blocks': 2,
        'hidden_dim': 16,
        'flow_type': 'maf',
    }


@pytest.fixture
def model_config_3node():
    """Model configuration for 3-node tests."""
    return {
        'num_nodes': 3,
        'flow_hidden': [16, 16],
        'flow_n_blocks': 2,
        'hidden_dim': 16,
        'flow_type': 'maf',
    }


@pytest.fixture
def noise_scales_2node():
    """Fixed noise scales for 2-node model."""
    return jnp.array([0.1, 0.1], dtype=jnp.float32)


@pytest.fixture
def noise_scales_3node():
    """Fixed noise scales for 3-node model."""
    return jnp.array([0.1, 0.1, 0.1], dtype=jnp.float32)


# =============================================================================
# Temperature and Annealing Fixtures
# =============================================================================

@pytest.fixture
def temperatures():
    """Standard temperature values for testing."""
    return {
        'T_B': 0.2,
        'tau_sink': 0.3,
    }


@pytest.fixture
def alpha_beta_2node(uniform_prior_2node):
    """Alpha and beta matrices for 2-node uniform prior."""
    from svidag.utils import compute_alpha_beta_from_prior
    return compute_alpha_beta_from_prior(uniform_prior_2node)


@pytest.fixture
def alpha_beta_3node(uniform_prior_3node):
    """Alpha and beta matrices for 3-node uniform prior."""
    from svidag.utils import compute_alpha_beta_from_prior
    return compute_alpha_beta_from_prior(uniform_prior_3node)


# =============================================================================
# Order Potential (r) Fixtures
# =============================================================================

@pytest.fixture
def sample_r_2node(rng_key):
    """Sample order potential vector for 2 nodes."""
    return jrand.normal(rng_key, (2,))


@pytest.fixture
def sample_r_3node(rng_key):
    """Sample order potential vector for 3 nodes."""
    return jrand.normal(rng_key, (3,))


@pytest.fixture
def particles_2node(rng_key):
    """Sample particles for 2-node SVGD. Shape: (20, 2)"""
    return jrand.normal(rng_key, (20, 2))


@pytest.fixture
def particles_3node(rng_key):
    """Sample particles for 3-node SVGD. Shape: (20, 3)"""
    return jrand.normal(rng_key, (20, 3))


# =============================================================================
# Flow-related Fixtures
# =============================================================================

@pytest.fixture
def gamma_flat_2node(rng_key):
    """Sample gamma logits for 2 nodes (2 off-diagonal elements)."""
    return jrand.normal(rng_key, (2,))


@pytest.fixture
def gamma_flat_3node(rng_key):
    """Sample gamma logits for 3 nodes (6 off-diagonal elements)."""
    return jrand.normal(rng_key, (6,))


# =============================================================================
# Helper Functions as Fixtures
# =============================================================================

@pytest.fixture
def assert_allclose():
    """Fixture providing a tolerance-aware array comparison function."""
    def _assert_allclose(actual, expected, rtol=1e-5, atol=1e-6, msg=""):
        actual_np = np.asarray(actual)
        expected_np = np.asarray(expected)
        np.testing.assert_allclose(actual_np, expected_np, rtol=rtol, atol=atol, err_msg=msg)
    return _assert_allclose


@pytest.fixture
def assert_shape():
    """Fixture providing a shape assertion function."""
    def _assert_shape(array, expected_shape, msg=""):
        actual_shape = np.asarray(array).shape
        assert actual_shape == expected_shape, f"{msg} Expected shape {expected_shape}, got {actual_shape}"
    return _assert_shape


# =============================================================================
# Device Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def jax_device():
    """Returns the default JAX device (cached per session)."""
    return jax.devices()[0]


# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def clear_jax_cache():
    """Clears JAX compilation cache between tests to prevent memory buildup."""
    yield
    jax.clear_caches()
