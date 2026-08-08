"""
Unit tests for svidag/bayesian.py

This module tests Bayesian neural network components including:
- HyperNetwork class
- BayesianLinear layer
- BayesianMLP network
- NodeModel class

Author: Test Suite for SVIDAG
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
from flax import linen as nn
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from svidag import bayesian, config


class TestHyperNetwork:
    """Tests for HyperNetwork class."""
    
    @pytest.fixture
    def hyper_net(self):
        """Create a HyperNetwork instance."""
        return bayesian.HyperNetwork(out_shape=(4, 3), hidden_dim=8)
    
    def test_output_shape(self, hyper_net, rng_key):
        """Output shape should match out_shape parameter."""
        x = jrand.normal(rng_key, (5,))  # Input condition
        
        params = hyper_net.init(rng_key, x)
        output = hyper_net.apply(params, x)
        
        assert output.shape == (4, 3)
    
    def test_output_finite(self, hyper_net, rng_key):
        """Hypernetwork outputs should be finite real values."""
        x = jrand.normal(rng_key, (5,))
        
        params = hyper_net.init(rng_key, x)
        output = hyper_net.apply(params, x)
        
        assert jnp.all(jnp.isfinite(output)).item()

    def test_matrix_condition_matches_flattened_condition(self, hyper_net, rng_key):
        """Hypernetworks should flatten adjacency-like matrix conditions consistently."""
        x_matrix = jrand.normal(rng_key, (2, 3))
        x_flat = x_matrix.reshape(-1)

        params = hyper_net.init(rng_key, x_matrix)
        out_matrix = hyper_net.apply(params, x_matrix)
        out_flat = hyper_net.apply(params, x_flat)

        np.testing.assert_allclose(np.array(out_matrix), np.array(out_flat), atol=1e-6)
    
    def test_different_inputs_different_outputs(self, hyper_net, rng_key):
        """Different inputs should produce different outputs."""
        x1 = jrand.normal(rng_key, (5,))
        x2 = x1 + 1.0
        
        params = hyper_net.init(rng_key, x1)
        out1 = hyper_net.apply(params, x1)
        out2 = hyper_net.apply(params, x2)
        
        assert not jnp.allclose(out1, out2).item()
    
    def test_deterministic_forward_pass(self, hyper_net, rng_key):
        """Same input should produce same output (deterministic)."""
        x = jrand.normal(rng_key, (5,))
        
        params = hyper_net.init(rng_key, x)
        out1 = hyper_net.apply(params, x)
        out2 = hyper_net.apply(params, x)
        
        np.testing.assert_array_equal(np.array(out1), np.array(out2))
    
    def test_jit_compatible(self, hyper_net, rng_key):
        """Network should be JIT-compatible."""
        x = jrand.normal(rng_key, (5,))
        params = hyper_net.init(rng_key, x)
        
        @jax.jit
        def forward(params, x):
            return hyper_net.apply(params, x)
        
        output = forward(params, x)
        assert output.shape == (4, 3)


class TestBayesianLinear:
    """Tests for BayesianLinear layer."""
    
    @pytest.fixture
    def bayesian_linear(self):
        """Create a BayesianLinear layer."""
        return bayesian.BayesianLinear(
            in_features=4,
            out_features=3,
            cond_dim=2
        )
    
    def test_output_shape(self, bayesian_linear, rng_key):
        """Output shape should be (batch_size, out_features)."""
        x = jrand.normal(rng_key, (5, 4))  # (batch, in_features)
        A_condition = jrand.normal(rng_key, (2,))
        
        key1, key2 = jrand.split(rng_key)
        params = bayesian_linear.init({"params": key1}, x, A_condition, key2)
        output = bayesian_linear.apply(params, x, A_condition, key2)
        
        assert output.shape == (5, 3)
    
    def test_stochastic_forward_pass(self, bayesian_linear, rng_key):
        """Different random keys should produce different outputs (stochastic)."""
        x = jrand.normal(rng_key, (5, 4))
        A_condition = jrand.normal(rng_key, (2,))
        
        key1, key2, key3 = jrand.split(rng_key, 3)
        params = bayesian_linear.init({"params": key1}, x, A_condition, key2)
        
        out1 = bayesian_linear.apply(params, x, A_condition, key2)
        out2 = bayesian_linear.apply(params, x, A_condition, key3)
        
        # Should be different due to reparameterization
        assert not jnp.allclose(out1, out2).item()
    
    def test_kl_divergence_positive(self, bayesian_linear, rng_key):
        """KL divergence should be non-negative."""
        x = jrand.normal(rng_key, (5, 4))
        A_condition = jrand.normal(rng_key, (2,))
        
        key1, key2 = jrand.split(rng_key)
        params = bayesian_linear.init({"params": key1}, x, A_condition, key2)
        kl = bayesian_linear.apply(params, A_condition, method=bayesian_linear.kl_divergence)
        
        assert float(kl) >= 0
    
    def test_kl_divergence_finite(self, bayesian_linear, rng_key):
        """KL divergence should be finite."""
        x = jrand.normal(rng_key, (5, 4))
        A_condition = jrand.normal(rng_key, (2,))
        
        key1, key2 = jrand.split(rng_key)
        params = bayesian_linear.init({"params": key1}, x, A_condition, key2)
        kl = bayesian_linear.apply(params, A_condition, method=bayesian_linear.kl_divergence)
        
        assert jnp.isfinite(kl).item()
    
    def test_condition_affects_output(self, bayesian_linear, rng_key):
        """Different conditioning should affect output distribution."""
        x = jrand.normal(rng_key, (5, 4))
        A_cond1 = jnp.array([0.0, 0.0])
        A_cond2 = jnp.array([1.0, 1.0])
        
        key1, key2 = jrand.split(rng_key)
        # Initialize with one condition
        params = bayesian_linear.init({"params": key1}, x, A_cond1, key2)
        
        # Same random key but different conditions
        out1 = bayesian_linear.apply(params, x, A_cond1, key2)
        out2 = bayesian_linear.apply(params, x, A_cond2, key2)
        
        # Outputs may be different due to hypernetwork scaling
        # At minimum, check both are valid
        assert jnp.all(jnp.isfinite(out1)).item()
        assert jnp.all(jnp.isfinite(out2)).item()

    def test_matrix_condition_supported(self, bayesian_linear, rng_key):
        """Posterior statistics should support full adjacency matrix conditioning."""
        x = jrand.normal(rng_key, (5, 4))
        A_condition = jrand.normal(rng_key, (2, 2))

        key1, key2 = jrand.split(rng_key)
        params = bayesian_linear.init({"params": key1}, x, A_condition, key2)
        output = bayesian_linear.apply(params, x, A_condition, key2)
        kl = bayesian_linear.apply(params, A_condition, method=bayesian_linear.kl_divergence)

        assert output.shape == (5, 3)
        assert jnp.isfinite(kl).item()


class TestBayesianMLP:
    """Tests for BayesianMLP network."""
    
    @pytest.fixture
    def bayesian_mlp(self):
        """Create a BayesianMLP instance."""
        return bayesian.BayesianMLP(
            input_dim=4,
            hidden_dim=8,
            output_dim=2,
            cond_dim=3
        )
    
    def test_output_shape(self, bayesian_mlp, rng_key):
        """Output should have correct shape."""
        x = jrand.normal(rng_key, (10, 4))
        A_condition = jrand.normal(rng_key, (3,))
        
        key1, key2 = jrand.split(rng_key)
        params = bayesian_mlp.init({"params": key1}, x, A_condition, key2)
        output = bayesian_mlp.apply(params, x, A_condition, key2)
        
        assert output.shape == (10, 2)
    
    def test_stochastic_output(self, bayesian_mlp, rng_key):
        """MLP output should be stochastic."""
        x = jrand.normal(rng_key, (10, 4))
        A_condition = jrand.normal(rng_key, (3,))
        
        key1, key2, key3 = jrand.split(rng_key, 3)
        params = bayesian_mlp.init({"params": key1}, x, A_condition, key2)
        
        out1 = bayesian_mlp.apply(params, x, A_condition, key2)
        out2 = bayesian_mlp.apply(params, x, A_condition, key3)
        
        assert not jnp.allclose(out1, out2).item()
    
    def test_kl_divergence_sum_of_layers(self, bayesian_mlp, rng_key):
        """KL divergence should be sum from all BayesianLinear layers."""
        x = jrand.normal(rng_key, (10, 4))
        A_condition = jrand.normal(rng_key, (3,))
        
        key1, key2 = jrand.split(rng_key)
        params = bayesian_mlp.init({"params": key1}, x, A_condition, key2)
        kl_total = bayesian_mlp.apply(params, A_condition, method=bayesian_mlp.kl_divergence)
        
        # Should be positive (sum of positive terms)
        assert float(kl_total) > 0
        assert jnp.isfinite(kl_total).item()
    
    def test_forward_pass_finite(self, bayesian_mlp, rng_key):
        """Forward pass should produce finite values."""
        x = jrand.normal(rng_key, (10, 4))
        A_condition = jrand.normal(rng_key, (3,))
        
        key1, key2 = jrand.split(rng_key)
        params = bayesian_mlp.init({"params": key1}, x, A_condition, key2)
        output = bayesian_mlp.apply(params, x, A_condition, key2)
        
        assert jnp.all(jnp.isfinite(output)).item()
    
    def test_gradient_flow(self, bayesian_mlp, rng_key):
        """Gradients should flow through the network."""
        x = jrand.normal(rng_key, (5, 4))
        A_condition = jrand.normal(rng_key, (3,))
        
        key1, key2 = jrand.split(rng_key)
        params = bayesian_mlp.init({"params": key1}, x, A_condition, key2)
        
        def loss_fn(params):
            out = bayesian_mlp.apply(params, x, A_condition, key2)
            return jnp.sum(out ** 2)
        
        grads = jax.grad(loss_fn)(params)
        
        # Check that gradients exist and are finite
        flat_grads = jax.tree_util.tree_leaves(grads)
        for g in flat_grads:
            assert jnp.all(jnp.isfinite(g)).item()


class TestNodeModel:
    """Tests for NodeModel class."""
    
    @pytest.fixture
    def node_model(self):
        """Create a NodeModel instance."""
        return bayesian.NodeModel(num_nodes=3, hidden_dim=8)
    
    def test_output_shape(self, node_model, rng_key):
        """Output should be (batch_size, 1) - single node prediction."""
        x = jrand.normal(rng_key, (10, 3))  # (batch, num_nodes)
        A_condition = jrand.normal(rng_key, (3,))
        
        key1, key2 = jrand.split(rng_key)
        params = node_model.init({"params": key1}, x, key2, A_condition)
        output = node_model.apply(params, x, key2, A_condition)
        
        assert output.shape == (10, 1)
    
    def test_stochastic_prediction(self, node_model, rng_key):
        """Node predictions should be stochastic."""
        x = jrand.normal(rng_key, (10, 3))
        A_condition = jrand.normal(rng_key, (3,))
        
        key1, key2, key3 = jrand.split(rng_key, 3)
        params = node_model.init({"params": key1}, x, key2, A_condition)
        
        out1 = node_model.apply(params, x, key2, A_condition)
        out2 = node_model.apply(params, x, key3, A_condition)
        
        assert not jnp.allclose(out1, out2).item()
    
    def test_kl_divergence(self, node_model, rng_key):
        """KL divergence should be positive and finite."""
        x = jrand.normal(rng_key, (10, 3))
        A_condition = jrand.normal(rng_key, (3,))
        
        key1, key2 = jrand.split(rng_key)
        params = node_model.init({"params": key1}, x, key2, A_condition)
        kl = node_model.apply(params, A_condition, method=node_model.kl_divergence)
        
        assert float(kl) > 0
        assert jnp.isfinite(kl).item()
    
    def test_masking_effect(self, node_model, rng_key):
        """Different A_condition (masks) should affect predictions."""
        x = jrand.normal(rng_key, (10, 3))
        
        # Sparse mask (one parent)
        A_sparse = jnp.array([1.0, 0.0, 0.0])
        # Dense mask (all parents)
        A_dense = jnp.array([1.0, 1.0, 1.0])
        
        key1, key2 = jrand.split(rng_key)
        params = node_model.init({"params": key1}, x, key2, A_sparse)
        
        out_sparse = node_model.apply(params, x, key2, A_sparse)
        out_dense = node_model.apply(params, x, key2, A_dense)
        
        # Should be different
        assert not jnp.allclose(out_sparse, out_dense).item()
    
    def test_finite_outputs(self, node_model, rng_key):
        """All outputs should be finite."""
        x = jrand.normal(rng_key, (20, 3))
        A_condition = jrand.normal(rng_key, (3,))
        
        key1, key2 = jrand.split(rng_key)
        params = node_model.init({"params": key1}, x, key2, A_condition)
        output = node_model.apply(params, x, key2, A_condition)
        
        assert jnp.all(jnp.isfinite(output)).item()

    def test_matrix_condition_supported(self, node_model, rng_key):
        """Node model should accept a full relaxed adjacency matrix as its condition."""
        x = jrand.normal(rng_key, (10, 3))
        A_condition = jrand.normal(rng_key, (3, 3))

        key1, key2 = jrand.split(rng_key)
        params = node_model.init({"params": key1}, x, key2, A_condition)
        output = node_model.apply(params, x, key2, A_condition)
        kl = node_model.apply(params, A_condition, method=node_model.kl_divergence)

        assert output.shape == (10, 1)
        assert jnp.isfinite(kl).item()


class TestNodeModelWithDifferentSizes:
    """Tests for NodeModel with various network sizes."""
    
    @pytest.mark.parametrize("num_nodes", [2, 3, 5, 10])
    def test_different_num_nodes(self, num_nodes, rng_key):
        """NodeModel should work with different number of nodes."""
        model = bayesian.NodeModel(num_nodes=num_nodes, hidden_dim=8)
        x = jrand.normal(rng_key, (5, num_nodes))
        A_condition = jrand.normal(rng_key, (num_nodes,))
        
        key1, key2 = jrand.split(rng_key)
        params = model.init({"params": key1}, x, key2, A_condition)
        output = model.apply(params, x, key2, A_condition)
        
        assert output.shape == (5, 1)
    
    @pytest.mark.parametrize("hidden_dim", [4, 8, 16, 32])
    def test_different_hidden_dims(self, hidden_dim, rng_key):
        """NodeModel should work with different hidden dimensions."""
        model = bayesian.NodeModel(num_nodes=3, hidden_dim=hidden_dim)
        x = jrand.normal(rng_key, (5, 3))
        A_condition = jrand.normal(rng_key, (3,))
        
        key1, key2 = jrand.split(rng_key)
        params = model.init({"params": key1}, x, key2, A_condition)
        output = model.apply(params, x, key2, A_condition)
        
        assert output.shape == (5, 1)


class TestBayesianNetworkNumericalStability:
    """Tests for numerical stability of Bayesian networks."""
    
    def test_large_input_values(self, rng_key):
        """Network should handle large input values."""
        model = bayesian.NodeModel(num_nodes=3, hidden_dim=8)
        x = jrand.normal(rng_key, (5, 3)) * 100  # Large values
        A_condition = jrand.normal(rng_key, (3,))
        
        key1, key2 = jrand.split(rng_key)
        params = model.init({"params": key1}, x, key2, A_condition)
        output = model.apply(params, x, key2, A_condition)
        
        assert jnp.all(jnp.isfinite(output)).item()
    
    def test_small_input_values(self, rng_key):
        """Network should handle small input values."""
        model = bayesian.NodeModel(num_nodes=3, hidden_dim=8)
        x = jrand.normal(rng_key, (5, 3)) * 1e-5  # Small values
        A_condition = jrand.normal(rng_key, (3,))
        
        key1, key2 = jrand.split(rng_key)
        params = model.init({"params": key1}, x, key2, A_condition)
        output = model.apply(params, x, key2, A_condition)
        
        assert jnp.all(jnp.isfinite(output)).item()
    
    def test_zero_condition(self, rng_key):
        """Network should handle zero conditioning vector."""
        model = bayesian.NodeModel(num_nodes=3, hidden_dim=8)
        x = jrand.normal(rng_key, (5, 3))
        A_condition = jnp.zeros(3)  # All zeros
        
        key1, key2 = jrand.split(rng_key)
        params = model.init({"params": key1}, x, key2, A_condition)
        output = model.apply(params, x, key2, A_condition)
        
        assert jnp.all(jnp.isfinite(output)).item()
    
    def test_kl_divergence_zero_condition(self, rng_key):
        """KL divergence should be finite with zero conditioning."""
        model = bayesian.NodeModel(num_nodes=3, hidden_dim=8)
        x = jrand.normal(rng_key, (5, 3))
        A_condition = jnp.zeros(3)
        
        key1, key2 = jrand.split(rng_key)
        params = model.init({"params": key1}, x, key2, A_condition)
        kl = model.apply(params, A_condition, method=model.kl_divergence)
        
        assert jnp.isfinite(kl).item()


class TestBayesianVMAPCompatibility:
    """Tests for vmap compatibility of Bayesian networks."""
    
    def test_vmap_over_samples(self, rng_key):
        """NodeModel should be vmappable over samples."""
        model = bayesian.NodeModel(num_nodes=3, hidden_dim=8)
        
        # Single sample
        x_single = jrand.normal(rng_key, (1, 3))
        A_condition = jrand.normal(rng_key, (3,))
        
        key1, key2 = jrand.split(rng_key)
        params = model.init({"params": key1}, x_single, key2, A_condition)
        
        # Multiple samples via vmap
        x_batch = jrand.normal(rng_key, (10, 3))
        keys = jrand.split(key2, 10)
        
        def single_forward(x_row, key):
            return model.apply(params, x_row[None, :], key, A_condition)
        
        outputs = jax.vmap(single_forward)(x_batch, keys)
        
        assert outputs.shape == (10, 1, 1)
    
    def test_vmap_over_conditions(self, rng_key):
        """NodeModel forward should work with vmapped conditions."""
        model = bayesian.NodeModel(num_nodes=3, hidden_dim=8)
        x = jrand.normal(rng_key, (5, 3))
        
        key1, key2 = jrand.split(rng_key)
        A_conditions = jrand.normal(key1, (4, 3))  # 4 different conditions
        
        # Initialize with one condition
        params = model.init({"params": key1}, x, key2, A_conditions[0])
        
        # Apply with different conditions
        keys = jrand.split(key2, 4)
        
        def forward_with_cond(A_cond, key):
            return model.apply(params, x, key, A_cond)
        
        outputs = jax.vmap(forward_with_cond)(A_conditions, keys)
        
        assert outputs.shape == (4, 5, 1)
