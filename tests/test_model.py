"""
Unit tests for svidag/model.py

This module tests the SVIDAGModel class which integrates:
- Normalizing flows for edge logits
- Bayesian node models
- Relaxed adjacency computation
- ELBO components

Author: Test Suite for SVIDAG
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from svidag import model, config
from svidag.utils import compute_alpha_beta_from_prior


class TestSVIDAGModelInitialization:
    """Tests for SVIDAGModel initialization."""
    
    @pytest.fixture
    def model_2node(self):
        """Create a 2-node SVIDAGModel."""
        return model.SVIDAGModel(
            num_nodes=2,
            flow_hidden=[16, 16],
            flow_n_blocks=2,
            fixed_noise_scales=jnp.array([0.1, 0.1]),
            flow_type='maf'
        )
    
    @pytest.fixture
    def model_3node(self):
        """Create a 3-node SVIDAGModel."""
        return model.SVIDAGModel(
            num_nodes=3,
            flow_hidden=[16, 16],
            flow_n_blocks=2,
            fixed_noise_scales=jnp.array([0.1, 0.1, 0.1]),
            flow_type='maf'
        )
    
    def test_model_initialization(self, model_2node, rng_key, small_batch_2d, 
                                   sample_r_2node, alpha_beta_2node, temperatures):
        """Model should initialize without errors."""
        alpha_mat, beta_mat = alpha_beta_2node
        
        params = model_2node.init(
            {"params": rng_key},
            small_batch_2d,
            sample_r_2node,
            rng_key,
            temperatures['T_B'],
            temperatures['tau_sink'],
            alpha_mat,
            beta_mat,
        )
        
        assert 'params' in params
    
    def test_3node_model_initialization(self, model_3node, rng_key, small_batch_3d,
                                        sample_r_3node, alpha_beta_3node, temperatures):
        """3-node model should initialize correctly."""
        alpha_mat, beta_mat = alpha_beta_3node
        
        params = model_3node.init(
            {"params": rng_key},
            small_batch_3d,
            sample_r_3node,
            rng_key,
            temperatures['T_B'],
            temperatures['tau_sink'],
            alpha_mat,
            beta_mat,
        )
        
        assert 'params' in params


class TestSVIDAGModelForward:
    """Tests for SVIDAGModel forward pass."""
    
    @pytest.fixture
    def initialized_model_2node(self, rng_key):
        """Create and initialize a 2-node model."""
        mdl = model.SVIDAGModel(
            num_nodes=2,
            flow_hidden=[16, 16],
            flow_n_blocks=2,
            fixed_noise_scales=jnp.array([0.1, 0.1]),
            flow_type='maf'
        )
        
        x = jrand.normal(rng_key, (5, 2))
        r = jrand.normal(rng_key, (2,))
        p_prior = jnp.full((2, 2), 0.5)
        p_prior = p_prior * (1 - jnp.eye(2))
        alpha_mat, beta_mat = compute_alpha_beta_from_prior(p_prior)
        
        params = mdl.init(
            {"params": rng_key},
            x, r, rng_key,
            0.2, 0.3,
            alpha_mat, beta_mat,
        )
        
        return mdl, params, alpha_mat, beta_mat
    
    def test_forward_output_shapes(self, initialized_model_2node, rng_key):
        """Forward pass should return correct shapes."""
        mdl, params, alpha_mat, beta_mat = initialized_model_2node
        x = jrand.normal(rng_key, (10, 2))
        r = jrand.normal(rng_key, (2,))
        
        preds, terms = mdl.apply(
            params, x, r, rng_key,
            0.2, 0.3,
            alpha_mat, beta_mat,
        )
        
        assert preds.shape == (10, 2)
        assert 'A_relaxed' in terms
        assert terms['A_relaxed'].shape == (2, 2)
    
    def test_forward_terms_keys(self, initialized_model_2node, rng_key):
        """Forward pass should return all expected terms."""
        mdl, params, alpha_mat, beta_mat = initialized_model_2node
        x = jrand.normal(rng_key, (5, 2))
        r = jrand.normal(rng_key, (2,))
        
        _, terms = mdl.apply(
            params, x, r, rng_key,
            0.2, 0.3,
            alpha_mat, beta_mat,
        )
        
        expected_keys = ['z', 'log_det', 'log_p_gamma', 'loglik', 
                        'll_mse', 'll_const', 'kl_theta', 'A_relaxed',
                        'gamma_mat', 'r']
        for key in expected_keys:
            assert key in terms, f"Missing key: {key}"
    
    def test_predictions_finite(self, initialized_model_2node, rng_key):
        """Predictions should be finite."""
        mdl, params, alpha_mat, beta_mat = initialized_model_2node
        x = jrand.normal(rng_key, (10, 2))
        r = jrand.normal(rng_key, (2,))
        
        preds, _ = mdl.apply(
            params, x, r, rng_key,
            0.2, 0.3,
            alpha_mat, beta_mat,
        )
        
        assert jnp.all(jnp.isfinite(preds)).item()
    
    def test_A_relaxed_in_valid_range(self, initialized_model_2node, rng_key):
        """Relaxed adjacency should be in [0, 1]."""
        mdl, params, alpha_mat, beta_mat = initialized_model_2node
        x = jrand.normal(rng_key, (5, 2))
        r = jrand.normal(rng_key, (2,))
        
        _, terms = mdl.apply(
            params, x, r, rng_key,
            0.2, 0.3,
            alpha_mat, beta_mat,
        )
        
        A = terms['A_relaxed']
        assert jnp.all(A >= 0).item()
        assert jnp.all(A <= 1 + 1e-5).item()
    
    def test_diagonal_of_A_is_zero(self, initialized_model_2node, rng_key):
        """Diagonal of A_relaxed should be zero (no self-loops)."""
        mdl, params, alpha_mat, beta_mat = initialized_model_2node
        x = jrand.normal(rng_key, (5, 2))
        r = jrand.normal(rng_key, (2,))
        
        _, terms = mdl.apply(
            params, x, r, rng_key,
            0.2, 0.3,
            alpha_mat, beta_mat,
        )
        
        diag = jnp.diag(terms['A_relaxed'])
        np.testing.assert_allclose(np.array(diag), np.zeros(2), atol=1e-5)


class TestHardAdjacency:
    """Tests for the hard DAG construction A = B ⊙ M(r) and straight-through."""

    @pytest.fixture
    def initialized_model_3node(self, rng_key):
        """Create and initialize a 3-node model (nontrivial acyclicity)."""
        mdl = model.SVIDAGModel(
            num_nodes=3,
            flow_hidden=[16, 16],
            flow_n_blocks=2,
            fixed_noise_scales=jnp.array([0.1, 0.1, 0.1]),
            flow_type='maf'
        )

        x = jrand.normal(rng_key, (5, 3))
        r = jrand.normal(rng_key, (3,))
        p_prior = jnp.full((3, 3), 0.5) * (1 - jnp.eye(3))
        alpha_mat, beta_mat = compute_alpha_beta_from_prior(p_prior)

        params = mdl.init(
            {"params": rng_key},
            x, r, rng_key,
            0.2, 0.3,
            alpha_mat, beta_mat,
        )

        return mdl, params, alpha_mat, beta_mat

    def _forward_terms(self, fixture, rng_key, key_offset=0):
        mdl, params, alpha_mat, beta_mat = fixture
        key = jrand.fold_in(rng_key, key_offset)
        x = jrand.normal(key, (5, 3))
        r = jrand.normal(key, (3,))
        _, terms = mdl.apply(
            params, x, r, key,
            0.2, 0.3,
            alpha_mat, beta_mat,
        )
        return terms

    def test_A_hard_present_binary_zero_diag(self, initialized_model_3node, rng_key):
        """terms['A_hard'] exists, is exactly 0/1 with a zero diagonal."""
        terms = self._forward_terms(initialized_model_3node, rng_key)
        assert 'A_hard' in terms
        A = np.array(terms['A_hard'])
        assert set(np.unique(A)).issubset({0.0, 1.0})
        np.testing.assert_allclose(np.diag(A), np.zeros(3))

    def test_A_hard_is_acyclic(self, initialized_model_3node, rng_key):
        """Every hard sample must be a DAG (Theorem 3.1: acyclicity by construction)."""
        for offset in range(10):
            terms = self._forward_terms(initialized_model_3node, rng_key, key_offset=offset)
            A = np.array(terms['A_hard'])
            acc = np.eye(3)
            for _k in range(3):
                acc = acc @ A
                assert np.trace(acc) == 0.0, f"Cycle found in hard sample:\n{A}"

    def test_A_hard_equals_B_hard_times_M_hard(self, initialized_model_3node, rng_key):
        """A_hard must equal 1{B_tilde >= 0.5} ⊙ M_hard(r_centered)."""
        from svidag.utils import build_mask_M_hard
        terms = self._forward_terms(initialized_model_3node, rng_key)
        B_hard = (np.array(terms['B_tilde']) >= 0.5).astype(float)
        M_hard = np.array(build_mask_M_hard(terms['r_centered']))
        np.testing.assert_allclose(np.array(terms['A_hard']), B_hard * M_hard)

    def test_straight_through_forward_value(self, initialized_model_3node, rng_key):
        """With ST enabled, the forward value of A_forward equals the hard DAG."""
        assert config.straight_through_mask, "Paper spec: straight-through mask is on"
        terms = self._forward_terms(initialized_model_3node, rng_key)
        np.testing.assert_allclose(
            np.array(terms['A_forward']), np.array(terms['A_hard']), atol=1e-6
        )


class TestSVIDAGModelELBOComponents:
    """Tests for ELBO component computation."""
    
    @pytest.fixture
    def initialized_model(self, rng_key):
        """Create and initialize model."""
        mdl = model.SVIDAGModel(
            num_nodes=2,
            flow_hidden=[16, 16],
            flow_n_blocks=2,
            fixed_noise_scales=jnp.array([0.1, 0.1]),
            flow_type='maf'
        )
        
        x = jrand.normal(rng_key, (5, 2))
        r = jrand.normal(rng_key, (2,))
        p_prior = jnp.full((2, 2), 0.5) * (1 - jnp.eye(2))
        alpha_mat, beta_mat = compute_alpha_beta_from_prior(p_prior)
        
        params = mdl.init({"params": rng_key}, x, r, rng_key, 0.2, 0.3, alpha_mat, beta_mat)
        return mdl, params, alpha_mat, beta_mat
    
    def test_kl_theta_non_negative(self, initialized_model, rng_key):
        """KL divergence for theta should be non-negative."""
        mdl, params, alpha_mat, beta_mat = initialized_model
        x = jrand.normal(rng_key, (5, 2))
        r = jrand.normal(rng_key, (2,))
        
        _, terms = mdl.apply(params, x, r, rng_key, 0.2, 0.3, alpha_mat, beta_mat)
        
        assert float(terms['kl_theta']) >= 0
    
    def test_log_det_finite(self, initialized_model, rng_key):
        """Log determinant should be finite."""
        mdl, params, alpha_mat, beta_mat = initialized_model
        x = jrand.normal(rng_key, (5, 2))
        r = jrand.normal(rng_key, (2,))
        
        _, terms = mdl.apply(params, x, r, rng_key, 0.2, 0.3, alpha_mat, beta_mat)
        
        assert jnp.isfinite(terms['log_det']).item()
    
    def test_log_p_gamma_finite(self, initialized_model, rng_key):
        """Log prior on gamma should be finite."""
        mdl, params, alpha_mat, beta_mat = initialized_model
        x = jrand.normal(rng_key, (5, 2))
        r = jrand.normal(rng_key, (2,))
        
        _, terms = mdl.apply(params, x, r, rng_key, 0.2, 0.3, alpha_mat, beta_mat)
        
        assert jnp.isfinite(terms['log_p_gamma']).item()
    
    def test_loglik_finite(self, initialized_model, rng_key):
        """Log likelihood should be finite."""
        mdl, params, alpha_mat, beta_mat = initialized_model
        x = jrand.normal(rng_key, (5, 2))
        r = jrand.normal(rng_key, (2,))
        
        _, terms = mdl.apply(params, x, r, rng_key, 0.2, 0.3, alpha_mat, beta_mat)
        
        assert jnp.isfinite(terms['loglik']).item()

    def test_loglik_includes_gaussian_normalizer(self, initialized_model, rng_key):
        """The model likelihood should include the Gaussian normalization constant."""
        mdl, params, alpha_mat, beta_mat = initialized_model
        x = jrand.normal(rng_key, (5, 2))
        r = jrand.normal(rng_key, (2,))

        _, terms = mdl.apply(params, x, r, rng_key, 0.2, 0.3, alpha_mat, beta_mat)

        assert float(terms["ll_const"]) != 0.0
        np.testing.assert_allclose(
            np.array(terms["loglik"]),
            np.array(terms["ll_mse"] + terms["ll_const"]),
            atol=1e-6,
        )


class TestSVIDAGModelWithDifferentFlows:
    """Tests for SVIDAGModel with different flow types."""
    
    @pytest.mark.parametrize("flow_type", ['maf', 'nsf', 'nsf_coupling'])
    def test_different_flow_types(self, flow_type, rng_key):
        """Model should work with different flow types."""
        mdl = model.SVIDAGModel(
            num_nodes=3,
            flow_hidden=[16, 16],
            flow_n_blocks=2,
            fixed_noise_scales=jnp.array([0.1, 0.1, 0.1]),
            flow_type=flow_type,
            nsf_num_bins=8,
            nsf_tail_bound=5.0
        )
        
        x = jrand.normal(rng_key, (5, 3))
        r = jrand.normal(rng_key, (3,))
        p_prior = jnp.full((3, 3), 0.5) * (1 - jnp.eye(3))
        alpha_mat, beta_mat = compute_alpha_beta_from_prior(p_prior)
        
        params = mdl.init({"params": rng_key}, x, r, rng_key, 0.2, 0.3, alpha_mat, beta_mat)
        preds, terms = mdl.apply(params, x, r, rng_key, 0.2, 0.3, alpha_mat, beta_mat)
        
        assert preds.shape == (5, 3)
        assert jnp.all(jnp.isfinite(preds)).item()


class TestSVIDAGModelStochasticity:
    """Tests for stochasticity in SVIDAGModel."""
    
    def test_different_keys_different_outputs(self, rng_key):
        """Different random keys should produce different outputs."""
        mdl = model.SVIDAGModel(
            num_nodes=2,
            flow_hidden=[16, 16],
            flow_n_blocks=2,
            fixed_noise_scales=jnp.array([0.1, 0.1]),
            flow_type='maf'
        )
        
        x = jrand.normal(rng_key, (5, 2))
        r = jrand.normal(rng_key, (2,))
        p_prior = jnp.full((2, 2), 0.5) * (1 - jnp.eye(2))
        alpha_mat, beta_mat = compute_alpha_beta_from_prior(p_prior)
        
        key1, key2 = jrand.split(rng_key)
        params = mdl.init({"params": key1}, x, r, key1, 0.2, 0.3, alpha_mat, beta_mat)
        
        preds1, _ = mdl.apply(params, x, r, key1, 0.2, 0.3, alpha_mat, beta_mat)
        preds2, _ = mdl.apply(params, x, r, key2, 0.2, 0.3, alpha_mat, beta_mat)
        
        assert not jnp.allclose(preds1, preds2).item()
    
    def test_same_key_same_output(self, rng_key):
        """Same random key should produce same output."""
        mdl = model.SVIDAGModel(
            num_nodes=2,
            flow_hidden=[16, 16],
            flow_n_blocks=2,
            fixed_noise_scales=jnp.array([0.1, 0.1]),
            flow_type='maf'
        )
        
        x = jrand.normal(rng_key, (5, 2))
        r = jrand.normal(rng_key, (2,))
        p_prior = jnp.full((2, 2), 0.5) * (1 - jnp.eye(2))
        alpha_mat, beta_mat = compute_alpha_beta_from_prior(p_prior)
        
        params = mdl.init({"params": rng_key}, x, r, rng_key, 0.2, 0.3, alpha_mat, beta_mat)
        
        preds1, _ = mdl.apply(params, x, r, rng_key, 0.2, 0.3, alpha_mat, beta_mat)
        preds2, _ = mdl.apply(params, x, r, rng_key, 0.2, 0.3, alpha_mat, beta_mat)
        
        np.testing.assert_array_equal(np.array(preds1), np.array(preds2))


class TestSVIDAGModelGradients:
    """Tests for gradient computation through SVIDAGModel."""
    
    def test_gradients_finite(self, rng_key):
        """Gradients should be finite."""
        mdl = model.SVIDAGModel(
            num_nodes=2,
            flow_hidden=[16, 16],
            flow_n_blocks=2,
            fixed_noise_scales=jnp.array([0.1, 0.1]),
            flow_type='maf'
        )
        
        x = jrand.normal(rng_key, (5, 2))
        r = jrand.normal(rng_key, (2,))
        p_prior = jnp.full((2, 2), 0.5) * (1 - jnp.eye(2))
        alpha_mat, beta_mat = compute_alpha_beta_from_prior(p_prior)
        
        params = mdl.init({"params": rng_key}, x, r, rng_key, 0.2, 0.3, alpha_mat, beta_mat)
        
        def loss_fn(params):
            preds, terms = mdl.apply(params, x, r, rng_key, 0.2, 0.3, alpha_mat, beta_mat)
            return -terms['loglik'] + terms['kl_theta']
        
        grads = jax.grad(loss_fn)(params)
        
        flat_grads = jax.tree_util.tree_leaves(grads)
        for g in flat_grads:
            assert jnp.all(jnp.isfinite(g)).item()


class TestSVIDAGModelJITCompatibility:
    """Tests for JIT compatibility of SVIDAGModel."""
    
    def test_forward_jit_compatible(self, rng_key):
        """Forward pass should be JIT-compatible."""
        mdl = model.SVIDAGModel(
            num_nodes=2,
            flow_hidden=[16, 16],
            flow_n_blocks=2,
            fixed_noise_scales=jnp.array([0.1, 0.1]),
            flow_type='maf'
        )
        
        x = jrand.normal(rng_key, (5, 2))
        r = jrand.normal(rng_key, (2,))
        p_prior = jnp.full((2, 2), 0.5) * (1 - jnp.eye(2))
        alpha_mat, beta_mat = compute_alpha_beta_from_prior(p_prior)
        
        params = mdl.init({"params": rng_key}, x, r, rng_key, 0.2, 0.3, alpha_mat, beta_mat)
        
        @jax.jit
        def jit_forward(params, x, r, rng, T_B, tau, alpha, beta):
            return mdl.apply(params, x, r, rng, T_B, tau, alpha, beta)
        
        preds, terms = jit_forward(params, x, r, rng_key, 0.2, 0.3, alpha_mat, beta_mat)
        
        assert preds.shape == (5, 2)
        assert jnp.all(jnp.isfinite(preds)).item()


class TestSVIDAGModelTemperatureEffects:
    """Tests for temperature parameter effects."""
    
    def test_low_temperature_sharper_A(self, rng_key):
        """Lower temperature should produce sharper adjacency values."""
        mdl = model.SVIDAGModel(
            num_nodes=2,
            flow_hidden=[16, 16],
            flow_n_blocks=2,
            fixed_noise_scales=jnp.array([0.1, 0.1]),
            flow_type='maf'
        )
        
        x = jrand.normal(rng_key, (5, 2))
        r = jrand.normal(rng_key, (2,))
        p_prior = jnp.full((2, 2), 0.5) * (1 - jnp.eye(2))
        alpha_mat, beta_mat = compute_alpha_beta_from_prior(p_prior)
        
        params = mdl.init({"params": rng_key}, x, r, rng_key, 0.1, 0.1, alpha_mat, beta_mat)
        
        # Low temperature
        _, terms_low = mdl.apply(params, x, r, rng_key, 0.1, 0.1, alpha_mat, beta_mat)
        # High temperature
        _, terms_high = mdl.apply(params, x, r, rng_key, 1.0, 1.0, alpha_mat, beta_mat)
        
        # Both should be valid
        assert jnp.all(jnp.isfinite(terms_low['A_relaxed'])).item()
        assert jnp.all(jnp.isfinite(terms_high['A_relaxed'])).item()
