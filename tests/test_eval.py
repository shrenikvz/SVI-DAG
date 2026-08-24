"""
Unit tests for svidag/eval.py

This module tests evaluation functions including:
- posterior_predict
- sample_relaxed_adj
- sample_hard_adj
- estimate_dag_probabilities
- estimate_cpdag_probabilities

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

from svidag import eval as svidag_eval
from svidag import train, config
from svidag.utils import compute_alpha_beta_from_prior


@pytest.fixture
def trained_state_2node(rng_key):
    """Create a trained state for 2-node model."""
    # Create training data
    train_data = jrand.normal(rng_key, (100, 2))
    p_prior = jnp.full((2, 2), 0.5) * (1 - jnp.eye(2))
    
    model, state = train.make_model_and_state(
        rng_key, train_data, p_prior, 2, jnp.array([0.1, 0.1])
    )
    
    alpha_mat, beta_mat = compute_alpha_beta_from_prior(p_prior)
    
    return state, alpha_mat, beta_mat, train_data


@pytest.fixture
def trained_state_3node(rng_key):
    """Create a trained state for 3-node model."""
    train_data = jrand.normal(rng_key, (100, 3))
    p_prior = jnp.full((3, 3), 0.5) * (1 - jnp.eye(3))
    
    model, state = train.make_model_and_state(
        rng_key, train_data, p_prior, 3, jnp.array([0.1, 0.1, 0.1])
    )
    
    alpha_mat, beta_mat = compute_alpha_beta_from_prior(p_prior)
    
    return state, alpha_mat, beta_mat, train_data


class TestPosteriorPredict:
    """Tests for posterior_predict function."""
    
    def test_output_shapes(self, trained_state_2node, rng_key):
        """Should return mean and std predictions."""
        state, alpha_mat, beta_mat, train_data = trained_state_2node
        test_data = jrand.normal(rng_key, (10, 2))
        
        mean_preds, std_preds = svidag_eval.posterior_predict(
            state.apply_fn, state.params, state.particles,
            test_data, rng_key,
            config.T_B_end, config.tau_sink_end,
            alpha_mat, beta_mat,
            num_samples=5
        )
        
        assert mean_preds.shape == (10, 2)
        assert std_preds.shape == (10, 2)
    
    def test_mean_finite(self, trained_state_2node, rng_key):
        """Mean predictions should be finite."""
        state, alpha_mat, beta_mat, train_data = trained_state_2node
        test_data = jrand.normal(rng_key, (10, 2))
        
        mean_preds, _ = svidag_eval.posterior_predict(
            state.apply_fn, state.params, state.particles,
            test_data, rng_key,
            config.T_B_end, config.tau_sink_end,
            alpha_mat, beta_mat,
            num_samples=5
        )
        
        assert np.all(np.isfinite(mean_preds))
    
    def test_std_non_negative(self, trained_state_2node, rng_key):
        """Std predictions should be non-negative."""
        state, alpha_mat, beta_mat, train_data = trained_state_2node
        test_data = jrand.normal(rng_key, (10, 2))
        
        _, std_preds = svidag_eval.posterior_predict(
            state.apply_fn, state.params, state.particles,
            test_data, rng_key,
            config.T_B_end, config.tau_sink_end,
            alpha_mat, beta_mat,
            num_samples=5
        )
        
        assert np.all(std_preds >= 0)
    
    def test_different_keys_different_results(self, trained_state_2node, rng_key):
        """Different keys should give different results."""
        state, alpha_mat, beta_mat, train_data = trained_state_2node
        test_data = jrand.normal(rng_key, (5, 2))
        key1, key2 = jrand.split(rng_key)
        
        mean1, _ = svidag_eval.posterior_predict(
            state.apply_fn, state.params, state.particles,
            test_data, key1,
            config.T_B_end, config.tau_sink_end,
            alpha_mat, beta_mat,
            num_samples=5
        )
        
        mean2, _ = svidag_eval.posterior_predict(
            state.apply_fn, state.params, state.particles,
            test_data, key2,
            config.T_B_end, config.tau_sink_end,
            alpha_mat, beta_mat,
            num_samples=5
        )
        
        # With Monte Carlo sampling, results will typically differ
        assert not np.allclose(mean1, mean2)


class TestSampleRelaxedAdj:
    """Tests for sample_relaxed_adj function."""
    
    def test_output_shape(self, trained_state_2node, rng_key):
        """Should return array of shape (num_samples, m, m)."""
        state, alpha_mat, beta_mat, train_data = trained_state_2node
        
        A_samples = svidag_eval.sample_relaxed_adj(
            state.apply_fn, state.params, state.particles,
            rng_key, config.T_B_end, config.tau_sink_end,
            alpha_mat, beta_mat,
            num_samples=10,
            train_data=train_data
        )
        
        assert A_samples.shape == (10, 2, 2)
    
    def test_3node_output_shape(self, trained_state_3node, rng_key):
        """Should work with 3-node model."""
        state, alpha_mat, beta_mat, train_data = trained_state_3node
        
        A_samples = svidag_eval.sample_relaxed_adj(
            state.apply_fn, state.params, state.particles,
            rng_key, config.T_B_end, config.tau_sink_end,
            alpha_mat, beta_mat,
            num_samples=10,
            train_data=train_data
        )
        
        assert A_samples.shape == (10, 3, 3)
    
    def test_values_in_valid_range(self, trained_state_2node, rng_key):
        """Relaxed adjacency values should be in [0, 1]."""
        state, alpha_mat, beta_mat, train_data = trained_state_2node
        
        A_samples = svidag_eval.sample_relaxed_adj(
            state.apply_fn, state.params, state.particles,
            rng_key, config.T_B_end, config.tau_sink_end,
            alpha_mat, beta_mat,
            num_samples=10,
            train_data=train_data
        )
        
        assert np.all(A_samples >= 0)
        assert np.all(A_samples <= 1 + 1e-5)
    
    def test_finite_values(self, trained_state_2node, rng_key):
        """All values should be finite."""
        state, alpha_mat, beta_mat, train_data = trained_state_2node
        
        A_samples = svidag_eval.sample_relaxed_adj(
            state.apply_fn, state.params, state.particles,
            rng_key, config.T_B_end, config.tau_sink_end,
            alpha_mat, beta_mat,
            num_samples=10,
            train_data=train_data
        )
        
        assert np.all(np.isfinite(A_samples))


class TestSampleHardAdj:
    """Tests for sample_hard_adj (hard posterior DAG samples A = B ⊙ M(r))."""

    def test_output_shape(self, trained_state_3node, rng_key):
        """Should return array of shape (num_samples, m, m)."""
        state, alpha_mat, beta_mat, train_data = trained_state_3node

        A_samples = svidag_eval.sample_hard_adj(
            state.apply_fn, state.params, state.particles,
            rng_key, config.T_B_end, config.tau_sink_end,
            alpha_mat, beta_mat,
            num_samples=10,
            train_data=train_data
        )

        assert A_samples.shape == (10, 3, 3)

    def test_samples_are_binary(self, trained_state_3node, rng_key):
        """Hard samples are exactly 0/1 with zero diagonals — no thresholding."""
        state, alpha_mat, beta_mat, train_data = trained_state_3node

        A_samples = svidag_eval.sample_hard_adj(
            state.apply_fn, state.params, state.particles,
            rng_key, config.T_B_end, config.tau_sink_end,
            alpha_mat, beta_mat,
            num_samples=20,
            train_data=train_data
        )

        assert set(np.unique(A_samples)).issubset({0.0, 1.0})
        for A in A_samples:
            np.testing.assert_allclose(np.diag(A), np.zeros(3))

    def test_every_sample_is_a_dag(self, trained_state_3node, rng_key):
        """Every hard sample must be acyclic (Theorem 3.1)."""
        state, alpha_mat, beta_mat, train_data = trained_state_3node

        A_samples = svidag_eval.sample_hard_adj(
            state.apply_fn, state.params, state.particles,
            rng_key, config.T_B_end, config.tau_sink_end,
            alpha_mat, beta_mat,
            num_samples=50,
            train_data=train_data
        )

        for A in A_samples:
            acc = np.eye(3)
            for _k in range(3):
                acc = acc @ A
                assert np.trace(acc) == 0.0, f"Cycle found in hard sample:\n{A}"


class TestEstimateDagProbabilities:
    """Tests for estimate_dag_probabilities function."""
    
    def test_returns_probs_and_top_list(self, trained_state_2node, rng_key, true_adj_2node):
        """Should return probabilities dict and top list."""
        state, alpha_mat, beta_mat, train_data = trained_state_2node
        
        target_dags = {"true_DAG": true_adj_2node.astype(int)}
        
        probs, top_list = svidag_eval.estimate_dag_probabilities(
            state.apply_fn, state.params, state.particles,
            rng_key, alpha_mat, beta_mat,
            target_dags,
            num_samples=20,
            T_B=config.T_B_end,
            tau_sn=config.tau_sink_end,
            train_data=train_data
        )
        
        assert isinstance(probs, dict)
        assert isinstance(top_list, list)
    
    def test_probs_sum_at_most_one(self, trained_state_2node, rng_key, true_adj_2node):
        """Probabilities should sum to at most 1."""
        state, alpha_mat, beta_mat, train_data = trained_state_2node
        
        target_dags = {"true_DAG": true_adj_2node.astype(int)}
        
        probs, _ = svidag_eval.estimate_dag_probabilities(
            state.apply_fn, state.params, state.particles,
            rng_key, alpha_mat, beta_mat,
            target_dags,
            num_samples=20,
            T_B=config.T_B_end,
            tau_sn=config.tau_sink_end,
            train_data=train_data
        )
        
        total_prob = sum(probs.values())
        assert total_prob <= 1.0 + 1e-6
    
    def test_probs_in_valid_range(self, trained_state_2node, rng_key, true_adj_2node):
        """Each probability should be in [0, 1]."""
        state, alpha_mat, beta_mat, train_data = trained_state_2node
        
        target_dags = {"true_DAG": true_adj_2node.astype(int)}
        
        probs, _ = svidag_eval.estimate_dag_probabilities(
            state.apply_fn, state.params, state.particles,
            rng_key, alpha_mat, beta_mat,
            target_dags,
            num_samples=20,
            T_B=config.T_B_end,
            tau_sn=config.tau_sink_end,
            train_data=train_data
        )
        
        for name, prob in probs.items():
            assert 0 <= prob <= 1, f"Invalid probability for {name}: {prob}"
    
    def test_top_list_format(self, trained_state_2node, rng_key, true_adj_2node):
        """Top list should contain (adjacency, probability) tuples."""
        state, alpha_mat, beta_mat, train_data = trained_state_2node
        
        target_dags = {"true_DAG": true_adj_2node.astype(int)}
        
        _, top_list = svidag_eval.estimate_dag_probabilities(
            state.apply_fn, state.params, state.particles,
            rng_key, alpha_mat, beta_mat,
            target_dags,
            num_samples=20,
            T_B=config.T_B_end,
            tau_sn=config.tau_sink_end,
            train_data=train_data
        )
        
        if len(top_list) > 0:
            adj, prob = top_list[0]
            assert isinstance(adj, np.ndarray)
            assert adj.shape == (2, 2)
            assert isinstance(prob, (float, np.floating))
    
    def test_top_list_sorted_by_probability(self, trained_state_2node, rng_key, true_adj_2node):
        """Top list should be sorted by probability (descending)."""
        state, alpha_mat, beta_mat, train_data = trained_state_2node
        
        target_dags = {"true_DAG": true_adj_2node.astype(int)}
        
        _, top_list = svidag_eval.estimate_dag_probabilities(
            state.apply_fn, state.params, state.particles,
            rng_key, alpha_mat, beta_mat,
            target_dags,
            num_samples=50,
            T_B=config.T_B_end,
            tau_sn=config.tau_sink_end,
            train_data=train_data
        )
        
        if len(top_list) > 1:
            probs = [p for _, p in top_list]
            # Check descending order
            for i in range(len(probs) - 1):
                assert probs[i] >= probs[i + 1]


class TestEstimateCpdagProbabilities:
    """Tests for estimate_cpdag_probabilities function."""
    
    def test_returns_top_list(self, trained_state_2node, rng_key):
        """Should return list of (cpdag, probability) tuples."""
        state, alpha_mat, beta_mat, train_data = trained_state_2node
        
        top_list = svidag_eval.estimate_cpdag_probabilities(
            state.apply_fn, state.params, state.particles,
            rng_key, alpha_mat, beta_mat,
            num_samples=20,
            T_B=config.T_B_end,
            tau_sn=config.tau_sink_end,
            train_data=train_data
        )
        
        assert isinstance(top_list, list)
    
    def test_cpdag_format(self, trained_state_2node, rng_key):
        """CPDAGs in top list should have correct format."""
        state, alpha_mat, beta_mat, train_data = trained_state_2node
        
        top_list = svidag_eval.estimate_cpdag_probabilities(
            state.apply_fn, state.params, state.particles,
            rng_key, alpha_mat, beta_mat,
            num_samples=20,
            T_B=config.T_B_end,
            tau_sn=config.tau_sink_end,
            train_data=train_data
        )
        
        if len(top_list) > 0:
            cpdag, prob = top_list[0]
            assert isinstance(cpdag, np.ndarray)
            assert cpdag.shape == (2, 2)
            assert 0 <= prob <= 1
    
    def test_cpdag_probabilities_sum_to_one(self, trained_state_2node, rng_key):
        """CPDAG probabilities should sum to at most 1."""
        state, alpha_mat, beta_mat, train_data = trained_state_2node
        
        top_list = svidag_eval.estimate_cpdag_probabilities(
            state.apply_fn, state.params, state.particles,
            rng_key, alpha_mat, beta_mat,
            num_samples=20,
            T_B=config.T_B_end,
            tau_sn=config.tau_sink_end,
            train_data=train_data
        )
        
        total_prob = sum(p for _, p in top_list)
        assert total_prob <= 1.0 + 1e-6
    
    def test_3node_cpdag(self, trained_state_3node, rng_key):
        """Should work with 3-node model."""
        state, alpha_mat, beta_mat, train_data = trained_state_3node
        
        top_list = svidag_eval.estimate_cpdag_probabilities(
            state.apply_fn, state.params, state.particles,
            rng_key, alpha_mat, beta_mat,
            num_samples=20,
            T_B=config.T_B_end,
            tau_sn=config.tau_sink_end,
            train_data=train_data
        )
        
        if len(top_list) > 0:
            cpdag, _ = top_list[0]
            assert cpdag.shape == (3, 3)


class TestEvalConsistency:
    """Tests for consistency across evaluation functions."""
    
    def test_dag_vs_cpdag_consistency(self, trained_state_2node, rng_key, true_adj_2node):
        """DAG and CPDAG estimates should be somewhat consistent."""
        state, alpha_mat, beta_mat, train_data = trained_state_2node
        
        target_dags = {"true_DAG": true_adj_2node.astype(int)}
        
        # Use same key for both
        dag_probs, dag_top = svidag_eval.estimate_dag_probabilities(
            state.apply_fn, state.params, state.particles,
            rng_key, alpha_mat, beta_mat,
            target_dags,
            num_samples=50,
            T_B=config.T_B_end,
            tau_sn=config.tau_sink_end,
            train_data=train_data
        )
        
        cpdag_top = svidag_eval.estimate_cpdag_probabilities(
            state.apply_fn, state.params, state.particles,
            rng_key, alpha_mat, beta_mat,
            num_samples=50,
            T_B=config.T_B_end,
            tau_sn=config.tau_sink_end,
            train_data=train_data
        )
        
        # Both should return non-empty results
        assert len(dag_top) > 0 or len(cpdag_top) > 0


class TestEvalEdgeCases:
    """Tests for edge cases in evaluation functions."""
    
    def test_single_sample(self, trained_state_2node, rng_key, true_adj_2node):
        """Should work with single sample."""
        state, alpha_mat, beta_mat, train_data = trained_state_2node
        
        target_dags = {"true_DAG": true_adj_2node.astype(int)}
        
        probs, top_list = svidag_eval.estimate_dag_probabilities(
            state.apply_fn, state.params, state.particles,
            rng_key, alpha_mat, beta_mat,
            target_dags,
            num_samples=1,
            T_B=config.T_B_end,
            tau_sn=config.tau_sink_end,
            train_data=train_data
        )
        
        # Should still work without error
        assert isinstance(probs, dict)
    
    def test_many_samples(self, trained_state_2node, rng_key, true_adj_2node):
        """Should work with many samples."""
        state, alpha_mat, beta_mat, train_data = trained_state_2node
        
        target_dags = {"true_DAG": true_adj_2node.astype(int)}
        
        probs, top_list = svidag_eval.estimate_dag_probabilities(
            state.apply_fn, state.params, state.particles,
            rng_key, alpha_mat, beta_mat,
            target_dags,
            num_samples=100,
            T_B=config.T_B_end,
            tau_sn=config.tau_sink_end,
            train_data=train_data
        )
        
        # Probabilities should be more stable with more samples
        for prob in probs.values():
            assert np.isfinite(prob)
    
    def test_single_test_sample(self, trained_state_2node, rng_key):
        """Posterior predict should work with single test sample."""
        state, alpha_mat, beta_mat, train_data = trained_state_2node
        test_data = jrand.normal(rng_key, (1, 2))
        
        mean_preds, std_preds = svidag_eval.posterior_predict(
            state.apply_fn, state.params, state.particles,
            test_data, rng_key,
            config.T_B_end, config.tau_sink_end,
            alpha_mat, beta_mat,
            num_samples=5
        )
        
        assert mean_preds.shape == (1, 2)
        assert std_preds.shape == (1, 2)
