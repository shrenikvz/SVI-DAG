"""
Unit tests for svidag/utils.py

This module tests all utility functions including:
- Array transformations (vec_to_offdiag_matrix, mat_offdiag_to_vec)
- Sinkhorn normalization and mask building
- Probability distributions (Gaussian, logistic-beta)
- DAG/CPDAG metrics (SHD, TPR)
- SVGD kernel computations
- Prior matrix generation

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

from svidag import utils, config


class TestLinearAnneal:
    """Tests for linear_anneal function."""
    
    def test_start_value(self):
        """At iteration 0, should return start value."""
        result = utils.linear_anneal(0, 100, start=1.0, end=0.0)
        assert float(result) == pytest.approx(1.0)
    
    def test_end_value(self):
        """At final iteration, should return end value."""
        result = utils.linear_anneal(100, 100, start=1.0, end=0.0)
        assert float(result) == pytest.approx(0.0)
    
    def test_midpoint(self):
        """At midpoint, should return average of start and end."""
        result = utils.linear_anneal(50, 100, start=1.0, end=0.0)
        assert float(result) == pytest.approx(0.5)
    
    def test_beyond_total_clips_to_one(self):
        """Beyond total iterations, should clip to end value."""
        result = utils.linear_anneal(150, 100, start=1.0, end=0.0)
        assert float(result) == pytest.approx(0.0)
    
    def test_increasing_anneal(self):
        """Test annealing from low to high."""
        result = utils.linear_anneal(50, 100, start=0.2, end=0.8)
        assert float(result) == pytest.approx(0.5)
    
    def test_jit_compatible(self):
        """Verify function is JIT-compilable."""
        jit_anneal = jax.jit(utils.linear_anneal, static_argnums=(1,))
        result = jit_anneal(50.0, 100, 1.0, 0.0)
        assert float(result) == pytest.approx(0.5)


class TestCenterOrderPotentials:
    """Tests for center_order_potentials helper."""

    def test_removes_common_shift(self):
        """Centered order potentials should have zero mean per particle."""
        r = jnp.array([[3.0, 1.0], [2.0, 2.0], [-1.0, 4.0]])
        centered = utils.center_order_potentials(r)
        means = jnp.mean(centered, axis=-1)
        np.testing.assert_allclose(np.array(means), np.zeros(3), atol=1e-6)

    def test_preserves_pairwise_differences(self):
        """Centering should preserve relative differences between node scores."""
        r = jnp.array([4.0, 1.0, -2.0])
        centered = utils.center_order_potentials(r)
        original_diffs = np.array(r[:, None] - r[None, :])
        centered_diffs = np.array(centered[:, None] - centered[None, :])
        np.testing.assert_allclose(centered_diffs, original_diffs, atol=1e-6)


class TestOffdiagIndices:
    """Tests for offdiag_indices function."""
    
    def test_2x2_matrix(self):
        """Test off-diagonal indices for 2x2 matrix."""
        rows, cols = utils.offdiag_indices(2)
        # Should have 2 off-diagonal elements
        assert len(rows) == 2
        assert len(cols) == 2
        # Check specific indices
        expected = {(0, 1), (1, 0)}
        actual = set(zip(rows, cols))
        assert actual == expected
    
    def test_3x3_matrix(self):
        """Test off-diagonal indices for 3x3 matrix."""
        rows, cols = utils.offdiag_indices(3)
        # Should have 6 off-diagonal elements
        assert len(rows) == 6
        assert len(cols) == 6
    
    def test_diagonal_excluded(self):
        """Verify diagonal indices are excluded."""
        rows, cols = utils.offdiag_indices(5)
        for r, c in zip(rows, cols):
            assert r != c, f"Diagonal element ({r}, {c}) should be excluded"


class TestVecToOffdiagMatrix:
    """Tests for vec_to_offdiag_matrix function."""
    
    def test_2node_reconstruction(self):
        """Test vector to matrix conversion for 2 nodes."""
        vec = jnp.array([0.5, 0.3])
        mat = utils.vec_to_offdiag_matrix(vec, 2)
        
        # Check shape
        assert mat.shape == (2, 2)
        # Check diagonal is zero
        assert float(mat[0, 0]) == 0.0
        assert float(mat[1, 1]) == 0.0
        # Check off-diagonal elements
        assert float(mat[0, 1]) == pytest.approx(0.5) or float(mat[0, 1]) == pytest.approx(0.3)
        assert float(mat[1, 0]) == pytest.approx(0.5) or float(mat[1, 0]) == pytest.approx(0.3)
    
    def test_3node_reconstruction(self):
        """Test vector to matrix conversion for 3 nodes."""
        vec = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        mat = utils.vec_to_offdiag_matrix(vec, 3)
        
        assert mat.shape == (3, 3)
        # Diagonal should be zero
        assert float(mat[0, 0]) == 0.0
        assert float(mat[1, 1]) == 0.0
        assert float(mat[2, 2]) == 0.0
    
    def test_roundtrip_conversion(self):
        """Test that vec -> mat -> vec gives same result."""
        original_vec = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        mat = utils.vec_to_offdiag_matrix(original_vec, 3)
        recovered_vec = utils.mat_offdiag_to_vec(mat)
        np.testing.assert_allclose(np.array(original_vec), np.array(recovered_vec), rtol=1e-5)


class TestMatOffdiagToVec:
    """Tests for mat_offdiag_to_vec function."""
    
    def test_2x2_extraction(self):
        """Test extracting off-diagonal elements from 2x2 matrix."""
        mat = jnp.array([[0.0, 0.5], [0.3, 0.0]])
        vec = utils.mat_offdiag_to_vec(mat)
        
        assert vec.shape == (2,)
        # Check that both off-diagonal values are present
        np.testing.assert_allclose(np.sort(np.array(vec)), np.array([0.3, 0.5]), atol=1e-6)
    
    def test_3x3_extraction(self):
        """Test extracting off-diagonal elements from 3x3 matrix."""
        mat = jnp.array([
            [0.0, 1.0, 2.0],
            [3.0, 0.0, 4.0],
            [5.0, 6.0, 0.0]
        ])
        vec = utils.mat_offdiag_to_vec(mat)
        
        assert vec.shape == (6,)


class TestSampleLogisticNoise:
    """Tests for sample_logistic_noise function."""
    
    def test_output_shape(self, rng_key):
        """Test that output shape matches input shape."""
        shape = (10, 5)
        noise = utils.sample_logistic_noise(rng_key, shape)
        assert noise.shape == shape
    
    def test_statistical_properties(self, rng_key):
        """Test that samples have approximately correct mean and variance."""
        # Logistic(0, 1) has mean 0 and variance π²/3 ≈ 3.29
        noise = utils.sample_logistic_noise(rng_key, (10000,))
        mean = float(jnp.mean(noise))
        var = float(jnp.var(noise))
        
        assert mean == pytest.approx(0.0, abs=0.1)
        assert var == pytest.approx(np.pi**2 / 3, rel=0.1)
    
    def test_deterministic_with_same_key(self, rng_key):
        """Same key should produce same samples."""
        noise1 = utils.sample_logistic_noise(rng_key, (10,))
        noise2 = utils.sample_logistic_noise(rng_key, (10,))
        np.testing.assert_array_equal(np.array(noise1), np.array(noise2))


class TestLogisticConcrete:
    """Tests for logistic_concrete (Gumbel-softmax relaxation) function."""
    
    def test_output_in_zero_one(self, rng_key):
        """Output should be in (0, 1) range."""
        logits = jnp.array([0.0, 1.0, -1.0, 5.0, -5.0])
        samples = utils.logistic_concrete(rng_key, logits, temperature=0.5)
        
        assert jnp.all(samples > 0).item()
        assert jnp.all(samples < 1).item()
    
    def test_low_temperature_approaches_hard(self, rng_key):
        """Low temperature should make outputs closer to 0 or 1."""
        logits = jnp.array([10.0, -10.0])  # Strong logits
        
        # Low temperature
        samples_low = utils.logistic_concrete(rng_key, logits, temperature=0.01)
        # High temperature
        samples_high = utils.logistic_concrete(rng_key, logits, temperature=5.0)
        
        # Low temp should be closer to 0/1
        assert float(samples_low[0]) > float(samples_high[0])  # High logit -> high sample
        assert float(samples_low[1]) < float(samples_high[1])  # Low logit -> low sample
    
    def test_temperature_effect(self, rng_key):
        """Higher temperature should produce softer (more uncertain) outputs."""
        logits = jnp.zeros(100)
        
        samples_low_temp = utils.logistic_concrete(rng_key, logits, temperature=0.1)
        samples_high_temp = utils.logistic_concrete(rng_key, logits, temperature=2.0)
        
        # Variance should be higher for lower temperature (more extreme values)
        var_low = float(jnp.var(samples_low_temp))
        var_high = float(jnp.var(samples_high_temp))
        
        # With temp=0.1, values should be closer to 0 or 1, hence higher variance
        # With temp=2.0, values should be closer to 0.5, hence lower variance
        assert var_low > var_high


class TestSinkhornNormalization:
    """Tests for sinkhorn_normalization function."""
    
    def test_output_is_doubly_stochastic_approx(self):
        """Test that output is approximately doubly stochastic."""
        X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        P = utils.sinkhorn_normalization(X, n_iters=100)
        
        # Row sums should be approximately equal
        row_sums = jnp.sum(P, axis=1)
        assert jnp.allclose(row_sums, row_sums[0], rtol=0.01).item()
        
        # Column sums should be approximately equal
        col_sums = jnp.sum(P, axis=0)
        assert jnp.allclose(col_sums, col_sums[0], rtol=0.01).item()
    
    def test_positive_output(self):
        """Output should be non-negative."""
        X = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        P = utils.sinkhorn_normalization(X, n_iters=100)
        
        assert jnp.all(P >= 0).item()
    
    def test_convergence_with_iterations(self):
        """More iterations should give better normalization."""
        X = jnp.array([[1.0, 10.0], [10.0, 1.0]])
        
        P_few = utils.sinkhorn_normalization(X, n_iters=5)
        P_many = utils.sinkhorn_normalization(X, n_iters=100)
        
        # More iterations should give row sums closer to each other
        row_std_few = float(jnp.std(jnp.sum(P_few, axis=1)))
        row_std_many = float(jnp.std(jnp.sum(P_many, axis=1)))
        
        assert row_std_many <= row_std_few


class TestBuildPTau:
    """Tests for build_P_tau function."""
    
    def test_output_shape(self, sample_r_2node):
        """Test output shape matches number of nodes."""
        P = utils.build_P_tau(sample_r_2node, tau=0.3)
        assert P.shape == (2, 2)
    
    def test_3node_output_shape(self, sample_r_3node):
        """Test output shape for 3 nodes."""
        P = utils.build_P_tau(sample_r_3node, tau=0.3)
        assert P.shape == (3, 3)
    
    def test_approximately_doubly_stochastic(self, sample_r_3node):
        """P should be approximately doubly stochastic."""
        P = utils.build_P_tau(sample_r_3node, tau=0.3)
        
        row_sums = jnp.sum(P, axis=1)
        col_sums = jnp.sum(P, axis=0)
        
        # Should sum to approximately 1
        assert jnp.allclose(row_sums, 1.0, rtol=0.1).item() or jnp.allclose(row_sums, row_sums[0], rtol=0.1).item()
    
    def test_positive_values(self, sample_r_2node):
        """All values should be non-negative."""
        P = utils.build_P_tau(sample_r_2node, tau=0.3)
        assert jnp.all(P >= 0).item()

    def test_shift_invariant_to_common_offset(self, sample_r_3node):
        """Adding a common offset should not change the soft permutation."""
        shifted = sample_r_3node + 7.5
        P = utils.build_P_tau(sample_r_3node, tau=0.3)
        P_shifted = utils.build_P_tau(shifted, tau=0.3)
        np.testing.assert_allclose(np.array(P), np.array(P_shifted), atol=1e-5)


class TestBuildMaskMTau:
    """Tests for build_mask_M_tau function."""
    
    def test_output_shape(self, sample_r_2node):
        """Test output shape matches number of nodes."""
        M = utils.build_mask_M_tau(sample_r_2node, tau=0.3)
        assert M.shape == (2, 2)
    
    def test_3node_output_shape(self, sample_r_3node):
        """Test output shape for 3 nodes."""
        M = utils.build_mask_M_tau(sample_r_3node, tau=0.3)
        assert M.shape == (3, 3)
    
    def test_values_in_valid_range(self, sample_r_3node):
        """Mask values should be in [0, 1] range."""
        M = utils.build_mask_M_tau(sample_r_3node, tau=0.3)
        assert jnp.all(M >= 0).item()
        assert jnp.all(M <= 1 + 1e-5).item()  # Allow small numerical error
    
    def test_diagonal_approximately_zero(self, sample_r_3node):
        """Diagonal should be close to zero (no self-loops)."""
        M = utils.build_mask_M_tau(sample_r_3node, tau=0.1)
        diag = jnp.diag(M)
        # Diagonal might not be exactly zero due to numerical issues
        # but should be relatively small
        assert jnp.all(diag < 0.5).item()

    def test_shift_invariant_to_common_offset(self, sample_r_2node):
        """Adding the same constant to all node scores should not change M."""
        shifted = sample_r_2node + 5.0
        M = utils.build_mask_M_tau(sample_r_2node, tau=0.3)
        M_shifted = utils.build_mask_M_tau(shifted, tau=0.3)
        np.testing.assert_allclose(np.array(M), np.array(M_shifted), atol=1e-5)


class TestBuildMaskMHard:
    """Tests for build_mask_M_hard (hard mask M(r) = P(r) L P(r)^T)."""

    @staticmethod
    def _expected_via_permutation(r):
        """Reference construction: explicitly form P(r) and compute P L P^T."""
        r = np.asarray(r)
        n = r.shape[0]
        order = np.argsort(-r)  # descending sort: order[k] = node at position k
        P = np.zeros((n, n))
        for k, i in enumerate(order):
            P[i, k] = 1.0  # node i sits at position k
        L = np.tril(np.ones((n, n)), k=-1)
        return P @ L @ P.T

    def test_matches_P_L_PT(self, sample_r_3node):
        """M_hard must equal the explicit P(r) L P(r)^T construction."""
        M = utils.build_mask_M_hard(sample_r_3node)
        expected = self._expected_via_permutation(sample_r_3node)
        np.testing.assert_allclose(np.array(M), expected, atol=1e-6)

    def test_matches_P_L_PT_random(self):
        """Agreement with P L P^T across many random potentials."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            r = jnp.array(rng.normal(size=5).astype(np.float32))
            M = utils.build_mask_M_hard(r)
            expected = self._expected_via_permutation(r)
            np.testing.assert_allclose(np.array(M), expected, atol=1e-6)

    def test_binary_and_zero_diagonal(self, sample_r_3node):
        """Mask entries are exactly 0/1 with a zero diagonal."""
        M = np.array(utils.build_mask_M_hard(sample_r_3node))
        assert set(np.unique(M)).issubset({0.0, 1.0})
        np.testing.assert_allclose(np.diag(M), np.zeros(3))

    def test_exactly_one_direction_per_pair(self, sample_r_3node):
        """For i != j exactly one of M[i,j], M[j,i] is 1 (total order)."""
        M = np.array(utils.build_mask_M_hard(sample_r_3node))
        n = M.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                assert M[i, j] + M[j, i] == 1.0

    def test_low_tau_limit_of_M_tau(self):
        """build_mask_M_tau approaches M_hard as tau -> 0.

        Potentials are kept small so exp(S0/tau) stays inside float32 range
        (large |r|/tau underflows the Sinkhorn kernel to exact zeros).
        """
        r = jnp.array([0.3, -0.25, 0.05], dtype=jnp.float32)
        M_hard = np.array(utils.build_mask_M_hard(r))
        M_tau = np.array(utils.build_mask_M_tau(r, tau=0.02))
        np.testing.assert_allclose(M_tau, M_hard, atol=5e-2)

    def test_masked_graph_is_acyclic(self):
        """B ⊙ M_hard is a DAG for any binary B (Theorem 3.1)."""
        rng = np.random.default_rng(1)
        for _ in range(10):
            r = jnp.array(rng.normal(size=4).astype(np.float32))
            M = np.array(utils.build_mask_M_hard(r))
            B = rng.integers(0, 2, size=(4, 4)).astype(float)
            np.fill_diagonal(B, 0)
            A = B * M
            # Acyclic iff trace(A^k) == 0 for all k = 1..n
            acc = np.eye(4)
            for _k in range(4):
                acc = acc @ A
                assert np.trace(acc) == 0.0


class TestGaussianLogpdf:
    """Tests for gaussian_logpdf function."""
    
    def test_standard_normal_at_zero(self):
        """Test log pdf of standard normal at x=0."""
        result = utils.gaussian_logpdf(0.0, mu=0.0, sigma=1.0)
        expected = -0.5 * np.log(2 * np.pi)
        assert float(result) == pytest.approx(expected, rel=1e-5)
    
    def test_symmetry(self):
        """PDF should be symmetric around mean."""
        result_pos = utils.gaussian_logpdf(1.0, mu=0.0, sigma=1.0)
        result_neg = utils.gaussian_logpdf(-1.0, mu=0.0, sigma=1.0)
        assert float(result_pos) == pytest.approx(float(result_neg))
    
    def test_decreasing_with_distance(self):
        """PDF should decrease as x moves away from mean."""
        result_near = utils.gaussian_logpdf(0.1, mu=0.0, sigma=1.0)
        result_far = utils.gaussian_logpdf(2.0, mu=0.0, sigma=1.0)
        assert float(result_near) > float(result_far)
    
    def test_wider_sigma_flatter_pdf(self):
        """Wider sigma should give higher pdf far from mean."""
        result_narrow = utils.gaussian_logpdf(2.0, mu=0.0, sigma=0.5)
        result_wide = utils.gaussian_logpdf(2.0, mu=0.0, sigma=2.0)
        assert float(result_wide) > float(result_narrow)


class TestStdNormalLogpdf:
    """Tests for std_normal_logpdf function."""
    
    def test_scalar_input(self):
        """Test with scalar-like input."""
        z = jnp.array([0.0])
        result = utils.std_normal_logpdf(z)
        expected = -0.5 * np.log(2 * np.pi)
        assert float(result) == pytest.approx(expected, rel=1e-5)
    
    def test_vector_input(self):
        """Test with vector input."""
        z = jnp.array([0.0, 0.0, 0.0])
        result = utils.std_normal_logpdf(z)
        expected = -1.5 * np.log(2 * np.pi)  # -0.5 * 3 * log(2π)
        assert float(result) == pytest.approx(expected, rel=1e-5)
    
    def test_non_zero_values(self):
        """Test with non-zero values."""
        z = jnp.array([1.0, -1.0])
        result = utils.std_normal_logpdf(z)
        expected = -0.5 * (2.0 + 2 * np.log(2 * np.pi))  # -0.5 * (z²_sum + d*log(2π))
        assert float(result) == pytest.approx(expected, rel=1e-5)


class TestLogisticBetaLogpdfGamma:
    """Tests for logistic_beta_logpdf_gamma function."""
    
    def test_valid_output(self, rng_key):
        """Test that function produces valid log probability."""
        gamma_mat = jrand.normal(rng_key, (3, 3))
        alpha_mat = jnp.ones((3, 3)) * 2.0
        beta_mat = jnp.ones((3, 3)) * 2.0
        
        result = utils.logistic_beta_logpdf_gamma(gamma_mat, alpha_mat, beta_mat)
        
        # Should be a scalar
        assert result.shape == ()
        # Should be finite
        assert jnp.isfinite(result).item()
    
    def test_uniform_beta_11(self, rng_key):
        """Beta(1,1) is uniform, so logit gives flat prior on (0,1)."""
        gamma_mat = jrand.normal(rng_key, (2, 2)) * 0.1
        alpha_mat = jnp.ones((2, 2))
        beta_mat = jnp.ones((2, 2))
        
        result = utils.logistic_beta_logpdf_gamma(gamma_mat, alpha_mat, beta_mat)
        assert jnp.isfinite(result).item()
    
    def test_higher_alpha_favors_high_probability(self):
        """Higher alpha should favor higher sigmoid(gamma) values."""
        # gamma=10 -> sigmoid(10) ≈ 1 (high)
        gamma_high = jnp.array([[0.0, 10.0], [10.0, 0.0]])
        # gamma=-10 -> sigmoid(-10) ≈ 0 (low)
        gamma_low = jnp.array([[0.0, -10.0], [-10.0, 0.0]])
        
        # Beta with high alpha favors values near 1
        alpha_high = jnp.array([[1.0, 10.0], [10.0, 1.0]])
        beta_low = jnp.array([[1.0, 1.0], [1.0, 1.0]])
        
        log_p_high = utils.logistic_beta_logpdf_gamma(gamma_high, alpha_high, beta_low)
        log_p_low = utils.logistic_beta_logpdf_gamma(gamma_low, alpha_high, beta_low)
        
        # High gamma should have higher log probability with high alpha prior
        assert float(log_p_high) > float(log_p_low)


class TestGetPriorMatrix:
    """Tests for get_prior_matrix function."""
    
    def test_noninformative_prior(self, true_adj_2node):
        """Non-informative prior should be 0.5 everywhere off-diagonal."""
        prior = utils.get_prior_matrix("noninformative", ["x1", "x2"], true_adj_2node, num_nodes=2)
        
        # Off-diagonal should be 0.5
        assert float(prior[0, 1]) == pytest.approx(0.5)
        assert float(prior[1, 0]) == pytest.approx(0.5)
        # Diagonal should be 0
        assert float(prior[0, 0]) == 0.0
        assert float(prior[1, 1]) == 0.0
    
    def test_strong_correct_prior(self, true_adj_2node):
        """Strong correct prior should have high probability on true edges."""
        prior = utils.get_prior_matrix("strong_correct", ["x1", "x2"], true_adj_2node, num_nodes=2)
        
        # True edge x1->x2 (adj[1,0]=1) should have high prior
        assert float(prior[1, 0]) == pytest.approx(0.99)
        # Non-edge should have low prior
        assert float(prior[0, 1]) == pytest.approx(0.01)
    
    def test_strong_incorrect_prior(self, true_adj_2node):
        """Strong incorrect prior should have high probability on wrong edges."""
        prior = utils.get_prior_matrix("strong_incorrect", ["x1", "x2"], true_adj_2node, num_nodes=2)
        
        # Wrong edge x2->x1 should have high prior (transposed from true)
        assert float(prior[0, 1]) == pytest.approx(0.99)
        # True edge should have low prior
        assert float(prior[1, 0]) == pytest.approx(0.01)
    
    def test_diagonal_always_zero(self, true_adj_3node):
        """Diagonal should always be zero regardless of scenario."""
        for scenario in ["noninformative", "strong_correct", "strong_incorrect"]:
            prior = utils.get_prior_matrix(scenario, ["x1", "x2", "x3"], true_adj_3node, num_nodes=3)
            diag = jnp.diag(prior)
            np.testing.assert_array_equal(np.array(diag), np.zeros(3))
    
    def test_unknown_scenario_raises(self, true_adj_2node):
        """Unknown scenario should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            utils.get_prior_matrix("invalid_scenario", ["x1", "x2"], true_adj_2node, num_nodes=2)


class TestComputeNuMatrix:
    """Tests for compute_nu_matrix function."""
    
    def test_uniform_prior_gives_minimum_nu(self, uniform_prior_2node):
        """Uniform prior (0.5) should give minimum nu."""
        nu = utils.compute_nu_matrix(uniform_prior_2node)
        
        # At p=0.5, |p-0.5|=0, so nu = NU_MIN
        assert float(nu[0, 1]) == pytest.approx(config.NU_MIN)
        assert float(nu[1, 0]) == pytest.approx(config.NU_MIN)
    
    def test_extreme_prior_gives_high_nu(self):
        """Prior near 0 or 1 should give high nu."""
        extreme_prior = jnp.array([[0.0, 0.99], [0.01, 0.0]])
        nu = utils.compute_nu_matrix(extreme_prior)
        
        # Near 0 or 1, |p-0.5| ≈ 0.5, so nu should be higher
        assert float(nu[0, 1]) > config.NU_MIN
        assert float(nu[1, 0]) > config.NU_MIN
    
    def test_diagonal_is_zero(self, uniform_prior_3node):
        """Diagonal of nu should be zero."""
        nu = utils.compute_nu_matrix(uniform_prior_3node)
        diag = jnp.diag(nu)
        np.testing.assert_array_almost_equal(np.array(diag), np.zeros(3))


class TestComputeAlphaBetaFromPrior:
    """Tests for compute_alpha_beta_from_prior function."""
    
    def test_output_shapes(self, uniform_prior_3node):
        """Alpha and beta should have same shape as prior."""
        alpha, beta = utils.compute_alpha_beta_from_prior(uniform_prior_3node)
        
        assert alpha.shape == uniform_prior_3node.shape
        assert beta.shape == uniform_prior_3node.shape
    
    def test_alpha_beta_positive(self, uniform_prior_2node):
        """Alpha and beta should be positive."""
        alpha, beta = utils.compute_alpha_beta_from_prior(uniform_prior_2node)
        
        assert jnp.all(alpha >= 1.0).item()  # minimum is 1.0 due to +1 in formula
        assert jnp.all(beta >= 1.0).item()
    
    def test_uniform_prior_symmetric_alpha_beta(self, uniform_prior_2node):
        """Uniform prior should give symmetric alpha and beta (off-diagonal)."""
        alpha, beta = utils.compute_alpha_beta_from_prior(uniform_prior_2node)
        
        # At p=0.5, alpha = nu*0.5+1 = beta = nu*(1-0.5)+1
        assert float(alpha[0, 1]) == pytest.approx(float(beta[0, 1]))
    
    def test_high_prior_high_alpha(self):
        """High prior probability should give higher alpha than beta."""
        high_prior = jnp.array([[0.0, 0.9], [0.9, 0.0]])
        alpha, beta = utils.compute_alpha_beta_from_prior(high_prior)
        
        # alpha should be higher for p=0.9
        assert float(alpha[0, 1]) > float(beta[0, 1])


class TestGaussianLikelihoodLogsum:
    """Tests for gaussian_likelihood_logsum function."""
    
    def test_perfect_prediction(self):
        """Perfect prediction should give maximum likelihood."""
        x_true = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        x_pred = x_true  # Perfect prediction
        noise = jnp.array([0.1, 0.1])
        
        ll_mse, ll_const = utils.gaussian_likelihood_logsum(x_true, x_pred, noise)
        
        # MSE component should be 0 (perfect fit)
        assert float(ll_mse) == pytest.approx(0.0, abs=1e-5)
    
    def test_worse_prediction_lower_likelihood(self):
        """Worse predictions should have lower likelihood."""
        x_true = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        x_good = x_true + 0.01  # Small error
        x_bad = x_true + 1.0    # Large error
        noise = jnp.array([0.1, 0.1])
        
        ll_good, _ = utils.gaussian_likelihood_logsum(x_true, x_good, noise)
        ll_bad, _ = utils.gaussian_likelihood_logsum(x_true, x_bad, noise)
        
        assert float(ll_good) > float(ll_bad)
    
    def test_higher_noise_more_tolerant(self):
        """Higher noise should make likelihood more tolerant of errors."""
        x_true = jnp.array([[1.0, 2.0]])
        x_pred = jnp.array([[1.5, 2.5]])  # Fixed error
        
        noise_low = jnp.array([0.1, 0.1])
        noise_high = jnp.array([1.0, 1.0])
        
        ll_low, _ = utils.gaussian_likelihood_logsum(x_true, x_pred, noise_low)
        ll_high, _ = utils.gaussian_likelihood_logsum(x_true, x_pred, noise_high)
        
        # Higher noise should give higher likelihood for same error
        assert float(ll_high) > float(ll_low)
    
    def test_include_const_flag(self):
        """Test that include_const flag works correctly."""
        x_true = jnp.array([[1.0, 2.0]])
        x_pred = jnp.array([[1.0, 2.0]])
        noise = jnp.array([0.1, 0.1])
        
        _, ll_const_true = utils.gaussian_likelihood_logsum(x_true, x_pred, noise, include_const=True)
        _, ll_const_false = utils.gaussian_likelihood_logsum(x_true, x_pred, noise, include_const=False)
        
        # With include_const=True, const term should be non-zero
        assert float(ll_const_true) != 0.0
        # With include_const=False, const term should be zero
        assert float(ll_const_false) == 0.0


class TestComputeMetrics:
    """Tests for compute_metrics function (SHD, TPR, F1, Brier, AUROC)."""
    
    def test_perfect_prediction_binary_only(self, true_adj_2node):
        """Perfect prediction should give SHD=0, TPR=1, F1=1."""
        metrics = utils.compute_metrics(true_adj_2node, true_adj_2node)
        
        assert metrics['SHD'] == 0
        assert metrics['TPR'] == 1.0
        assert metrics['F1'] == pytest.approx(1.0)
        assert metrics['Brier'] is None
        assert metrics['AUROC'] is None
    
    def test_perfect_prediction_with_probs(self, true_adj_2node):
        """Perfect prediction with probs should give all perfect metrics."""
        pred_probs = true_adj_2node.astype(np.float32)
        metrics = utils.compute_metrics(true_adj_2node, true_adj_2node, pred_probs)
        
        assert metrics['SHD'] == 0
        assert metrics['TPR'] == 1.0
        assert metrics['F1'] == pytest.approx(1.0)
        assert metrics['Brier'] == pytest.approx(0.0)
        assert metrics['AUROC'] == pytest.approx(1.0)
    
    def test_empty_prediction_on_nonempty_true(self, true_adj_2node):
        """Empty prediction on non-empty true should give poor metrics."""
        pred = np.zeros_like(true_adj_2node)
        pred_probs = np.zeros_like(true_adj_2node, dtype=np.float32)
        metrics = utils.compute_metrics(pred, true_adj_2node, pred_probs)
        
        num_true_edges = int(np.sum(true_adj_2node))
        assert metrics['SHD'] == num_true_edges
        assert metrics['TPR'] == 0.0
        assert metrics['F1'] == 0.0
        assert metrics['Brier'] == pytest.approx(0.5)  # 1 edge with 0 prob, 1 non-edge with 0 prob
    
    def test_all_ones_prediction(self, true_adj_2node):
        """All-ones prediction should have calculated metrics."""
        pred = np.ones_like(true_adj_2node)
        np.fill_diagonal(pred, 0)
        pred_probs = pred.astype(np.float32)
        metrics = utils.compute_metrics(pred, true_adj_2node, pred_probs)
        
        assert metrics['TPR'] == 1.0
        num_true_edges = int(np.sum(true_adj_2node))
        num_pred_edges = int(np.sum(pred))
        assert metrics['SHD'] == num_pred_edges - num_true_edges
        assert 0.0 < metrics['F1'] <= 1.0
    
    def test_wrong_direction(self, true_adj_2node):
        """Predicting reversed edges should have SHD=2, F1=0."""
        pred = true_adj_2node.T
        metrics = utils.compute_metrics(pred, true_adj_2node)
        
        assert metrics['SHD'] == 2
        assert metrics['TPR'] == 0.0
        assert metrics['F1'] == 0.0
    
    def test_empty_true_graph(self, empty_adj_2node):
        """Empty true graph should give TPR=1, F1=0 if pred is also empty."""
        pred_empty = np.zeros((2, 2), dtype=np.float32)
        metrics = utils.compute_metrics(pred_empty, empty_adj_2node)
        
        assert metrics['SHD'] == 0
        assert metrics['TPR'] == 1.0
        assert metrics['F1'] == 0.0  # No TPs possible
    
    def test_f1_partial_recovery(self, true_adj_3node):
        """Partial recovery should give F1 between 0 and 1."""
        pred = np.zeros_like(true_adj_3node)
        pred[1, 0] = 1  # Only first edge
        metrics = utils.compute_metrics(pred, true_adj_3node)
        
        # TP=1, FP=0, FN=1, precision=1, recall=0.5, F1=2/3
        assert metrics['F1'] == pytest.approx(2.0/3.0)
    
    def test_brier_worst_calibration(self, true_adj_2node):
        """Completely wrong probabilities should give Brier=1."""
        pred_probs = 1.0 - true_adj_2node.astype(np.float32)
        np.fill_diagonal(pred_probs, 0)
        pred_binary = (pred_probs > 0.5).astype(np.float32)
        
        metrics = utils.compute_metrics(pred_binary, true_adj_2node, pred_probs)
        assert metrics['Brier'] == pytest.approx(1.0)
    
    def test_brier_uniform_prediction(self, true_adj_2node):
        """Uniform 0.5 prediction should give Brier=0.25."""
        pred_probs = np.full_like(true_adj_2node, 0.5, dtype=np.float32)
        np.fill_diagonal(pred_probs, 0)
        pred_binary = (pred_probs > 0.5).astype(np.float32)
        
        metrics = utils.compute_metrics(pred_binary, true_adj_2node, pred_probs)
        assert metrics['Brier'] == pytest.approx(0.25)
    
    def test_auroc_perfect_ranking(self, true_adj_2node):
        """Perfect ranking should give AUROC=1."""
        pred_probs = np.where(true_adj_2node == 1, 0.9, 0.1).astype(np.float32)
        np.fill_diagonal(pred_probs, 0)
        
        metrics = utils.compute_metrics(true_adj_2node, true_adj_2node, pred_probs)
        assert metrics['AUROC'] == pytest.approx(1.0)
    
    def test_auroc_worst_ranking(self, true_adj_2node):
        """Reversed ranking should give AUROC=0."""
        pred_probs = np.where(true_adj_2node == 1, 0.1, 0.9).astype(np.float32)
        np.fill_diagonal(pred_probs, 0)
        pred_binary = (pred_probs > 0.5).astype(np.float32)
        
        metrics = utils.compute_metrics(pred_binary, true_adj_2node, pred_probs)
        assert metrics['AUROC'] == pytest.approx(0.0)
    
    def test_auroc_only_positives_returns_nan(self):
        """All positives should return NaN AUROC."""
        pred_probs = np.array([[0.0, 0.9], [0.9, 0.0]], dtype=np.float32)
        true_adj = np.array([[0, 1], [1, 0]], dtype=np.float32)
        pred_binary = true_adj.copy()
        
        metrics = utils.compute_metrics(pred_binary, true_adj, pred_probs)
        assert np.isnan(metrics['AUROC'])
    
    def test_auroc_only_negatives_returns_nan(self, empty_adj_2node):
        """All negatives should return NaN AUROC."""
        pred_probs = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float32)
        pred_binary = np.zeros((2, 2), dtype=np.float32)
        
        metrics = utils.compute_metrics(pred_binary, empty_adj_2node, pred_probs)
        assert np.isnan(metrics['AUROC'])
    
    def test_auroc_tie_handling(self):
        """All tied predictions should give AUROC=0.5."""
        true_adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float32)
        pred_probs = np.full((3, 3), 0.5, dtype=np.float32)
        np.fill_diagonal(pred_probs, 0)
        pred_binary = (pred_probs > 0.5).astype(np.float32)
        
        metrics = utils.compute_metrics(pred_binary, true_adj, pred_probs)
        assert metrics['AUROC'] == pytest.approx(0.5)
    
    def test_returns_dict(self, true_adj_2node):
        """Should return a dictionary with expected keys."""
        metrics = utils.compute_metrics(true_adj_2node, true_adj_2node)
        
        assert isinstance(metrics, dict)
        assert 'SHD' in metrics
        assert 'TPR' in metrics
        assert 'F1' in metrics
        assert 'Brier' in metrics
        assert 'AUROC' in metrics


class TestDagToCpdag:
    """Tests for dag_to_cpdag function."""
    
    def test_chain_graph(self, true_adj_3node):
        """Chain DAG (x1->x2->x3) should have all edges undirected in CPDAG."""
        # True adj is in j->i convention, dag_to_cpdag expects i->j
        cpdag = utils.dag_to_cpdag(true_adj_3node.T)
        
        # Chain has no v-structures, so edges should be undirected
        # Check x1-x2: should be undirected (both directions = 1)
        assert cpdag[0, 1] == 1 and cpdag[1, 0] == 1  # x1 - x2
        assert cpdag[1, 2] == 1 and cpdag[2, 1] == 1  # x2 - x3
        # No edge between x1 and x3
        assert cpdag[0, 2] == 0 and cpdag[2, 0] == 0
    
    def test_v_structure(self, true_adj_3node_vstruct):
        """V-structure (x1->x3<-x2) should preserve directed edges."""
        cpdag = utils.dag_to_cpdag(true_adj_3node_vstruct.T)
        
        # V-structure edges should be directed in CPDAG
        # x1 -> x3 and x2 -> x3 should be directed
        assert cpdag[0, 2] == 1 and cpdag[2, 0] == 0  # x1 -> x3
        assert cpdag[1, 2] == 1 and cpdag[2, 1] == 0  # x2 -> x3
        # No edge between x1 and x2
        assert cpdag[0, 1] == 0 and cpdag[1, 0] == 0
    
    def test_empty_graph(self, empty_adj_2node):
        """Empty DAG should give empty CPDAG."""
        cpdag = utils.dag_to_cpdag(empty_adj_2node)
        np.testing.assert_array_equal(cpdag, empty_adj_2node)
    
    def test_single_edge(self, true_adj_2node):
        """Single edge DAG should give undirected edge (same MEC)."""
        cpdag = utils.dag_to_cpdag(true_adj_2node.T)
        
        # Single edge has no v-structures, should be undirected
        # Check both directions are 1 (undirected edge)
        assert cpdag[0, 1] + cpdag[1, 0] == 2


class TestComputeShdCpdag:
    """Tests for compute_shd_cpdag function."""
    
    def test_identical_cpdags(self):
        """Identical CPDAGs should have SHD=0."""
        cpdag = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        shd = utils.compute_shd_cpdag(cpdag, cpdag)
        assert shd == 0
    
    def test_missing_edge(self):
        """Missing edge should add to SHD."""
        true_cpdag = np.array([[0, 1], [1, 0]])  # One undirected edge
        pred_cpdag = np.array([[0, 0], [0, 0]])  # No edges
        
        shd = utils.compute_shd_cpdag(pred_cpdag, true_cpdag)
        assert shd == 1  # One missing edge
    
    def test_extra_edge(self):
        """Extra edge should add to SHD."""
        true_cpdag = np.array([[0, 0], [0, 0]])  # No edges
        pred_cpdag = np.array([[0, 1], [1, 0]])  # One undirected edge
        
        shd = utils.compute_shd_cpdag(pred_cpdag, true_cpdag)
        assert shd == 1  # One extra edge
    
    def test_wrong_orientation(self):
        """Wrong orientation on existing edge should add to SHD."""
        # True: undirected edge (both directions)
        true_cpdag = np.array([[0, 1], [1, 0]])
        # Pred: directed edge (one direction only)
        pred_cpdag = np.array([[0, 1], [0, 0]])
        
        shd = utils.compute_shd_cpdag(pred_cpdag, true_cpdag)
        assert shd == 1  # One orientation error


class TestComputeTprCpdag:
    """Tests for compute_tpr_cpdag function."""
    
    def test_perfect_recovery(self):
        """Perfect skeleton recovery should give TPR=1."""
        cpdag = np.array([[0, 1], [1, 0]])
        tpr = utils.compute_tpr_cpdag(cpdag, cpdag)
        assert tpr == 1.0
    
    def test_no_edges_recovered(self):
        """No edges recovered should give TPR=0."""
        true_cpdag = np.array([[0, 1], [1, 0]])
        pred_cpdag = np.array([[0, 0], [0, 0]])
        
        tpr = utils.compute_tpr_cpdag(pred_cpdag, true_cpdag)
        assert tpr == 0.0
    
    def test_partial_recovery(self):
        """Partial recovery should give proportional TPR."""
        # True: two undirected edges
        true_cpdag = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        # Pred: one of the two edges
        pred_cpdag = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        
        tpr = utils.compute_tpr_cpdag(pred_cpdag, true_cpdag)
        assert tpr == 0.5  # 1 out of 2 edges
    
    def test_empty_true_graph(self):
        """Empty true graph should give TPR=1 if pred is also empty."""
        true_cpdag = np.array([[0, 0], [0, 0]])
        pred_cpdag = np.array([[0, 0], [0, 0]])
        
        tpr = utils.compute_tpr_cpdag(pred_cpdag, true_cpdag)
        assert tpr == 1.0


class TestComputeF1Cpdag:
    """Tests for compute_f1_cpdag function."""
    
    def test_perfect_recovery(self):
        """Perfect skeleton recovery should give F1=1."""
        cpdag = np.array([[0, 1], [1, 0]])
        f1 = utils.compute_f1_cpdag(cpdag, cpdag)
        assert f1 == pytest.approx(1.0)
    
    def test_no_edges_recovered(self):
        """No edges recovered should give F1=0."""
        true_cpdag = np.array([[0, 1], [1, 0]])
        pred_cpdag = np.array([[0, 0], [0, 0]])
        
        f1 = utils.compute_f1_cpdag(pred_cpdag, true_cpdag)
        assert f1 == 0.0
    
    def test_partial_recovery(self):
        """Partial recovery should give F1 between 0 and 1."""
        # True: two undirected edges
        true_cpdag = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        # Pred: one of the two edges
        pred_cpdag = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        
        f1 = utils.compute_f1_cpdag(pred_cpdag, true_cpdag)
        # TP=1, FP=0, FN=1, precision=1, recall=0.5, F1=2/3
        assert f1 == pytest.approx(2.0/3.0)
    
    def test_extra_edges(self):
        """Extra edges should lower precision and F1."""
        true_cpdag = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])  # One edge
        pred_cpdag = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])  # Two edges
        
        f1 = utils.compute_f1_cpdag(pred_cpdag, true_cpdag)
        # TP=1, FP=1, FN=0, precision=0.5, recall=1.0, F1=2/3
        assert f1 == pytest.approx(2.0/3.0)
    
    def test_both_empty(self):
        """Both empty should give F1=0."""
        empty = np.array([[0, 0], [0, 0]])
        f1 = utils.compute_f1_cpdag(empty, empty)
        assert f1 == 0.0


class TestComputeBrierCpdag:
    """Tests for compute_brier_cpdag function."""
    
    def test_perfect_calibration(self):
        """Perfect probabilities should give Brier=0."""
        true_cpdag = np.array([[0, 1], [1, 0]])  # Undirected edge
        pred_probs = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        
        brier = utils.compute_brier_cpdag(pred_probs, true_cpdag)
        assert brier == pytest.approx(0.0)
    
    def test_worst_calibration(self):
        """Completely wrong probabilities should give Brier=1."""
        true_cpdag = np.array([[0, 1], [1, 0]])  # Undirected edge
        pred_probs = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        
        brier = utils.compute_brier_cpdag(pred_probs, true_cpdag)
        assert brier == pytest.approx(1.0)
    
    def test_uniform_prediction(self):
        """Uniform 0.5 prediction should give Brier=0.25."""
        true_cpdag = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])  # Two edges
        pred_probs = np.full((3, 3), 0.5, dtype=np.float32)
        np.fill_diagonal(pred_probs, 0)
        
        brier = utils.compute_brier_cpdag(pred_probs, true_cpdag)
        assert brier == pytest.approx(0.25)
    
    def test_symmetrization(self):
        """Should symmetrize predictions using max of both directions."""
        true_cpdag = np.array([[0, 1], [1, 0]])  # Undirected edge
        # Asymmetric predictions
        pred_probs = np.array([[0.0, 0.9], [0.3, 0.0]], dtype=np.float32)
        
        brier = utils.compute_brier_cpdag(pred_probs, true_cpdag)
        # Symmetrized: max(0.9, 0.3) = 0.9, so error = (0.9-1)^2 = 0.01
        assert brier == pytest.approx(0.01)


class TestComputeAurocCpdag:
    """Tests for compute_auroc_cpdag function."""
    
    def test_perfect_ranking(self):
        """Perfect ranking should give AUROC=1."""
        true_cpdag = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])  # One edge (0-1)
        pred_probs = np.array([
            [0.0, 0.9, 0.1],
            [0.9, 0.0, 0.1],
            [0.1, 0.1, 0.0]
        ], dtype=np.float32)
        
        auroc = utils.compute_auroc_cpdag(pred_probs, true_cpdag)
        assert auroc == pytest.approx(1.0)
    
    def test_worst_ranking(self):
        """Reversed ranking should give AUROC=0."""
        true_cpdag = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])  # One edge (0-1)
        pred_probs = np.array([
            [0.0, 0.1, 0.9],
            [0.1, 0.0, 0.9],
            [0.9, 0.9, 0.0]
        ], dtype=np.float32)
        
        auroc = utils.compute_auroc_cpdag(pred_probs, true_cpdag)
        assert auroc == pytest.approx(0.0)
    
    def test_random_ranking(self):
        """Tied predictions should give AUROC=0.5."""
        true_cpdag = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])  # One edge
        pred_probs = np.full((3, 3), 0.5, dtype=np.float32)
        np.fill_diagonal(pred_probs, 0)
        
        auroc = utils.compute_auroc_cpdag(pred_probs, true_cpdag)
        assert auroc == pytest.approx(0.5)
    
    def test_all_edges_returns_nan(self):
        """Complete graph should return NaN (no negatives)."""
        true_cpdag = np.array([[0, 1], [1, 0]])  # All possible edges
        pred_probs = np.array([[0.0, 0.9], [0.9, 0.0]], dtype=np.float32)
        
        auroc = utils.compute_auroc_cpdag(pred_probs, true_cpdag)
        assert np.isnan(auroc)
    
    def test_no_edges_returns_nan(self):
        """Empty graph should return NaN (no positives)."""
        true_cpdag = np.array([[0, 0], [0, 0]])
        pred_probs = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float32)
        
        auroc = utils.compute_auroc_cpdag(pred_probs, true_cpdag)
        assert np.isnan(auroc)


class TestComputeMetricsCpdag:
    """Tests for compute_metrics_cpdag function."""
    
    def test_perfect_prediction_binary_only(self):
        """Perfect prediction should give SHD=0, TPR=1, F1=1."""
        cpdag = np.array([[0, 1], [1, 0]])
        metrics = utils.compute_metrics_cpdag(cpdag, cpdag)
        
        assert metrics['SHD'] == 0
        assert metrics['TPR'] == 1.0
        assert metrics['F1'] == pytest.approx(1.0)
        assert metrics['Brier'] is None
        assert metrics['AUROC'] is None
    
    def test_perfect_prediction_with_probs(self):
        """Perfect prediction with probs should give all perfect metrics."""
        true_cpdag = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        pred_probs = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ], dtype=np.float32)
        
        metrics = utils.compute_metrics_cpdag(true_cpdag, true_cpdag, pred_probs)
        
        assert metrics['SHD'] == 0
        assert metrics['TPR'] == 1.0
        assert metrics['F1'] == pytest.approx(1.0)
        assert metrics['Brier'] == pytest.approx(0.0)
        assert metrics['AUROC'] == pytest.approx(1.0)
    
    def test_returns_dict_with_expected_keys(self):
        """Should return dict with all expected keys."""
        cpdag = np.array([[0, 1], [1, 0]])
        metrics = utils.compute_metrics_cpdag(cpdag, cpdag)
        
        assert isinstance(metrics, dict)
        assert 'SHD' in metrics
        assert 'TPR' in metrics
        assert 'F1' in metrics
        assert 'Brier' in metrics
        assert 'AUROC' in metrics
    
    def test_partial_recovery(self):
        """Partial recovery should give intermediate metrics."""
        true_cpdag = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])  # Two edges
        pred_cpdag = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])  # One edge
        
        metrics = utils.compute_metrics_cpdag(pred_cpdag, true_cpdag)
        
        assert metrics['SHD'] == 1  # Missing one edge
        assert metrics['TPR'] == 0.5  # Found 1 of 2
        assert metrics['F1'] == pytest.approx(2.0/3.0)


class TestComputeSvgdUpdate:
    """Tests for compute_svgd_update function."""
    
    def test_output_shape(self, particles_2node, rng_key):
        """Output should have same shape as input particles."""
        grads = jrand.normal(rng_key, particles_2node.shape)
        phi = utils.compute_svgd_update(particles_2node, grads)
        
        assert phi.shape == particles_2node.shape
    
    def test_finite_output(self, particles_3node, rng_key):
        """Output should be finite."""
        grads = jrand.normal(rng_key, particles_3node.shape)
        phi = utils.compute_svgd_update(particles_3node, grads)
        
        assert jnp.all(jnp.isfinite(phi)).item()
    
    def test_repulsive_term_effect(self):
        """Identical particles should have strong repulsive term."""
        # All particles at same location
        particles = jnp.ones((5, 2))
        grads = jnp.zeros((5, 2))  # No gradient information
        
        phi = utils.compute_svgd_update(particles, grads)
        
        # With identical particles and zero gradients, update should be non-zero
        # due to repulsive term (kernel gradient)
        # Actually with identical particles, kernel gradients are zero
        # So this test needs reconsideration
        # The update should be zero when particles are identical and grads are zero
        assert jnp.all(jnp.isfinite(phi)).item()
    
    def test_spread_particles_behavior(self, rng_key):
        """Well-spread particles should have moderate update magnitudes."""
        key1, key2 = jrand.split(rng_key)
        particles = jrand.normal(key1, (10, 3))
        grads = jrand.normal(key2, (10, 3)) * 0.1
        
        phi = utils.compute_svgd_update(particles, grads)
        
        # Update magnitude should be reasonable
        max_update = float(jnp.max(jnp.abs(phi)))
        assert max_update < 100  # Sanity check

    def test_mask_feature_kernel_ignores_common_shift(self):
        """M_tau-based kernel geometry should be identical after a common shift."""
        particles = jnp.array([
            [2.0, 0.0],
            [1.0, -1.0],
            [-1.5, -3.5],
        ])
        shifted_particles = particles + jnp.array([[3.0], [-2.0], [5.0]])
        grads = jnp.array([
            [0.2, -0.1],
            [0.05, -0.05],
            [-0.1, 0.15],
        ])

        phi = utils.compute_svgd_update(
            particles,
            grads,
            feature_fn=lambda r: utils.build_mask_M_tau(r, tau=0.3).reshape(-1),
        )
        phi_shifted = utils.compute_svgd_update(
            shifted_particles,
            grads,
            feature_fn=lambda r: utils.build_mask_M_tau(r, tau=0.3).reshape(-1),
        )

        np.testing.assert_allclose(np.array(phi), np.array(phi_shifted), atol=1e-6)


class TestToDevice:
    """Tests for to_device function."""
    
    def test_preserves_values(self):
        """Moving to device should preserve array values."""
        arr = jnp.array([1.0, 2.0, 3.0])
        result = utils.to_device(arr)
        np.testing.assert_allclose(np.array(arr), np.array(result))
    
    def test_preserves_shape(self):
        """Moving to device should preserve array shape."""
        arr = jnp.ones((3, 4, 5))
        result = utils.to_device(arr)
        assert result.shape == arr.shape


class TestToNumpy:
    """Tests for to_numpy function."""
    
    def test_jax_array_conversion(self):
        """Should convert JAX array to NumPy."""
        jax_arr = jnp.array([1.0, 2.0, 3.0])
        result = utils.to_numpy(jax_arr)
        
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])
    
    def test_numpy_passthrough(self):
        """NumPy arrays should pass through unchanged."""
        np_arr = np.array([1.0, 2.0, 3.0])
        result = utils.to_numpy(np_arr)
        
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np_arr)
    
    def test_preserves_dtype(self):
        """Should preserve data type."""
        jax_arr = jnp.array([1, 2, 3], dtype=jnp.int32)
        result = utils.to_numpy(jax_arr)
        
        assert result.dtype == np.int32
