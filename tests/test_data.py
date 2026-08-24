"""
Unit tests for svidag/data.py

This module tests data handling components including:
- IdentityScaler class
- Dataset dataclass
- Data generators (see test_main.py for two_node_generator_nonlinear)
- generate_synthetic_dataset function
- Helper functions (complex_func1, complex_func2)

Author: Test Suite for SVIDAG
"""

import pytest
import numpy as np
import jax.numpy as jnp
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from svidag import data, config

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import main as main_mod
two_node_generator_nonlinear = main_mod.two_node_generator_nonlinear


class TestIdentityScaler:
    """Tests for IdentityScaler class."""
    
    def test_fit_sets_attributes(self):
        """Fit should set scale_ and mean_ attributes."""
        scaler = data.IdentityScaler()
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        
        scaler.fit(X)
        
        np.testing.assert_array_equal(scaler.scale_, np.ones(2))
        np.testing.assert_array_equal(scaler.mean_, np.zeros(2))
    
    def test_transform_returns_identity(self):
        """Transform should return input unchanged."""
        scaler = data.IdentityScaler()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        scaler.fit(X)
        X_transformed = scaler.transform(X)
        
        np.testing.assert_array_equal(X_transformed, X)
    
    def test_inverse_transform_returns_identity(self):
        """Inverse transform should return input unchanged."""
        scaler = data.IdentityScaler()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        scaler.fit(X)
        X_inv = scaler.inverse_transform(X)
        
        np.testing.assert_array_equal(X_inv, X)
    
    def test_fit_returns_self(self):
        """Fit should return self for method chaining."""
        scaler = data.IdentityScaler()
        X = np.array([[1.0, 2.0]])
        
        result = scaler.fit(X)
        
        assert result is scaler
    
    def test_roundtrip_preserves_data(self):
        """Transform followed by inverse_transform should preserve data."""
        scaler = data.IdentityScaler()
        X = np.random.randn(100, 5)
        
        scaler.fit(X)
        X_roundtrip = scaler.inverse_transform(scaler.transform(X))
        
        np.testing.assert_array_equal(X_roundtrip, X)


class TestComplexFunctions:
    """Tests for complex_func1 and complex_func2."""
    
    def test_complex_func1_at_zero(self):
        """Test complex_func1 at x=0."""
        result = data.complex_func1(0.0)
        # At x=0: exp(0)*sin(0) + 0*cos(0) + sin(0) = 0 + 0 + 0 = 0
        assert result == pytest.approx(0.0, abs=1e-10)
    
    def test_complex_func1_vectorized(self):
        """Test that complex_func1 works with arrays."""
        x = np.array([-1.0, 0.0, 1.0])
        result = data.complex_func1(x)
        
        assert result.shape == x.shape
        assert np.all(np.isfinite(result))
    
    def test_complex_func2_at_zero(self):
        """Test complex_func2 at x=0."""
        result = data.complex_func2(0.0)
        # At x=0: exp(0)*sin(0) + 0*cos(0) + sin(0) = 1*0 + 0 + 0 = 0
        assert result == pytest.approx(0.0, abs=1e-10)
    
    def test_complex_func2_vectorized(self):
        """Test that complex_func2 works with arrays."""
        x = np.array([-1.0, 0.0, 1.0])
        result = data.complex_func2(x)
        
        assert result.shape == x.shape
        assert np.all(np.isfinite(result))
    
    def test_functions_produce_different_outputs(self):
        """complex_func1 and complex_func2 should produce different outputs."""
        x = np.array([0.5, 1.0, 1.5])
        
        result1 = data.complex_func1(x)
        result2 = data.complex_func2(x)
        
        # Should be different (not identical functions)
        assert not np.allclose(result1, result2)
    
    def test_functions_are_continuous(self):
        """Functions should be continuous (no large jumps for small changes)."""
        x = np.linspace(-2, 2, 1000)
        
        y1 = data.complex_func1(x)
        y2 = data.complex_func2(x)
        
        # Check that consecutive differences are bounded
        max_diff1 = np.max(np.abs(np.diff(y1)))
        max_diff2 = np.max(np.abs(np.diff(y2)))
        
        # Differences should be small for small step sizes
        assert max_diff1 < 0.1  # Reasonable bound
        assert max_diff2 < 1.0


class TestDataset:
    """Tests for Dataset dataclass."""
    
    def test_dataset_has_required_fields(self, sample_obs_data, true_adj_2node):
        """Dataset should have all required fields."""
        # Create a minimal dataset
        train_data = jnp.array(sample_obs_data[:-10])
        test_data = jnp.array(sample_obs_data[-10:])
        
        ds = data.Dataset(
            train_data=train_data,
            test_data_scaled=test_data,
            dataset_size=len(train_data),
            noise_scales=jnp.array([0.1, 0.1]),
            obs_train_orig=sample_obs_data[:-10],
            obs_test_orig=sample_obs_data[-10:],
            scaler=StandardScaler(),
            true_adj_np=true_adj_2node,
            true_DAG=jnp.array(true_adj_2node),
            node_names=["x1", "x2"],
            num_nodes=2
        )
        
        assert hasattr(ds, 'train_data')
        assert hasattr(ds, 'test_data_scaled')
        assert hasattr(ds, 'dataset_size')
        assert hasattr(ds, 'noise_scales')
        assert hasattr(ds, 'obs_train_orig')
        assert hasattr(ds, 'obs_test_orig')
        assert hasattr(ds, 'scaler')
        assert hasattr(ds, 'true_adj_np')
        assert hasattr(ds, 'true_DAG')
        assert hasattr(ds, 'node_names')
        assert hasattr(ds, 'num_nodes')
    
    def test_dataset_size_consistency(self, sample_obs_data, true_adj_2node):
        """Dataset size should match train_data length."""
        train = sample_obs_data[:-10]
        test = sample_obs_data[-10:]
        
        ds = data.Dataset(
            train_data=jnp.array(train),
            test_data_scaled=jnp.array(test),
            dataset_size=len(train),
            noise_scales=jnp.array([0.1, 0.1]),
            obs_train_orig=train,
            obs_test_orig=test,
            scaler=StandardScaler().fit(train),
            true_adj_np=true_adj_2node,
            true_DAG=jnp.array(true_adj_2node),
            node_names=["x1", "x2"],
            num_nodes=2
        )
        
        assert ds.dataset_size == ds.train_data.shape[0]



class TestGenerateSyntheticDataset:
    """Tests for generate_synthetic_dataset function."""
    
    def test_returns_dataset_object(self):
        """Should return a Dataset object."""
        ds = data.generate_synthetic_dataset(generator_fn=two_node_generator_nonlinear)
        
        assert isinstance(ds, data.Dataset)
    
    def test_default_generator_produces_2_nodes(self):
        """Default generator should produce 2-node dataset."""
        ds = data.generate_synthetic_dataset(generator_fn=two_node_generator_nonlinear)
        
        assert ds.num_nodes == 2
        assert ds.train_data.shape[1] == 2
    
    def test_train_test_split(self):
        """Train and test should be properly split."""
        ds = data.generate_synthetic_dataset(generator_fn=two_node_generator_nonlinear)
        
        # Test should have n_test_samples samples
        assert ds.test_data_scaled.shape[0] == config.n_test_samples
        # Train should have the rest
        assert ds.dataset_size == ds.obs_train_orig.shape[0]
    
    def test_scaler_is_fitted(self):
        """Scaler should be fitted to training data."""
        ds = data.generate_synthetic_dataset(generator_fn=two_node_generator_nonlinear)
        
        # StandardScaler should have mean_ attribute after fitting
        assert hasattr(ds.scaler, 'mean_')
        assert hasattr(ds.scaler, 'scale_')
    
    def test_scaled_data_has_unit_variance_approx(self):
        """Scaled training data should have approximately unit variance."""
        ds = data.generate_synthetic_dataset(
            generator_fn=two_node_generator_nonlinear,
            normalize=True,
        )
        
        train_std = np.std(np.array(ds.train_data), axis=0)
        # Should be close to 1 (scaled)
        np.testing.assert_allclose(train_std, 1.0, rtol=0.1)
    
    def test_scaled_data_has_zero_mean_approx(self):
        """Scaled training data should have approximately zero mean."""
        ds = data.generate_synthetic_dataset(
            generator_fn=two_node_generator_nonlinear,
            normalize=True,
        )
        
        train_mean = np.mean(np.array(ds.train_data), axis=0)
        # Should be close to 0 (centered)
        np.testing.assert_allclose(train_mean, 0.0, atol=0.1)
    
    def test_noise_scales_shape(self):
        """Noise scales should have correct shape."""
        ds = data.generate_synthetic_dataset(generator_fn=two_node_generator_nonlinear)
        
        assert ds.noise_scales.shape == (ds.num_nodes,)
    
    def test_node_names_match_num_nodes(self):
        """Number of node names should match num_nodes."""
        ds = data.generate_synthetic_dataset(generator_fn=two_node_generator_nonlinear)
        
        assert len(ds.node_names) == ds.num_nodes
    
    def test_true_dag_shape(self):
        """True DAG should have correct shape."""
        ds = data.generate_synthetic_dataset(generator_fn=two_node_generator_nonlinear)
        
        assert ds.true_DAG.shape == (ds.num_nodes, ds.num_nodes)
        assert ds.true_adj_np.shape == (ds.num_nodes, ds.num_nodes)
    
    def test_custom_generator(self):
        """Should accept custom generator function."""
        def custom_gen():
            rng = np.random.default_rng(0)
            obs = rng.normal(size=(100, 3)).astype(np.float32)
            adj = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
            names = ["a", "b", "c"]
            return obs, adj, names
        
        ds = data.generate_synthetic_dataset(generator_fn=custom_gen)
        
        assert ds.num_nodes == 3
        assert ds.node_names == ["a", "b", "c"]
    
    def test_custom_node_names_override(self):
        """Custom node_names should override generator's names."""
        ds = data.generate_synthetic_dataset(generator_fn=two_node_generator_nonlinear, node_names=["node_a", "node_b"])
        
        assert ds.node_names == ["node_a", "node_b"]
    
    def test_preserves_original_data(self):
        """Original (unscaled) data should be preserved."""
        ds = data.generate_synthetic_dataset(generator_fn=two_node_generator_nonlinear)
        
        # Original data should not be scaled (different from train_data)
        orig_mean = np.mean(ds.obs_train_orig, axis=0)
        # Original data likely has non-zero mean
        # Just check that it's stored and has correct shape
        assert ds.obs_train_orig.shape[0] == ds.dataset_size
        assert ds.obs_test_orig.shape[0] == config.n_test_samples
    

class TestGenerateSyntheticDatasetEdgeCases:
    """Edge case tests for generate_synthetic_dataset."""
    
    def test_generator_returning_no_names(self):
        """Generator returning None for names should use defaults."""
        def gen_no_names():
            rng = np.random.default_rng(0)
            obs = rng.normal(size=(100, 2)).astype(np.float32)
            adj = np.array([[0, 0], [1, 0]], dtype=np.float32)
            return obs, adj, None  # No names
        
        ds = data.generate_synthetic_dataset(generator_fn=gen_no_names)
        
        # Should generate default names
        assert ds.node_names == ["x1", "x2"]


class TestDataArrayTypes:
    """Tests for array type consistency in Dataset."""
    
    def test_train_data_is_jax_array(self):
        """train_data should be JAX array."""
        ds = data.generate_synthetic_dataset(generator_fn=two_node_generator_nonlinear)
        
        assert isinstance(ds.train_data, jnp.ndarray)
    
    def test_test_data_is_jax_array(self):
        """test_data_scaled should be JAX array."""
        ds = data.generate_synthetic_dataset(generator_fn=two_node_generator_nonlinear)
        
        assert isinstance(ds.test_data_scaled, jnp.ndarray)
    
    def test_noise_scales_is_jax_array(self):
        """noise_scales should be JAX array."""
        ds = data.generate_synthetic_dataset(generator_fn=two_node_generator_nonlinear)
        
        assert isinstance(ds.noise_scales, jnp.ndarray)
    
    def test_true_dag_is_jax_array(self):
        """true_DAG should be JAX array."""
        ds = data.generate_synthetic_dataset(generator_fn=two_node_generator_nonlinear)
        
        assert isinstance(ds.true_DAG, jnp.ndarray)
    
    def test_original_data_is_numpy_array(self):
        """Original data should be NumPy array."""
        ds = data.generate_synthetic_dataset(generator_fn=two_node_generator_nonlinear)
        
        assert isinstance(ds.obs_train_orig, np.ndarray)
        assert isinstance(ds.obs_test_orig, np.ndarray)


class TestScalerIntegration:
    """Tests for scaler integration with Dataset."""
    
    def test_scaler_inverse_transform_recovers_original(self):
        """Inverse transform of scaled data should recover original."""
        ds = data.generate_synthetic_dataset(generator_fn=two_node_generator_nonlinear)
        
        # Inverse transform training data
        recovered = ds.scaler.inverse_transform(np.array(ds.train_data))
        
        # Should be close to original (allowing for float precision)
        np.testing.assert_allclose(recovered, ds.obs_train_orig, rtol=1e-5, atol=1e-5)
    
    def test_scaler_works_on_test_data(self):
        """Scaler should work correctly on test data."""
        ds = data.generate_synthetic_dataset(generator_fn=two_node_generator_nonlinear)
        
        # Transform test data manually
        test_scaled_manual = ds.scaler.transform(ds.obs_test_orig)
        
        # Should match stored scaled test data
        np.testing.assert_allclose(
            test_scaled_manual, 
            np.array(ds.test_data_scaled), 
            rtol=1e-5, 
            atol=1e-5
        )


class TestDataValidation:
    """Tests for data validation and integrity."""
    
    def test_no_nan_in_train_data(self):
        """Training data should not contain NaN."""
        ds = data.generate_synthetic_dataset(generator_fn=two_node_generator_nonlinear)
        
        assert not np.any(np.isnan(np.array(ds.train_data)))
    
    def test_no_nan_in_test_data(self):
        """Test data should not contain NaN."""
        ds = data.generate_synthetic_dataset(generator_fn=two_node_generator_nonlinear)
        
        assert not np.any(np.isnan(np.array(ds.test_data_scaled)))
    
    def test_no_inf_in_train_data(self):
        """Training data should not contain Inf."""
        ds = data.generate_synthetic_dataset(generator_fn=two_node_generator_nonlinear)
        
        assert not np.any(np.isinf(np.array(ds.train_data)))
    
    def test_no_inf_in_test_data(self):
        """Test data should not contain Inf."""
        ds = data.generate_synthetic_dataset(generator_fn=two_node_generator_nonlinear)
        
        assert not np.any(np.isinf(np.array(ds.test_data_scaled)))
    
    def test_adjacency_diagonal_is_zero(self):
        """True adjacency diagonal should be zero (no self-loops)."""
        ds = data.generate_synthetic_dataset(generator_fn=two_node_generator_nonlinear)
        
        diag = np.diag(np.array(ds.true_DAG))
        np.testing.assert_array_equal(diag, np.zeros(ds.num_nodes))
    
    def test_adjacency_is_binary(self):
        """True adjacency should be binary (0 or 1)."""
        ds = data.generate_synthetic_dataset(generator_fn=two_node_generator_nonlinear)
        
        adj = np.array(ds.true_DAG)
        unique_vals = np.unique(adj)
        assert set(unique_vals).issubset({0.0, 1.0})


class TestCustomGeneratorIntegration:
    """Tests for custom generator integration."""
    
    def test_single_node_generator(self):
        """Should handle single-node generator (edge case)."""
        def single_node_gen():
            rng = np.random.default_rng(0)
            obs = rng.normal(size=(100, 1)).astype(np.float32)
            adj = np.zeros((1, 1), dtype=np.float32)
            return obs, adj, ["x"]
        
        ds = data.generate_synthetic_dataset(generator_fn=single_node_gen)
        
        assert ds.num_nodes == 1
        assert ds.true_DAG.shape == (1, 1)
    
    def test_larger_network_generator(self):
        """Should handle larger networks."""
        def large_gen():
            n_nodes = 5
            rng = np.random.default_rng(0)
            obs = rng.normal(size=(100, n_nodes)).astype(np.float32)
            # Simple chain: x1->x2->x3->x4->x5
            adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
            for i in range(n_nodes - 1):
                adj[i + 1, i] = 1.0
            names = [f"x{i+1}" for i in range(n_nodes)]
            return obs, adj, names
        
        ds = data.generate_synthetic_dataset(generator_fn=large_gen)
        
        assert ds.num_nodes == 5
        assert len(ds.node_names) == 5
        assert ds.true_DAG.shape == (5, 5)
        # Check chain structure
        assert np.sum(np.array(ds.true_DAG)) == 4  # 4 edges in 5-node chain


class TestSimulateSemNoiseTypes:
    """Tests for the noise_type parameter of simulate_sem (cases 7-9)."""

    @staticmethod
    def _chain_adj(m=4):
        adj = np.zeros((m, m), dtype=np.int32)
        for i in range(m - 1):
            adj[i + 1, i] = 1
        return adj

    @staticmethod
    def _edgeless(m=5):
        return np.zeros((m, m), dtype=np.int32)

    def test_default_gaussian_backwards_compatible(self):
        """Omitting noise_type must reproduce the historical Gaussian draws."""
        adj = self._chain_adj()
        d0, _ = data.simulate_sem(adj, 2000, sem_type='linear', rng_seed=7)
        d1, _ = data.simulate_sem(adj, 2000, sem_type='linear', rng_seed=7,
                                  noise_type='gaussian')
        np.testing.assert_array_equal(d0, d1)

    def test_all_noise_types_zero_mean_unit_std(self):
        """On an edgeless graph the data IS the noise: zero mean, std ~ scale."""
        iso = self._edgeless()
        for nt in ('gaussian', 'gumbel', 'exponential'):
            d, _ = data.simulate_sem(iso, 100_000, sem_type='linear',
                                     rng_seed=1, noise_type=nt)
            assert abs(float(d.mean())) < 0.03, f"{nt}: mean {d.mean()}"
            np.testing.assert_allclose(d.std(axis=0), 1.0, atol=0.05,
                                       err_msg=f"{nt}: std not ~1")

    def test_hetero_gaussian_per_node_scales(self):
        """hetero_gaussian draws per-node scales in [0.5, 2.0] * noise_scale."""
        iso = self._edgeless(8)
        d, _ = data.simulate_sem(iso, 50_000, sem_type='linear',
                                 rng_seed=3, noise_type='hetero_gaussian')
        stds = d.std(axis=0)
        assert abs(float(d.mean())) < 0.05
        assert np.all(stds > 0.4) and np.all(stds < 2.2)
        # Across-node heteroscedasticity: the scales must actually differ.
        assert stds.max() - stds.min() > 0.2

    def test_gumbel_and_exponential_are_skewed(self):
        """Non-Gaussian noise must show its positive skew (Gumbel ~1.14, Exp ~2)."""
        iso = self._edgeless(1)
        for nt, lo, hi in (('gumbel', 0.9, 1.4), ('exponential', 1.7, 2.3)):
            d, _ = data.simulate_sem(iso, 200_000, sem_type='linear',
                                     rng_seed=2, noise_type=nt)
            x = (d[:, 0] - d[:, 0].mean()) / d[:, 0].std()
            skew = float(np.mean(x ** 3))
            assert lo < skew < hi, f"{nt}: skew {skew}"

    def test_noise_type_reproducible(self):
        """Same seed + noise_type must give identical data."""
        adj = self._chain_adj()
        for nt in ('gumbel', 'exponential', 'hetero_gaussian'):
            d0, _ = data.simulate_sem(adj, 500, rng_seed=11, noise_type=nt)
            d1, _ = data.simulate_sem(adj, 500, rng_seed=11, noise_type=nt)
            np.testing.assert_array_equal(d0, d1)

    def test_nonlinear_and_mim_accept_noise_type(self):
        """All SEM branches route noise through the shared sampler."""
        adj = self._chain_adj()
        for st in ('nonlinear', 'mim'):
            d, _ = data.simulate_sem(adj, 200, sem_type=st, rng_seed=5,
                                     noise_type='exponential')
            assert np.all(np.isfinite(d))

    def test_unknown_noise_type_raises(self):
        with pytest.raises(ValueError, match="noise_type"):
            data.simulate_sem(self._edgeless(), 10, noise_type='cauchy')
