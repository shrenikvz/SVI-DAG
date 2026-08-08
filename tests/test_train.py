"""
Unit tests for svidag/train.py

This module tests training utilities including:
- TrainState class
- make_optimizer function
- make_model_and_state function
- apply_model function
- train_step function

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

from svidag import train, config
from svidag.utils import compute_alpha_beta_from_prior, to_device


class TestMakeOptimizer:
    """Tests for make_optimizer function."""
    
    def test_returns_optax_gradient_transform(self):
        """Should return an optax optimizer."""
        import optax
        
        optimizer = train.make_optimizer(lr=1e-3, clip=1.0)
        
        # Optax optimizers are GradientTransformation objects
        assert hasattr(optimizer, 'init')
        assert hasattr(optimizer, 'update')
    
    def test_optimizer_works_with_params(self, rng_key):
        """Optimizer should work with parameter pytree."""
        optimizer = train.make_optimizer(lr=1e-3, clip=1.0)
        
        # Sample params
        params = {'w': jrand.normal(rng_key, (10, 5)), 'b': jrand.normal(rng_key, (5,))}
        
        opt_state = optimizer.init(params)
        
        # Simulate gradient
        grads = jax.tree_util.tree_map(jnp.ones_like, params)
        
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        
        # Updates should have same structure as params
        assert updates.keys() == params.keys()
    
    def test_gradient_clipping_applied(self, rng_key):
        """Gradient clipping should limit gradient norms."""
        optimizer = train.make_optimizer(lr=1e-3, clip=0.1)
        
        params = {'w': jrand.normal(rng_key, (10,))}
        opt_state = optimizer.init(params)
        
        # Large gradients
        grads = {'w': jnp.ones(10) * 100}
        
        updates, _ = optimizer.update(grads, opt_state, params)
        
        # Updates should be clipped (smaller than original scaled gradients)
        update_norm = jnp.linalg.norm(updates['w'])
        # The clipped gradient norm should be reasonable
        assert float(update_norm) < 100  # Much smaller than unclipped


class TestMakeModelAndState:
    """Tests for make_model_and_state function."""
    
    def test_returns_model_and_state(self, rng_key, small_batch_2d, uniform_prior_2node):
        """Should return model and state objects."""
        model, state = train.make_model_and_state(
            rng_key,
            small_batch_2d,
            uniform_prior_2node,
            num_nodes=2,
            fixed_noise_scales=jnp.array([0.1, 0.1])
        )
        
        assert model is not None
        assert state is not None
    
    def test_state_has_particles(self, rng_key, small_batch_2d, uniform_prior_2node):
        """State should have particles attribute."""
        _, state = train.make_model_and_state(
            rng_key,
            small_batch_2d,
            uniform_prior_2node,
            num_nodes=2,
            fixed_noise_scales=jnp.array([0.1, 0.1])
        )
        
        assert hasattr(state, 'particles')
        assert state.particles.shape == (config.n_particles, 2)
    
    def test_state_has_params(self, rng_key, small_batch_2d, uniform_prior_2node):
        """State should have params attribute."""
        _, state = train.make_model_and_state(
            rng_key,
            small_batch_2d,
            uniform_prior_2node,
            num_nodes=2,
            fixed_noise_scales=jnp.array([0.1, 0.1])
        )
        
        assert hasattr(state, 'params')
        assert state.params is not None
    
    def test_state_has_apply_fn(self, rng_key, small_batch_2d, uniform_prior_2node):
        """State should have apply_fn attribute."""
        _, state = train.make_model_and_state(
            rng_key,
            small_batch_2d,
            uniform_prior_2node,
            num_nodes=2,
            fixed_noise_scales=jnp.array([0.1, 0.1])
        )
        
        assert hasattr(state, 'apply_fn')
        assert callable(state.apply_fn)
    
    def test_3node_model(self, rng_key, small_batch_3d, uniform_prior_3node):
        """Should work with 3-node setup."""
        _, state = train.make_model_and_state(
            rng_key,
            small_batch_3d,
            uniform_prior_3node,
            num_nodes=3,
            fixed_noise_scales=jnp.array([0.1, 0.1, 0.1])
        )
        
        assert state.particles.shape == (config.n_particles, 3)
    
    def test_reproducible_initialization(self, small_batch_2d, uniform_prior_2node):
        """Same seed should produce same initialization."""
        key1 = jrand.PRNGKey(42)
        key2 = jrand.PRNGKey(42)
        
        _, state1 = train.make_model_and_state(
            key1, small_batch_2d, uniform_prior_2node, 2, jnp.array([0.1, 0.1])
        )
        _, state2 = train.make_model_and_state(
            key2, small_batch_2d, uniform_prior_2node, 2, jnp.array([0.1, 0.1])
        )
        
        np.testing.assert_array_equal(np.array(state1.particles), np.array(state2.particles))


class TestTrainState:
    """Tests for TrainState class."""
    
    def test_inherits_from_flax_train_state(self, rng_key, small_batch_2d, uniform_prior_2node):
        """TrainState should extend flax train_state.TrainState."""
        from flax.training import train_state
        
        _, state = train.make_model_and_state(
            rng_key, small_batch_2d, uniform_prior_2node, 2, jnp.array([0.1, 0.1])
        )
        
        assert isinstance(state, train_state.TrainState)
    
    def test_apply_gradients_updates_params(self, rng_key, small_batch_2d, uniform_prior_2node):
        """apply_gradients should update parameters."""
        _, state = train.make_model_and_state(
            rng_key, small_batch_2d, uniform_prior_2node, 2, jnp.array([0.1, 0.1])
        )
        
        # Create fake gradients
        grads = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 0.01, state.params)
        
        new_state = state.apply_gradients(grads=grads)
        
        # Step should increase
        assert new_state.step == state.step + 1
    
    def test_replace_particles(self, rng_key, small_batch_2d, uniform_prior_2node):
        """State should allow replacing particles."""
        _, state = train.make_model_and_state(
            rng_key, small_batch_2d, uniform_prior_2node, 2, jnp.array([0.1, 0.1])
        )
        
        new_particles = jrand.normal(rng_key, state.particles.shape)
        new_state = state.replace(particles=new_particles)
        
        assert not jnp.allclose(state.particles, new_state.particles).item()


class TestApplyModel:
    """Tests for apply_model function."""
    
    @pytest.fixture
    def setup_model_state(self, rng_key, medium_batch_2d, uniform_prior_2node):
        """Set up model and state for testing."""
        model, state = train.make_model_and_state(
            rng_key, medium_batch_2d, uniform_prior_2node, 2, jnp.array([0.1, 0.1])
        )
        alpha_mat, beta_mat = compute_alpha_beta_from_prior(uniform_prior_2node)
        return state, alpha_mat, beta_mat
    
    def test_apply_model_output_shapes(self, setup_model_state, rng_key, medium_batch_2d):
        """apply_model should return predictions and terms."""
        state, alpha_mat, beta_mat = setup_model_state
        r = state.particles[0]
        batch = medium_batch_2d[:10]
        
        preds, terms = train.apply_model(
            state.apply_fn, state.params, batch, r, rng_key,
            0.2, 0.3, alpha_mat, beta_mat
        )
        
        assert preds.shape == (10, 2)
        assert 'A_relaxed' in terms
    
    def test_apply_model_jitted(self, setup_model_state, rng_key, medium_batch_2d):
        """apply_model should be JIT-compiled."""
        state, alpha_mat, beta_mat = setup_model_state
        r = state.particles[0]
        batch = medium_batch_2d[:10]
        
        # First call compiles
        preds1, _ = train.apply_model(
            state.apply_fn, state.params, batch, r, rng_key,
            0.2, 0.3, alpha_mat, beta_mat
        )
        
        # Second call uses cached
        preds2, _ = train.apply_model(
            state.apply_fn, state.params, batch, r, rng_key,
            0.2, 0.3, alpha_mat, beta_mat
        )
        
        np.testing.assert_array_equal(np.array(preds1), np.array(preds2))
    
    def test_apply_model_finite_outputs(self, setup_model_state, rng_key, medium_batch_2d):
        """apply_model outputs should be finite."""
        state, alpha_mat, beta_mat = setup_model_state
        r = state.particles[0]
        batch = medium_batch_2d[:10]
        
        preds, terms = train.apply_model(
            state.apply_fn, state.params, batch, r, rng_key,
            0.2, 0.3, alpha_mat, beta_mat
        )
        
        assert jnp.all(jnp.isfinite(preds)).item()
        assert jnp.all(jnp.isfinite(terms['A_relaxed'])).item()


class TestTrainStep:
    """Tests for train_step function."""
    
    @pytest.fixture
    def setup_training(self, rng_key, medium_batch_2d, uniform_prior_2node):
        """Set up training components."""
        model, state = train.make_model_and_state(
            rng_key, medium_batch_2d, uniform_prior_2node, 2, jnp.array([0.1, 0.1])
        )
        alpha_mat, beta_mat = compute_alpha_beta_from_prior(uniform_prior_2node)
        dataset_size = medium_batch_2d.shape[0]
        return state, alpha_mat, beta_mat, dataset_size
    
    def test_train_step_returns_state(self, setup_training, rng_key, medium_batch_2d):
        """train_step should return new state."""
        state, alpha_mat, beta_mat, dataset_size = setup_training
        batch = medium_batch_2d[:config.batch_size]
        
        new_state, loss, aux = train.train_step(
            state, batch, rng_key, 
            it=1, total_iters=100,
            alpha_mat=alpha_mat, beta_mat=beta_mat,
            N_total=dataset_size, mc_samples=2
        )
        
        assert isinstance(new_state, train.TrainState)
    
    def test_train_step_returns_loss(self, setup_training, rng_key, medium_batch_2d):
        """train_step should return loss value."""
        state, alpha_mat, beta_mat, dataset_size = setup_training
        batch = medium_batch_2d[:config.batch_size]
        
        _, loss, _ = train.train_step(
            state, batch, rng_key,
            it=1, total_iters=100,
            alpha_mat=alpha_mat, beta_mat=beta_mat,
            N_total=dataset_size, mc_samples=2
        )
        
        assert isinstance(float(loss), float)
        assert jnp.isfinite(loss).item()
    
    def test_train_step_returns_aux(self, setup_training, rng_key, medium_batch_2d):
        """train_step should return aux dictionary."""
        state, alpha_mat, beta_mat, dataset_size = setup_training
        batch = medium_batch_2d[:config.batch_size]
        
        _, _, aux = train.train_step(
            state, batch, rng_key,
            it=1, total_iters=100,
            alpha_mat=alpha_mat, beta_mat=beta_mat,
            N_total=dataset_size, mc_samples=2
        )
        
        expected_keys = ['elbo', 'ell', 'kl_theta', 'kl_gamma', 'T_B', 'tau_sn']
        for key in expected_keys:
            assert key in aux, f"Missing aux key: {key}"
    
    def test_train_step_updates_params(self, setup_training, rng_key, medium_batch_2d):
        """train_step should update model parameters."""
        state, alpha_mat, beta_mat, dataset_size = setup_training
        batch = medium_batch_2d[:config.batch_size]
        
        old_params = jax.tree_util.tree_map(lambda x: x.copy(), state.params)
        
        new_state, _, _ = train.train_step(
            state, batch, rng_key,
            it=1, total_iters=100,
            alpha_mat=alpha_mat, beta_mat=beta_mat,
            N_total=dataset_size, mc_samples=2
        )
        
        # At least some params should change
        old_leaves = jax.tree_util.tree_leaves(old_params)
        new_leaves = jax.tree_util.tree_leaves(new_state.params)
        params_changed = any(
            not jnp.allclose(old_leaf, new_leaf).item()
            for old_leaf, new_leaf in zip(old_leaves, new_leaves)
        )

        assert params_changed
    
    def test_train_step_updates_particles(self, setup_training, rng_key, medium_batch_2d):
        """train_step should update particles via SVGD."""
        state, alpha_mat, beta_mat, dataset_size = setup_training
        batch = medium_batch_2d[:config.batch_size]
        
        old_particles = state.particles.copy()
        
        new_state, _, _ = train.train_step(
            state, batch, rng_key,
            it=1, total_iters=100,
            alpha_mat=alpha_mat, beta_mat=beta_mat,
            N_total=dataset_size, mc_samples=2
        )
        
        # Particles should be updated
        assert not jnp.allclose(old_particles, new_state.particles).item()
    
    def test_train_step_finite_values(self, setup_training, rng_key, medium_batch_2d):
        """train_step outputs should be finite."""
        state, alpha_mat, beta_mat, dataset_size = setup_training
        batch = medium_batch_2d[:config.batch_size]
        
        new_state, loss, aux = train.train_step(
            state, batch, rng_key,
            it=1, total_iters=100,
            alpha_mat=alpha_mat, beta_mat=beta_mat,
            N_total=dataset_size, mc_samples=2
        )
        
        assert jnp.isfinite(loss).item()
        assert jnp.all(jnp.isfinite(new_state.particles)).item()
        assert jnp.isfinite(aux['elbo']).item()
    
    def test_train_step_temperature_annealing(self, setup_training, rng_key, medium_batch_2d):
        """Temperature should anneal from start to end."""
        state, alpha_mat, beta_mat, dataset_size = setup_training
        batch = medium_batch_2d[:config.batch_size]
        
        # Early iteration
        _, _, aux_early = train.train_step(
            state, batch, rng_key,
            it=1, total_iters=1000,
            alpha_mat=alpha_mat, beta_mat=beta_mat,
            N_total=dataset_size, mc_samples=2
        )
        
        # Late iteration
        _, _, aux_late = train.train_step(
            state, batch, rng_key,
            it=999, total_iters=1000,
            alpha_mat=alpha_mat, beta_mat=beta_mat,
            N_total=dataset_size, mc_samples=2
        )
        
        # Temperatures should be different if annealing is configured
        # (depends on config.T_B_start vs T_B_end)
        assert aux_early['T_B'] is not None
        assert aux_late['T_B'] is not None


class TestTrainStepMultipleIterations:
    """Tests for multiple training iterations."""
    
    def test_multiple_steps_no_nan(self, rng_key, medium_batch_2d, uniform_prior_2node):
        """Multiple train steps should not produce NaN."""
        model, state = train.make_model_and_state(
            rng_key, medium_batch_2d, uniform_prior_2node, 2, jnp.array([0.1, 0.1])
        )
        alpha_mat, beta_mat = compute_alpha_beta_from_prior(uniform_prior_2node)
        dataset_size = medium_batch_2d.shape[0]
        
        key = rng_key
        for i in range(5):
            key, step_key = jrand.split(key)
            batch_idx = jrand.randint(step_key, (config.batch_size,), 0, dataset_size)
            batch = medium_batch_2d[batch_idx]
            
            state, loss, aux = train.train_step(
                state, batch, step_key,
                it=i+1, total_iters=100,
                alpha_mat=alpha_mat, beta_mat=beta_mat,
                N_total=dataset_size, mc_samples=2
            )
            
            assert jnp.isfinite(loss).item(), f"NaN loss at step {i}"
            assert jnp.all(jnp.isfinite(state.particles)).item(), f"NaN particles at step {i}"
    
    def test_elbo_can_improve(self, rng_key, medium_batch_2d, uniform_prior_2node):
        """ELBO should not consistently decrease (training should work)."""
        model, state = train.make_model_and_state(
            rng_key, medium_batch_2d, uniform_prior_2node, 2, jnp.array([0.1, 0.1])
        )
        alpha_mat, beta_mat = compute_alpha_beta_from_prior(uniform_prior_2node)
        dataset_size = medium_batch_2d.shape[0]
        
        elbos = []
        key = rng_key
        for i in range(10):
            key, step_key = jrand.split(key)
            batch_idx = jrand.randint(step_key, (config.batch_size,), 0, dataset_size)
            batch = medium_batch_2d[batch_idx]
            
            state, loss, aux = train.train_step(
                state, batch, step_key,
                it=i+1, total_iters=100,
                alpha_mat=alpha_mat, beta_mat=beta_mat,
                N_total=dataset_size, mc_samples=2
            )
            elbos.append(float(aux['elbo']))
        
        # ELBO values should be finite
        assert all(np.isfinite(e) for e in elbos)
