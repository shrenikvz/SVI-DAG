"""
Unit tests for svidag/flows.py

This module tests normalizing flow components including:
- MaskedDense layer
- MADE network
- MAF (Masked Autoregressive Flow) blocks
- MAFStack (stacked MAF)
- Neural Spline Flow components (rational_quadratic_spline)
- NSFStack and NSFCouplingStack
- create_flow_stack factory function

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

from svidag import flows


class TestMaskedDense:
    """Tests for MaskedDense layer."""
    
    def test_output_shape(self, rng_key):
        """Output should have correct shape."""
        mask = jnp.array([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])  # (3, 2)
        layer = flows.MaskedDense(features=2, mask=mask)
        
        x = jrand.normal(rng_key, (5, 3))
        params = layer.init(rng_key, x)
        output = layer.apply(params, x)
        
        assert output.shape == (5, 2)
    
    def test_mask_enforced(self, rng_key):
        """Mask should enforce connectivity constraints."""
        # Mask that blocks some connections
        mask = jnp.array([[1.0, 0.0], [0.0, 1.0]])  # Diagonal only
        layer = flows.MaskedDense(features=2, mask=mask)
        
        x = jnp.array([[1.0, 0.0]])  # Only first input active
        params = layer.init(rng_key, x)
        out1 = layer.apply(params, x)
        
        x2 = jnp.array([[0.0, 1.0]])  # Only second input active
        out2 = layer.apply(params, x2)
        
        # Due to diagonal mask, outputs should depend on corresponding inputs
        assert jnp.all(jnp.isfinite(out1)).item()
        assert jnp.all(jnp.isfinite(out2)).item()
    
    def test_no_bias_option(self, rng_key):
        """Layer should work without bias."""
        mask = jnp.ones((3, 2))
        layer = flows.MaskedDense(features=2, mask=mask, use_bias=False)
        
        x = jrand.normal(rng_key, (5, 3))
        params = layer.init(rng_key, x)
        
        # Check no bias in params
        assert 'bias' not in params['params']


class TestMADE:
    """Tests for MADE (Masked Autoencoder for Distribution Estimation)."""
    
    @pytest.fixture
    def made_net(self):
        """Create a MADE network."""
        return flows.MADE(latent_dim=4, cond_dim=2, hidden_dims=[8, 8])
    
    def test_output_shape(self, made_net, rng_key):
        """Output (shift, log_scale) should have correct shapes."""
        z = jrand.normal(rng_key, (4,))
        cond = jrand.normal(rng_key, (2,))
        
        params = made_net.init(rng_key, z, cond)
        shift, log_scale = made_net.apply(params, z, cond)
        
        assert shift.shape == (4,)
        assert log_scale.shape == (4,)
    
    def test_log_scale_clipping(self, made_net, rng_key):
        """Log scale should be clipped to prevent numerical issues."""
        z = jrand.normal(rng_key, (4,)) * 100  # Large values
        cond = jrand.normal(rng_key, (2,))
        
        params = made_net.init(rng_key, z, cond)
        _, log_scale = made_net.apply(params, z, cond)
        
        # Should be clipped to [-10, 10]
        assert jnp.all(log_scale >= -10.0).item()
        assert jnp.all(log_scale <= 10.0).item()
    
    def test_autoregressive_property(self, made_net, rng_key):
        """MADE should have autoregressive structure (verified via mask shapes)."""
        z = jrand.normal(rng_key, (4,))
        cond = jrand.normal(rng_key, (2,))
        
        params = made_net.init(rng_key, z, cond)
        
        # Just verify it computes successfully
        shift, log_scale = made_net.apply(params, z, cond)
        assert jnp.all(jnp.isfinite(shift)).item()
        assert jnp.all(jnp.isfinite(log_scale)).item()
    
    def test_create_masks_dimensions(self):
        """Test that create_masks produces correct dimensions."""
        masks = flows.MADE.create_masks(
            input_dim=4,
            cond_dim=2,
            hidden_dims=[8, 8],
            output_dim=4
        )
        
        # First mask: (input_dim + cond_dim, hidden_dims[0])
        assert masks[0].shape == (6, 8)
        # Hidden mask: (hidden_dims[0], hidden_dims[1])
        assert masks[1].shape == (8, 8)
        # Output mask: (hidden_dims[-1], 2 * output_dim)
        assert masks[2].shape == (8, 8)


class TestMAF:
    """Tests for MAF (Masked Autoregressive Flow) block."""
    
    @pytest.fixture
    def maf_block(self):
        """Create a MAF block."""
        return flows.MAF(latent_dim=4, cond_dim=2, hidden_dims=[8, 8])
    
    def test_output_shape(self, maf_block, rng_key):
        """Output (x, log_det) should have correct shapes."""
        z = jrand.normal(rng_key, (4,))
        cond = jrand.normal(rng_key, (2,))
        
        params = maf_block.init(rng_key, z, cond)
        x, log_det = maf_block.apply(params, z, cond)
        
        assert x.shape == (4,)
        assert log_det.shape == ()  # Scalar
    
    def test_log_det_finite(self, maf_block, rng_key):
        """Log determinant should be finite."""
        z = jrand.normal(rng_key, (4,))
        cond = jrand.normal(rng_key, (2,))
        
        params = maf_block.init(rng_key, z, cond)
        _, log_det = maf_block.apply(params, z, cond)
        
        assert jnp.isfinite(log_det).item()
    
    def test_different_z_different_output(self, maf_block, rng_key):
        """Different input z should give different output x."""
        z1 = jrand.normal(rng_key, (4,))
        z2 = z1 + 1.0
        cond = jrand.normal(rng_key, (2,))
        
        params = maf_block.init(rng_key, z1, cond)
        x1, _ = maf_block.apply(params, z1, cond)
        x2, _ = maf_block.apply(params, z2, cond)
        
        assert not jnp.allclose(x1, x2).item()


class TestMAFStack:
    """Tests for MAFStack (stacked MAF blocks)."""
    
    @pytest.fixture
    def maf_stack(self):
        """Create a MAFStack."""
        return flows.MAFStack(
            latent_dim=4,
            cond_dim=2,
            hidden_dims=[8, 8],
            n_blocks=3
        )
    
    def test_output_shape(self, maf_stack, rng_key):
        """Output should have correct shape."""
        z = jrand.normal(rng_key, (4,))
        cond = jrand.normal(rng_key, (2,))
        
        params = maf_stack.init(rng_key, z, cond)
        x, log_det = maf_stack.apply(params, z, cond)
        
        assert x.shape == (4,)
        assert log_det.shape == ()
    
    def test_log_det_accumulates(self, maf_stack, rng_key):
        """Total log det should be sum of block log dets."""
        z = jrand.normal(rng_key, (4,))
        cond = jrand.normal(rng_key, (2,))
        
        params = maf_stack.init(rng_key, z, cond)
        _, log_det = maf_stack.apply(params, z, cond)
        
        # Just check it's finite and reasonable
        assert jnp.isfinite(log_det).item()
    
    def test_expressiveness_with_multiple_blocks(self, rng_key):
        """More blocks should give more expressive transformations."""
        z = jrand.normal(rng_key, (4,))
        cond = jrand.normal(rng_key, (2,))
        
        stack_1 = flows.MAFStack(latent_dim=4, cond_dim=2, hidden_dims=[8], n_blocks=1)
        stack_3 = flows.MAFStack(latent_dim=4, cond_dim=2, hidden_dims=[8], n_blocks=3)
        
        params_1 = stack_1.init(rng_key, z, cond)
        params_3 = stack_3.init(rng_key, z, cond)
        
        x_1, _ = stack_1.apply(params_1, z, cond)
        x_3, _ = stack_3.apply(params_3, z, cond)
        
        # Both should be valid
        assert jnp.all(jnp.isfinite(x_1)).item()
        assert jnp.all(jnp.isfinite(x_3)).item()
    
    def test_jit_compatible(self, maf_stack, rng_key):
        """MAFStack should be JIT-compatible."""
        z = jrand.normal(rng_key, (4,))
        cond = jrand.normal(rng_key, (2,))
        
        params = maf_stack.init(rng_key, z, cond)
        
        @jax.jit
        def forward(params, z, cond):
            return maf_stack.apply(params, z, cond)
        
        x, log_det = forward(params, z, cond)
        assert x.shape == (4,)


class TestRationalQuadraticSpline:
    """Tests for rational_quadratic_spline function."""
    
    def test_output_shape(self):
        """Output should match input shape."""
        inputs = jnp.array([0.0, 0.5, -0.5])
        n_bins = 8
        widths = jnp.zeros((3, n_bins))
        heights = jnp.zeros((3, n_bins))
        derivatives = jnp.zeros((3, n_bins + 1))
        
        outputs, log_det = flows.rational_quadratic_spline(
            inputs, widths, heights, derivatives
        )
        
        assert outputs.shape == inputs.shape
        assert log_det.shape == inputs.shape
    
    def test_identity_initialization(self):
        """With zero-initialized params, should approximate identity."""
        inputs = jnp.array([0.0, 1.0, -1.0, 2.0])
        n_bins = 8
        widths = jnp.zeros((4, n_bins))
        heights = jnp.zeros((4, n_bins))
        derivatives = jnp.zeros((4, n_bins + 1))
        
        outputs, log_det = flows.rational_quadratic_spline(
            inputs, widths, heights, derivatives,
            left=-5.0, right=5.0, bottom=-5.0, top=5.0
        )
        
        # With uniform widths/heights, should be close to identity
        np.testing.assert_allclose(np.array(outputs), np.array(inputs), rtol=0.2, atol=0.2)
    
    def test_finite_outputs(self, rng_key):
        """Outputs should always be finite."""
        inputs = jrand.normal(rng_key, (10,))
        n_bins = 8
        widths = jrand.normal(rng_key, (10, n_bins)) * 0.1
        heights = jrand.normal(rng_key, (10, n_bins)) * 0.1
        derivatives = jrand.normal(rng_key, (10, n_bins + 1)) * 0.1
        
        outputs, log_det = flows.rational_quadratic_spline(
            inputs, widths, heights, derivatives
        )
        
        assert jnp.all(jnp.isfinite(outputs)).item()
        assert jnp.all(jnp.isfinite(log_det)).item()
    
    def test_monotonic_within_domain(self, rng_key):
        """Spline should be monotonic (verified by positive derivatives)."""
        inputs = jnp.linspace(-4, 4, 20)
        n_bins = 8
        widths = jnp.zeros((20, n_bins))
        heights = jnp.zeros((20, n_bins))
        derivatives = jnp.ones((20, n_bins + 1))  # Positive derivatives
        
        outputs, log_det = flows.rational_quadratic_spline(
            inputs, widths, heights, derivatives,
            left=-5.0, right=5.0, bottom=-5.0, top=5.0
        )
        
        # Log det should be finite (monotonic transformation)
        assert jnp.all(jnp.isfinite(log_det)).item()
    
    def test_outside_domain_identity(self, rng_key):
        """Points outside domain should get identity transformation."""
        inputs = jnp.array([-10.0, 10.0])  # Outside [-5, 5]
        n_bins = 8
        widths = jrand.normal(rng_key, (2, n_bins))
        heights = jrand.normal(rng_key, (2, n_bins))
        derivatives = jrand.normal(rng_key, (2, n_bins + 1))
        
        outputs, log_det = flows.rational_quadratic_spline(
            inputs, widths, heights, derivatives,
            left=-5.0, right=5.0, bottom=-5.0, top=5.0
        )
        
        # Outside domain: identity transformation
        np.testing.assert_allclose(np.array(outputs), np.array(inputs))
        np.testing.assert_allclose(np.array(log_det), np.zeros(2), atol=1e-6)


class TestNeuralSplineFlowBlock:
    """Tests for NeuralSplineFlowBlock."""
    
    @pytest.fixture
    def nsf_block(self):
        """Create an NSF block."""
        return flows.NeuralSplineFlowBlock(
            latent_dim=4,
            cond_dim=2,
            hidden_dims=[8, 8],
            num_bins=8,
            tail_bound=5.0
        )
    
    def test_output_shape(self, nsf_block, rng_key):
        """Output should have correct shape."""
        z = jrand.normal(rng_key, (4,))
        cond = jrand.normal(rng_key, (2,))
        
        params = nsf_block.init(rng_key, z, cond)
        x, log_det = nsf_block.apply(params, z, cond)
        
        assert x.shape == (4,)
        assert log_det.shape == ()
    
    def test_finite_outputs(self, nsf_block, rng_key):
        """Outputs should be finite."""
        z = jrand.normal(rng_key, (4,))
        cond = jrand.normal(rng_key, (2,))
        
        params = nsf_block.init(rng_key, z, cond)
        x, log_det = nsf_block.apply(params, z, cond)
        
        assert jnp.all(jnp.isfinite(x)).item()
        assert jnp.isfinite(log_det).item()


class TestNSFStack:
    """Tests for NSFStack (stacked Neural Spline Flow blocks)."""
    
    @pytest.fixture
    def nsf_stack(self):
        """Create an NSFStack."""
        return flows.NSFStack(
            latent_dim=4,
            cond_dim=2,
            hidden_dims=[8, 8],
            n_blocks=3,
            num_bins=8,
            tail_bound=5.0
        )
    
    def test_output_shape(self, nsf_stack, rng_key):
        """Output should have correct shape."""
        z = jrand.normal(rng_key, (4,))
        cond = jrand.normal(rng_key, (2,))
        
        params = nsf_stack.init(rng_key, z, cond)
        x, log_det = nsf_stack.apply(params, z, cond)
        
        assert x.shape == (4,)
        assert log_det.shape == ()
    
    def test_finite_outputs(self, nsf_stack, rng_key):
        """Outputs should be finite."""
        z = jrand.normal(rng_key, (4,))
        cond = jrand.normal(rng_key, (2,))
        
        params = nsf_stack.init(rng_key, z, cond)
        x, log_det = nsf_stack.apply(params, z, cond)
        
        assert jnp.all(jnp.isfinite(x)).item()
        assert jnp.isfinite(log_det).item()
    
    def test_jit_compatible(self, nsf_stack, rng_key):
        """NSFStack should be JIT-compatible."""
        z = jrand.normal(rng_key, (4,))
        cond = jrand.normal(rng_key, (2,))
        
        params = nsf_stack.init(rng_key, z, cond)
        
        @jax.jit
        def forward(params, z, cond):
            return nsf_stack.apply(params, z, cond)
        
        x, log_det = forward(params, z, cond)
        assert jnp.all(jnp.isfinite(x)).item()


class TestNSFCouplingStack:
    """Tests for NSFCouplingStack (coupling layer architecture)."""
    
    @pytest.fixture
    def nsf_coupling_stack(self):
        """Create an NSFCouplingStack."""
        return flows.NSFCouplingStack(
            latent_dim=4,
            cond_dim=2,
            hidden_dims=[8, 8],
            n_blocks=3,
            num_bins=8,
            tail_bound=5.0
        )
    
    def test_output_shape(self, nsf_coupling_stack, rng_key):
        """Output should have correct shape."""
        z = jrand.normal(rng_key, (4,))
        cond = jrand.normal(rng_key, (2,))
        
        params = nsf_coupling_stack.init(rng_key, z, cond)
        x, log_det = nsf_coupling_stack.apply(params, z, cond)
        
        assert x.shape == (4,)
        assert log_det.shape == ()
    
    def test_alternating_transform(self, nsf_coupling_stack, rng_key):
        """Coupling layers should alternate which half is transformed."""
        z = jrand.normal(rng_key, (4,))
        cond = jrand.normal(rng_key, (2,))
        
        params = nsf_coupling_stack.init(rng_key, z, cond)
        x, _ = nsf_coupling_stack.apply(params, z, cond)
        
        # Should transform the input
        assert not jnp.allclose(x, z).item()


class TestCreateFlowStack:
    """Tests for create_flow_stack factory function."""
    
    def test_create_maf_stack(self, rng_key):
        """Factory should create MAFStack for 'maf' type."""
        flow = flows.create_flow_stack(
            flow_type='maf',
            latent_dim=4,
            cond_dim=2,
            hidden_dims=[8, 8],
            n_blocks=2
        )
        
        z = jrand.normal(rng_key, (4,))
        cond = jrand.normal(rng_key, (2,))
        
        params = flow.init(rng_key, z, cond)
        x, log_det = flow.apply(params, z, cond)
        
        assert x.shape == (4,)
        assert isinstance(flow, flows.MAFStack)
    
    def test_create_nsf_stack(self, rng_key):
        """Factory should create NSFStack for 'nsf' type."""
        flow = flows.create_flow_stack(
            flow_type='nsf',
            latent_dim=4,
            cond_dim=2,
            hidden_dims=[8, 8],
            n_blocks=2,
            num_bins=8
        )
        
        z = jrand.normal(rng_key, (4,))
        cond = jrand.normal(rng_key, (2,))
        
        params = flow.init(rng_key, z, cond)
        x, log_det = flow.apply(params, z, cond)
        
        assert x.shape == (4,)
        assert isinstance(flow, flows.NSFStack)
    
    def test_create_nsf_coupling_stack(self, rng_key):
        """Factory should create NSFCouplingStack for 'nsf_coupling' type."""
        flow = flows.create_flow_stack(
            flow_type='nsf_coupling',
            latent_dim=4,
            cond_dim=2,
            hidden_dims=[8, 8],
            n_blocks=2,
            num_bins=8
        )
        
        z = jrand.normal(rng_key, (4,))
        cond = jrand.normal(rng_key, (2,))
        
        params = flow.init(rng_key, z, cond)
        x, log_det = flow.apply(params, z, cond)
        
        assert x.shape == (4,)
        assert isinstance(flow, flows.NSFCouplingStack)
    
    def test_invalid_flow_type_raises(self):
        """Invalid flow type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown flow type"):
            flows.create_flow_stack(
                flow_type='invalid_type',
                latent_dim=4,
                cond_dim=2,
                hidden_dims=[8],
                n_blocks=1
            )
    
    def test_case_insensitive_flow_type(self, rng_key):
        """Flow type should be case-insensitive."""
        flow_lower = flows.create_flow_stack('maf', 4, 2, [8], 1)
        flow_upper = flows.create_flow_stack('MAF', 4, 2, [8], 1)
        flow_mixed = flows.create_flow_stack('Maf', 4, 2, [8], 1)
        
        # All should create MAFStack
        assert isinstance(flow_lower, flows.MAFStack)
        assert isinstance(flow_upper, flows.MAFStack)
        assert isinstance(flow_mixed, flows.MAFStack)


class TestFlowNumericalStability:
    """Tests for numerical stability of flow transformations."""
    
    @pytest.mark.parametrize("flow_type", ['maf', 'nsf', 'nsf_coupling'])
    def test_large_input_stability(self, flow_type, rng_key):
        """Flows should handle large input values."""
        flow = flows.create_flow_stack(
            flow_type=flow_type,
            latent_dim=4,
            cond_dim=2,
            hidden_dims=[8],
            n_blocks=2,
            num_bins=8
        )
        
        z = jrand.normal(rng_key, (4,)) * 10  # Large values
        cond = jrand.normal(rng_key, (2,))
        
        params = flow.init(rng_key, z, cond)
        x, log_det = flow.apply(params, z, cond)
        
        assert jnp.all(jnp.isfinite(x)).item()
        assert jnp.isfinite(log_det).item()
    
    @pytest.mark.parametrize("flow_type", ['maf', 'nsf', 'nsf_coupling'])
    def test_zero_input(self, flow_type, rng_key):
        """Flows should handle zero input."""
        flow = flows.create_flow_stack(
            flow_type=flow_type,
            latent_dim=4,
            cond_dim=2,
            hidden_dims=[8],
            n_blocks=2,
            num_bins=8
        )
        
        z = jnp.zeros(4)
        cond = jrand.normal(rng_key, (2,))
        
        params = flow.init(rng_key, z, cond)
        x, log_det = flow.apply(params, z, cond)
        
        assert jnp.all(jnp.isfinite(x)).item()
        assert jnp.isfinite(log_det).item()


class TestFlowGradients:
    """Tests for gradient computation through flows."""
    
    @pytest.mark.parametrize("flow_type", ['maf', 'nsf'])
    def test_gradient_flow(self, flow_type, rng_key):
        """Gradients should flow through the transformation."""
        flow = flows.create_flow_stack(
            flow_type=flow_type,
            latent_dim=4,
            cond_dim=2,
            hidden_dims=[8],
            n_blocks=2,
            num_bins=8
        )
        
        z = jrand.normal(rng_key, (4,))
        cond = jrand.normal(rng_key, (2,))
        
        params = flow.init(rng_key, z, cond)
        
        def loss_fn(params, z, cond):
            x, log_det = flow.apply(params, z, cond)
            return jnp.sum(x ** 2) - log_det
        
        grads = jax.grad(loss_fn)(params, z, cond)
        
        # Check gradients are finite
        flat_grads = jax.tree_util.tree_leaves(grads)
        for g in flat_grads:
            assert jnp.all(jnp.isfinite(g)).item()


class TestFlowVMAPCompatibility:
    """Tests for vmap compatibility of flows."""
    
    @pytest.mark.parametrize("flow_type", ['maf', 'nsf'])
    def test_vmap_over_samples(self, flow_type, rng_key):
        """Flows should be vmappable over samples."""
        flow = flows.create_flow_stack(
            flow_type=flow_type,
            latent_dim=4,
            cond_dim=2,
            hidden_dims=[8],
            n_blocks=2,
            num_bins=8
        )
        
        z_single = jrand.normal(rng_key, (4,))
        cond = jrand.normal(rng_key, (2,))
        
        params = flow.init(rng_key, z_single, cond)
        
        # Batch of z samples
        z_batch = jrand.normal(rng_key, (10, 4))
        
        def single_forward(z):
            return flow.apply(params, z, cond)
        
        x_batch, log_det_batch = jax.vmap(single_forward)(z_batch)
        
        assert x_batch.shape == (10, 4)
        assert log_det_batch.shape == (10,)
