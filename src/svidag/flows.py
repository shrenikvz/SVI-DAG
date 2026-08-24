'''
Normalizing Flows module for the SVIDAG.

Current coded NF's: 
1. Masked Autoregressive Flows (MAF)
2. Neural Spline Flows (NSF) - Rational Quadratic Splines

Author: Shrenik Zinage
'''

from typing import Tuple
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
from flax import linen as nn

# =============================================================================
# Constants for Neural Spline Flows
# =============================================================================
# Minimum values ensure numerical stability and invertibility
DEFAULT_MIN_BIN_WIDTH = 1e-3   # Prevents degenerate bins with zero width
DEFAULT_MIN_BIN_HEIGHT = 1e-3  # Prevents flat regions with zero derivative
DEFAULT_MIN_DERIVATIVE = 1e-3  # Ensures strict monotonicity


# =============================================================================
# Masked Autoregressive Flow (MAF) Components
# =============================================================================

class MaskedDense(nn.Module):
    """
    Dense layer with fixed binary mask enforcing autoregressive connectivity.
    
    Mask[i,j] = 1 allows connection from input i to output j; 0 blocks it.
    Used in MADE to ensure output[i] depends only on inputs[0:i].
    """
    features: int
    mask: jnp.ndarray
    use_bias: bool = True

    @nn.compact
    def __call__(self, inputs):
        k = self.param("kernel", nn.initializers.lecun_normal(), (inputs.shape[-1], self.features))
        # Element-wise masking zeroes out forbidden connections
        masked_kernel = k * self.mask
        y = jnp.dot(inputs, masked_kernel)
        if self.use_bias:
            b = self.param("bias", nn.initializers.zeros, (self.features,))
            y = y + b
        return y


class MADE(nn.Module):
    """Masked Autoencoder for Distribution Estimation.
    
    Outputs affine transform parameters (shift, log_scale) such that
    output[i] depends only on input[0:i-1], enabling tractable density.
    
    The conditioning input (r) is fully visible to all hidden units,
    allowing the transform to adapt based on order potentials.
    """
    latent_dim: int
    cond_dim: int
    hidden_dims: list

    @staticmethod
    def create_masks(input_dim, cond_dim, hidden_dims, output_dim):
        """
        Create autoregressive masks using degree ordering.
        
        Each unit is assigned a "degree" in [0, input_dim). A connection
        from unit with degree d1 to unit with degree d2 is allowed iff d1 < d2
        (strict for output) or d1 <= d2 (for hidden layers).
        """
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        
        # Input degrees: 0, 1, ..., input_dim-1 (autoregressive ordering)
        input_degrees = np.arange(input_dim)
        hidden_degrees = []
        
        # Input layer mask: [z, cond] -> hidden[0]
        m_in = np.zeros((input_dim + cond_dim, hidden_dims[0]))
        
        # Hidden unit degrees randomly assigned in valid range
        hd0 = np.sort(rng.integers(0, input_dim, hidden_dims[0]))
        hidden_degrees.append(hd0)
        
        # z part: autoregressive (input_deg <= hidden_deg)
        for i in range(input_dim):
            for j in range(hidden_dims[0]):
                m_in[i, j] = input_degrees[i] <= hd0[j]
        
        # Conditioning input is fully connected (no autoregressive constraint)
        m_in[input_dim:, :] = 1.0 
        
        # Hidden layer masks
        m_hiddens = []
        for layer_idx in range(len(hidden_dims) - 1):
            m_h = np.zeros((hidden_dims[layer_idx], hidden_dims[layer_idx + 1]))
            hd_next = np.sort(rng.integers(0, input_dim, hidden_dims[layer_idx + 1]))
            hidden_degrees.append(hd_next)
            for i in range(hidden_dims[layer_idx]):
                for j in range(hidden_dims[layer_idx + 1]):
                    m_h[i, j] = hidden_degrees[layer_idx][i] <= hd_next[j]
            m_hiddens.append(m_h)

        # Output layer mask: hidden[-1] -> [shift, log_scale]
        # STRICT inequality ensures output[i] depends on inputs[0:i-1] only
        m_out = np.zeros((hidden_dims[-1], 2 * output_dim))
        for i in range(output_dim):
            for j in range(hidden_dims[-1]):
                connect = hidden_degrees[-1][j] < input_degrees[i]
                m_out[j, i] = connect                    # shift params
                m_out[j, i + output_dim] = connect       # log_scale params

        masks = [jnp.array(m_in)] + [jnp.array(m) for m in m_hiddens] + [jnp.array(m_out)]
        return masks

    def setup(self):
        masks = MADE.create_masks(self.latent_dim, self.cond_dim, self.hidden_dims, self.latent_dim)
        self.input_layer = MaskedDense(features=self.hidden_dims[0], mask=masks[0])
        self.hidden_layers = [MaskedDense(features=self.hidden_dims[i + 1], mask=masks[i + 1])
                              for i in range(len(self.hidden_dims) - 1)]
        self.output_layer = MaskedDense(features=2 * self.latent_dim, mask=masks[-1])

    def __call__(self, z, cond):
        inputs = jnp.concatenate([z, cond], axis=-1)
        x = nn.relu(self.input_layer(inputs))
        for layer in self.hidden_layers:
            x = nn.relu(layer(x))
        outputs = self.output_layer(x)
        shift, log_scale = jnp.split(outputs, 2, axis=-1)
        # Clip log_scale to prevent numerical issues in exp()
        log_scale = jnp.clip(log_scale, -10.0, 10.0)
        return shift, log_scale


class MAF(nn.Module):
    """
    Single Masked Autoregressive Flow block.
    
    Applies element-wise affine transform: x = z * exp(s(z)) + μ(z)
    where μ, s are computed by MADE ensuring x[i] depends only on z[0:i-1].
    """
    latent_dim: int
    cond_dim: int
    hidden_dims: list

    def setup(self):
        self.made = MADE(self.latent_dim, self.cond_dim, self.hidden_dims)

    def __call__(self, z, cond):
        shift, log_scale = self.made(z, cond)
        # Affine transform: x = z * exp(log_scale) + shift
        x = z * jnp.exp(log_scale) + shift
        # Log-det of Jacobian is sum of log_scales (diagonal Jacobian)
        log_det = jnp.sum(log_scale, axis=-1)
        return x, log_det


class MAFStack(nn.Module):
    """
    Stack of MAF blocks with random permutations between layers.
    
    Permutations break the autoregressive ordering, allowing each dimension
    to influence all others across the stack (universal approximation).
    """
    latent_dim: int
    cond_dim: int
    hidden_dims: list
    n_blocks: int

    def setup(self):
        self.flows = [MAF(latent_dim=self.latent_dim, cond_dim=self.cond_dim, 
                          hidden_dims=self.hidden_dims, name=f"maf_{k}")
                      for k in range(self.n_blocks)]
        # Fixed random permutations for reproducibility
        perms = []
        key = jrand.PRNGKey(42)
        for _ in range(self.n_blocks):
            key, sub = jrand.split(key)
            perms.append(jrand.permutation(sub, self.latent_dim))
        self.perms = perms

    def __call__(self, z, cond):
        total_log_det = 0.0
        for flow, perm in zip(self.flows, self.perms):
            z = z[..., perm]  # Permute dimensions before each block
            z, ld = flow(z, cond)
            total_log_det += ld
        return z, total_log_det


# =============================================================================
# Neural Spline Flow (NSF) Components - Rational Quadratic Splines
# =============================================================================

def rational_quadratic_spline(
    inputs: jnp.ndarray,
    unnormalized_widths: jnp.ndarray,
    unnormalized_heights: jnp.ndarray,
    unnormalized_derivatives: jnp.ndarray,
    inverse: bool = False,
    left: float = -5.0,
    right: float = 5.0,
    bottom: float = -5.0,
    top: float = 5.0,
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Rational quadratic spline transformation (Durkan et al., 2019).
    
    The spline is defined by K bins with learned widths, heights, and derivatives
    at knot points. Within each bin, a rational quadratic function provides a
    smooth, monotonic, invertible mapping with analytically tractable Jacobian.
    
    Key properties:
    - Monotonicity guaranteed by positive derivatives at knots
    - Smooth C1 continuity across bin boundaries
    - Identity transform outside [left, right] (tails)
    - Expressivity controlled by number of bins
    
    Args:
        inputs: Values to transform [..., dim]
        unnormalized_widths: Unconstrained bin widths [..., dim, num_bins]
        unnormalized_heights: Unconstrained bin heights [..., dim, num_bins]
        unnormalized_derivatives: Unconstrained knot derivatives [..., dim, num_bins+1]
        inverse: If True, compute inverse transformation (x→z instead of z→x)
        left, right: Input domain boundaries
        bottom, top: Output domain boundaries
        min_*: Minimum values for numerical stability
        
    Returns:
        outputs: Transformed values, same shape as inputs
        log_abs_det: Log |det(Jacobian)| for density computation
    """
    num_bins = unnormalized_widths.shape[-1]
    
    # === Constrain bin widths: softmax ensures sum=1, then scale to [left, right] ===
    widths = jax.nn.softmax(unnormalized_widths, axis=-1)
    # Ensure minimum width for numerical stability
    widths = min_bin_width + (1.0 - min_bin_width * num_bins) * widths
    cumwidths = jnp.cumsum(widths, axis=-1)
    # Prepend 0 to get knot positions: [0, w1, w1+w2, ..., 1]
    cumwidths = jnp.pad(cumwidths, [(0, 0)] * (cumwidths.ndim - 1) + [(1, 0)], 
                        constant_values=0.0)
    # Scale to actual domain: [left, ..., right]
    cumwidths = (right - left) * cumwidths + left
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]
    
    # === Constrain bin heights similarly ===
    heights = jax.nn.softmax(unnormalized_heights, axis=-1)
    heights = min_bin_height + (1.0 - min_bin_height * num_bins) * heights
    cumheights = jnp.cumsum(heights, axis=-1)
    cumheights = jnp.pad(cumheights, [(0, 0)] * (cumheights.ndim - 1) + [(1, 0)], 
                         constant_values=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    heights = cumheights[..., 1:] - cumheights[..., :-1]
    
    # === Constrain derivatives: softplus ensures positivity for monotonicity ===
    derivatives = min_derivative + jax.nn.softplus(unnormalized_derivatives)
    
    # === Determine which inputs are inside the spline domain ===
    # Outside the domain, apply identity transform (common in NSF for tails)
    if inverse:
        inside_interval = (inputs >= bottom) & (inputs <= top)
    else:
        inside_interval = (inputs >= left) & (inputs <= right)
    
    # === Find which bin each input belongs to ===
    # Use broadcasting comparison since JAX searchsorted requires 1D
    if inverse:
        # For inverse: find bin by y-coordinate (cumheights)
        bin_idx = jnp.sum(cumheights <= inputs[..., None], axis=-1) - 1
    else:
        # For forward: find bin by x-coordinate (cumwidths)
        bin_idx = jnp.sum(cumwidths <= inputs[..., None], axis=-1) - 1
    
    bin_idx = jnp.clip(bin_idx, 0, num_bins - 1)
    
    # Helper to gather parameters for the relevant bin
    def gather_1d(arr, idx):
        """Index into last dimension using idx."""
        return jnp.take_along_axis(arr, idx[..., None], axis=-1)[..., 0]
    
    # === Gather bin-specific parameters ===
    input_cumwidths = gather_1d(cumwidths, bin_idx)      # x_k (left boundary)
    input_bin_widths = gather_1d(widths, bin_idx)        # w_k
    input_cumheights = gather_1d(cumheights, bin_idx)    # y_k (bottom boundary)
    input_bin_heights = gather_1d(heights, bin_idx)      # h_k
    input_delta = input_bin_heights / input_bin_widths   # Slope s_k = h_k / w_k
    input_derivatives = gather_1d(derivatives, bin_idx)  # d_k (left derivative)
    input_derivatives_plus_one = gather_1d(derivatives[..., 1:], bin_idx)  # d_{k+1}
    
    if inverse:
        # === Inverse transform: solve quadratic for θ given y ===
        # RQ spline equation is quadratic in θ; use quadratic formula
        # Cross-multiplying the RQ equation and collecting powers of θ gives:
        #   A*θ² + B*θ + C = 0
        # where the (inputs - input_cumheights) correction terms in A and B
        # arise from the denominator of the RQ spline (Durkan et al., 2019).
        xi = inputs - input_cumheights
        delta_term = input_derivatives + input_derivatives_plus_one - 2.0 * input_delta
        a = input_bin_heights * (input_delta - input_derivatives) + xi * delta_term
        b = input_bin_heights * input_derivatives - xi * delta_term
        c = -input_delta * xi
        
        discriminant = b ** 2 - 4.0 * a * c
        discriminant = jnp.maximum(discriminant, 0.0)  # Clamp for numerical safety
        
        # Select correct root (always the one in [0,1])
        theta = (2.0 * c) / (-b - jnp.sqrt(discriminant))
        theta = jnp.clip(theta, 0.0, 1.0)
        
        # Convert θ back to x
        outputs = theta * input_bin_widths + input_cumwidths
        
        # Compute Jacobian (same formula as forward, then negate log)
        theta_one_minus_theta = theta * (1.0 - theta)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2.0 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta ** 2 * (
            input_derivatives_plus_one * theta ** 2
            + 2.0 * input_delta * theta_one_minus_theta
            + input_derivatives * (1.0 - theta) ** 2
        )
        log_abs_det = jnp.log(derivative_numerator + 1e-8) - 2.0 * jnp.log(denominator + 1e-8)
        log_abs_det = -log_abs_det  # Inverse: negate log-det
    else:
        # === Forward transform: compute y from x via rational quadratic ===
        # θ = (x - x_k) / w_k ∈ [0, 1] is normalized position within bin
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta = jnp.clip(theta, 0.0, 1.0)
        theta_one_minus_theta = theta * (1.0 - theta)
        
        # RQ spline formula: y = y_k + h_k * [s_k*θ² + d_k*θ(1-θ)] / [s_k + (d_k+d_{k+1}-2s_k)*θ(1-θ)]
        numerator = input_bin_heights * (
            input_delta * theta ** 2 + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2.0 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / (denominator + 1e-8)
        
        # Analytical derivative of RQ spline (used for log-det)
        derivative_numerator = input_delta ** 2 * (
            input_derivatives_plus_one * theta ** 2
            + 2.0 * input_delta * theta_one_minus_theta
            + input_derivatives * (1.0 - theta) ** 2
        )
        log_abs_det = jnp.log(derivative_numerator + 1e-8) - 2.0 * jnp.log(denominator + 1e-8)
    
    # === Apply identity transform outside the spline domain (tails) ===
    outputs = jnp.where(inside_interval, outputs, inputs)
    log_abs_det = jnp.where(inside_interval, log_abs_det, 0.0)
    
    return outputs, log_abs_det


class SplineConditioner(nn.Module):
    """
    Neural network that outputs spline parameters conditioned on input.
    
    This is a standard MLP that outputs unnormalized widths, heights, and derivatives
    for the rational quadratic spline transformation.
    """
    output_dim: int
    hidden_dims: list
    num_bins: int = 8
    
    @nn.compact
    def __call__(self, cond_input: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
            cond_input: Conditioning input [batch_dims..., cond_dim]
            
        Returns:
            unnormalized_widths: [batch_dims..., output_dim, num_bins]
            unnormalized_heights: [batch_dims..., output_dim, num_bins]
            unnormalized_derivatives: [batch_dims..., output_dim, num_bins + 1]
        """
        # Total output size: output_dim * (3 * num_bins + 1)
        # widths: num_bins, heights: num_bins, derivatives: num_bins + 1
        total_params_per_dim = 3 * self.num_bins + 1
        total_output = self.output_dim * total_params_per_dim
        
        x = cond_input
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(hidden_dim, name=f"hidden_{i}")(x)
            x = nn.relu(x)
        
        # Output layer with zero initialization for stable training
        outputs = nn.Dense(
            total_output, 
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="output"
        )(x)
        
        # Reshape to [batch_dims..., output_dim, total_params_per_dim]
        output_shape = outputs.shape[:-1] + (self.output_dim, total_params_per_dim)
        outputs = outputs.reshape(output_shape)
        
        # Split into widths, heights, derivatives
        unnormalized_widths = outputs[..., :self.num_bins]
        unnormalized_heights = outputs[..., self.num_bins:2 * self.num_bins]
        unnormalized_derivatives = outputs[..., 2 * self.num_bins:]
        
        return unnormalized_widths, unnormalized_heights, unnormalized_derivatives


class MaskedSplineConditioner(nn.Module):
    """
    Autoregressive conditioner using masked connections for spline parameters.
    
    Similar to MADE but outputs spline parameters instead of shift/scale.
    """
    latent_dim: int
    cond_dim: int
    hidden_dims: list
    num_bins: int = 8
    
    @staticmethod
    def create_masks(input_dim, cond_dim, hidden_dims, output_dim, num_params_per_output):
        """Create autoregressive masks for spline parameter prediction."""
        rng = np.random.default_rng(42)
        
        input_degrees = np.arange(input_dim)
        hidden_degrees = []
        
        # Input layer mask
        m_in = np.zeros((input_dim + cond_dim, hidden_dims[0]))
        hd0 = np.sort(rng.integers(0, input_dim, hidden_dims[0]))
        hidden_degrees.append(hd0)
        
        for i in range(input_dim):
            for j in range(hidden_dims[0]):
                m_in[i, j] = input_degrees[i] <= hd0[j]
        m_in[input_dim:, :] = 1.0
        
        # Hidden layer masks
        m_hiddens = []
        for layer_idx in range(len(hidden_dims) - 1):
            m_h = np.zeros((hidden_dims[layer_idx], hidden_dims[layer_idx + 1]))
            hd_next = np.sort(rng.integers(0, input_dim, hidden_dims[layer_idx + 1]))
            hidden_degrees.append(hd_next)
            for i in range(hidden_dims[layer_idx]):
                for j in range(hidden_dims[layer_idx + 1]):
                    m_h[i, j] = hidden_degrees[layer_idx][i] <= hd_next[j]
            m_hiddens.append(m_h)
        
        # Output layer mask: each output dimension gets num_params_per_output parameters
        total_output = output_dim * num_params_per_output
        m_out = np.zeros((hidden_dims[-1], total_output))
        for i in range(output_dim):
            for j in range(hidden_dims[-1]):
                connect = hidden_degrees[-1][j] < input_degrees[i]
                for p in range(num_params_per_output):
                    m_out[j, i * num_params_per_output + p] = connect
        
        masks = [jnp.array(m_in)] + [jnp.array(m) for m in m_hiddens] + [jnp.array(m_out)]
        return masks
    
    def setup(self):
        num_params_per_output = 3 * self.num_bins + 1
        masks = MaskedSplineConditioner.create_masks(
            self.latent_dim, self.cond_dim, self.hidden_dims, 
            self.latent_dim, num_params_per_output
        )
        
        self.input_layer = MaskedDense(features=self.hidden_dims[0], mask=masks[0])
        self.hidden_layers = [
            MaskedDense(features=self.hidden_dims[i + 1], mask=masks[i + 1])
            for i in range(len(self.hidden_dims) - 1)
        ]
        self.output_layer = MaskedDense(
            features=self.latent_dim * (3 * self.num_bins + 1), 
            mask=masks[-1]
        )
    
    def __call__(self, z, cond):
        """
        Args:
            z: Input to transform [batch_dims..., latent_dim]
            cond: Conditioning input [batch_dims..., cond_dim]
            
        Returns:
            unnormalized_widths: [batch_dims..., latent_dim, num_bins]
            unnormalized_heights: [batch_dims..., latent_dim, num_bins]
            unnormalized_derivatives: [batch_dims..., latent_dim, num_bins + 1]
        """
        inputs = jnp.concatenate([z, cond], axis=-1)
        x = nn.relu(self.input_layer(inputs))
        for layer in self.hidden_layers:
            x = nn.relu(layer(x))
        outputs = self.output_layer(x)
        
        # Reshape to [batch_dims..., latent_dim, params_per_dim]
        params_per_dim = 3 * self.num_bins + 1
        output_shape = outputs.shape[:-1] + (self.latent_dim, params_per_dim)
        outputs = outputs.reshape(output_shape)
        
        unnormalized_widths = outputs[..., :self.num_bins]
        unnormalized_heights = outputs[..., self.num_bins:2 * self.num_bins]
        unnormalized_derivatives = outputs[..., 2 * self.num_bins:]
        
        return unnormalized_widths, unnormalized_heights, unnormalized_derivatives


class NeuralSplineFlowBlock(nn.Module):
    """
    Single Neural Spline Flow block using autoregressive masking.
    
    This applies a rational quadratic spline transformation where the spline
    parameters are predicted autoregressively conditioned on previous dimensions.
    """
    latent_dim: int
    cond_dim: int
    hidden_dims: list
    num_bins: int = 8
    tail_bound: float = 5.0
    
    def setup(self):
        self.conditioner = MaskedSplineConditioner(
            latent_dim=self.latent_dim,
            cond_dim=self.cond_dim,
            hidden_dims=self.hidden_dims,
            num_bins=self.num_bins
        )
    
    def __call__(self, z: jnp.ndarray, cond: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward transformation: z -> x
        
        Args:
            z: Base distribution samples [latent_dim]
            cond: Conditioning input [cond_dim]
            
        Returns:
            x: Transformed samples [latent_dim]
            log_det: Log determinant of Jacobian (scalar)
        """
        widths, heights, derivatives = self.conditioner(z, cond)
        
        # Apply spline transformation element-wise
        x, log_det = rational_quadratic_spline(
            z, widths, heights, derivatives,
            inverse=False,
            left=-self.tail_bound,
            right=self.tail_bound,
            bottom=-self.tail_bound,
            top=self.tail_bound
        )
        
        # Sum log determinants across dimensions
        total_log_det = jnp.sum(log_det, axis=-1)
        
        return x, total_log_det


class NeuralSplineCouplingBlock(nn.Module):
    """
    Neural Spline Flow block using coupling layer architecture.
    
    Splits the input into two parts. One part is kept unchanged and used to
    condition the spline transformation of the other part.
    """
    latent_dim: int
    cond_dim: int
    hidden_dims: list
    num_bins: int = 8
    tail_bound: float = 5.0
    transform_first_half: bool = True
    
    def setup(self):
        # Determine split dimensions
        self.d1 = self.latent_dim // 2
        self.d2 = self.latent_dim - self.d1
        
        # The conditioner takes the unchanged part + external conditioning
        if self.transform_first_half:
            cond_input_dim = self.d2 + self.cond_dim
            output_dim = self.d1
        else:
            cond_input_dim = self.d1 + self.cond_dim
            output_dim = self.d2
        
        self.conditioner = SplineConditioner(
            output_dim=output_dim,
            hidden_dims=self.hidden_dims,
            num_bins=self.num_bins
        )
    
    def __call__(self, z: jnp.ndarray, cond: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward transformation using coupling layer.
        
        Args:
            z: Base distribution samples [latent_dim]
            cond: Conditioning input [cond_dim]
            
        Returns:
            x: Transformed samples [latent_dim]
            log_det: Log determinant of Jacobian (scalar)
        """
        if self.transform_first_half:
            z_transform = z[..., :self.d1]
            z_identity = z[..., self.d1:]
            cond_input = jnp.concatenate([z_identity, cond], axis=-1)
        else:
            z_identity = z[..., :self.d1]
            z_transform = z[..., self.d1:]
            cond_input = jnp.concatenate([z_identity, cond], axis=-1)
        
        widths, heights, derivatives = self.conditioner(cond_input)
        
        x_transform, log_det = rational_quadratic_spline(
            z_transform, widths, heights, derivatives,
            inverse=False,
            left=-self.tail_bound,
            right=self.tail_bound,
            bottom=-self.tail_bound,
            top=self.tail_bound
        )
        
        if self.transform_first_half:
            x = jnp.concatenate([x_transform, z_identity], axis=-1)
        else:
            x = jnp.concatenate([z_identity, x_transform], axis=-1)
        
        total_log_det = jnp.sum(log_det, axis=-1)
        
        return x, total_log_det


class NSFStack(nn.Module):
    """
    Stack of Neural Spline Flow blocks with random permutations.
    
    Uses autoregressive masking for the spline parameter prediction.
    This is the recommended NSF architecture for most use cases.
    """
    latent_dim: int
    cond_dim: int
    hidden_dims: list
    n_blocks: int
    num_bins: int = 8
    tail_bound: float = 5.0
    
    def setup(self):
        self.flows = [
            NeuralSplineFlowBlock(
                latent_dim=self.latent_dim,
                cond_dim=self.cond_dim,
                hidden_dims=self.hidden_dims,
                num_bins=self.num_bins,
                tail_bound=self.tail_bound,
                name=f"nsf_{k}"
            )
            for k in range(self.n_blocks)
        ]
        
        # Create random permutations for each layer
        perms = []
        key = jrand.PRNGKey(42)
        for _ in range(self.n_blocks):
            key, sub = jrand.split(key)
            perms.append(jrand.permutation(sub, self.latent_dim))
        self.perms = perms
    
    def __call__(self, z: jnp.ndarray, cond: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward transformation through the stack.
        
        Args:
            z: Base distribution samples [latent_dim]
            cond: Conditioning input [cond_dim]
            
        Returns:
            x: Transformed samples [latent_dim]
            total_log_det: Sum of log determinants (scalar)
        """
        total_log_det = 0.0
        for flow, perm in zip(self.flows, self.perms):
            z = z[..., perm]
            z, ld = flow(z, cond)
            total_log_det += ld
        return z, total_log_det


class NSFCouplingStack(nn.Module):
    """
    Stack of Neural Spline Flow blocks using coupling layers.
    
    Alternates between transforming the first and second halves of the input.
    This architecture can be faster for large dimensions but may be less expressive.
    """
    latent_dim: int
    cond_dim: int
    hidden_dims: list
    n_blocks: int
    num_bins: int = 8
    tail_bound: float = 5.0
    
    def setup(self):
        self.flows = [
            NeuralSplineCouplingBlock(
                latent_dim=self.latent_dim,
                cond_dim=self.cond_dim,
                hidden_dims=self.hidden_dims,
                num_bins=self.num_bins,
                tail_bound=self.tail_bound,
                transform_first_half=(k % 2 == 0),
                name=f"nsf_coupling_{k}"
            )
            for k in range(self.n_blocks)
        ]
        
        # Create random permutations
        perms = []
        key = jrand.PRNGKey(42)
        for _ in range(self.n_blocks):
            key, sub = jrand.split(key)
            perms.append(jrand.permutation(sub, self.latent_dim))
        self.perms = perms
    
    def __call__(self, z: jnp.ndarray, cond: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward transformation through the coupling stack.
        
        Args:
            z: Base distribution samples [latent_dim]
            cond: Conditioning input [cond_dim]
            
        Returns:
            x: Transformed samples [latent_dim]
            total_log_det: Sum of log determinants (scalar)
        """
        total_log_det = 0.0
        for flow, perm in zip(self.flows, self.perms):
            z = z[..., perm]
            z, ld = flow(z, cond)
            total_log_det += ld
        return z, total_log_det


# =============================================================================
# Factory function for creating flow stacks
# =============================================================================

def create_flow_stack(
    flow_type: str,
    latent_dim: int,
    cond_dim: int,
    hidden_dims: list,
    n_blocks: int,
    num_bins: int = 8,
    tail_bound: float = 5.0,
    name: str = "flow"
) -> nn.Module:
    """
    Factory function to create a flow stack based on the specified type.
    
    Args:
        flow_type: Type of flow ('maf', 'nsf', 'nsf_coupling')
        latent_dim: Dimension of the latent space
        cond_dim: Dimension of the conditioning input
        hidden_dims: List of hidden layer dimensions
        n_blocks: Number of flow blocks
        num_bins: Number of bins for spline flows (NSF only)
        tail_bound: Boundary for spline transformation (NSF only)
        name: Name for the module
        
    Returns:
        A flow stack module (MAFStack, NSFStack, or NSFCouplingStack)
    """
    flow_type = flow_type.lower()
    
    if flow_type == 'maf':
        return MAFStack(
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            hidden_dims=hidden_dims,
            n_blocks=n_blocks,
            name=name
        )
    elif flow_type == 'nsf':
        return NSFStack(
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            hidden_dims=hidden_dims,
            n_blocks=n_blocks,
            num_bins=num_bins,
            tail_bound=tail_bound,
            name=name
        )
    elif flow_type == 'nsf_coupling':
        return NSFCouplingStack(
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            hidden_dims=hidden_dims,
            n_blocks=n_blocks,
            num_bins=num_bins,
            tail_bound=tail_bound,
            name=name
        )
    else:
        raise ValueError(
            f"Unknown flow type: {flow_type}. "
            f"Supported types: 'maf', 'nsf', 'nsf_coupling'"
        )

