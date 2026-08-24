'''
Neural network module for the SVIDAG.

Author: Shrenik Zinage
'''


import numpy as np
from typing import Tuple
import jax.numpy as jnp
import jax.random as jrand
from flax import linen as nn

from . import config


class HyperNetwork(nn.Module):
    """
    Generic hypernetwork that maps an adjacency condition to a tensor output.

    The formulation requires the variational posterior over network parameters
    to be produced by a differentiable hypernetwork conditioned on the current
    relaxed adjacency. We therefore flatten the condition explicitly so the
    network can consume either a row vector or a full adjacency matrix.
    """
    out_shape: tuple
    hidden_dim: int = 16

    @nn.compact
    def __call__(self, x):
        x = jnp.ravel(x)
        h = nn.relu(nn.Dense(self.hidden_dim)(x))
        h = nn.relu(nn.Dense(self.hidden_dim)(h))
        out = nn.Dense(np.prod(self.out_shape))(h)
        return jnp.reshape(out, self.out_shape)


class PosteriorStatsHyperNetwork(nn.Module):
    """
    Hypernetwork that returns the mean and rho parameters of a diagonal Gaussian.

    This directly implements the formulation
        (mu_phi(A), rho_phi(A)) = H_phi(A),
        sigma_phi(A) = softplus(rho_phi(A)).
    """

    out_shape: tuple
    hidden_dim: int = 16
    rho_bias_init: float = -2.0

    @nn.compact
    def __call__(self, x) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.ravel(x)
        h = nn.relu(nn.Dense(self.hidden_dim, name="hidden_0")(x))
        h = nn.relu(nn.Dense(self.hidden_dim, name="hidden_1")(h))

        output_size = int(np.prod(self.out_shape))
        mu = nn.Dense(output_size, name="mu_head")(h)
        rho = nn.Dense(
            output_size,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.constant(self.rho_bias_init),
            name="rho_head",
        )(h)

        return jnp.reshape(mu, self.out_shape), jnp.reshape(rho, self.out_shape)


class BayesianLinear(nn.Module):
    """
    Linear layer with adjacency-conditioned Gaussian posterior over parameters.

    The guide matches the formulation
        q_phi(theta | A) = N(mu_phi(A), diag(sigma_phi(A)^2))
    where both the means and the unconstrained rho parameters are produced by
    hypernetworks and sigma_phi(A) = softplus(rho_phi(A)).
    """
    in_features: int
    out_features: int
    cond_dim: int
    # NOTE: these are dataclass field defaults, so they are bound ONCE at
    # import time.  Assigning config.prior_theta_sigma at runtime therefore
    # does NOT reach them -- any experiment that "varied" the weight prior by
    # setting the config attribute after import was silently a no-op.  The
    # KL below reads the live config value instead, so the knob works; these
    # fields remain as explicit per-instance overrides.
    prior_mu: float = None
    prior_sigma: float = None

    def setup(self):
        self.weight_posterior = PosteriorStatsHyperNetwork(
            out_shape=(self.out_features, self.in_features),
            hidden_dim=16,
        )
        self.bias_posterior = PosteriorStatsHyperNetwork(
            out_shape=(self.out_features,),
            hidden_dim=16,
        )

    def posterior_stats(self, A_condition) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return posterior moments conditioned on the current relaxed adjacency."""
        weight_mu, weight_rho = self.weight_posterior(A_condition)
        bias_mu, bias_rho = self.bias_posterior(A_condition)
        weight_sigma = nn.softplus(weight_rho) + 1e-6
        bias_sigma = nn.softplus(bias_rho) + 1e-6
        return weight_mu, bias_mu, weight_sigma, bias_sigma

    def sample_from_stats(self, x, rng, stats):
        """Sample weights from cached posterior moments and apply the layer."""
        weight_mu, bias_mu, weight_sigma, bias_sigma = stats
        rng_w, rng_b = jrand.split(rng)
        eps_w = jrand.normal(rng_w, weight_mu.shape)
        eps_b = jrand.normal(rng_b, bias_mu.shape)
        W = weight_mu + weight_sigma * eps_w
        b = bias_mu + bias_sigma * eps_b
        return jnp.dot(x, W.T) + b

    def kl_from_stats(self, stats):
        """Closed-form KL from cached posterior moments."""
        weight_mu, bias_mu, weight_sigma, bias_sigma = stats
        # Read the prior from config at call time (see the note on the fields
        # above): the dataclass defaults freeze at import, so this is what
        # actually makes config.prior_theta_sigma a working knob.
        _p_mu = config.prior_theta_mu if self.prior_mu is None else self.prior_mu
        _p_sigma = config.prior_theta_sigma if self.prior_sigma is None else self.prior_sigma
        kl_w = jnp.sum(
            jnp.log(_p_sigma / weight_sigma)
            + (weight_sigma**2 + (weight_mu - _p_mu) ** 2) / (2 * _p_sigma**2)
            - 0.5
        )
        kl_b = jnp.sum(
            jnp.log(_p_sigma / bias_sigma)
            + (bias_sigma**2 + (bias_mu - _p_mu) ** 2) / (2 * _p_sigma**2)
            - 0.5
        )
        return kl_w + kl_b

    def __call__(self, x, A_condition, rng):
        """Forward pass with weight sampling via reparameterization trick."""
        stats = self.posterior_stats(A_condition)
        return self.sample_from_stats(x, rng, stats)

    def kl_divergence(self, A_condition):
        """
        Closed-form KL(q(W)||p(W)) for diagonal Gaussian variational posterior
        and isotropic Gaussian prior: KL = Σ[log(σ_p/σ_q) + (σ_q² + (μ_q-μ_p)²)/(2σ_p²) - 0.5]
        """
        stats = self.posterior_stats(A_condition)
        return self.kl_from_stats(stats)


class BayesianMLP(nn.Module):
    """
    3-layer Bayesian MLP with tanh activations.
    Each layer has a diagonal Gaussian posterior conditioned on adjacency.
    """
    input_dim: int
    hidden_dim: int
    output_dim: int
    cond_dim: int

    def setup(self):
        self.b1 = BayesianLinear(self.input_dim, self.hidden_dim, cond_dim=self.cond_dim)
        self.b2 = BayesianLinear(self.hidden_dim, self.hidden_dim, cond_dim=self.cond_dim)
        self.b3 = BayesianLinear(self.hidden_dim, self.output_dim, cond_dim=self.cond_dim)

    def __call__(self, x, A_condition, rng):
        # Independent random keys for each layer's weight sampling
        r1, r2, r3 = jrand.split(rng, 3)
        x = nn.tanh(self.b1(x, A_condition, r1))
        x = nn.tanh(self.b2(x, A_condition, r2))
        return self.b3(x, A_condition, r3)

    def forward_and_kl(self, x, A_condition, rng):
        """Run the sampled forward pass and KL accumulation in one pass."""
        r1, r2, r3 = jrand.split(rng, 3)

        stats1 = self.b1.posterior_stats(A_condition)
        x = nn.tanh(self.b1.sample_from_stats(x, r1, stats1))
        kl = self.b1.kl_from_stats(stats1)

        stats2 = self.b2.posterior_stats(A_condition)
        x = nn.tanh(self.b2.sample_from_stats(x, r2, stats2))
        kl = kl + self.b2.kl_from_stats(stats2)

        stats3 = self.b3.posterior_stats(A_condition)
        x = self.b3.sample_from_stats(x, r3, stats3)
        kl = kl + self.b3.kl_from_stats(stats3)

        return x, kl

    def kl_divergence(self, A_condition):
        """Total KL is sum of per-layer KL divergences."""
        return self.b1.kl_divergence(A_condition) + self.b2.kl_divergence(A_condition) + self.b3.kl_divergence(A_condition)


class NodeModel(nn.Module):
    """
    Structural equation model for a single node.
    
    Predicts x_i given parent values: x_i = f_i(pa(x_i)) + ε_i
    The parent set is soft-gated by A_condition (row of adjacency matrix).
    Input x is masked by A_condition before being passed to the network.
    """
    num_nodes: int
    hidden_dim: int = config.hidden_dim

    def setup(self):
        # MLP: takes all node values (masked) → predicts single node value
        self.model = BayesianMLP(
            self.num_nodes,
            self.hidden_dim,
            1,
            cond_dim=self.num_nodes * self.num_nodes,
        )

    def __call__(self, x, rng, A_condition):
        """x: parent values (pre-masked by caller), A_condition: adjacency row for this node."""
        return self.model(x, A_condition, rng)

    def forward_and_kl(self, x, rng, A_condition):
        """Forward pass plus KL using shared posterior statistics."""
        return self.model.forward_and_kl(x, A_condition, rng)

    def kl_divergence(self, A_condition):
        return self.model.kl_divergence(A_condition)
