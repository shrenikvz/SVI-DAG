'''
Model module for the SVIDAG.

Author: Shrenik Zinage
'''


import numpy as np
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jrand
from flax import linen as nn

from . import config
from .bayesian import NodeModel, PosteriorStatsHyperNetwork
from .flows import create_flow_stack
from .utils import (
    logistic_beta_logpdf_gamma,
    logistic_concrete,
    vec_to_offdiag_matrix,
    build_mask_M_tau,
    build_mask_M_hard,
    gaussian_likelihood_logsum,
    center_order_potentials,
)


class SVIDAGModel(nn.Module):
    num_nodes: int
    flow_hidden: list
    flow_n_blocks: int
    fixed_noise_scales: Optional[jnp.ndarray]
    flow_type: str = 'nsf'
    nsf_num_bins: int = 8
    nsf_tail_bound: float = 5.0

    def setup(self):
        # Flow dimension = number of possible directed edges (excluding self-loops)
        d = self.num_nodes * (self.num_nodes - 1)
        # Normalizing flow: maps base z ~ N(0,I) to edge logits γ, conditioned on r
        self.flow = create_flow_stack(
            flow_type=self.flow_type,
            latent_dim=d,
            cond_dim=self.num_nodes,  # Conditioning on order potentials r
            hidden_dims=self.flow_hidden,
            n_blocks=self.flow_n_blocks,
            num_bins=self.nsf_num_bins,
            tail_bound=self.nsf_tail_bound,
            name="flow"
        )
        # One Bayesian MLP per node to model structural equations
        self.node_models = [NodeModel(self.num_nodes, hidden_dim=config.hidden_dim, name=f"node_model_{i}") for i in range(self.num_nodes)]
        # Graph-conditioned posterior over per-node log noise scales.
        if config.learn_likelihood_noise:
            self.noise_posterior = PosteriorStatsHyperNetwork(
                out_shape=(self.num_nodes,),
                hidden_dim=16,
                rho_bias_init=-2.0,
                name="noise_posterior",
            )
        # Precompute diagonal mask to enforce no self-loops
        diag_mask_np = np.eye(self.num_nodes, dtype=np.float32)
        self.diag_mask = jnp.array(diag_mask_np)

    def _sample_likelihood_noise(self, A_condition, rng):
        """Sample graph-conditioned per-node observation noise scales."""
        if config.learn_likelihood_noise:
            log_noise_mu, log_noise_rho = self.noise_posterior(A_condition)
            log_noise_sigma = nn.softplus(log_noise_rho) + 1e-6
            eps = jrand.normal(rng, log_noise_mu.shape)
            log_noise = log_noise_mu + log_noise_sigma * eps
            noise_scales = jnp.exp(log_noise)

            prior_mu = jnp.full_like(log_noise_mu, jnp.log(config.obs_noise_scale))
            prior_sigma = jnp.asarray(config.prior_log_noise_sigma, dtype=log_noise_mu.dtype)
            kl_noise = jnp.sum(
                jnp.log(prior_sigma / log_noise_sigma)
                + (log_noise_sigma**2 + (log_noise_mu - prior_mu) ** 2) / (2 * prior_sigma**2)
                - 0.5
            )
            return noise_scales, kl_noise, log_noise_mu, log_noise_sigma

        if self.fixed_noise_scales is not None:
            noise_scales = self.fixed_noise_scales
        else:
            noise_scales = jnp.ones((self.num_nodes,)) * config.obs_noise_scale
        noise_scales = jnp.asarray(noise_scales)
        log_noise = jnp.log(noise_scales)
        zeros = jnp.zeros_like(log_noise)
        return noise_scales, jnp.asarray(0.0, dtype=noise_scales.dtype), log_noise, zeros

    def forward_sample(self, x, r, rng, T_B, tau_sink, alpha_mat, beta_mat, st_weight=1.0) -> Tuple[jnp.ndarray, dict]:
        """
        Single forward pass: sample graph structure, compute predictions and ELBO terms.

        Args:
            x: Observed data batch [N, m]
            r: Order potentials [m] determining node ordering
            rng: JAX PRNG key
            T_B: Temperature for Concrete relaxation of edges
            tau_sink: Temperature for Sinkhorn soft permutation
            alpha_mat, beta_mat: Beta prior parameters for edges

        Returns:
            preds: Model predictions [N, m]
            terms: Dict of ELBO components and intermediate quantities
        """
        rng, k_z, k_B, k_noise = jrand.split(rng, 4)

        # Only relative node potentials matter for the induced ordering.
        r_centered = center_order_potentials(r)

        # Step 1: Sample edge logits γ from flow q(γ|r)
        d = self.num_nodes * (self.num_nodes - 1)
        z = jrand.normal(k_z, (d,))  # Base distribution sample

        gamma_flat, log_det = self.flow(z, cond=r_centered)  # Transform z → γ
        gamma_flat = jnp.clip(gamma_flat, -15.0, 15.0)  # Prevent numerical overflow in sigmoid

        gamma_mat = vec_to_offdiag_matrix(gamma_flat, self.num_nodes)

        # Step 2: Relaxed binary edges B̃ via Gumbel-Softmax/Concrete
        B_tilde_vec = logistic_concrete(k_B, gamma_flat, T_B)
        B_tilde = vec_to_offdiag_matrix(B_tilde_vec, self.num_nodes)

        # Step 3: Construct adjacency matrices.
        # Relaxed surrogate: Ã = B̃ ⊙ M_τ(r) ⊙ (1 - I) — generally cyclic,
        # used only to route gradients.
        M_tau = build_mask_M_tau(r_centered, tau_sink)
        A_relaxed = (B_tilde * M_tau) * (1 - self.diag_mask)

        # Hard DAG: A = B ⊙ M(r) with B = 1{γ + R > 0} and the hard
        # permutation mask M(r) = P(r) L P(r)^T. Thresholding B̃ at 0.5 is
        # exactly 1{γ + R > 0} for any temperature T > 0, so the hard sample
        # and its relaxation share the same logistic noise.
        B_hard = jnp.where(B_tilde >= 0.5, 1.0, 0.0)
        M_hard = build_mask_M_hard(r_centered)
        A_hard = B_hard * M_hard  # Binary DAG by construction

        # Straight-through estimator: A_ST = Ã + sg(A - Ã) evaluates the hard
        # DAG in the forward pass while backpropagating through Ã.
        # ``st_weight`` interpolates the forward VALUE between the relaxed
        # surrogate (0) and the hard DAG (1). The stop-gradient term carries no
        # derivative, so the gradient is the same at every weight -- only what
        # the likelihood is evaluated on changes. See config.st_warmup_frac.
        if config.straight_through_mask:
            A_forward = A_relaxed + st_weight * jax.lax.stop_gradient(A_hard - A_relaxed)
        else:
            A_forward = A_relaxed

        # Step 4: Compute predictions via structural equations
        preds = []
        kl_theta = 0.0
        _, *node_keys = jrand.split(rng, self.num_nodes + 1)
        for i, node in enumerate(self.node_models):
            # The hypernetworks H_phi receive A_ST: its forward value is
            # exactly the hard DAG A, so this evaluates the same density as
            # q_phi(theta | A) while routing surrogate derivatives through
            # the relaxed adjacency \tilde{A}.
            A_i = A_forward[i]
            # Mask inputs: only parent values contribute
            x_mask = x * A_i
            # theta_i parameterises f_i : pa(i) -> x_i, so row i is the part of
            # A it may depend on. Conditioning on the whole matrix also routes
            # dELL/dA through every other node's hypernetwork, which buries the
            # per-edge signal. See config.node_cond_row_only.
            A_cond = A_i if config.node_cond_row_only else A_forward
            pred_i, kl_i = node.forward_and_kl(x_mask, node_keys[i], A_condition=A_cond)
            preds.append(pred_i)
            kl_theta += kl_i
        preds = jnp.concatenate(preds, axis=1)

        current_noise_scales, kl_noise, noise_log_mu, noise_log_sigma = self._sample_likelihood_noise(
            A_forward,
            k_noise,
        )
        kl_theta = kl_theta + kl_noise

        # Step 5: Compute log-likelihood p(x | predictions, σ)
        ll_mse, ll_const = gaussian_likelihood_logsum(
            x,
            preds,
            current_noise_scales,
            include_const=True,
        )
        loglik = ll_mse + ll_const

        # Step 6: Compute log prior p(γ) under Logistic-Beta
        log_p_gamma = logistic_beta_logpdf_gamma(gamma_mat, alpha_mat, beta_mat)

        # Return all quantities needed for ELBO computation
        terms = {
            "z": z,                    # Base sample for KL(q(γ)||p(γ))
            "log_det": log_det,        # Flow Jacobian for change-of-variables
            "log_p_gamma": log_p_gamma,# Log prior on edge logits
            "loglik": loglik,          # Data log-likelihood
            "ll_mse": ll_mse,          # MSE component of likelihood
            "ll_const": ll_const,      # Normalizing constant term
            "kl_theta": kl_theta,      # KL for node model weights
            "kl_noise": kl_noise,      # KL for graph-conditioned likelihood noise
            "B_tilde": B_tilde,        # Relaxed Bernoulli edges (Concrete samples)
            "M_tau": M_tau,            # Relaxed order mask from Sinkhorn
            "A_relaxed": A_relaxed,    # Soft adjacency surrogate Ã (gradients)
            "A_hard": A_hard,          # Hard DAG A = B ⊙ M(r) (binary, acyclic)
            "A_forward": A_forward,    # A_ST = Ã + sg(A − Ã) when ST is on
            "gamma_mat": gamma_mat,    # Edge logits in matrix form
            "noise_scales": current_noise_scales,  # Sampled per-node observation noise
            "noise_log_mu": noise_log_mu,          # Posterior mean of log noise
            "noise_log_sigma": noise_log_sigma,    # Posterior std of log noise
            "r": r,                    # Raw order potentials (for logging)
            "r_centered": r_centered,  # Shift-invariant order representation
        }
        return preds, terms

    def sample_A_hard(self, r, rng, T_B) -> jnp.ndarray:
        """
        Draw the hard DAG A = B ⊙ M(r) only.

        This is the exact ``A_hard`` of ``forward_sample`` -- the same PRNG
        split order, the same flow call, the same threshold -- but it skips
        everything ``A_hard`` does not depend on: the per-node Bayesian MLPs,
        the noise posterior, the likelihood, and the Sinkhorn relaxation
        M_τ (the hard mask uses ranks, not P_τ). Posterior sampling only ever
        reads ``A_hard``, so this returns bit-identical draws for a fraction
        of the work.
        """
        rng, k_z, k_B, _k_noise = jrand.split(rng, 4)
        r_centered = center_order_potentials(r)

        d = self.num_nodes * (self.num_nodes - 1)
        z = jrand.normal(k_z, (d,))
        gamma_flat, _log_det = self.flow(z, cond=r_centered)
        gamma_flat = jnp.clip(gamma_flat, -15.0, 15.0)

        B_tilde_vec = logistic_concrete(k_B, gamma_flat, T_B)
        B_tilde = vec_to_offdiag_matrix(B_tilde_vec, self.num_nodes)
        B_hard = jnp.where(B_tilde >= 0.5, 1.0, 0.0)
        return B_hard * build_mask_M_hard(r_centered)

    def sample_B_hard(self, r, rng, T_B) -> jnp.ndarray:
        """
        Draw the edge indicators B = 1{gamma + R > 0} WITHOUT the DAG mask.

        Same construction as ``sample_A_hard`` minus the ``M(r)`` factor, so it
        reports what the flow believes about each ordered pair irrespective of
        whether the current ordering permits it. Used to seed the SVGD
        particles from the learned edge structure (see
        ``svidag.train.warmstart_particles``); the returned matrix is generally
        cyclic and is never used as a posterior sample.
        """
        rng, k_z, k_B, _k_noise = jrand.split(rng, 4)
        r_centered = center_order_potentials(r)
        d = self.num_nodes * (self.num_nodes - 1)
        z = jrand.normal(k_z, (d,))
        gamma_flat, _log_det = self.flow(z, cond=r_centered)
        gamma_flat = jnp.clip(gamma_flat, -15.0, 15.0)
        B_tilde_vec = logistic_concrete(k_B, gamma_flat, T_B)
        B_tilde = vec_to_offdiag_matrix(B_tilde_vec, self.num_nodes)
        return jnp.where(B_tilde >= 0.5, 1.0, 0.0)

    def __call__(self, x, r, rng, T_B, tau_sink, alpha_mat, beta_mat, st_weight=1.0):
        """Flax __call__ delegates to forward_sample."""
        return self.forward_sample(x, r, rng, T_B, tau_sink, alpha_mat, beta_mat, st_weight)
