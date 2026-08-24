'''
Evaluation module for the SVIDAG.

Author: Shrenik Zinage
'''

import numpy as np
import jax
import jax.random as jrand

from . import config
from .train import apply_model
from .utils import to_numpy, dag_to_cpdag


def _select_particle_indices(rng, num_samples, distinct_particles=False):
    """
    Select particle indices for posterior sampling.

    When distinct_particles=True and the requested sample count is no larger than
    the particle count, indices are drawn without replacement so visualization
    can reflect different order-potential particles rather than repeated draws
    from the same particle.
    """
    if distinct_particles and num_samples <= config.n_particles:
        return jrand.permutation(rng, config.n_particles)[:num_samples]
    return jrand.randint(rng, (num_samples,), 0, config.n_particles)


def posterior_predict(apply_fn, params, particles, scaled_test_data, rng, T_B_final, tau_final, alpha_mat, beta_mat, num_samples=50):
    """
    Generate posterior predictive samples for test data.
    
    Samples graph structures from the posterior q(A) and computes predictions
    under each sampled structure. Returns mean and std across samples.
    
    Args:
        apply_fn: Model apply function
        params: Trained model parameters
        particles: SVGD particles [K, m] for order potentials
        scaled_test_data: Standardized test observations [N_test, m]
        rng: JAX PRNG key
        T_B_final, tau_final: Final temperature values
        alpha_mat, beta_mat: Beta prior parameters
        num_samples: Number of posterior samples to draw
        
    Returns:
        mean_preds: Mean prediction [N_test, m]
        std_preds: Standard deviation [N_test, m] capturing epistemic uncertainty
    """
    # Uniformly sample from SVGD particles (approximate posterior over r)
    idx = jrand.randint(rng, (num_samples,), 0, config.n_particles)
    selected_r = particles[idx]
    keys = jrand.split(rng, num_samples)

    def single_eval(r_val, key):
        preds, _ = apply_model(apply_fn, params, scaled_test_data, r_val, key, T_B_final, tau_final, alpha_mat, beta_mat)
        return preds

    # Vectorized evaluation over all samples
    preds_samples = to_numpy(jax.vmap(single_eval)(selected_r, keys)).astype(np.float32, copy=False)
    mean_preds = np.mean(preds_samples, axis=0, dtype=np.float32)
    std_preds = np.std(preds_samples, axis=0, dtype=np.float32)
    return mean_preds, std_preds


def sample_relaxed_adj(
    apply_fn,
    params,
    particles,
    rng,
    T_B,
    tau_sn,
    alpha_mat,
    beta_mat,
    num_samples,
    train_data,
    distinct_particles=False,
):
    """
    Sample relaxed (soft) adjacency matrices from the posterior.
    
    Each sample represents a draw from q(A|r) where r is sampled from
    the SVGD particle approximation to p(r|data).
    
    Args:
        num_samples: Number of adjacency samples to draw
        train_data: Training data (only first sample used as reference)
        
    Returns:
        A_samples: [num_samples, m, m] soft adjacency matrices with values in [0,1]
    """
    idx = _select_particle_indices(rng, num_samples, distinct_particles=distinct_particles)
    selected_r = particles[idx]
    keys = jrand.split(rng, num_samples)
    # Need a reference batch for model forward pass; use single sample
    reference_batch = train_data[:1]

    def single_eval(r_val, key):
        _, terms = apply_model(apply_fn, params, reference_batch, r_val, key, T_B, tau_sn, alpha_mat, beta_mat)
        return terms["A_relaxed"]

    A_samples = to_numpy(jax.vmap(single_eval)(selected_r, keys)).astype(np.float32, copy=False)
    return A_samples  # [S, m, m]


def sample_hard_adj(
    apply_fn,
    params,
    particles,
    rng,
    T_B,
    tau_sn,
    alpha_mat,
    beta_mat,
    num_samples,
    train_data,
    distinct_particles=False,
):
    """
    Sample hard (binary) DAG adjacency matrices from the posterior.

    Each draw follows the generative construction of the formulation exactly:
    r uniform over the SVGD particles, z ~ p0, γ = T_φ(z; r),
    B = 1{γ + R > 0}, A = B ⊙ M(r) with the hard permutation mask
    M(r) = P(r) L P(r)^T. Every sample is a binary DAG by construction —
    no thresholding is involved.

    Args:
        num_samples: Number of adjacency samples to draw
        train_data: Training data (only first sample used as reference)

    Returns:
        A_samples: [num_samples, m, m] binary DAG adjacency matrices
    """
    idx = _select_particle_indices(rng, num_samples, distinct_particles=distinct_particles)
    selected_r = particles[idx]
    keys = jrand.split(rng, num_samples)
    reference_batch = train_data[:1]

    def single_eval(r_val, key):
        _, terms = apply_model(apply_fn, params, reference_batch, r_val, key, T_B, tau_sn, alpha_mat, beta_mat)
        return terms["A_hard"]

    A_samples = to_numpy(jax.vmap(single_eval)(selected_r, keys)).astype(np.float32, copy=False)
    return A_samples  # [S, m, m]


def sample_hard_adj_fast(apply_fn, params, particles, rng, T_B, num_samples, distinct_particles=False):
    """
    Same draws as ``sample_hard_adj``, via the model's ``sample_A_hard`` method.

    ``sample_hard_adj`` runs the full ``forward_sample`` -- all m per-node
    Bayesian MLPs, their hypernetworks, the noise posterior and the 300-step
    Sinkhorn -- once per sample, then discards everything except ``A_hard``.
    None of that enters A = B ⊙ M(r), so this path skips it. Identical PRNG
    split order, so the samples are bit-identical.
    """
    idx = _select_particle_indices(rng, num_samples, distinct_particles=distinct_particles)
    selected_r = particles[idx]
    keys = jrand.split(rng, num_samples)

    def single_eval(r_val, key):
        return apply_fn({"params": params}, r_val, key, T_B, method=lambda m, *a: m.sample_A_hard(*a))

    A_samples = to_numpy(jax.jit(jax.vmap(single_eval))(selected_r, keys))
    return A_samples.astype(np.float32, copy=False)  # [S, m, m]


def sample_relaxed_B(
    apply_fn,
    params,
    particles,
    rng,
    T_B,
    tau_sn,
    alpha_mat,
    beta_mat,
    num_samples,
    train_data,
    distinct_particles=False,
):
    """
    Sample relaxed Bernoulli matrices (B) from the posterior.
    
    Args:
        num_samples: Number of samples to draw
        train_data: Training data (only first sample used as reference)
        
    Returns:
        B_samples: [num_samples, m, m] relaxed Bernoulli matrices
    """
    idx = _select_particle_indices(rng, num_samples, distinct_particles=distinct_particles)
    selected_r = particles[idx]
    keys = jrand.split(rng, num_samples)
    reference_batch = train_data[:1]

    def single_eval(r_val, key):
        _, terms = apply_model(apply_fn, params, reference_batch, r_val, key, T_B, tau_sn, alpha_mat, beta_mat)
        return terms["B_tilde"]

    B_samples = to_numpy(jax.vmap(single_eval)(selected_r, keys)).astype(np.float32, copy=False)
    return B_samples


def sample_mask_M(
    apply_fn,
    params,
    particles,
    rng,
    T_B,
    tau_sn,
    alpha_mat,
    beta_mat,
    num_samples,
    train_data,
    distinct_particles=False,
):
    """
    Sample DAG masks (M) from the posterior.
    
    Args:
        num_samples: Number of samples to draw
        train_data: Training data (only first sample used as reference)
        
    Returns:
        M_samples: [num_samples, m, m] DAG masks
    """
    idx = _select_particle_indices(rng, num_samples, distinct_particles=distinct_particles)
    selected_r = particles[idx]
    keys = jrand.split(rng, num_samples)
    reference_batch = train_data[:1]

    def single_eval(r_val, key):
        _, terms = apply_model(apply_fn, params, reference_batch, r_val, key, T_B, tau_sn, alpha_mat, beta_mat)
        return terms["M_tau"]

    M_samples = to_numpy(jax.vmap(single_eval)(selected_r, keys)).astype(np.float32, copy=False)
    return M_samples


def sample_single_ABC(apply_fn, params, particles, rng, T_B, tau_sn, alpha_mat, beta_mat, train_data):
    """
    Sample one random posterior draw of A, B, and M matrices.
    
    Args:
        train_data: Training data (only first sample used as reference)
        
    Returns:
        A_sample: [m, m] relaxed adjacency sample
        B_sample: [m, m] relaxed Bernoulli sample before mask
        M_sample: [m, m] Sinkhorn mask sample
    """
    idx = jrand.randint(rng, (1,), 0, config.n_particles)
    selected_r = particles[idx]
    _, single_key = jrand.split(rng)
    reference_batch = train_data[:1]

    def single_eval(r_val, key):
        _, terms = apply_model(apply_fn, params, reference_batch, r_val, key, T_B, tau_sn, alpha_mat, beta_mat)
        return terms["A_relaxed"], terms["B_tilde"], terms["M_tau"]

    A_sample, B_sample, M_sample = single_eval(selected_r[0], single_key)
    A_sample = to_numpy(A_sample).astype(np.float32)
    B_sample = to_numpy(B_sample).astype(np.float32)
    M_sample = to_numpy(M_sample).astype(np.float32)
    return A_sample, B_sample, M_sample


def estimate_dag_probabilities(apply_fn, params, particles, rng, alpha_mat, beta_mat, target_dags_dict, num_samples, T_B, tau_sn, train_data):
    """
    Estimate posterior probabilities of specific DAG structures.

    Draws hard (binary) DAG samples A = B ⊙ M(r) from the posterior and
    counts matches against target DAGs of interest.

    Args:
        target_dags_dict: Dict mapping DAG names to adjacency matrices
        num_samples: Number of Monte Carlo samples

    Returns:
        probs: Dict mapping DAG names to estimated probabilities
        top_list: List of (adjacency_matrix, probability) for top 10 most frequent DAGs
    """
    target_arrays = {name: np.array(adj, dtype=np.int8) for name, adj in target_dags_dict.items()}
    counts = {name: 0 for name in target_arrays.keys()}
    unique_count = {}  # Track all unique DAGs seen
    first_target = next(iter(target_arrays.values()))
    m = first_target.shape[0]

    # Sample hard DAGs from posterior (binary by construction)
    A_bin_samples = sample_hard_adj(
        apply_fn, params, particles, rng, T_B, tau_sn, alpha_mat, beta_mat,
        num_samples, train_data,
    ).astype(np.int8)

    # Count matches and track unique DAGs
    for A_bin in A_bin_samples:
        for name, A_target in target_arrays.items():
            if np.array_equal(A_bin, A_target):
                counts[name] += 1
        # Use bytes representation as hash key for uniqueness
        key = A_bin.tobytes()
        unique_count[key] = unique_count.get(key, 0) + 1
    
    # Convert counts to probabilities
    probs = {name: counts[name] / num_samples for name in counts}
    # Get top 10 most frequent DAGs
    top = sorted(unique_count.items(), key=lambda kv: kv[1], reverse=True)[:10]
    top_list = [(np.frombuffer(k, dtype=np.int8).reshape(m, m), c / num_samples) for k, c in top]
    return probs, top_list


def estimate_cpdag_probabilities(apply_fn, params, particles, rng, alpha_mat, beta_mat, num_samples, T_B, tau_sn, train_data):
    """
    Estimate posterior probabilities over Markov Equivalence Classes (MECs).

    Unlike DAG probabilities, this groups DAGs into their equivalence classes
    (CPDAGs) before counting. Two DAGs in the same MEC imply the same
    conditional independence relationships.

    Args:
        num_samples: Number of Monte Carlo samples

    Returns:
        top_list: List of (cpdag_matrix, probability) for top 10 most frequent MECs
    """
    # Sample hard DAGs from posterior (binary by construction)
    A_bin_samples = sample_hard_adj(
        apply_fn, params, particles, rng, T_B, tau_sn, alpha_mat, beta_mat,
        num_samples, train_data,
    ).astype(np.int8)
    m = A_bin_samples.shape[1]

    cpdag_counts = {}
    
    for A_bin in A_bin_samples:
        # Convert DAG → CPDAG (essential graph representing MEC)
        # Convention conversion: our A has j→i, dag_to_cpdag expects i→j
        # So transpose before and after
        cpdag = dag_to_cpdag(A_bin.T).T
        cpdag = cpdag.astype(np.int8)
        
        # Use CPDAG as unique identifier for the equivalence class
        key = cpdag.tobytes()
        cpdag_counts[key] = cpdag_counts.get(key, 0) + 1

    # Return top 10 most probable equivalence classes
    top = sorted(cpdag_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    top_list = [(np.frombuffer(k, dtype=np.int8).reshape(m, m), c / num_samples) for k, c in top]
    
    return top_list
