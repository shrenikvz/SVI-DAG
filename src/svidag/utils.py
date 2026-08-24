'''
Utility functions for the SVIDAG.

Author: Shrenik Zinage
'''

from functools import lru_cache
import numpy as np
from typing import Tuple
import jax
import jax.numpy as jnp
import jax.random as jrand
import jax.scipy as jsc
import jax.scipy.special as jsp

from . import config

# Singleton to cache the compute device across calls
_compute_device = None


def log_device_info():
    """Print device information and return the preferred compute device."""
    print("JAX devices:", jax.devices())
    print("Default backend:", jax.default_backend())
    try:
        device = jax.devices("gpu")[0]
        print(f"✓ GPU detected: {device}")
    except (RuntimeError, IndexError):
        # Fall back to CPU if no GPU available
        device = jax.devices()[0]
        print(f"⚠ No GPU found - running on {device.platform.upper()}")
    return device


def get_compute_device():
    """Lazily determine the compute device to mirror original behavior."""
    global _compute_device
    if _compute_device is None:
        _compute_device = log_device_info()
    return _compute_device


def to_device(array):
    """Move data to the default compute device (GPU when available)."""
    return jax.device_put(array, device=get_compute_device())


def to_numpy(array):
    """Safely move JAX arrays back to host NumPy for plotting/logging."""
    if isinstance(array, np.ndarray):
        return array
    try:
        # device_get handles async transfer from GPU to host
        return np.asarray(jax.device_get(array))
    except (TypeError, ValueError):
        return np.asarray(array)


def linear_anneal(it: float, total: float, start: float, end: float):
    """
    Linearly anneals the value from start to end over total iterations.
    """
    # Clip ensures t ∈ [0, 1] even if it > total
    t = jnp.clip(it / total, 0.0, 1.0)
    return (1.0 - t) * start + t * end


def center_order_potentials(r):
    """
    Remove the mean shift from each order-potential vector.

    The Sinkhorn ordering construction only depends on relative node scores, so
    the common-shift direction in r is structurally unidentifiable. Centering r
    makes that explicit and keeps downstream geometry focused on order-relevant
    differences between nodes.
    """
    return r - jnp.mean(r, axis=-1, keepdims=True)


@lru_cache(maxsize=None)
def offdiag_indices(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the off-diagonal indices of an n x n matrix.
    Used for vectorizing operations on edge parameters (excluding self-loops).
    """
    mask = np.ones((n, n), dtype=np.float32) - np.eye(n, dtype=np.float32)
    return np.where(mask)


@lru_cache(maxsize=None)
def _strict_lower_triangular_mask(n: int):
    """Cache the DAG-order mask constant used by Sinkhorn-based ordering."""
    return jnp.tril(jnp.ones((n, n), dtype=jnp.float32), k=-1)


def vec_to_offdiag_matrix(vec, n: int):
    """
    Converts a flat vector of n*(n-1) elements to an n×n matrix with zeros on diagonal.
    Maps flow outputs (gamma logits) to adjacency-shaped matrices.
    """
    idx = offdiag_indices(n)
    mat = jnp.zeros((n, n), dtype=vec.dtype)
    # JAX functional update: mat.at[idx].set(vec) returns new array
    return mat.at[idx].set(vec)


def mat_offdiag_to_vec(mat):
    """
    Extracts off-diagonal elements from matrix to flat vector.
    Inverse of vec_to_offdiag_matrix.
    """
    n = mat.shape[-1]
    idx = offdiag_indices(n)
    return mat[idx]


def sample_logistic_noise(key, shape):
    """
    Samples from the standard Logistic(0,1) distribution using inverse CDF.
    Used as the noise source for Gumbel-Softmax / Concrete relaxation.
    """
    # Clamp to (1e-6, 1-1e-6) to avoid log(0) numerical issues
    u = jrand.uniform(key, shape=shape, minval=1e-6, maxval=1.0 - 1e-6)
    # Inverse CDF of logistic: log(u / (1-u))
    return jnp.log(u) - jnp.log1p(-u)


def logistic_concrete(key, logits, temperature):
    """
    Gumbel-Softmax / Concrete relaxation of Bernoulli variables.
    Returns continuous samples in (0, 1) that approximate hard 0/1 as temperature → 0.

    Args:
        key: JAX PRNG key
        logits: Log-odds γ for edge probabilities
        temperature: Controls sharpness; lower = closer to discrete

    Returns:
        Relaxed Bernoulli samples ∈ (0, 1)
    """
    r = sample_logistic_noise(key, logits.shape)
    # Sigmoid of (logit + noise) / temp gives Concrete relaxation
    return jax.nn.sigmoid((logits + r) / temperature)


def sinkhorn_normalization(X, n_iters=config.sinkhorn_iters, eps=config.sinkhorn_eps):
    """
    Iteratively normalizes rows and columns to produce a doubly-stochastic matrix.
    Used to relax permutation matrices for differentiable DAG ordering.

    Args:
        X: Non-negative matrix to normalize
        n_iters: Number of alternating row/column normalization iterations
        eps: Small constant for numerical stability in division

    Returns:
        Approximately doubly-stochastic matrix (rows and columns sum to ~1/n)
    """
    def body(_, x):
        # Alternating projections: normalize rows, then columns
        x = x / (jnp.sum(x, axis=1, keepdims=True) + eps)
        x = x / (jnp.sum(x, axis=0, keepdims=True) + eps)
        return x

    # Initial global normalization ensures bounded starting point
    X = X / (jnp.sum(X, axis=(0, 1), keepdims=True) + eps)
    # fori_loop is JIT-friendly fixed iteration count loop
    return jax.lax.fori_loop(0, n_iters, body, X)


def log_sinkhorn_normalization(log_K, n_iters=config.sinkhorn_iters):
    """
    Doubly-stochastic projection carried out entirely in log space.

    The probability-space version above is only safe while ``exp`` of the
    score matrix stays in float32 range. Here the scores are
    ``S0(r)/tau`` with ``|S0| ~ ||r||*m``, so the spread grows like
    ``||r||*m/tau`` -- 250*||r|| at m=25, tau=0.1. Past a spread of ~87 the
    smallest entries of ``exp(logits - max)`` underflow to *exactly* zero,
    and because the row/column step maps ``0 -> 0/(0+eps) = 0``, zeros are an
    absorbing state: no number of further iterations can recover them. The
    result stops being doubly stochastic (observed row sums 0.0 to 3.0 at
    ||r||=1.5) and M_tau decorrelates from the hard mask it is supposed to
    relax, which zeroes the only gradient path to the edge logits.

    Working in log space with logsumexp removes the underflow entirely and
    agrees with the probability-space result to ~1e-6 wherever that one is
    still valid.
    """
    def body(_, lk):
        lk = lk - jsp.logsumexp(lk, axis=1, keepdims=True)
        lk = lk - jsp.logsumexp(lk, axis=0, keepdims=True)
        return lk

    log_K = log_K - jsp.logsumexp(log_K)
    return jax.lax.fori_loop(0, n_iters, body, log_K)


def build_P_tau(r, tau):
    """
    Constructs soft permutation matrix P_τ(r) from order potentials r.
    P_τ approaches a hard permutation as τ → 0.

    The construction follows: P = Sinkhorn(exp(S/τ)) where S = r ⊗ o^T
    with o = (m, m-1, ..., 1) so that P sorts indices by descending potential.

    Args:
        r: Order potential vector of shape (m,) - encodes node ordering
        tau: Temperature controlling softness of permutation

    Returns:
        Soft permutation matrix P_τ of shape (m, m)
    """
    r = center_order_potentials(r)
    n = r.shape[0]

    if config.sinkhorn_scale_invariant:
        # M(r) reads only the ranks of r, so rescaling r leaves the hard mask
        # untouched -- but it rescales these logits, and with them how soft the
        # relaxation is. Normalising makes tau the only control on softness,
        # instead of tau/rms(r) drifting as the particles spread.
        rms = jnp.sqrt(jnp.mean(r ** 2))
        r = r / jnp.maximum(rms, 1e-6)

    o = jnp.arange(n, 0, -1, dtype=r.dtype)  # Position weights: (m, m-1, ..., 1)
    S0 = jnp.outer(r, o)
    
    logits = S0 / tau
    if config.sinkhorn_log_space:
        # Normalise in log space: exp() of these logits underflows to exactly
        # zero once ||r|| grows, and the probability-space iteration cannot
        # recover from a zero. See log_sinkhorn_normalization.
        return jnp.exp(log_sinkhorn_normalization(logits, n_iters=config.sinkhorn_iters))
    logits = logits - jnp.max(logits)
    K = jnp.exp(logits)
    return sinkhorn_normalization(K, n_iters=config.sinkhorn_iters, eps=config.sinkhorn_eps)


def build_mask_M_tau(r, tau):
    """
    Constructs the DAG acyclicity mask M_τ from order potentials.
    
    M_τ = P @ L_strict @ P^T where L_strict is strict lower triangular.
    This ensures M_τ[i,j] ≈ 1 only if node j precedes node i in the ordering,
    enforcing acyclicity: edges can only go from earlier to later nodes.

    Args:
        r: Order potential vector determining node ordering
        tau: Temperature for soft permutation

    Returns:
        Mask matrix M_τ with M[i,j] ≈ 1 if edge j→i is allowed (DAG-consistent)
    """
    n = r.shape[0]
    P = build_P_tau(r, tau)
    l_strict = _strict_lower_triangular_mask(n)
    # P @ L @ P^T transforms the constraint to the soft permutation basis
    return P @ l_strict @ P.T


def build_mask_M_hard(r):
    """
    Constructs the hard DAG acyclicity mask M(r) = P(r) L P(r)^T, where P(r)
    is the permutation that sorts indices by descending potential and L is the
    strictly lower triangular matrix of ones.

    Computed without materializing P(r): with ranks[i] the position of node i
    in the descending sort of r, P L P^T has entry [i, j] equal to
    1{ranks[i] > ranks[j]}, i.e. an edge j -> i (A[i, j] = 1) is allowed
    exactly when j precedes i in the topological order.

    Args:
        r: Order potential vector of shape (m,)

    Returns:
        Binary mask M(r) of shape (m, m); B ⊙ M(r) is a DAG for any binary B.
    """
    ranks = jnp.argsort(jnp.argsort(-r))
    return (ranks[:, None] > ranks[None, :]).astype(jnp.float32)


def gaussian_logpdf(x, mu, sigma):
    """Element-wise Gaussian log-PDF: log N(x | mu, sigma²)."""
    var = sigma**2
    return -0.5 * ((x - mu) ** 2 / var + jnp.log(2 * jnp.pi * var))


def gaussian_logpdf_diag(x, mu, sigma_vec):
    """Log-PDF of diagonal Gaussian: log N(x | mu, diag(sigma_vec²))."""
    var = sigma_vec**2
    return -0.5 * (
        jnp.sum(((x - mu) ** 2) / var, axis=-1) + jnp.sum(jnp.log(2 * jnp.pi * var))
    )


def std_normal_logpdf(z):
    """Log-PDF of standard normal N(0, I). Used for base distribution in flows."""
    return -0.5 * (jnp.sum(z**2, axis=-1) + z.shape[-1] * jnp.log(2 * jnp.pi))


def logistic_beta_logpdf_gamma(gamma_mat, alpha_mat, beta_mat):
    """
    Computes log p(γ) under Logistic-Beta prior: sigmoid(γ) ~ Beta(α, β).
    
    This is the domain-informed prior where α, β encode expert beliefs about
    edge probabilities. The logistic transform maps real-valued γ to (0,1).

    Args:
        gamma_mat: Edge logits γ in matrix form (m×m)
        alpha_mat: Beta distribution α parameters (m×m)
        beta_mat: Beta distribution β parameters (m×m)

    Returns:
        Scalar log-probability summed over all off-diagonal edges
    """
    # Extract off-diagonal elements (exclude self-loops)
    gamma = mat_offdiag_to_vec(gamma_mat)
    alpha = mat_offdiag_to_vec(alpha_mat)
    beta = mat_offdiag_to_vec(beta_mat)
    # Transform to (0,1) via sigmoid for Beta density evaluation
    s = jax.nn.sigmoid(gamma)
    s = jnp.clip(s, 1e-8, 1 - 1e-8)  # Numerical stability for log
    # Beta log-PDF: log B(α,β)^-1 + (α-1)log(s) + (β-1)log(1-s)
    # Here we use the simplified form with α*log(s) + β*log(1-s)
    term = -jsp.betaln(alpha, beta) + alpha * jnp.log(s) + beta * jnp.log1p(-s)
    return jnp.sum(term)


def get_prior_matrix(scenario: str, node_names, true_adj_np, num_nodes=2):
    """
    Constructs prior edge probability matrix p_ij for different experimental scenarios.
    
    Scenarios test how domain knowledge (correct vs incorrect priors) affects learning:
    - strong_correct: High confidence on true edges (0.99 true, 0.01 false)
    - strong_incorrect: High confidence on reversed edges (misleading prior)
    - noninformative: Uniform 0.5 (no domain knowledge)

    Args:
        scenario: One of the above scenario names
        node_names: List of node names (unused, kept for API compatibility)
        true_adj_np: Ground truth adjacency matrix
        num_nodes: Number of nodes in the graph

    Returns:
        p_prior: Matrix of prior edge probabilities with zeros on diagonal
    """
    diag_mask = jnp.eye(num_nodes)
    true = jnp.array(true_adj_np)
    # "Wrong" prior assigns high probability to reversed edges (transposed)
    wrong = jnp.transpose(true) * (1 - diag_mask)

    if scenario == "strong_correct":
        p_prior = true * 0.99 + (1 - true) * 0.01
    elif scenario == "strong_incorrect":
        p_prior = wrong * 0.99 + (1 - wrong) * 0.01
    elif scenario == "noninformative":
        p_prior = jnp.full((num_nodes, num_nodes), 0.5, dtype=jnp.float32)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    # Zero out diagonal: no self-loops allowed
    return p_prior * (1 - diag_mask)


def compute_nu_matrix(p_prior):
    """
    Computes concentration parameter ν for Beta prior from edge probabilities.
    
    ν controls how peaked the Beta(α, β) distribution is around p_prior.
    Higher ν = stronger prior belief. Formula: ν = ν_min + κ * |p - 0.5|^γ
    
    This makes priors closer to 0 or 1 more confident (higher ν).

    Args:
        p_prior: Prior edge probability matrix

    Returns:
        nu: Concentration parameter matrix (zeros on diagonal)
    """
    # Distance from uncertainty (0.5) determines confidence
    abs_term = jnp.abs(p_prior - 0.5)
    nu = config.NU_MIN + config.PR_PREC_KAPPA * jnp.power(abs_term, config.PR_PREC_GAMMA)
    diag_mask = jnp.eye(p_prior.shape[0], dtype=jnp.float32)
    return nu * (1 - diag_mask)


def compute_alpha_beta_from_prior(p_prior):
    """
    Converts prior probabilities to Beta distribution parameters.
    
    For Beta(α, β): mean = α/(α+β) = p, concentration = α + β ≈ ν.
    We use α = ν*p + 1, β = ν*(1-p) + 1 to ensure α, β ≥ 1.

    Args:
        p_prior: Prior edge probability matrix

    Returns:
        alpha, beta: Beta distribution parameters, shape matches p_prior
    """
    nu = compute_nu_matrix(p_prior)
    # +1 ensures proper Beta distribution (α, β > 0)
    alpha = nu * p_prior + 1.0
    beta = nu * (1.0 - p_prior) + 1.0
    return alpha, beta


def gaussian_likelihood_logsum(x_true, x_pred, noise_scales, include_const=True):
    """
    Computes Gaussian observation log-likelihood: Σ log N(x_true | x_pred, σ²).
    
    Separates MSE term from normalizing constant for optional exclusion
    (constant can be dropped when noise_scales are fixed, not learned).

    Args:
        x_true: Observed data [N, m]
        x_pred: Model predictions [N, m]
        noise_scales: Per-node observation noise σ [m]
        include_const: If False, omit -0.5*N*log(2πσ²) term
 
    Returns:
        ll_mse: Data-fit term -0.5 * Σ((x-μ)²/σ²)
        ll_const: Normalizing constant (0 if include_const=False)
    """
    var = noise_scales**2 + 1e-6  # Small epsilon prevents division by zero
    ll_mse = -0.5 * jnp.sum(((x_true - x_pred) ** 2) / var)
    ll_const = 0.0
    if include_const:
        # Constant term: -0.5 * N * Σ_i log(2πσ_i²)
        ll_const = -0.5 * x_true.shape[0] * jnp.sum(jnp.log(2 * jnp.pi * var))
    return ll_mse, ll_const


def _compute_auroc_from_flat(pred_flat, true_flat):
    """
    Internal helper to compute AUROC from flattened prediction and label arrays.
    
    Uses the Mann-Whitney U statistic formulation with proper tie handling.
    
    Args:
        pred_flat: Flattened predicted probabilities
        true_flat: Flattened binary labels
    
    Returns:
        auroc: AUROC score, or NaN if only one class is present.
    """
    pred_flat = np.asarray(pred_flat, dtype=np.float64)
    true_flat = np.asarray(true_flat, dtype=np.int32)
    
    n_pos = np.sum(true_flat == 1)
    n_neg = np.sum(true_flat == 0)
    
    # AUROC is undefined when only one class is present
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    
    # Sort predictions and corresponding labels
    order = np.argsort(pred_flat)
    pred_sorted = pred_flat[order]
    true_sorted = true_flat[order]
    
    # Compute ranks with tie handling (average ranks for tied values)
    n = len(pred_flat)
    ranks = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        # Find all elements with the same predicted value (ties)
        while j < n and pred_sorted[j] == pred_sorted[i]:
            j += 1
        # Assign average rank to all tied elements (1-indexed)
        avg_rank = (i + 1 + j) / 2.0
        ranks[i:j] = avg_rank
        i = j
    
    # Sum of ranks for positive examples
    sum_ranks_pos = np.sum(ranks[true_sorted == 1])
    
    # Mann-Whitney U statistic formula for AUROC
    # U = R - n_pos*(n_pos+1)/2, where R is sum of positive ranks
    # AUROC = U / (n_pos * n_neg)
    auroc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    
    return float(auroc)


def compute_metrics(pred_adj, true_adj, pred_probs=None):
    """
    Computes comprehensive DAG learning metrics.
    
    Computes both binary classification metrics (SHD, TPR, F1) and, if soft
    predictions are provided, probabilistic metrics (Brier Score, AUROC).
    
    Args:
        pred_adj: Predicted binary adjacency matrix [m, m]
        true_adj: Ground truth binary adjacency matrix [m, m]
        pred_probs: Optional predicted edge probabilities (soft adjacency) [m, m].
                    Values should be in [0, 1]. Required for Brier Score and AUROC.

    Returns:
        dict with keys:
            - 'SHD': Structural Hamming Distance (lower is better, 0 = perfect)
            - 'TPR': True Positive Rate (higher is better, 1 = all true edges found)
            - 'F1': F1 Score (higher is better, 1 = perfect)
            - 'Brier': Brier Score (lower is better, 0 = perfect). None if pred_probs not provided.
            - 'AUROC': Area Under ROC Curve (higher is better, 1 = perfect). None if pred_probs not provided.
    """
    # Compute TP, FP, FN for binary metrics
    TP = np.sum((pred_adj == 1) & (true_adj == 1))
    FP = np.sum((pred_adj == 1) & (true_adj == 0))
    FN = np.sum((pred_adj == 0) & (true_adj == 1))
    
    # SHD: Structural Hamming Distance
    SHD = int(FP + FN)
    
    # TPR: True Positive Rate (Recall)
    n_true_edges = np.sum(true_adj == 1)
    TPR = float(TP / n_true_edges) if n_true_edges > 0 else 1.0
    
    # F1 Score: Harmonic mean of precision and recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    if precision + recall == 0:
        F1 = 0.0
    else:
        F1 = float(2.0 * (precision * recall) / (precision + recall))
    
    # Probabilistic metrics (require soft predictions)
    Brier = None
    AUROC = None
    
    if pred_probs is not None:
        m = pred_probs.shape[0]
        # Off-diagonal mask to exclude self-loops
        mask = np.ones((m, m), dtype=np.float32) - np.eye(m, dtype=np.float32)
        
        pred_flat = pred_probs[mask == 1].astype(np.float64)
        true_flat = true_adj[mask == 1].astype(np.float64)
        
        # Brier Score: Mean squared error between probabilities and binary labels
        Brier = float(np.mean((pred_flat - true_flat) ** 2))
        
        # AUROC: Area Under ROC Curve
        AUROC = _compute_auroc_from_flat(pred_flat, true_flat.astype(np.int32))
    
    return {
        'SHD': SHD,
        'TPR': TPR,
        'F1': F1,
        'Brier': Brier,
        'AUROC': AUROC
    }


def dag_to_cpdag(dag_adj):
    """
    Converts a DAG to its Markov Equivalence Class representation (CPDAG/Essential Graph).
    
    A CPDAG has directed edges where all DAGs in the MEC agree, and undirected edges
    where orientation varies across equivalent DAGs. Uses v-structure detection 
    followed by Meek's orientation rules.
    
    Args:
        dag_adj: (m, m) numpy array where dag_adj[i, j] = 1 means i -> j.
        
    Returns:
        cpdag_adj: (m, m) numpy array.
                   1 at [i, j] and 0 at [j, i] => i -> j (Directed, compelled)
                   1 at [i, j] and 1 at [j, i] => i - j (Undirected, reversible)
    """
    m = dag_adj.shape[0]
    # Step 1: Extract skeleton (edge existence ignoring direction)
    skeleton = ((dag_adj + dag_adj.T) > 0).astype(int)
    # Initialize CPDAG with all edges undirected (symmetric 1s)
    cpdag = skeleton.copy()
    
    # Step 2: Identify and orient v-structures (immoralities)
    # V-structure: i -> k <- j where i and j are NOT adjacent
    # These orientations are compelled (identical across MEC)
    for k in range(m):
        parents = np.where(dag_adj[:, k] == 1)[0]
        if len(parents) < 2:
            continue
        
        # Check all pairs of parents for v-structure
        for idx1 in range(len(parents)):
            for idx2 in range(idx1 + 1, len(parents)):
                i = parents[idx1]
                j = parents[idx2]
                
                # V-structure requires non-adjacency between parents
                if skeleton[i, j] == 0:
                    # Orient both edges into k (compelled by v-structure)
                    cpdag[i, k] = 1
                    cpdag[k, i] = 0
                    cpdag[j, k] = 1
                    cpdag[k, j] = 0
    
    # Step 3: Apply Meek's rules iteratively until convergence
    # These rules propagate orientations to avoid creating new v-structures or cycles
    while True:
        changed = False
        
        # Rule 1: i -> k - j and i ⊥ j => orient k -> j
        # Prevents creating new v-structure i -> k <- j
        directed_edges = np.argwhere((cpdag == 1) & (cpdag.T == 0))
        for i, k in directed_edges:
            # Find undirected neighbors of k
            neighbors_k = np.where((cpdag[k, :] == 1) & (cpdag[:, k] == 1))[0]
            for j in neighbors_k:
                if skeleton[i, j] == 0:  # i and j non-adjacent
                    cpdag[k, j] = 1
                    cpdag[j, k] = 0
                    changed = True
        
        # Rule 2: i -> k -> j and i - j => orient i -> j
        # Prevents creating cycle i -> k -> j -> i
        for k in range(m):
            parents_k = np.where((cpdag[:, k] == 1) & (cpdag[k, :] == 0))[0]
            children_k = np.where((cpdag[k, :] == 1) & (cpdag[:, k] == 0))[0]
            
            for i in parents_k:
                for j in children_k:
                    if cpdag[i, j] == 1 and cpdag[j, i] == 1:  # i - j undirected
                        cpdag[i, j] = 1
                        cpdag[j, i] = 0
                        changed = True
                        
        # Rule 3: i - k -> j, i - l -> j, k ⊥ l, i - j => orient i -> j
        # Two non-adjacent paths force orientation
        undirected_edges = np.argwhere(np.triu((cpdag == 1) & (cpdag.T == 1)))
        for idx in range(len(undirected_edges)):
            i, j = undirected_edges[idx]
            
            # Track whether this specific edge was oriented (not the global flag)
            edge_oriented = False
            
            # Find neighbors of i that have directed edge to j
            neighbors_i = np.where((cpdag[i, :] == 1) & (cpdag[:, i] == 1))[0]
            candidates = []
            for node in neighbors_i:
                if node == j: continue
                if cpdag[node, j] == 1 and cpdag[j, node] == 0:
                    candidates.append(node)
            
            # Need two non-adjacent candidates
            if len(candidates) >= 2:
                found_pair = False
                for c1_idx in range(len(candidates)):
                    for c2_idx in range(c1_idx + 1, len(candidates)):
                        k = candidates[c1_idx]
                        l = candidates[c2_idx]
                        if skeleton[k, l] == 0:
                            cpdag[i, j] = 1
                            cpdag[j, i] = 0
                            changed = True
                            edge_oriented = True
                            found_pair = True
                            break
                    if found_pair: break
            
            # Symmetric check: maybe j -> i is the correct orientation
            # Use per-edge flag so other rules' changes don't skip this check
            if not edge_oriented:
                neighbors_j = np.where((cpdag[j, :] == 1) & (cpdag[:, j] == 1))[0]
                candidates_j = []
                for node in neighbors_j:
                    if node == i: continue
                    if cpdag[node, i] == 1 and cpdag[i, node] == 0:
                        candidates_j.append(node)
                
                if len(candidates_j) >= 2:
                    found_pair = False
                    for c1_idx in range(len(candidates_j)):
                        for c2_idx in range(c1_idx + 1, len(candidates_j)):
                            k = candidates_j[c1_idx]
                            l = candidates_j[c2_idx]
                            if skeleton[k, l] == 0:
                                cpdag[j, i] = 1
                                cpdag[i, j] = 0
                                changed = True
                                found_pair = True
                                break
                        if found_pair: break

        if not changed:
            break
            
    return cpdag


def compute_shd_cpdag(pred_cpdag, true_cpdag):
    """
    Compute Structural Hamming Distance between two CPDAGs.
    
    SHD for CPDAGs counts: (1) missing/extra edges in skeleton, 
    (2) orientation mismatches on common edges.
    
    Convention: 
      A[i,j]=1, A[j,i]=0 => directed i->j
      A[i,j]=1, A[j,i]=1 => undirected i-j
    """
    # Extract skeletons (ignore orientation)
    pred_skel = ((pred_cpdag + pred_cpdag.T) > 0).astype(int)
    true_skel = ((true_cpdag + true_cpdag.T) > 0).astype(int)
    
    # Skeleton errors: divide by 2 since symmetric matrices double-count edges
    diff_skel = np.abs(pred_skel - true_skel)
    skeleton_errors = np.sum(diff_skel) / 2
    
    # Orientation errors on edges present in both graphs
    intersection = (pred_skel * true_skel)
    orientation_errors = 0
    m = pred_cpdag.shape[0]
    
    # Only check upper triangle to avoid double-counting
    for i in range(m):
        for j in range(i+1, m):
            if intersection[i, j] == 1:
                # Encode orientation: 0=undirected, 1=i->j, 2=j->i
                pred_dir = 0
                if pred_cpdag[i,j]==1 and pred_cpdag[j,i]==1: pred_dir = 0
                elif pred_cpdag[i,j]==1: pred_dir = 1
                elif pred_cpdag[j,i]==1: pred_dir = 2
                
                true_dir = 0
                if true_cpdag[i,j]==1 and true_cpdag[j,i]==1: true_dir = 0
                elif true_cpdag[i,j]==1: true_dir = 1
                elif true_cpdag[j,i]==1: true_dir = 2
                
                if pred_dir != true_dir:
                    orientation_errors += 1
                    
    return skeleton_errors + orientation_errors


def compute_tpr_cpdag(pred_cpdag, true_cpdag):
    """
    Compute skeleton-level True Positive Rate (adjacency recall) for CPDAGs.
    An edge is a TP if it exists in both skeletons, regardless of orientation.
    
    Returns:
        TPR = (# edges in both) / (# edges in true graph)
    """
    pred_skel = ((pred_cpdag + pred_cpdag.T) > 0).astype(int)
    true_skel = ((true_cpdag + true_cpdag.T) > 0).astype(int)
    
    # Divide by 2 since symmetric matrices count each edge twice
    TP = np.sum((pred_skel == 1) & (true_skel == 1)) / 2
    P = np.sum(true_skel) / 2
    
    # Handle edge case of empty true graph
    if P == 0:
        return 1.0 if np.sum(pred_skel) == 0 else 0.0
        
    return TP / P


def compute_f1_cpdag(pred_cpdag, true_cpdag):
    """
    Compute skeleton-level F1 Score for CPDAGs.
    
    F1 is the harmonic mean of precision and recall, computed on the skeleton
    (ignoring edge orientations). An edge is a TP if it exists in both skeletons.
    
    Args:
        pred_cpdag: Predicted CPDAG adjacency matrix [m, m]
        true_cpdag: Ground truth CPDAG adjacency matrix [m, m]
    
    Returns:
        F1 score in [0, 1]. Higher is better (1 = perfect skeleton recovery).
    """
    # Extract skeletons (symmetric adjacency, ignoring orientation)
    pred_skel = ((pred_cpdag + pred_cpdag.T) > 0).astype(int)
    true_skel = ((true_cpdag + true_cpdag.T) > 0).astype(int)
    
    # Use upper triangle to count edges (avoid double-counting)
    pred_upper = np.triu(pred_skel, k=1)
    true_upper = np.triu(true_skel, k=1)
    
    TP = np.sum((pred_upper == 1) & (true_upper == 1))
    FP = np.sum((pred_upper == 1) & (true_upper == 0))
    FN = np.sum((pred_upper == 0) & (true_upper == 1))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2.0 * (precision * recall) / (precision + recall)
    return float(f1)


def compute_brier_cpdag(pred_probs, true_cpdag):
    """
    Compute skeleton-level Brier Score for CPDAGs.
    
    Measures calibration quality at the skeleton level: mean squared error between
    predicted edge probabilities and true skeleton edges. For CPDAGs, we symmetrize
    both predictions (max of both directions) and true skeleton, then compute
    Brier score on the upper triangle.
    
    Args:
        pred_probs: Predicted edge probabilities (soft adjacency) [m, m].
                    Values should be in [0, 1].
        true_cpdag: Ground truth CPDAG adjacency matrix [m, m]
    
    Returns:
        Brier score in [0, 1]. Lower is better (0 = perfect calibration).
    """
    m = pred_probs.shape[0]
    
    # Symmetrize predictions: take max of both directions for skeleton probability
    pred_skel_probs = np.maximum(pred_probs, pred_probs.T)
    
    # Extract true skeleton (binary)
    true_skel = ((true_cpdag + true_cpdag.T) > 0).astype(np.float64)
    
    # Upper triangle mask (exclude diagonal and lower triangle)
    upper_mask = np.triu(np.ones((m, m), dtype=np.float32), k=1)
    
    pred_flat = pred_skel_probs[upper_mask == 1].astype(np.float64)
    true_flat = true_skel[upper_mask == 1].astype(np.float64)
    
    # Brier score = mean squared error
    brier = np.mean((pred_flat - true_flat) ** 2)
    return float(brier)


def compute_auroc_cpdag(pred_probs, true_cpdag):
    """
    Compute skeleton-level AUROC for CPDAGs.
    
    Measures ranking quality at the skeleton level: probability that a randomly
    chosen true skeleton edge has a higher predicted probability than a randomly
    chosen non-edge. Predictions are symmetrized (max of both directions).
    
    Args:
        pred_probs: Predicted edge probabilities (soft adjacency) [m, m].
                    Values should be in [0, 1].
        true_cpdag: Ground truth CPDAG adjacency matrix [m, m]
    
    Returns:
        AUROC score in [0, 1]. Higher is better (1 = perfect ranking,
        0.5 = random). Returns NaN if only one class is present.
    """
    m = pred_probs.shape[0]
    
    # Symmetrize predictions: take max of both directions for skeleton probability
    pred_skel_probs = np.maximum(pred_probs, pred_probs.T)
    
    # Extract true skeleton (binary)
    true_skel = ((true_cpdag + true_cpdag.T) > 0).astype(np.int32)
    
    # Upper triangle mask (exclude diagonal and lower triangle)
    upper_mask = np.triu(np.ones((m, m), dtype=np.float32), k=1)
    
    pred_flat = pred_skel_probs[upper_mask == 1].astype(np.float64)
    true_flat = true_skel[upper_mask == 1].astype(np.int32)
    
    return _compute_auroc_from_flat(pred_flat, true_flat)


def compute_metrics_cpdag(pred_cpdag, true_cpdag, pred_probs=None):
    """
    Computes comprehensive CPDAG (skeleton-level) metrics.
    
    Computes both binary classification metrics (SHD, TPR, F1) on the skeleton
    and, if soft predictions are provided, probabilistic metrics (Brier Score, AUROC).
    
    Args:
        pred_cpdag: Predicted CPDAG adjacency matrix [m, m]
        true_cpdag: Ground truth CPDAG adjacency matrix [m, m]
        pred_probs: Optional predicted edge probabilities (soft adjacency) [m, m].
                    Values should be in [0, 1]. Required for Brier Score and AUROC.

    Returns:
        dict with keys:
            - 'SHD': Structural Hamming Distance for CPDAGs (lower is better)
            - 'TPR': Skeleton-level True Positive Rate (higher is better)
            - 'F1': Skeleton-level F1 Score (higher is better)
            - 'Brier': Skeleton-level Brier Score (lower is better). None if pred_probs not provided.
            - 'AUROC': Skeleton-level AUROC (higher is better). None if pred_probs not provided.
    """
    SHD = compute_shd_cpdag(pred_cpdag, true_cpdag)
    TPR = compute_tpr_cpdag(pred_cpdag, true_cpdag)
    F1 = compute_f1_cpdag(pred_cpdag, true_cpdag)
    
    Brier = None
    AUROC = None
    
    if pred_probs is not None:
        Brier = compute_brier_cpdag(pred_probs, true_cpdag)
        AUROC = compute_auroc_cpdag(pred_probs, true_cpdag)
    
    return {
        'SHD': SHD,
        'TPR': TPR,
        'F1': F1,
        'Brier': Brier,
        'AUROC': AUROC
    }


def compute_expected_metrics(A_samples_binary, true_adj, pred_probs=None):
    """
    Computes expected DAG metrics by averaging over posterior samples.
    
    For each binary adjacency sample, computes SHD, TPR, and F1 against the
    ground truth, then averages to get expected values under the posterior.
    AUROC and Brier Score are computed on the soft predictions (posterior mean).
    
    Args:
        A_samples_binary: Array of binary adjacency samples [S, m, m]
        true_adj: Ground truth binary adjacency matrix [m, m]
        pred_probs: Optional posterior mean (soft adjacency) [m, m] for AUROC/Brier.
    
    Returns:
        dict with keys:
            - 'E_SHD': Expected Structural Hamming Distance
            - 'E_TPR': Expected True Positive Rate
            - 'E_F1': Expected F1 Score
            - 'Brier': Brier Score (on pred_probs). None if pred_probs not provided.
            - 'AUROC': AUROC (on pred_probs). None if pred_probs not provided.
    """
    num_samples = A_samples_binary.shape[0]
    true_adj = true_adj.astype(int)
    
    shd_list = []
    tpr_list = []
    f1_list = []
    
    n_true_edges = np.sum(true_adj == 1)
    
    for i in range(num_samples):
        pred_adj = A_samples_binary[i].astype(int)
        
        # Compute TP, FP, FN
        TP = np.sum((pred_adj == 1) & (true_adj == 1))
        FP = np.sum((pred_adj == 1) & (true_adj == 0))
        FN = np.sum((pred_adj == 0) & (true_adj == 1))
        
        # SHD
        shd = int(FP + FN)
        shd_list.append(shd)
        
        # TPR
        tpr = float(TP / n_true_edges) if n_true_edges > 0 else 1.0
        tpr_list.append(tpr)
        
        # F1
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * (precision * recall) / (precision + recall)
        f1_list.append(f1)
    
    # Compute expected values
    E_SHD = float(np.mean(shd_list))
    E_TPR = float(np.mean(tpr_list))
    E_F1 = float(np.mean(f1_list))
    
    # Probabilistic metrics on soft predictions
    Brier = None
    AUROC = None
    
    if pred_probs is not None:
        m = pred_probs.shape[0]
        mask = np.ones((m, m), dtype=np.float32) - np.eye(m, dtype=np.float32)
        
        pred_flat = pred_probs[mask == 1].astype(np.float64)
        true_flat = true_adj[mask == 1].astype(np.float64)
        
        Brier = float(np.mean((pred_flat - true_flat) ** 2))
        AUROC = _compute_auroc_from_flat(pred_flat, true_flat.astype(np.int32))
    
    return {
        'E_SHD': E_SHD,
        'E_TPR': E_TPR,
        'E_F1': E_F1,
        'Brier': Brier,
        'AUROC': AUROC
    }


def _get_unique_dags_and_probs(A_samples_binary):
    """
    Helper to extract unique DAGs and their empirical probabilities from samples.
    """
    num_samples = A_samples_binary.shape[0]
    unique_counts = {}
    
    for i in range(num_samples):
        # Use bytes as hashable key
        key = A_samples_binary[i].tobytes()
        unique_counts[key] = unique_counts.get(key, 0) + 1
        
    unique_dags = []
    probs = []
    
    m = A_samples_binary.shape[1]
    
    for key, count in unique_counts.items():
        dag = np.frombuffer(key, dtype=A_samples_binary.dtype).reshape(m, m)
        unique_dags.append(dag)
        probs.append(count / num_samples)
        
    return unique_dags, np.array(probs)


def compute_posterior_entropy(A_samples_binary):
    """
    Computes the entropy of the posterior distribution over DAGs.
    H(P) = - sum(p * log(p))
    
    Higher entropy indicates higher uncertainty (posterior is more spread out).
    Lower entropy indicates the model is more confident (concentrated on fewer DAGs).
    
    Args:
        A_samples_binary: Array of binary adjacency samples [S, m, m]
        
    Returns:
        Entropy value (float)
    """
    _, probs = _get_unique_dags_and_probs(A_samples_binary)
    # Add epsilon to avoid log(0)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return float(entropy)


def compute_coverage_probability(A_samples_binary, true_adj, k_list=[1, 5, 10]):
    """
    Computes whether the true DAG is within the top-K most probable DAGs (Credible Set).
    
    Args:
        A_samples_binary: Array of binary adjacency samples [S, m, m]
        true_adj: Ground truth binary adjacency matrix [m, m]
        k_list: List of K values to check (e.g., is true DAG in top-1, top-5, top-10?)
        
    Returns:
        dict: {f'top_{k}': 1.0 if true DAG in top-K else 0.0}
        prob_true: Posterior probability of the true DAG
    """
    unique_dags, probs = _get_unique_dags_and_probs(A_samples_binary)
    
    # Sort by probability descending
    sorted_indices = np.argsort(-probs)
    sorted_dags = [unique_dags[i] for i in sorted_indices]
    sorted_probs = probs[sorted_indices]
    
    true_adj = true_adj.astype(A_samples_binary.dtype)
    true_in_top_k = {}
    
    # Find rank of true DAG
    true_rank = -1
    prob_true = 0.0
    
    for i, dag in enumerate(sorted_dags):
        if np.array_equal(dag, true_adj):
            true_rank = i + 1
            prob_true = sorted_probs[i]
            break
            
    for k in k_list:
        # If true DAG was found and its rank is <= k
        if true_rank != -1 and true_rank <= k:
            true_in_top_k[f'top_{k}'] = 1.0
        else:
            true_in_top_k[f'top_{k}'] = 0.0
            
    return true_in_top_k, float(prob_true)


def compute_expected_metrics_cpdag(A_samples_binary, true_cpdag, pred_probs=None):
    """
    Computes expected CPDAG (skeleton-level) metrics by averaging over posterior samples.
    
    For each binary adjacency sample, converts to CPDAG, then computes SHD, TPR, 
    and F1 against the ground truth CPDAG, then averages to get expected values.
    AUROC and Brier Score are computed on the soft predictions (posterior mean).
    
    Args:
        A_samples_binary: Array of binary adjacency samples [S, m, m]
        true_cpdag: Ground truth CPDAG adjacency matrix [m, m]
        pred_probs: Optional posterior mean (soft adjacency) [m, m] for AUROC/Brier.
    
    Returns:
        dict with keys:
            - 'E_SHD': Expected CPDAG Structural Hamming Distance
            - 'E_TPR': Expected CPDAG TPR (skeleton recall)
            - 'E_F1': Expected CPDAG F1 Score (skeleton F1)
            - 'Brier': CPDAG Brier Score (on pred_probs). None if pred_probs not provided.
            - 'AUROC': CPDAG AUROC (on pred_probs). None if pred_probs not provided.
    """
    num_samples = A_samples_binary.shape[0]
    
    shd_list = []
    tpr_list = []
    f1_list = []
    
    for i in range(num_samples):
        pred_adj = A_samples_binary[i].astype(int)
        # Convert DAG to CPDAG
        # Convention: our adjacency is j→i, dag_to_cpdag expects i→j
        pred_cpdag = dag_to_cpdag(pred_adj.T).T
        
        # Compute CPDAG metrics
        shd = compute_shd_cpdag(pred_cpdag, true_cpdag)
        tpr = compute_tpr_cpdag(pred_cpdag, true_cpdag)
        f1 = compute_f1_cpdag(pred_cpdag, true_cpdag)
        
        shd_list.append(shd)
        tpr_list.append(tpr)
        f1_list.append(f1)
    
    # Compute expected values
    E_SHD = float(np.mean(shd_list))
    E_TPR = float(np.mean(tpr_list))
    E_F1 = float(np.mean(f1_list))
    
    # Probabilistic metrics on soft predictions (CPDAG level)
    Brier = None
    AUROC = None
    
    if pred_probs is not None:
        Brier = compute_brier_cpdag(pred_probs, true_cpdag)
        AUROC = compute_auroc_cpdag(pred_probs, true_cpdag)
    
    return {
        'E_SHD': E_SHD,
        'E_TPR': E_TPR,
        'E_F1': E_F1,
        'Brier': Brier,
        'AUROC': AUROC
    }


def _get_unique_cpdags_and_probs(A_samples_binary):
    """
    Helper to extract unique CPDAGs and their empirical probabilities from DAG samples.
    First converts each DAG sample to its CPDAG, then counts uniques.
    """
    num_samples = A_samples_binary.shape[0]
    unique_counts = {}
    
    for i in range(num_samples):
        # Convert DAG -> CPDAG
        # Convention: A_samples_binary is j->i, dag_to_cpdag expects i->j
        # So we transpose before and after
        dag = A_samples_binary[i].astype(int)
        cpdag = dag_to_cpdag(dag.T).T
        cpdag = cpdag.astype(np.int8)
        
        # Use bytes as hashable key
        key = cpdag.tobytes()
        unique_counts[key] = unique_counts.get(key, 0) + 1
        
    unique_cpdags = []
    probs = []
    
    m = A_samples_binary.shape[1]
    
    for key, count in unique_counts.items():
        cpdag = np.frombuffer(key, dtype=np.int8).reshape(m, m)
        unique_cpdags.append(cpdag)
        probs.append(count / num_samples)
        
    return unique_cpdags, np.array(probs)


def compute_posterior_entropy_cpdag(A_samples_binary):
    """
    Computes the entropy of the posterior distribution over Markov Equivalence Classes (CPDAGs).
    
    1. Converts all sampled DAGs to CPDAGs.
    2. Computes empirical distribution over unique CPDAGs.
    3. Calculates entropy H(P_cpdag).
    
    This is often more meaningful than DAG entropy because uncertainty within an MEC 
    is expected/unavoidable, while uncertainty across MECs represents true structural ambiguity.
    
    Args:
        A_samples_binary: Array of binary adjacency samples [S, m, m]
        
    Returns:
        Entropy value (float)
    """
    _, probs = _get_unique_cpdags_and_probs(A_samples_binary)
    # Add epsilon to avoid log(0)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return float(entropy)


def compute_coverage_probability_cpdag(A_samples_binary, true_cpdag, k_list=[1, 5, 10]):
    """
    Computes whether the true CPDAG is within the top-K most probable MECs (Credible Set).
    
    Args:
        A_samples_binary: Array of binary adjacency samples [S, m, m]
        true_cpdag: Ground truth CPDAG adjacency matrix [m, m]
        k_list: List of K values to check
        
    Returns:
        dict: {f'top_{k}': 1.0 if true CPDAG in top-K else 0.0}
        prob_true: Posterior probability of the true CPDAG
    """
    unique_cpdags, probs = _get_unique_cpdags_and_probs(A_samples_binary)
    
    # Sort by probability descending
    sorted_indices = np.argsort(-probs)
    sorted_cpdags = [unique_cpdags[i] for i in sorted_indices]
    sorted_probs = probs[sorted_indices]
    
    true_cpdag = true_cpdag.astype(np.int8)
    true_in_top_k = {}
    
    # Find rank of true CPDAG
    true_rank = -1
    prob_true = 0.0
    
    for i, cpdag in enumerate(sorted_cpdags):
        if np.array_equal(cpdag, true_cpdag):
            true_rank = i + 1
            prob_true = sorted_probs[i]
            break
            
    for k in k_list:
        if true_rank != -1 and true_rank <= k:
            true_in_top_k[f'top_{k}'] = 1.0
        else:
            true_in_top_k[f'top_{k}'] = 0.0
            
    return true_in_top_k, float(prob_true)


def _combine_svgd_terms(attract, repel, repel_scale=1.0):
    """
    Combine the attractive and repulsive halves of the SVGD update.

    ``config.svgd_repulsion_weight`` scales the repulsion; when
    ``config.svgd_repulsion_max_ratio`` is set, the repulsion is additionally
    capped at that multiple of the attraction norm, per particle. The cap
    exists because the median-heuristic bandwidth collapses as the particles
    agree, which makes the repulsion diverge exactly when consensus is forming
    while the attraction stays bounded by the particle-gradient clip. See the
    note in config.py.
    """
    repel = float(config.svgd_repulsion_weight) * repel_scale * repel
    ratio = config.svgd_repulsion_max_ratio
    if ratio is not None:
        a = jnp.linalg.norm(attract, axis=-1, keepdims=True)
        r = jnp.linalg.norm(repel, axis=-1, keepdims=True)
        repel = repel * jnp.minimum(1.0, float(ratio) * a / (r + 1e-12))
    return attract + repel


def compute_svgd_update(particles, grads_logp, kernel_points=None, feature_fn=None,
                        repel_scale=1.0):
    """
    Computes Stein Variational Gradient Descent update direction.
    
    SVGD approximates the gradient of KL divergence using kernel-based
    functional gradient descent. The update balances:
    - Attraction: particles follow gradient of log target (grads_logp)
    - Repulsion: particles spread out via kernel gradient term
    
    Uses RBF kernel with median heuristic for bandwidth selection.

    Args:
        particles: [K, m] array of K particles in m-dimensional space
        grads_logp: [K, m] gradients ∇log p(x) evaluated at each particle
        kernel_points: Optional [K, m] representation used only for the SVGD
            kernel geometry when the kernel is linear in the particle space.
        feature_fn: Optional differentiable feature map g(x). When provided,
            the RBF kernel is computed in feature space:
            k(x_j, x_i) = exp(-||g(x_j) - g(x_i)||² / h).
            The repulsive term then uses the exact chain rule
            ∇_{x_j} k(g(x_j), g(x_i)).

    Returns:
        phi: [K, m] update direction for each particle
    """
    K, m = particles.shape
    if feature_fn is not None:
        features = jax.vmap(feature_fn)(particles)  # [K, d]
        diff = features[:, None, :] - features[None, :, :]  # [K, K, d]
        dists_sq = jnp.sum(diff**2, axis=-1)

        med_sq = jnp.median(dists_sq)
        h = med_sq / jnp.log(K + 1.0)
        h = jnp.maximum(h, 1e-6)

        k_mat = jnp.exp(-dists_sq / h)  # [K, K]
        sum_score = jnp.dot(k_mat.T, grads_logp)  # [K, m]

        # Forward mode: the feature map g: R^m -> R^d is tall (m=25 -> d=625
        # for the M_tau feature), so jacfwd needs m JVPs where jacrev needs d
        # VJPs -- a d/m = 25x reduction in passes through the 300-step Sinkhorn
        # fori_loop, and no reverse-mode tape over those 300 states. The
        # Jacobian returned is identical.
        jacobians = jax.vmap(jax.jacfwd(feature_fn))(particles)  # [K, d, m]
        projected_diff = jnp.einsum("kdm,kid->kim", jacobians, diff)
        grad_k = -2.0 / h * k_mat[..., None] * projected_diff  # [K, K, m]
        sum_grad_k = jnp.sum(grad_k, axis=0)  # [K, m]
        return _combine_svgd_terms(sum_score / K, sum_grad_k / K, repel_scale)

    if kernel_points is None:
        kernel_points = particles

    # Compute pairwise squared distances: ||x_j - x_i||²
    diff = kernel_points[:, None, :] - kernel_points[None, :, :]  # [K, K, m]
    dists_sq = jnp.sum(diff**2, axis=-1)                  # [K, K]

    # Median heuristic: h = median(distances²) / log(K+1)
    # Adapts bandwidth to particle spread; +1 prevents log(1)=0
    med_sq = jnp.median(dists_sq)
    h = med_sq / jnp.log(K + 1.0)
    h = jnp.maximum(h, 1e-6)  # Prevent division by zero

    # RBF kernel: k(x_j, x_i) = exp(-||x_j - x_i||² / h)
    k_mat = jnp.exp(-dists_sq / h)  # [K, K]

    # Kernel gradient: ∇_{x_j} k(x_j, x_i) = -2/h * (x_j - x_i) * k(x_j, x_i)
    grad_k = -2.0 / h * diff * k_mat[..., None]  # [K, K, m]

    # Repulsive term: encourages particle diversity
    sum_grad_k = jnp.sum(grad_k, axis=0)         # [K, m]

    # Attractive term: kernel-weighted average of score functions
    sum_score = jnp.dot(k_mat.T, grads_logp)     # [K, m]

    # SVGD update: Φ(x_i) = (1/K) Σ_j [k(x_j, x_i) ∇log p(x_j) + ∇_{x_j} k(x_j, x_i)]
    return _combine_svgd_terms(sum_score / K, sum_grad_k / K, repel_scale)
