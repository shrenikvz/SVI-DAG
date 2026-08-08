'''
Data module for the SVIDAG.

Author: Shrenik Zinage
'''

from dataclasses import dataclass

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import jax.numpy as jnp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pandas as pd
import networkx as nx
try:
    from cdt.data import load_dataset
except ImportError:
    # CDT is optional; only needed for Sachs dataset
    load_dataset = None

from . import config
from .utils import to_device

#: Repository root, derived from this file's location (src/svidag/data.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The tab-separated Sachs file shipped with this repository: the 853-row,
#: purely *observational* subset that is the usual structure-learning
#: benchmark.  Resolved absolutely so it is reachable from any working
#: directory.
#:
#: NOTE: this is NOT the default.  ``load_sachs_dataset()`` defaults to the
#: 7,466-row full dataset bundled with cdt (all perturbation conditions
#: pooled), which is what every published result in
#: ``paper_results_reproduce/case_4`` was produced on -- see that directory's
#: README ("~7,466 samples").  The two sources differ in both row count and
#: column naming, so they are not interchangeable; pass this path explicitly
#: if you specifically want the observational subset.
OBSERVATIONAL_SACHS_PATH = _REPO_ROOT / "data" / "sachs" / "sachs.data.txt"


def _cdt_load_sachs_direct():
    """
    Return ``(dataframe, nx.DiGraph)`` exactly like ``cdt.data.load_dataset('sachs')``,
    by reading the resource CSVs that ship inside the installed cdt package.

    cdt's top-level import does ``from torch.utils.data import Dataset``
    unconditionally, so ``import cdt`` fails outright in a torch-free
    environment -- and torch is deliberately not part of this project's pinned
    package set (see requirements-direct.txt).  Reading the bundled resources
    directly keeps the Sachs ground truth available without pulling in a
    ~300 MB wheel that nothing else here uses.
    """
    import importlib.util as _ilu

    spec = _ilu.find_spec("cdt")
    if spec is None or spec.submodule_search_locations is None:
        raise ImportError(
            "cdt package not found on sys.path. Install the pinned package set "
            "with 'pip install -r requirements.txt'."
        )
    cdt_dir = Path(list(spec.submodule_search_locations)[0])
    data_csv = cdt_dir / "data" / "resources" / "cyto_full_data.csv"
    target_csv = cdt_dir / "data" / "resources" / "cyto_full_target.csv"
    df = pd.read_csv(data_csv)
    edges = pd.read_csv(target_csv)  # columns: Cause, Effect
    g = nx.DiGraph()
    g.add_nodes_from(df.columns.tolist())
    for _, row in edges.iterrows():
        g.add_edge(row["Cause"], row["Effect"])
    return df, g


def _load_sachs_reference():
    """Sachs measurements + consensus ground-truth graph, via cdt or its fallback."""
    if load_dataset is not None:
        return load_dataset("sachs")
    return _cdt_load_sachs_direct()


class IdentityScaler(BaseEstimator, TransformerMixin):
    """
    No-op scaler that implements sklearn transformer interface.
    Used when data should not be normalized (e.g., already standardized).
    """
    def __init__(self):
        self.scale_ = 1.0
        self.mean_ = 0.0

    def fit(self, X, y=None):
        # Store shape-compatible attributes for API consistency
        self.scale_ = np.ones(X.shape[1])
        self.mean_ = np.zeros(X.shape[1])
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X


@dataclass
class Dataset:
    """
    Bundles all data needed for SVIDAG training and evaluation.
    
    Convention: Adjacency A[i,j] = 1 means edge j → i (column causes row).
    """
    train_data: jnp.ndarray        # Scaled training data [N_train, m] on device
    test_data_scaled: jnp.ndarray  # Scaled test data [N_test, m] on device
    dataset_size: int              # N_train (for likelihood scaling)
    noise_scales: jnp.ndarray      # Per-node observation noise σ [m]
    obs_train_orig: np.ndarray     # Original (unscaled) training data
    obs_test_orig: np.ndarray      # Original (unscaled) test data
    scaler: Any                    # Fitted scaler for inverse transform
    true_adj_np: np.ndarray        # Ground truth adjacency matrix
    true_DAG: jnp.ndarray          # Ground truth as JAX array
    node_names: List[str]          # Human-readable node names
    num_nodes: int                 # Number of nodes m
    dataset_name: str = "dataset"  # Short identifier used for plot subdirectory naming


def complex_func1(x):
    """
    Nonlinear function for synthetic data: combines Gaussian envelope,
    polynomial oscillations, and periodic terms. Used to create 
    challenging nonlinear causal relationships.
    """
    return (
        np.exp(-x**2) * np.sin(5 * x**3)
        + np.abs(x)**0.5 * np.cos(4 * x)
        + np.sin(3 * np.pi * x)
    )

def complex_func2(x):
    """
    Second nonlinear function with different frequency/decay profile.
    Provides distinct nonlinearity for multi-hop causal chains.
    """
    return (
        np.exp(-x) * np.sin(3 * x**3)
        + np.abs(x)**2.5 * np.cos(2 * x)
        + np.sin(4 * np.pi * x)
    )



def two_node_generator_default():
    """
    Default 2-node synthetic data generator.

    Generates data from a simple nonlinear SCM:
        x1 ~ N(0, 1),  x2 = complex_func1(x1) + ε,  ε ~ N(0, 0.1²)

    Ground truth: x1 → x2  (A[1,0] = 1 in j→i convention).

    Returns:
        obs_data:    [N, 2] float32 observations
        true_adj_np: [2, 2] float32 adjacency matrix
        node_names:  ["x1", "x2"]
    """
    rng = np.random.default_rng(0)
    N = 1000
    x1 = rng.normal(loc=0.0, scale=1.0, size=N).astype(np.float32)
    x2 = (complex_func1(x1) + rng.normal(loc=0.0, scale=0.1, size=N)).astype(np.float32)
    obs_data = np.stack([x1, x2], axis=1)
    true_adj_np = np.zeros((2, 2), dtype=np.float32)
    true_adj_np[1, 0] = 1.0  # x1 → x2
    return obs_data, true_adj_np, ["x1", "x2"]


def generate_synthetic_dataset(generator_fn, node_names=None, normalize=False) -> Dataset:
    """
    Create a Dataset from a generator function.

    Handles:
    - Data generation via the provided generator
    - Train/test split
    - Optional standard scaling (fit on train, apply to test)
    - Moving data to compute device

    Args:
        generator_fn: Callable returning (obs_data [N,m], true_adj [m,m], names)
                     or (obs_data [N,m], true_adj [m,m], names, noise_stds [m])
        node_names: Override node names if generator doesn't provide them
        normalize: If False (default), use IdentityScaler and optimize on raw data scale.
                   If True, apply StandardScaler (zero mean, unit variance per feature).

    Returns:
        Dataset with all fields populated
    """
    # Generator: expect either:
    #   (obs_data, adj_matrix, optional_names)
    #   (obs_data, adj_matrix, optional_names, noise_stds)
    ret = generator_fn()
    if len(ret) == 4:
        obs_data, true_adj_np, maybe_names, generator_noise_stds = ret
    elif len(ret) == 3:
        obs_data, true_adj_np, maybe_names = ret
        generator_noise_stds = None
    else:
        raise ValueError(
            "Generator must return (obs_data, true_adj, names) "
            "or (obs_data, true_adj, names, noise_stds)."
        )
    # Fall back to generic names if not provided
    node_names = node_names or maybe_names or [f"x{i+1}" for i in range(obs_data.shape[1])]
    # Use the generator function's name as the dataset identifier
    dataset_name = getattr(generator_fn, "__name__", "custom")

    num_nodes = obs_data.shape[1]
    true_adj_np = true_adj_np.astype(np.float32, copy=False)
    true_DAG = jnp.array(true_adj_np)

    # Split: last n_test_samples for test, rest for train
    obs_train_orig = obs_data[: -config.n_test_samples]
    obs_test_orig = obs_data[-config.n_test_samples :]
    
    # Optionally apply StandardScaler (zero mean, unit variance); use IdentityScaler to keep raw scale
    if normalize:
        scaler = StandardScaler()
    else:
        scaler = IdentityScaler()
    obs_train_scaled = scaler.fit_transform(obs_train_orig)
    obs_test_scaled = scaler.transform(obs_test_orig)
    
    # Move to GPU/TPU if available
    train_data = to_device(jnp.array(obs_train_scaled, dtype=jnp.float32))
    test_data_scaled = to_device(jnp.array(obs_test_scaled, dtype=jnp.float32))
    dataset_size = train_data.shape[0]
    
    # Fixed observation noise for the likelihood, taken directly from config.
    # This is the same uniform value for every node, regardless of normalization
    # or the generator's intrinsic noise stds.  Adjust config.obs_noise_scale to
    # change the noise level without touching data generation code.
    noise_scale_base = np.ones((num_nodes,), dtype=np.float32) * config.obs_noise_scale

    # Keep a record of the per-node normalized noise scales for downstream use.
    config.obs_noise_scales = noise_scale_base.copy()
    noise_scales = to_device(jnp.array(noise_scale_base, dtype=jnp.float32))

    return Dataset(
        train_data=train_data,
        test_data_scaled=test_data_scaled,
        dataset_size=dataset_size,
        noise_scales=noise_scales,
        obs_train_orig=obs_train_orig,
        obs_test_orig=obs_test_orig,
        scaler=scaler,
        true_adj_np=true_adj_np,
        true_DAG=true_DAG,
        node_names=node_names,
        num_nodes=num_nodes,
        dataset_name=dataset_name,
    )


# =============================================================================
# Benchmark Graph Generation (Erdos-Renyi / Scale-Free)
# =============================================================================

def random_dag(num_nodes: int, graph_type: str = 'er', edge_prob: float = 0.3,
               sf_num_edges: int = 2, rng_seed: int = 0) -> np.ndarray:
    """
    Generate a random DAG using Erdos-Renyi or Scale-Free (Barabasi-Albert) model.

    Creates an undirected random graph via networkx, then converts to a DAG
    by imposing a random topological ordering (edges go from lower to higher
    order in the permutation).

    Args:
        num_nodes: Number of nodes m in the graph.
        graph_type: Graph model to use:
            'er' - Erdos-Renyi: each edge exists independently with probability p.
                   Expected number of edges ≈ m*(m-1)/2 * p.
            'sf' - Scale-Free (Barabasi-Albert): preferential attachment model.
                   Each new node connects to k existing nodes. Produces power-law
                   degree distribution. Total edges ≈ k * (m - k).
        edge_prob: Edge probability p (used only for ER graphs).
        sf_num_edges: Number of edges k from each new node (used only for SF graphs).
                      Must satisfy 1 ≤ k < num_nodes.
        rng_seed: Random seed for reproducibility.

    Returns:
        adj: [m, m] adjacency matrix with convention A[i,j]=1 means j→i.

    Examples:
        >>> adj_er = random_dag(10, graph_type='er', edge_prob=0.3)
        >>> adj_sf = random_dag(10, graph_type='sf', sf_num_edges=2)
    """
    if graph_type == 'er':
        G_undirected = nx.erdos_renyi_graph(num_nodes, edge_prob, seed=rng_seed)
    elif graph_type == 'sf':
        if sf_num_edges >= num_nodes:
            raise ValueError(
                f"sf_num_edges ({sf_num_edges}) must be < num_nodes ({num_nodes})"
            )
        G_undirected = nx.barabasi_albert_graph(num_nodes, sf_num_edges, seed=rng_seed)
    else:
        raise ValueError(
            f"Unknown graph_type: '{graph_type}'. Use 'er' or 'sf'."
        )

    # Impose random topological ordering to create DAG
    rng = np.random.default_rng(rng_seed)
    perm = list(rng.permutation(num_nodes))
    order = {node: i for i, node in enumerate(perm)}

    G_dag = nx.DiGraph()
    G_dag.add_nodes_from(range(num_nodes))
    for u, v in G_undirected.edges():
        if order[u] < order[v]:
            G_dag.add_edge(u, v)  # u → v
        else:
            G_dag.add_edge(v, u)  # v → u

    # NetworkX convention: adj_nx[i,j]=1 means i→j
    # Our convention: A[i,j]=1 means j→i, so transpose
    adj_nx = nx.to_numpy_array(G_dag, dtype=np.float32)
    return adj_nx.T


def simulate_sem(adj: np.ndarray, num_samples: int, sem_type: str = 'linear',
                 noise_scale: float = 1.0, weight_range: Tuple[float, float] = (0.5, 2.0),
                 rng_seed: int = 0,
                 noise_type: str = 'gaussian') -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Simulate observational data from a Structural Equation Model (SEM) on a DAG.

    Generates data by iterating through nodes in topological order, computing
    each node's value from its parents using the specified functional form,
    plus additive noise drawn from the distribution selected by ``noise_type``.

    Args:
        adj: [m, m] DAG adjacency matrix. A[i,j]=1 means j→i (column causes row).
        num_samples: Number of i.i.d. observational samples to generate.
        sem_type: Type of structural equations:
            'linear' - x_i = Σ_j w_ij * x_j + ε_i
                       Edge weights sampled from Uniform([-w_max, -w_min] ∪ [w_min, w_max]).
            'mlp' or 'nonlinear' - x_i = MLP(x_parents(i)) + ε_i
                       Random 1-hidden-layer MLP with ReLU activation and 10
                       hidden units (paper spec).  All MLP weights are drawn
                       from Uniform([-w_max, -w_min] ∪ [w_min, w_max]) -- the
                       same ``weight_range`` used by the linear SEM -- and
                       hidden-layer weights of each node's non-parents are
                       implicitly zero (only ``parents`` index ``W1``).
            'mim' - x_i = tanh(x_parents(i) · w + b) + ε_i
                    Multiplicative interaction model with random weights.
        noise_scale: Reference noise standard deviation s (see ``noise_type``).
        weight_range: (w_min, w_max) tuple for linear SEM edge weight bounds.
                      Weights are sampled with random sign to avoid bias.
        rng_seed: Random seed for reproducibility.
        noise_type: Distribution of the additive noise ε (all zero-mean):
            'gaussian' (default) - ε ~ N(0, s²).
            'hetero_gaussian' - across-node heteroscedastic Gaussian:
                per-node scales σ_i ~ Uniform(0.5, 2.0) * s are drawn once
                per dataset, then ε_i ~ N(0, σ_i²).
            'gumbel' - centered Gumbel with Var = s²:
                ε = Gumbel(0, β) − β·γ_E with β = s·√6/π.
            'exponential' - centered exponential with std = s:
                ε = Exp(scale=s) − s.

    Returns:
        data: [num_samples, m] observation matrix as float32.
        W: [m, m] weighted adjacency matrix (for linear SEM only; None for nonlinear).
    """
    rng = np.random.default_rng(rng_seed)
    m = adj.shape[0]

    # Validate DAG structure using networkx
    # adj.T converts our j→i convention to networkx i→j
    G = nx.DiGraph(adj.T)
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Input adjacency matrix does not represent a DAG.")
    topo_order = list(nx.topological_sort(G))

    # Per-node additive-noise sampler shared by every SEM type. For the
    # default 'gaussian' the underlying RNG stream is identical to the
    # historical rng.normal(0, noise_scale, ...) calls, so pre-existing
    # benchmark draws (cases 2/3) are unchanged.
    if noise_type == 'gaussian':
        def _noise(node):
            return rng.normal(0, noise_scale, size=num_samples).astype(np.float32)
    elif noise_type == 'hetero_gaussian':
        node_scales = (rng.uniform(0.5, 2.0, size=m) * noise_scale).astype(np.float32)

        def _noise(node):
            return rng.normal(0, node_scales[node], size=num_samples).astype(np.float32)
    elif noise_type == 'gumbel':
        gumbel_beta = noise_scale * np.sqrt(6.0) / np.pi

        def _noise(node):
            eps = rng.gumbel(0.0, gumbel_beta, size=num_samples)
            return (eps - gumbel_beta * np.euler_gamma).astype(np.float32)
    elif noise_type == 'exponential':
        def _noise(node):
            eps = rng.exponential(noise_scale, size=num_samples)
            return (eps - noise_scale).astype(np.float32)
    else:
        raise ValueError(
            f"Unknown noise_type: '{noise_type}'. Options: 'gaussian', "
            f"'hetero_gaussian', 'gumbel', 'exponential'."
        )

    data = np.zeros((num_samples, m), dtype=np.float32)
    W = None

    if sem_type == 'linear':
        w_min, w_max = weight_range

        # Build weighted adjacency: W[i,j] = weight of edge j→i
        W = np.zeros((m, m), dtype=np.float32)
        edge_rows, edge_cols = np.where(adj != 0)
        n_edges = len(edge_rows)
        if n_edges > 0:
            weights = rng.uniform(w_min, w_max, size=n_edges).astype(np.float32)
            signs = rng.choice(
                np.array([-1.0, 1.0], dtype=np.float32), size=n_edges
            )
            W[edge_rows, edge_cols] = weights * signs

        for node in topo_order:
            parents = np.where(adj[node, :] != 0)[0]
            if len(parents) > 0:
                # x_i = Σ_j W[i,j] * x_j
                data[:, node] = data[:, parents] @ W[node, parents]
            data[:, node] += _noise(node)

    elif sem_type in ('mlp', 'nonlinear'):
        # ----------------------------------------------------------------
        # Paper-spec nonlinear SEM:
        #   * single hidden layer of 10 neurons
        #   * ReLU activation
        #   * weights drawn from U([-w_max, -w_min] U [w_min, w_max]),
        #     i.e. magnitude ~ Uniform(w_min, w_max) with a random sign,
        #     using the same ``weight_range`` that controls the linear SEM
        #     (default (0.3, 0.7) per the paper)
        #   * hidden weights of node k's non-parents are zero -- enforced
        #     here by indexing ``W1`` only over actual parents
        # ----------------------------------------------------------------
        hidden_dim = 10
        w_min, w_max = weight_range

        def _signed_uniform(shape):
            mag = rng.uniform(w_min, w_max, size=shape).astype(np.float32)
            sign = rng.choice(
                np.array([-1.0, 1.0], dtype=np.float32), size=shape
            )
            return mag * sign

        for node in topo_order:
            parents = np.where(adj[node, :] != 0)[0]
            if len(parents) == 0:
                # Root node: sample from noise distribution
                data[:, node] = _noise(node)
            else:
                n_pa = len(parents)
                x_pa = data[:, parents]  # [N, n_pa]

                # Random 1-hidden-layer MLP: ReLU(X @ W1 + b1) @ W2
                W1 = _signed_uniform((n_pa, hidden_dim))
                b1 = _signed_uniform((hidden_dim,))
                W2 = _signed_uniform((hidden_dim,))

                pre = x_pa @ W1 + b1
                hidden = np.maximum(pre, 0.0)  # ReLU
                signal = hidden @ W2
                data[:, node] = signal + _noise(node)

    elif sem_type == 'mim':
        for node in topo_order:
            parents = np.where(adj[node, :] != 0)[0]
            if len(parents) == 0:
                data[:, node] = _noise(node)
            else:
                n_pa = len(parents)
                x_pa = data[:, parents]  # [N, n_pa]

                # Multiplicative interaction: tanh(X @ w + b)
                w = rng.standard_normal(n_pa).astype(np.float32)
                b = rng.standard_normal(1).astype(np.float32)
                signal = np.tanh(x_pa @ w + b)
                data[:, node] = signal + _noise(node)
    else:
        raise ValueError(
            f"Unknown sem_type: '{sem_type}'. "
            f"Options: 'linear', 'mlp'/'nonlinear', 'mim'."
        )

    return data, W


def generate_benchmark_dataset(
    num_nodes: int = 10,
    graph_type: str = 'er',
    sem_type: str = 'linear',
    num_samples: int = 1000,
    edge_prob: float = 0.3,
    sf_num_edges: int = 2,
    noise_scale: float = 1.0,
    weight_range: Tuple[float, float] = (0.5, 2.0),
    rng_seed: int = 0,
) -> Dataset:
    """
    Generate a complete benchmark dataset from a random DAG.

    Combines random DAG generation (ER or Scale-Free), SEM data simulation
    (linear or nonlinear), train/test splitting, standard scaling, and device
    placement into a single Dataset object ready for SVIDAG training.

    Args:
        num_nodes: Number of nodes m in the DAG.
        graph_type: 'er' (Erdos-Renyi) or 'sf' (Scale-Free / Barabasi-Albert).
        sem_type: Structural equation type:
            'linear' - Linear SEM with random edge weights.
            'mlp' or 'nonlinear' - Nonlinear SEM using random MLPs.
            'mim' - Multiplicative interaction model.
        num_samples: Total number of samples (will be split into train + test).
        edge_prob: Edge probability p for ER graphs.
        sf_num_edges: Edges per new node k for SF graphs.
        noise_scale: Noise standard deviation s for the SEM.
        weight_range: (w_min, w_max) for linear SEM edge weight bounds.
        rng_seed: Random seed for full reproducibility (graph + data).

    Returns:
        Dataset object with all fields populated for SVIDAG training.

    Examples:
        # 10-node linear ER graph with edge probability 0.3
        >>> ds = generate_benchmark_dataset(
        ...     num_nodes=10, graph_type='er', sem_type='linear',
        ...     num_samples=1000, edge_prob=0.3, noise_scale=1.0
        ... )

        # 20-node nonlinear Scale-Free graph
        >>> ds = generate_benchmark_dataset(
        ...     num_nodes=20, graph_type='sf', sem_type='nonlinear',
        ...     num_samples=2000, sf_num_edges=2, noise_scale=0.5
        ... )
    """
    # Step 1: Generate random DAG
    adj = random_dag(num_nodes, graph_type, edge_prob, sf_num_edges, rng_seed)

    # Step 2: Simulate data from SEM (use different seed for data vs graph)
    obs_data, W = simulate_sem(
        adj, num_samples, sem_type, noise_scale, weight_range, rng_seed + 1
    )

    # Log graph statistics
    n_edges = int(np.sum(adj))
    print(f"Generated {graph_type.upper()} DAG: {num_nodes} nodes, {n_edges} edges")
    print(f"SEM type: {sem_type}, noise scale: {noise_scale}, samples: {num_samples}")
    if W is not None:
        print(f"Edge weight range: [{W[W != 0].min():.3f}, {W[W != 0].max():.3f}]"
              if n_edges > 0 else "No edges in graph.")

    # Step 3: Build Dataset
    true_adj_np = adj.astype(np.float32)
    true_DAG = jnp.array(true_adj_np)
    node_names = [f"x{i + 1}" for i in range(num_nodes)]
    # e.g. "er_linear_10nodes" or "sf_nonlinear_20nodes"
    dataset_name = f"{graph_type}_{sem_type}_{num_nodes}nodes"

    # Train/test split
    if num_samples <= config.n_test_samples:
        raise ValueError(
            f"num_samples ({num_samples}) must be > n_test_samples ({config.n_test_samples})"
        )
    obs_train_orig = obs_data[: -config.n_test_samples]
    obs_test_orig = obs_data[-config.n_test_samples:]

    # Standard scaling (fit on train, apply to test)
    scaler = StandardScaler()
    obs_train_scaled = scaler.fit_transform(obs_train_orig)
    obs_test_scaled = scaler.transform(obs_test_orig)

    # Move to compute device (GPU if available)
    train_data = to_device(jnp.array(obs_train_scaled, dtype=jnp.float32))
    test_data_scaled = to_device(jnp.array(obs_test_scaled, dtype=jnp.float32))
    dataset_size = train_data.shape[0]
    noise_scales_device = to_device(
        jnp.ones((num_nodes,), dtype=jnp.float32) * config.obs_noise_scale
    )

    return Dataset(
        train_data=train_data,
        test_data_scaled=test_data_scaled,
        dataset_size=dataset_size,
        noise_scales=noise_scales_device,
        obs_train_orig=obs_train_orig,
        obs_test_orig=obs_test_orig,
        scaler=scaler,
        true_adj_np=true_adj_np,
        true_DAG=true_DAG,
        node_names=node_names,
        num_nodes=num_nodes,
        dataset_name=dataset_name,
    )


def load_sachs_dataset(data_path: Union[str, Path, None] = None) -> Dataset:
    """
    Load the Sachs protein signaling dataset.

    The Sachs dataset is a classic benchmark for causal discovery, containing
    flow cytometry measurements of 11 phosphoproteins/phospholipids in human
    immune cells under various perturbation conditions.

    Ground truth: Consensus network from the CDT package.

    Args:
        data_path: Path to a tab-separated Sachs data file.  ``None`` (the
            default) uses the 7,466-row full dataset bundled with cdt, which
            is what the case-4 benchmark and every published number were
            produced on.  Pass ``OBSERVATIONAL_SACHS_PATH`` to use the 853-row
            observational subset shipped in ``data/sachs/`` instead -- note
            that this changes the results, it is not merely a different route
            to the same data.

    Returns:
        Dataset with 11 nodes (proteins)
    """
    # Load observational measurements.
    if data_path is None:
        data_sachs_df, _ = _load_sachs_reference()
        data_sachs_df.dropna(inplace=True)
        obs_data = data_sachs_df.to_numpy(dtype=np.float32)
    else:
        data_sachs_table = pd.read_csv(data_path, sep="\t")
        obs_data = data_sachs_table.to_numpy(dtype=np.float32)

    # Load ground truth network from CDT (consensus from literature)
    data_sachs_df_cdt, graph_sachs = _load_sachs_reference()
    data_sachs_df_cdt.dropna(inplace=True)
    node_names = data_sachs_df_cdt.columns.tolist()
    # Transpose: NetworkX default is A[i,j]=1 means i→j, we use j→i
    true_adj_np = nx.to_numpy_array(graph_sachs, dtype=np.float32).T
    true_DAG = jnp.array(true_adj_np)
    num_nodes = true_DAG.shape[0]

    assert obs_data.shape[1] == num_nodes, f"Data/node count mismatch: {obs_data.shape[1]} vs {num_nodes}"

    # Split and scale
    obs_train_orig = obs_data[: -config.n_test_samples]
    obs_test_orig = obs_data[-config.n_test_samples :]

    # Debug output for verification
    print(f"obs_train_orig.shape: {obs_train_orig.shape}")
    print(f"obs_test_orig.shape: {obs_test_orig.shape}")
    print(f"obs_train_orig: {obs_train_orig}")
    print(f"obs_test_orig: {obs_test_orig}")
    print(f"true_adj_np: {true_adj_np}")
    print(f"true_DAG: {true_DAG}")
    print(f"node_names: {node_names}")
    print(f"num_nodes: {num_nodes}")
    
    scaler = StandardScaler()
    obs_train_scaled = scaler.fit_transform(obs_train_orig)
    obs_test_scaled = scaler.transform(obs_test_orig)

    train_data = to_device(jnp.array(obs_train_scaled, dtype=jnp.float32))
    test_data_scaled = to_device(jnp.array(obs_test_scaled, dtype=jnp.float32))
    dataset_size = train_data.shape[0]
    
    noise_scales = to_device(jnp.ones((num_nodes,), dtype=jnp.float32) * config.obs_noise_scale)

    return Dataset(
        train_data=train_data,
        test_data_scaled=test_data_scaled,
        dataset_size=dataset_size,
        noise_scales=noise_scales,
        obs_train_orig=obs_train_orig,
        obs_test_orig=obs_test_orig,
        scaler=scaler,
        true_adj_np=true_adj_np,
        true_DAG=true_DAG,
        node_names=node_names,
        num_nodes=num_nodes,
        dataset_name="sachs",
    )
