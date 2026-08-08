'''
Main entry point for running SVIDAG experiments.

This script provides a convenient interface to:
1. Select which prior scenarios to evaluate (domain-informed vs uninformative)
2. Choose between synthetic datasets or real-world data (e.g., Sachs)
3. Define custom data generators for experimentation

Usage:
    python main.py

Configuration is done via the CUSTOM_SCENARIOS and CUSTOM_DATA_BUILDER variables below.
Results and plots are saved to config.plot_dir (default: "plots_svidag/").
'''

import sys
from pathlib import Path
import numpy as np

# Add src/ to Python path for direct script execution (no pip install needed)
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from svidag.runner import run_all_scenarios
from svidag.data import (
    generate_synthetic_dataset,
    generate_benchmark_dataset,
    load_sachs_dataset,
    Dataset,
)

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
# Select which prior scenarios to evaluate. Options:
#   "noninformative", "strong_correct", "strong_incorrect"
# Set to None to run all scenarios defined in config.scenarios
CUSTOM_SCENARIOS = ["noninformative", "strong_correct", "strong_incorrect"] 

# Dataset configuration: provide a builder function
# CUSTOM_DATA_BUILDER = lambda: generate_synthetic_dataset(generator_fn=two_node_generator_nonlinear)

# === Sachs Dataset Example ===
# The Sachs protein signaling network (11 nodes) - a standard benchmark.
# Defaults to the 7,466-row full dataset bundled with cdt (the case-4 setting);
# pass svidag.data.OBSERVATIONAL_SACHS_PATH for the 853-row observational
# subset shipped in data/sachs/.
# CUSTOM_DATA_BUILDER = lambda: load_sachs_dataset()

# =============================================================================
# EXAMPLE CUSTOM DATA GENERATORS
# =============================================================================
# These functions demonstrate how to create custom synthetic datasets.
# Each generator returns either:
#   (obs_data [N, m], true_adj [m, m], node_names [m])
#   (obs_data [N, m], true_adj [m, m], node_names [m], noise_stds [m])

def two_node_generator_linear():
    """
    Simple linear 2-node SCM: x2 = a*x1 + noise
    
    Useful as a minimal sanity check for structure learning.
    Ground truth: x1 → x2
    """
    rng = np.random.default_rng(0)
    N = 1000
    a = 1.0  # Causal coefficient
    std_x1 = 1.0
    std_x2 = 1.0
    noise_stds = np.array([std_x1, std_x2], dtype=np.float32)
    x1 = rng.normal(loc=0.0, scale=std_x1, size=(N,))
    x2 = a * x1 + rng.normal(loc=0.0, scale=std_x2, size=(N,))
    obs_data = np.stack([x1, x2], axis=1).astype(np.float32)
    # Adjacency convention: A[i,j]=1 means j→i
    true_adj_np = np.zeros((2, 2), dtype=np.float32)
    true_adj_np[1, 0] = 1.0  # x1 -> x2 in (i,j) = j->i convention
    node_names = ["x1", "x2"]
    return obs_data, true_adj_np, node_names, noise_stds


def three_node_generator_linear():
    """
    Linear causal chain: x1 → x2 → x3
    
    Tests recovery of chain structure.
    """
    rng = np.random.default_rng(0)
    N = 1000
    a1 = 0.4  # x1 → x2 coefficient
    a2 = 0.8  # x2 → x3 coefficient
    std_x1 = 0.1
    std_x2 = 0.1
    std_x3 = 0.1
    noise_stds = np.array([std_x1, std_x2, std_x3], dtype=np.float32)
    # Generate x1 as AR(1) for more realistic temporal structure
    # phi1 = 0.8  # AR coefficient
    # x1 = np.zeros(N, dtype=np.float32)
    # x1[0] = rng.normal(loc=0.0, scale=std_x1)
    # for t in range(1, N):
    #     x1[t] = phi1 * x1[t - 1] + rng.normal(loc=0.0, scale=std_x1)
    x1 = rng.normal(loc=0.0, scale=std_x1, size=(N,))
    x2 = a1 * x1 + rng.normal(loc=0.0, scale=std_x2, size=(N,))
    x3 = a2 * x2 + rng.normal(loc=0.0, scale=std_x3, size=(N,))
    obs = np.stack([x1, x2, x3], axis=1).astype(np.float32)
    # Chain structure: x1→x2, x2→x3
    true_adj = np.array([[0, 0, 0],
                         [1, 0, 0],
                         [0, 1, 0]], dtype=np.float32)
    node_names = ["x1", "x2", "x3"]
    return obs, true_adj, node_names, noise_stds


def complex_func1(x):
    """Nonlinear function combining oscillations, exponential decay, and cusps."""
    return (
        np.exp(-x**2) * np.sin(5 * x**3)
        # + np.abs(x)**0.5 * np.cos(4 * x)
        + np.sin(3 * np.pi * x)
    )


def complex_func2(x):
    """Second nonlinear function with different characteristics for varied nonlinearity."""
    return (
        np.exp(-x) * np.sin(3 * x**3)
        # + np.abs(x)**2.5 * np.cos(2 * x)
        + np.sin(4 * np.pi * x)
    )


def two_node_generator_nonlinear():
    """
    2-node synthetic data generator with nonlinear AR(1) structure.

    Generates data from: x1 ~ AR(1), x2 = f(x1) + ε
    where f is a complex nonlinear function and AR(1) provides
    temporal correlation to test robustness beyond i.i.d. data.
    Ground truth: x1 → x2
    """
    rng = np.random.default_rng(123)
    N = 1000
    std_x1 = 1.0
    std_x2 = 0.1
    phi = 0.8  # AR coefficient
    # x1 = np.zeros(N, dtype=np.float32)
    eps = rng.normal(loc=0.0, scale=std_x1, size=(N,))
    # x1[0] = eps[0]
    x1 = eps
    # for t in range(1, N):
    #     x1[t] = phi * x1[t-1] + eps[t]
    x2 = complex_func1(x1) + rng.normal(loc=0.0, scale=std_x2, size=(N,))
    obs_data = np.stack([x1, x2], axis=1).astype(np.float32)
    true_adj_np = np.zeros((2, 2), dtype=np.float32)
    true_adj_np[1, 0] = 1.0  # x1 → x2
    node_names = ["x1", "x2"]
    return obs_data, true_adj_np, node_names


def three_node_generator_nonlinear():
    """
    Nonlinear causal chain: x1 → x2 → x3 with complex nonlinear functions.
    
    Tests the model's ability to capture nonlinear causal relationships.
    More challenging than linear case.
    """
    rng = np.random.default_rng(0)
    N = 1000
    std_x1 = 0.1
    std_x2 = 0.1  # Increased noise
    std_x3 = 0.1
    noise_stds = np.array([std_x1, std_x2, std_x3], dtype=np.float32)
    x1 = rng.normal(loc=0.0, scale=std_x1, size=(N,))
    x2 = complex_func1(x1) + rng.normal(loc=0.0, scale=std_x2, size=(N,))
    x3 = complex_func2(x2) + rng.normal(loc=0.0, scale=std_x3, size=(N,))
    obs = np.stack([x1, x2, x3], axis=1).astype(np.float32)
    true_adj = np.array([[0, 0, 0],
                         [1, 0, 0],
                         [0, 1, 0]], dtype=np.float32)
    node_names = ["x1", "x2", "x3"]
    return obs, true_adj, node_names, noise_stds


# =============================================================================
# BENCHMARK GRAPH EXAMPLES (Erdos-Renyi / Scale-Free)
# =============================================================================
# These examples demonstrate the new benchmark graph generation capabilities.
# Uncomment one to use instead of Sachs.

# --- Erdos-Renyi, Linear SEM ---
# 10 nodes, edge probability p=0.3, noise scale s=1.0, 1000 samples
# CUSTOM_DATA_BUILDER = lambda: generate_benchmark_dataset(
#     num_nodes=2, graph_type='er', sem_type='linear',
#     num_samples=1000, edge_prob=0.3, noise_scale=1.0, rng_seed=42,
# )

# --- Erdos-Renyi, Nonlinear SEM (MLP) ---
# 10 nodes, edge probability p=0.3, noise scale s=0.5, 2000 samples
# CUSTOM_DATA_BUILDER = lambda: generate_benchmark_dataset(
#     num_nodes=10, graph_type='er', sem_type='nonlinear',
#     num_samples=2000, edge_prob=0.3, noise_scale=0.5, rng_seed=42,
# )

# --- Scale-Free, Linear SEM ---
# 20 nodes, k=2 edges per new node, noise scale s=1.0, 2000 samples
# CUSTOM_DATA_BUILDER = lambda: generate_benchmark_dataset(
#     num_nodes=20, graph_type='sf', sem_type='linear',
#     num_samples=2000, sf_num_edges=2, noise_scale=1.0, rng_seed=42,
# )

# --- Scale-Free, Nonlinear SEM (MLP) ---
# 20 nodes, k=4 edges per new node, noise scale s=0.5, 5000 samples
# CUSTOM_DATA_BUILDER = lambda: generate_benchmark_dataset(
#     num_nodes=20, graph_type='sf', sem_type='nonlinear',
#     num_samples=5000, sf_num_edges=4, noise_scale=0.5, rng_seed=42,
# )

# --- Custom: large sparse ER graph ---
# 50 nodes, low edge probability p=0.1, linear SEM
# CUSTOM_DATA_BUILDER = lambda: generate_benchmark_dataset(
#     num_nodes=50, graph_type='er', sem_type='linear',
#     num_samples=5000, edge_prob=0.1, noise_scale=1.0, rng_seed=0,
# )

# Uncomment to use custom synthetic data instead of Sachs:
# CUSTOM_DATA_BUILDER = lambda: generate_synthetic_dataset(generator_fn=two_node_generator_linear, normalize=False)


def main():
    """Run SVIDAG experiments with configured scenarios and data."""
    # 2 node linear dataset
    # CUSTOM_DATA_BUILDER = lambda: generate_synthetic_dataset(generator_fn=two_node_generator_linear, normalize=True)
    # run_all_scenarios(scenarios=CUSTOM_SCENARIOS, dataset_builder=CUSTOM_DATA_BUILDER)
    # # 2 node nonlinear dataset
    # CUSTOM_DATA_BUILDER = lambda: generate_synthetic_dataset(generator_fn=two_node_generator_nonlinear, normalize=True)
    # run_all_scenarios(scenarios=CUSTOM_SCENARIOS, dataset_builder=CUSTOM_DATA_BUILDER)
    # # 3 node nonlinear dataset
    # CUSTOM_DATA_BUILDER = lambda: generate_synthetic_dataset(generator_fn=three_node_generator_nonlinear, normalize=True)
    # run_all_scenarios(scenarios=CUSTOM_SCENARIOS, dataset_builder=CUSTOM_DATA_BUILDER)
    # # 3 node linear dataset
    # CUSTOM_DATA_BUILDER = lambda: generate_synthetic_dataset(generator_fn=three_node_generator_linear, normalize=True)
    # run_all_scenarios(scenarios=CUSTOM_SCENARIOS, dataset_builder=CUSTOM_DATA_BUILDER)

    CUSTOM_DATA_BUILDER = lambda: load_sachs_dataset()
    run_all_scenarios(scenarios=CUSTOM_SCENARIOS, dataset_builder=CUSTOM_DATA_BUILDER)

if __name__ == "__main__":
    main()
