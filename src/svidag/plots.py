'''
Plotting module for the SVIDAG.

Author: Shrenik Zinage
'''

import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from . import config


def plot_posterior_predictive(scenario: str, mean_preds_orig: np.ndarray, std_preds_orig: np.ndarray, truth_orig: np.ndarray, num_nodes: int, output_dir: str = None):
    """
    Plot posterior predictive distributions against ground truth for each node.
    
    Shows mean prediction with 95% prediction interval (±2σ) to visualize
    both fit quality and uncertainty calibration.
    
    Args:
        mean_preds_orig: Mean predictions [N_test, num_nodes] in original scale
        std_preds_orig: Standard deviation [N_test, num_nodes]
        truth_orig: Ground truth test data [N_test, num_nodes]
    """
    if num_nodes > 12:
        print(f"Skipping posterior predictive plot for {num_nodes} nodes (current plotting limit is 12 nodes).")
        return None

    if output_dir is None:
        output_dir = config.plot_dir
    os.makedirs(output_dir, exist_ok=True)

    num_test_samples = mean_preds_orig.shape[0]
    fig, axes = plt.subplots(6, 2, figsize=(22, 20))
    axes = axes.flatten()
    for i in range(num_nodes):
        y_true = truth_orig[:, i]
        y_pred = mean_preds_orig[:, i]
        y_std = std_preds_orig[:, i]
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        mae = float(np.mean(np.abs(y_pred - y_true)))
        ss_res = np.sum((y_pred - y_true) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        ax = axes[i]
        xs = np.arange(num_test_samples)
        ax.plot(xs, y_true, label="Truth")
        ax.plot(xs, y_pred, "--", color="red", label="Pred median")
        # 95% prediction interval: median ± 2*std
        ax.fill_between(xs, y_pred - 2 * y_std, y_pred + 2 * y_std, alpha=0.2, color="green", label="95% PI")
        ax.set_title(f"{scenario} | Node {i+1} | RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
        ax.legend()
    # Hide unused subplots
    for k in range(num_nodes, len(axes)):
        axes[k].axis("off")
    plt.tight_layout()
    pf = os.path.join(output_dir, f"ppd_{scenario}.png")
    plt.savefig(pf, dpi=200)
    plt.close(fig)
    print(f"Saved posterior predictive plot: {pf}")
    return pf


def plot_adj_samples(scenario: str, A_samples_np: np.ndarray, n_samples_to_plot: int = 20, output_dir: str = None):
    """
    Visualize random samples from the posterior over relaxed adjacency matrices.
    
    Each heatmap shows a single sample A ∈ [0,1]^(m×m) where lighter colors
    indicate higher edge probability. Useful for inspecting posterior uncertainty
    and mode structure.
    """
    if output_dir is None:
        output_dir = config.plot_dir
    os.makedirs(output_dir, exist_ok=True)

    n_samples_to_plot = min(n_samples_to_plot, A_samples_np.shape[0])
    random_indices = np.random.choice(A_samples_np.shape[0], size=n_samples_to_plot, replace=False)
    A_random_samples = A_samples_np[random_indices]
    num_nodes = A_samples_np.shape[1]

    fig_samples, axes_samples = plt.subplots(4, 5, figsize=(20, 16))
    axes_samples = axes_samples.flatten()
    for idx, (ax, A_sample) in enumerate(zip(axes_samples, A_random_samples)):
        im = ax.imshow(A_sample, vmin=0, vmax=1, cmap="viridis")
        # ax.set_title(f"Sample {idx+1}")

        # Remove the default axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Draw grid lines for clarity
        for edge in range(num_nodes + 1):
            ax.axhline(edge - 0.5, color="w", linewidth=0.5)
            ax.axvline(edge - 0.5, color="w", linewidth=0.5)

        # Annotate the middle of each side with node numbers (1..num_nodes)
        for node in range(num_nodes):
            mid = num_nodes / 2.0
            fontsize = 14

            # Top (column label) - center above each column
            ax.text(node, -0.75, str(node + 1), ha="center", va="center", fontsize=fontsize, weight="bold")

            # Bottom (column label)
            ax.text(node, num_nodes - 0.25, str(node + 1), ha="center", va="center", fontsize=fontsize, weight="bold")

            # Left (row label) - center vertically at each row
            ax.text(-0.75, node, str(node + 1), ha="center", va="center", fontsize=fontsize, weight="bold")

            # Right (row label)
            ax.text(num_nodes - 0.25, node, str(node + 1), ha="center", va="center", fontsize=fontsize, weight="bold")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    samples_path = os.path.join(output_dir, f"20_random_adj_samples_{scenario}.png")
    plt.savefig(samples_path, dpi=200)
    plt.close(fig_samples)
    print(f"Saved 20 random adjacency samples plot: {samples_path}")
    return samples_path


def plot_B_samples(scenario: str, B_samples_np: np.ndarray, n_samples_to_plot: int = 20, output_dir: str = None):
    """
    Visualize random samples of the relaxed Bernoulli matrix B.
    """
    if output_dir is None:
        output_dir = config.plot_dir
    os.makedirs(output_dir, exist_ok=True)

    n_samples_to_plot = min(n_samples_to_plot, B_samples_np.shape[0])
    random_indices = np.random.choice(B_samples_np.shape[0], size=n_samples_to_plot, replace=False)
    B_random_samples = B_samples_np[random_indices]
    num_nodes = B_samples_np.shape[1]

    fig_samples, axes_samples = plt.subplots(4, 5, figsize=(20, 16))
    axes_samples = axes_samples.flatten()
    for idx, (ax, B_sample) in enumerate(zip(axes_samples, B_random_samples)):
        im = ax.imshow(B_sample, vmin=0, vmax=1, cmap="viridis")
        
        ax.set_xticks([])
        ax.set_yticks([])

        for edge in range(num_nodes + 1):
            ax.axhline(edge - 0.5, color="w", linewidth=0.5)
            ax.axvline(edge - 0.5, color="w", linewidth=0.5)

        for node in range(num_nodes):
            fontsize = 14
            ax.text(node, -0.75, str(node + 1), ha="center", va="center", fontsize=fontsize, weight="bold")
            ax.text(node, num_nodes - 0.25, str(node + 1), ha="center", va="center", fontsize=fontsize, weight="bold")
            ax.text(-0.75, node, str(node + 1), ha="center", va="center", fontsize=fontsize, weight="bold")
            ax.text(num_nodes - 0.25, node, str(node + 1), ha="center", va="center", fontsize=fontsize, weight="bold")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    samples_path = os.path.join(output_dir, f"20_random_B_samples_{scenario}.png")
    plt.savefig(samples_path, dpi=200)
    plt.close(fig_samples)
    print(f"Saved 20 random B samples plot: {samples_path}")
    return samples_path


def plot_M_samples(scenario: str, M_samples_np: np.ndarray, n_samples_to_plot: int = 20, output_dir: str = None):
    """
    Visualize random samples of the DAG mask M.
    """
    if output_dir is None:
        output_dir = config.plot_dir
    os.makedirs(output_dir, exist_ok=True)

    n_samples_to_plot = min(n_samples_to_plot, M_samples_np.shape[0])
    random_indices = np.random.choice(M_samples_np.shape[0], size=n_samples_to_plot, replace=False)
    M_random_samples = M_samples_np[random_indices]
    num_nodes = M_samples_np.shape[1]

    fig_samples, axes_samples = plt.subplots(4, 5, figsize=(20, 16))
    axes_samples = axes_samples.flatten()
    for idx, (ax, M_sample) in enumerate(zip(axes_samples, M_random_samples)):
        im = ax.imshow(M_sample, vmin=0, vmax=1, cmap="viridis")
        
        ax.set_xticks([])
        ax.set_yticks([])

        for edge in range(num_nodes + 1):
            ax.axhline(edge - 0.5, color="w", linewidth=0.5)
            ax.axvline(edge - 0.5, color="w", linewidth=0.5)

        for node in range(num_nodes):
            fontsize = 14
            ax.text(node, -0.75, str(node + 1), ha="center", va="center", fontsize=fontsize, weight="bold")
            ax.text(node, num_nodes - 0.25, str(node + 1), ha="center", va="center", fontsize=fontsize, weight="bold")
            ax.text(-0.75, node, str(node + 1), ha="center", va="center", fontsize=fontsize, weight="bold")
            ax.text(num_nodes - 0.25, node, str(node + 1), ha="center", va="center", fontsize=fontsize, weight="bold")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    samples_path = os.path.join(output_dir, f"20_random_M_samples_{scenario}.png")
    plt.savefig(samples_path, dpi=200)
    plt.close(fig_samples)
    print(f"Saved 20 random M samples plot: {samples_path}")
    return samples_path


def plot_abc_sample(scenario: str, A_sample: np.ndarray, B_sample: np.ndarray, M_sample: np.ndarray, output_dir: str = None):
    """
    Visualize one posterior sample of A, B and M in a single row.
    
    A: relaxed adjacency (A = B ⊙ M)
    B: relaxed Bernoulli sample from Concrete
    M: DAG mask from Sinkhorn relaxation
    """
    if output_dir is None:
        output_dir = config.plot_dir
    os.makedirs(output_dir, exist_ok=True)

    matrices = [
        (A_sample, "A (relaxed adjacency)"),
        (B_sample, "B (Concrete edge sample)"),
        (M_sample, "M (DAG mask)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax, (matrix, title) in zip(axes, matrices):
        num_nodes = matrix.shape[0]
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        for edge in range(num_nodes + 1):
            ax.axhline(edge - 0.5, color="w", linewidth=0.5)
            ax.axvline(edge - 0.5, color="w", linewidth=0.5)
        for node in range(num_nodes):
            ax.text(node, -0.75, str(node + 1), ha="center", va="center", fontsize=12, weight="bold")
            ax.text(node, num_nodes - 0.25, str(node + 1), ha="center", va="center", fontsize=12, weight="bold")
            ax.text(-0.75, node, str(node + 1), ha="center", va="center", fontsize=12, weight="bold")
            ax.text(num_nodes - 0.25, node, str(node + 1), ha="center", va="center", fontsize=12, weight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    sample_path = os.path.join(output_dir, f"sample_A_B_M_{scenario}.png")
    plt.savefig(sample_path, dpi=200)
    plt.close(fig)
    print(f"Saved A/B/M sample plot: {sample_path}")
    return sample_path


def plot_top_dags(scenario: str, top_list: List[Tuple[np.ndarray, float]], top_k: int = 5, output_dir: str = None):
    """Heatmaps for the top discovered DAGs, with node numbers labeled like plot_adj_samples()."""
    if output_dir is None:
        output_dir = config.plot_dir
    os.makedirs(output_dir, exist_ok=True)

    saved_paths = []
    for k, (A_top, p) in enumerate(top_list[:top_k]):
        num_nodes = A_top.shape[0]
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(A_top, vmin=0, vmax=1, cmap="viridis")
        ax.set_title(f"p={p:.4f}")

        # Remove the default axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Draw grid lines for clarity
        for edge in range(num_nodes + 1):
            ax.axhline(edge - 0.5, color="w", linewidth=0.5)
            ax.axvline(edge - 0.5, color="w", linewidth=0.5)

        # Annotate the middle of each side with node numbers (1..num_nodes)
        fontsize = 14
        for node in range(num_nodes):
            # Top (column label) - center above each column
            ax.text(node, -0.75, str(node + 1), ha="center", va="center", fontsize=fontsize, weight="bold")
            # Bottom (column label)
            ax.text(node, num_nodes - 0.25, str(node + 1), ha="center", va="center", fontsize=fontsize, weight="bold")
            # Left (row label) - center vertically at each row
            ax.text(-0.75, node, str(node + 1), ha="center", va="center", fontsize=fontsize, weight="bold")
            # Right (row label)
            ax.text(num_nodes - 0.25, node, str(node + 1), ha="center", va="center", fontsize=fontsize, weight="bold")

        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        tf = os.path.join(output_dir, f"top{k+1}_dag_{scenario}.png")
        plt.savefig(tf, dpi=200)
        plt.close()
        saved_paths.append(tf)
    return saved_paths


def plot_prior_vs_posterior(
    scenario: str,
    p_prior_np: np.ndarray,
    A_mean: np.ndarray,
    A_mode: np.ndarray = None,
    output_dir: str = None
):
    """Comparison of prior mean adjacency, posterior relaxed mean, and posterior mode."""
    if output_dir is None:
        output_dir = config.plot_dir
    os.makedirs(output_dir, exist_ok=True)

    if A_mode is not None:
        fig_compare, ax_compare = plt.subplots(1, 3, figsize=(15, 4))
    else:
        fig_compare, ax_compare = plt.subplots(1, 2, figsize=(10, 4))

    im_prior_cmp = ax_compare[0].imshow(p_prior_np, vmin=0, vmax=1, cmap="viridis")
    ax_compare[0].set_title("Prior mean (domain-informed)")
    plt.colorbar(im_prior_cmp, ax=ax_compare[0])
    im_post_cmp = ax_compare[1].imshow(A_mean, vmin=0, vmax=1, cmap="viridis")
    ax_compare[1].set_title("Posterior mean of relaxed A")
    plt.colorbar(im_post_cmp, ax=ax_compare[1])

    if A_mode is not None:
        im_mode = ax_compare[2].imshow(A_mode, vmin=0, vmax=1, cmap="viridis")
        ax_compare[2].set_title("Posterior mode DAG")
        plt.colorbar(im_mode, ax=ax_compare[2])

    plt.tight_layout()
    compare_path = os.path.join(output_dir, f"adj_prior_vs_posterior_{scenario}.pdf")
    plt.savefig(compare_path, dpi=200)
    plt.close(fig_compare)
    print(f"Saved prior vs posterior mean comparison plot: {compare_path}")
    return compare_path



def plot_losses(history: Dict[str, list], scenario: str, output_dir: str = None):
    """
    Plot ELBO and KL terms over iterations with linearly spaced y-axis ticks.
    """
    if output_dir is None:
        output_dir = config.plot_dir
    os.makedirs(output_dir, exist_ok=True)

    its = history["iter"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    metrics = [
        ("ELBO", history["elbo"]),
        ("ELL", history["ell"]),
        ("KLθ", history["kl_theta"]),
        ("KLγ", history["kl_gamma"]),
    ]

    for ax, (ylabel, values) in zip(axes, metrics):
        ax.plot(its, values, color="tab:blue")
        ax.set_ylabel(ylabel, fontsize=14)
        values_clean = np.asarray(values, dtype=np.float32)
        values_clean = values_clean[np.isfinite(values_clean)]
        if values_clean.size > 0:
            y_min = float(np.min(values_clean))
            y_max = float(np.max(values_clean))
            if y_min == y_max:
                eps = 1.0 if y_min == 0 else abs(y_min) * 0.05
                y_min -= eps
                y_max += eps
            ax.set_yticks(np.linspace(y_min, y_max, num=6))
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Iteration", fontsize=14)
    fig.tight_layout()
    of = os.path.join(output_dir, f"losses_{scenario}.png")
    fig.savefig(of, dpi=200)
    plt.close(fig)
    return of


def plot_ncm_parameter_evolution(history: Dict[str, list], scenario: str, output_dir: str = None):
    """Plot evolution of tracked NCM parameters across training epochs."""
    if output_dir is None:
        output_dir = config.plot_dir
    os.makedirs(output_dir, exist_ok=True)

    iters = history.get("iter", [])
    tracked_params = np.asarray(history.get("tracked_ncm_params", []), dtype=np.float32)
    param_labels = history.get("ncm_param_labels", [])

    if tracked_params.size == 0 or len(iters) == 0:
        return None

    fig_params, ax_params = plt.subplots(figsize=(11, 5))
    for idx in range(tracked_params.shape[1]):
        if idx < len(param_labels):
            curve_label = f"P{idx + 1}: {param_labels[idx]}"
        else:
            curve_label = f"P{idx + 1}"
        ax_params.plot(iters, tracked_params[:, idx], label=curve_label, linewidth=1.8)

    ax_params.set_xlabel("Iteration", fontsize=14)
    ax_params.set_ylabel("Parameter Value", fontsize=14)
    ax_params.set_title(f"Evolution of 5 Random NCM Parameters ({scenario})")
    ax_params.legend(fontsize=9, loc="best")
    ax_params.grid(True, alpha=0.3)
    fig_params.tight_layout()

    ncm_plot_path = os.path.join(output_dir, f"ncm_parameter_evolution_{scenario}.png")
    fig_params.savefig(ncm_plot_path, dpi=200)
    plt.close(fig_params)
    print(f"Saved NCM parameter evolution plot: {ncm_plot_path}")
    return ncm_plot_path


def plot_dag_probabilities(scenario: str, top_list: List[Tuple[np.ndarray, float]], output_dir: str = None):
    """
    Plot a bar chart of the posterior probabilities for the top DAG structures.
    """
    if output_dir is None:
        output_dir = config.plot_dir
    os.makedirs(output_dir, exist_ok=True)

    probs = [p for _, p in top_list]
    names = [f"DAG {i+1}" for i in range(len(probs))]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, probs, color='skyblue', edgecolor='navy')
    
    # Add probability labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Posterior probability', fontsize=12)
    ax.set_title(f'Posterior distribution over DAG structures ({scenario})', fontsize=14)
    ax.set_ylim(0, max(probs) * 1.15 if probs else 1.0)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    plot_path = os.path.join(output_dir, f"dag_probabilities_{scenario}.png")
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved DAG probabilities bar chart: {plot_path}")
    return plot_path
