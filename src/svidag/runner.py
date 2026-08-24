'''
Runner module for the SVIDAG.

Author: Shrenik Zinage
'''


import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
import time
import os
from flax.traverse_util import flatten_dict

from . import config
from .data import generate_synthetic_dataset
from .eval import posterior_predict, sample_relaxed_adj, sample_hard_adj, sample_single_ABC, sample_relaxed_B, sample_mask_M
from .train import make_model_and_state, train_step_donated
from .utils import (
    compute_alpha_beta_from_prior,
    compute_coverage_probability,
    compute_coverage_probability_cpdag,
    compute_expected_metrics,
    compute_expected_metrics_cpdag,
    compute_posterior_entropy,
    compute_posterior_entropy_cpdag,
    dag_to_cpdag,
    get_prior_matrix,
    to_device,
    to_numpy,
    get_compute_device,
)
from .plots import (
    plot_adj_samples,
    plot_B_samples,
    plot_M_samples,
    plot_abc_sample,
    plot_ncm_parameter_evolution,
    plot_posterior_predictive,
    plot_prior_vs_posterior,

    plot_losses,
)


def _clone_train_state(state):
    """Materialize fresh buffers for TrainState snapshots kept across donated steps."""

    def copy_leaf(x):
        if isinstance(x, (jax.Array, np.ndarray)):
            return jnp.array(x, copy=True)
        return x

    return jax.tree_util.tree_map(copy_leaf, state)


def _select_ncm_parameter_trackers(params, num_params_to_track, seed):
    """
    Select random scalar entries from node-model (NCM) parameters to track.

    Returns a list of dicts with:
      - path: tuple path into flattened params
      - flat_index: scalar index inside the flattened leaf tensor
      - label: readable parameter identifier
    """
    flat_params = flatten_dict(params)
    candidate_leaves = []
    total_scalars = 0

    for path, value in flat_params.items():
        path_str = "/".join(path)
        if "node_model_" not in path_str:
            continue
        size = int(np.prod(value.shape))
        if size <= 0:
            continue
        candidate_leaves.append((path, size, path_str))
        total_scalars += size

    if total_scalars == 0 or num_params_to_track <= 0:
        return []

    n_select = min(num_params_to_track, total_scalars)
    rng = np.random.default_rng(seed)
    selected_global_indices = np.sort(rng.choice(total_scalars, size=n_select, replace=False))

    trackers = []
    leaf_cursor = 0
    leaf_id = 0
    for global_idx in selected_global_indices:
        while leaf_id < len(candidate_leaves):
            path, size, path_str = candidate_leaves[leaf_id]
            if global_idx < leaf_cursor + size:
                local_idx = int(global_idx - leaf_cursor)
                trackers.append(
                    {
                        "path": path,
                        "flat_index": local_idx,
                        "label": f"{path_str}[{local_idx}]",
                    }
                )
                break
            leaf_cursor += size
            leaf_id += 1

    return trackers


def _snapshot_ncm_parameters(params, trackers):
    """Read the current scalar values for the selected NCM parameter entries."""
    if not trackers:
        return np.array([], dtype=np.float32)

    flat_params = flatten_dict(params)
    values = []
    for tracker in trackers:
        leaf = flat_params[tracker["path"]]
        flat_leaf = to_numpy(leaf).reshape(-1)
        values.append(float(flat_leaf[tracker["flat_index"]]))
    return np.asarray(values, dtype=np.float32)


def run_scenario(scenario, dataset):
    """
    Execute a complete SVIDAG experiment for a given prior scenario.
    
    Trains the model with the specified domain-informed prior, evaluates
    posterior over graph structures, and generates all result plots.

    Args:
        scenario: Prior scenario name (e.g., "strong_correct", "noninformative")
        dataset: Dataset object with train/test data and ground truth

    Returns:
        Dict with trained model, posterior summaries, metrics, and plot paths
    """
    print(f"\n--- Scenario: {scenario} ---")
    
    # === Setup domain-informed prior ===
    # Prior p_ij encodes expert belief about edge i→j probability
    p_prior = to_device(get_prior_matrix(scenario, dataset.node_names, dataset.true_adj_np, num_nodes=dataset.num_nodes))
    p_prior_np = to_numpy(p_prior).astype(np.float32, copy=False)
    # Sanitize: ensure valid probability values
    p_prior_np = np.nan_to_num(p_prior_np, nan=0.0)
    p_prior_np = np.clip(p_prior_np, 0.0, 1.0)
    p_prior_np = p_prior_np.astype(np.float32, copy=False)
    # Convert to Beta distribution parameters for Logistic-Beta prior
    alpha_mat, beta_mat = compute_alpha_beta_from_prior(p_prior)
    alpha_mat = to_device(alpha_mat)
    beta_mat = to_device(beta_mat)

    # === Initialize model and optimizer ===
    key = jrand.PRNGKey(config.seed)
    key, init_key = jrand.split(key)
    model, state = make_model_and_state(init_key, dataset.train_data, p_prior, dataset.num_nodes, fixed_noise_scales=dataset.noise_scales)

    # Early stopping tracking
    best_elbo = -np.inf
    best_state = _clone_train_state(state)
    no_improve = 0
    patience = 20000  # Stop if no improvement for this many iterations

    # History for plotting loss curves
    losses_hist = {
        "elbo": [],
        "ell": [],          # Expected log-likelihood
        "kl_theta": [],     # KL for node model weights
        "kl_noise": [],     # KL for graph-conditioned likelihood noise
        "kl_gamma": [],     # KL for edge logits (flow)
        "T_B": [],          # Temperature for Concrete relaxation
        "tau_sn": [],       # Temperature for Sinkhorn
        "iter": [],
        "epoch": [],
        "ncm_param_labels": [],
        "tracked_ncm_params": [],
    }

    scenario_seed = config.seed + sum(bytearray(scenario.encode("utf-8")))
    ncm_trackers = _select_ncm_parameter_trackers(state.params, num_params_to_track=5, seed=scenario_seed)
    losses_hist["ncm_param_labels"] = [tracker["label"] for tracker in ncm_trackers]

    scenario_plot_dir = os.path.join(config.plot_dir, dataset.dataset_name, scenario)
    os.makedirs(scenario_plot_dir, exist_ok=True)

    # === Training loop ===
    start_time = time.time()
    for it in range(1, config.num_iters + 1):
        key, kb, ks = jrand.split(key, 3)
        # Sample mini-batch uniformly from training data
        idx = jrand.randint(kb, (config.batch_size,), 0, dataset.dataset_size)
        batch = dataset.train_data[idx]

        # Single training step: updates both params and SVGD particles
        state, loss_val, aux = train_step_donated(
            state, batch, ks, it, config.num_iters, alpha_mat, beta_mat, dataset.dataset_size, config.ELBO_MC_SAMPLES
        )
        cur_elbo = float(aux["elbo"])

        if cur_elbo > best_elbo:
            best_elbo = cur_elbo
            best_state = _clone_train_state(state)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at iter {it}")
                state = best_state
                break

        if it % config.print_every == 0:
            end_time = time.time()
            elapsed = end_time - start_time
            start_time = end_time

            print(
                f"iter {it:6d} | ELBO {aux['elbo']:.3f} | ELL {aux['ell']:.3f} | "
                f"KLθ {aux['kl_theta']:.3f} | KLnoise {aux['kl_noise']:.3f} | "
                f"KLγ {aux['kl_gamma']:.3f} | T {aux['T_B']:.3f} τ {aux['tau_sn']:.3f} | time {elapsed:.2f}s"
            )
            losses_hist["elbo"].append(float(aux["elbo"]))
            losses_hist["ell"].append(float(aux["ell"]))
            losses_hist["kl_theta"].append(float(aux["kl_theta"]))
            losses_hist["kl_noise"].append(float(aux["kl_noise"]))
            losses_hist["kl_gamma"].append(float(aux["kl_gamma"]))
            losses_hist["T_B"].append(float(aux["T_B"]))
            losses_hist["tau_sn"].append(float(aux["tau_sn"]))
            losses_hist["iter"].append(it)
            losses_hist["epoch"].append((it * config.batch_size) / dataset.dataset_size)
            losses_hist["tracked_ncm_params"].append(_snapshot_ncm_parameters(state.params, ncm_trackers))

    # === Posterior Predictive Evaluation ===
    # Sample predictions from the learned posterior over graph structures
    key, pred_key = jrand.split(key)
    mean_preds_scaled, std_preds_scaled = posterior_predict(
        state.apply_fn,
        state.params,
        state.particles,
        dataset.test_data_scaled,
        pred_key,
        config.T_B_end,
        config.tau_sink_end,
        alpha_mat,
        beta_mat,
        num_samples=50,
    )
    # Transform predictions back to original scale for interpretable comparison
    mean_preds_orig = dataset.scaler.inverse_transform(to_numpy(mean_preds_scaled)).astype(np.float32, copy=False)
    # Scale std by the scaler's scale (std is not shifted, only scaled)
    std_preds_orig = (to_numpy(std_preds_scaled) * dataset.scaler.scale_).astype(np.float32, copy=False)
    truth_orig = dataset.obs_test_orig
    plot_posterior_predictive(scenario, mean_preds_orig, std_preds_orig, truth_orig, dataset.num_nodes, output_dir=scenario_plot_dir)

    viz_samples = min(20, config.n_particles)

    # === Sample hard posterior DAGs A = B ⊙ M(r) ===
    # Binary DAG samples by construction; the marginal edge probabilities
    # p̂_ij = E_q[A_ij] are the Monte Carlo mean of these binary samples.
    key, samp_key = jrand.split(key)
    A_samples = sample_hard_adj(
        state.apply_fn,
        state.params,
        state.particles,
        samp_key,
        config.T_B_end,
        config.tau_sink_end,
        alpha_mat,
        beta_mat,
        config.posterior_samples_A,
        dataset.train_data
    )
    A_bin_samples = to_numpy(A_samples).astype(np.int8)
    A_mean = np.mean(A_bin_samples, axis=0, dtype=np.float32)
    A_std = np.std(A_bin_samples.astype(np.float32), axis=0).astype(np.float32, copy=False)

    key, viz_adj_key = jrand.split(key)
    A_viz_samples = sample_relaxed_adj(
        state.apply_fn,
        state.params,
        state.particles,
        viz_adj_key,
        config.T_B_end,
        config.tau_sink_end,
        alpha_mat,
        beta_mat,
        viz_samples,
        dataset.train_data,
        distinct_particles=True,
    )
    A_viz_samples_np = to_numpy(A_viz_samples).astype(np.float32, copy=False)
    plot_adj_samples(scenario, A_viz_samples_np, n_samples_to_plot=viz_samples, output_dir=scenario_plot_dir)

    # === Sample relaxed B matrices ===
    key, viz_b_key = jrand.split(key)
    B_viz_samples = sample_relaxed_B(
        state.apply_fn,
        state.params,
        state.particles,
        viz_b_key,
        config.T_B_end,
        config.tau_sink_end,
        alpha_mat,
        beta_mat,
        viz_samples,
        dataset.train_data,
        distinct_particles=True
    )
    B_viz_samples_np = to_numpy(B_viz_samples).astype(np.float32, copy=False)
    plot_B_samples(scenario, B_viz_samples_np, n_samples_to_plot=viz_samples, output_dir=scenario_plot_dir)

    # === Sample M masks ===
    key, viz_m_key = jrand.split(key)
    M_viz_samples = sample_mask_M(
        state.apply_fn,
        state.params,
        state.particles,
        viz_m_key,
        config.T_B_end,
        config.tau_sink_end,
        alpha_mat,
        beta_mat,
        viz_samples,
        dataset.train_data,
        distinct_particles=True
    )
    M_viz_samples_np = to_numpy(M_viz_samples).astype(np.float32, copy=False)
    plot_M_samples(scenario, M_viz_samples_np, n_samples_to_plot=viz_samples, output_dir=scenario_plot_dir)

    # --- Single random sample heatmaps for A, B and M ---
    key, sample_abc_key = jrand.split(key)
    A_sample_np, B_sample_np, M_sample_np = sample_single_ABC(
        state.apply_fn,
        state.params,
        state.particles,
        sample_abc_key,
        config.T_B_end,
        config.tau_sink_end,
        alpha_mat,
        beta_mat,
        dataset.train_data,
    )
    plot_abc_sample(
        scenario=scenario,
        A_sample=A_sample_np,
        B_sample=B_sample_np,
        M_sample=M_sample_np,
        output_dir=scenario_plot_dir,
    )

    # Convert true DAG to its CPDAG representation for fair comparison
    # Handle convention: our adjacency is j→i, dag_to_cpdag expects i→j
    true_cpdag = dag_to_cpdag(dataset.true_adj_np.T).T
    
    # === Compute Expected Metrics ===
    # DAG-level expected metrics (averaged over all posterior samples)
    dag_metrics = compute_expected_metrics(
        A_bin_samples, 
        dataset.true_adj_np, 
        pred_probs=A_mean
    )
    
    # CPDAG-level expected metrics (averaged over all posterior samples)
    cpdag_metrics = compute_expected_metrics_cpdag(
        A_bin_samples, 
        true_cpdag, 
        pred_probs=A_mean
    )
    
    # === Compute Uncertainty Metrics ===
    # 1. DAG-Level Uncertainty
    entropy_dag = compute_posterior_entropy(A_bin_samples)
    coverage_dict_dag, prob_true_dag = compute_coverage_probability(
        A_bin_samples, 
        dataset.true_adj_np,
        k_list=[1, 5, 10]
    )
    
    # 2. CPDAG-Level Uncertainty (Markov Equivalence Class)
    entropy_cpdag = compute_posterior_entropy_cpdag(A_bin_samples)
    coverage_dict_cpdag, prob_true_cpdag = compute_coverage_probability_cpdag(
        A_bin_samples,
        true_cpdag,
        k_list=[1, 5, 10]
    )
    
    # === Print Metrics Report ===
    print("\n" + "="*70)
    print(f"  STRUCTURE LEARNING METRICS - Scenario: {scenario}")
    print(f"  (Expected metrics computed over {config.posterior_samples_A} posterior samples)")
    print("="*70)
    
    print("\n--- DAG-Level Metrics ---")
    print(f"  Expected SHD:    {dag_metrics['E_SHD']:.4f}")
    print(f"  Expected TPR:    {dag_metrics['E_TPR']:.4f}")
    print(f"  Expected F1:     {dag_metrics['E_F1']:.4f}")
    print(f"  Brier Score:     {dag_metrics['Brier']:.4f}")
    if dag_metrics['AUROC'] is not None and not np.isnan(dag_metrics['AUROC']):
        print(f"  AUROC:           {dag_metrics['AUROC']:.4f}")
    else:
        print(f"  AUROC:           N/A (single class)")
    
    print("\n--- CPDAG-Level Metrics ---")
    print(f"  Expected SHD:    {cpdag_metrics['E_SHD']:.4f}")
    print(f"  Expected TPR:    {cpdag_metrics['E_TPR']:.4f}")
    print(f"  Expected F1:     {cpdag_metrics['E_F1']:.4f}")
    print(f"  Brier Score:     {cpdag_metrics['Brier']:.4f}")
    if cpdag_metrics['AUROC'] is not None and not np.isnan(cpdag_metrics['AUROC']):
        print(f"  AUROC:           {cpdag_metrics['AUROC']:.4f}")
    else:
        print(f"  AUROC:           N/A (single class)")
    
    print("\n--- Epistemic Uncertainty Metrics (DAG) ---")
    print(f"  Posterior Entropy: {entropy_dag:.4f}")
    print(f"  Prob(True DAG):    {prob_true_dag:.4f}")
    print(f"  True in Top-1:     {int(coverage_dict_dag['top_1'])}")
    print(f"  True in Top-5:     {int(coverage_dict_dag['top_5'])}")
    print(f"  True in Top-10:    {int(coverage_dict_dag['top_10'])}")

    print("\n--- Epistemic Uncertainty Metrics (CPDAG) ---")
    print(f"  Posterior Entropy: {entropy_cpdag:.4f}")
    print(f"  Prob(True CPDAG):  {prob_true_cpdag:.4f}")
    print(f"  True in Top-1:     {int(coverage_dict_cpdag['top_1'])}")
    print(f"  True in Top-5:     {int(coverage_dict_cpdag['top_5'])}")
    print(f"  True in Top-10:    {int(coverage_dict_cpdag['top_10'])}")
    
    print("="*70 + "\n")

    if config.learn_likelihood_noise:
        print(
            f"Graph-conditioned likelihood noise enabled "
            f"(log-noise prior centered at σ={config.obs_noise_scale})."
        )
    else:
        print(f"Fixed noise scales (σ): {to_numpy(dataset.noise_scales)}")

    # plot_top_dags(scenario, top_list, output_dir=scenario_plot_dir)
    # plot_dag_probabilities(scenario, top_list, output_dir=scenario_plot_dir)

    prior_std = np.sqrt(p_prior_np * (1 - p_prior_np))
    prior_std = np.nan_to_num(prior_std, nan=0.0)
    prior_std = np.clip(prior_std, 0.0, 0.5)
    prior_std = prior_std.astype(np.float32, copy=False)

    A_mode = None
    mode_counts = {}
    for sample in A_bin_samples:
        key = sample.tobytes()
        mode_counts[key] = mode_counts.get(key, 0) + 1
    if len(mode_counts) > 0:
        mode_bytes = max(mode_counts.items(), key=lambda item: item[1])[0]
        A_mode = np.frombuffer(mode_bytes, dtype=np.int8).reshape(A_bin_samples.shape[1:])
    plot_prior_vs_posterior(
        scenario,
        p_prior_np,
        A_mean,
        A_mode=A_mode,
        output_dir=scenario_plot_dir
    )

    return {
        "model": model,
        "state": state,
        "A_mean": A_mean,
        "A_std": A_std,
        "p_prior": p_prior_np,
        "prior_std": prior_std,
        "losses": losses_hist,
        # DAG-level expected metrics
        "dag_metrics": dag_metrics,
        "E_SHD": dag_metrics['E_SHD'],
        "E_TPR": dag_metrics['E_TPR'],
        "E_F1": dag_metrics['E_F1'],
        "Brier": dag_metrics['Brier'],
        "AUROC": dag_metrics['AUROC'],
        # CPDAG-level expected metrics
        "cpdag_metrics": cpdag_metrics,
        "E_SHD_CPDAG": cpdag_metrics['E_SHD'],
        "E_TPR_CPDAG": cpdag_metrics['E_TPR'],
        "E_F1_CPDAG": cpdag_metrics['E_F1'],
        "Brier_CPDAG": cpdag_metrics['Brier'],
        "AUROC_CPDAG": cpdag_metrics['AUROC'],
        # Uncertainty Metrics (DAG)
        "posterior_entropy_dag": entropy_dag,
        "prob_true_dag": prob_true_dag,
        "true_in_top_1_dag": coverage_dict_dag['top_1'],
        "true_in_top_5_dag": coverage_dict_dag['top_5'],
        "true_in_top_10_dag": coverage_dict_dag['top_10'],
        # Uncertainty Metrics (CPDAG)
        "posterior_entropy_cpdag": entropy_cpdag,
        "prob_true_cpdag": prob_true_cpdag,
        "true_in_top_1_cpdag": coverage_dict_cpdag['top_1'],
        "true_in_top_5_cpdag": coverage_dict_cpdag['top_5'],
        "true_in_top_10_cpdag": coverage_dict_cpdag['top_10'],
        # Other
        "elbo_mc_samples": config.ELBO_MC_SAMPLES,
        "plot_dir": scenario_plot_dir,
    }


def run_all_scenarios(scenarios=None, dataset_builder=None, dataset=None):
    """
    Run one or more prior scenarios on a provided or generated dataset.

    Args:
        scenarios: optional list of scenario names to run (defaults to config.scenarios)
        dataset_builder: optional callable returning a Dataset; ignored if dataset is provided
        dataset: optional prebuilt Dataset instance to reuse
    """
    get_compute_device()  # triggers logging once
    print(f"Using {config.ELBO_MC_SAMPLES} Monte Carlo samples per iteration for ELBO estimation")
    active_scenarios = scenarios or config.scenarios
    if dataset is not None:
        dataset_obj = dataset
    elif dataset_builder is not None:
        dataset_obj = dataset_builder()
    else:
        raise ValueError("Either 'dataset' or 'dataset_builder' must be provided.")
    results = {}
    for sc in active_scenarios:
        results[sc] = run_scenario(sc, dataset_obj)

    for sc in active_scenarios:
        hist = results[sc]["losses"]
        if len(hist["iter"]) == 0:
            continue
        plot_losses(hist, sc, output_dir=results[sc]["plot_dir"])
        if len(hist["tracked_ncm_params"]) > 0:
            plot_ncm_parameter_evolution(hist, sc, output_dir=results[sc]["plot_dir"])

    return results


def main():
    run_all_scenarios()


if __name__ == "__main__":
    main()
