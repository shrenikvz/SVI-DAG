#!/usr/bin/env python
"""
Case 1: Effect of Domain-Informed Prior on Posterior Structure Inference
=========================================================================

Runs SVIDAG with the paper's 3 prior scenarios (incorrect / noninformative /
correct) on both linear and nonlinear 2-node synthetic data (N = 1000).

For each trained model, draws hard posterior DAG samples following the
generative construction exactly:

    r ~ uniform over SVGD particles,  z ~ p0,  γ = T_φ(z; r),
    B = 1{γ + R > 0},  A = B ⊙ M(r)   with   M(r) = P(r) L P(r)^T.

Every sample is a binary DAG by construction (Theorem 3.1), so exactly three
outcomes are possible for a 2-node graph and no thresholding step is involved:

    x1 -> x2    (A[1,0]=1, A[0,1]=0)
    x1 <- x2    (A[0,1]=1, A[1,0]=0)
    x1 ⊥  x2    (A[0,1]=0, A[1,0]=0)

A bidirectional sample would violate acyclicity-by-construction; the script
asserts that none occurs.

Outputs
-------
    case_1_results.json       – raw percentage numbers
    case_1_table.tex          – LaTeX table ready for the paper
    case_1_figure_data.csv    – long-form data for the domain_prior_svidag.pdf bar figure
"""

import sys
import os
import json
import time
import csv
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand

# ---------------------------------------------------------------------------
# Path setup – add project root so svidag package is importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJ_ROOT)

from src.svidag import config
from src.svidag.data import generate_synthetic_dataset
from src.svidag.train import make_model_and_state, train_step_donated, maybe_warmstart
from src.svidag.eval import sample_hard_adj
from src.svidag.utils import (
    get_prior_matrix,
    compute_alpha_beta_from_prior,
    to_device,
    to_numpy,
)
from src.svidag.runner import _clone_train_state

# Generator functions live in the top-level main.py
from main import two_node_generator_linear, two_node_generator_nonlinear

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
# The paper's 3 prior scenarios: p = 0.99 on the wrong direction, p = 0.5,
# and p = 0.99 on the true direction (see utils.get_prior_matrix).
SCENARIOS = [
    "strong_incorrect",
    "noninformative",
    "strong_correct",
]

SCENARIO_LABELS = {
    "strong_incorrect": "Incorrect",
    "noninformative":   "Noninformative",
    "strong_correct":   "Correct",
}

# ---------------------------------------------------------------------------
# Environment-driven hyperparameter profile (same convention as the case-2/3/4
# runners): run_case1.sh exports SVIDAG_* and they are applied to svidag.config
# before the model is built or train_step is traced.  With nothing exported,
# every value falls through to the committed config.py default.
# ---------------------------------------------------------------------------
_ENV_FLOAT = {
    "SVIDAG_LR": "lr",
    "SVIDAG_ETA_R": "eta_r",
    "SVIDAG_GRAD_CLIP": "grad_clip",
    "SVIDAG_PRIOR_R_SIGMA": "prior_r_sigma",
    "SVIDAG_T_B": ("T_B_start", "T_B_end"),
    "SVIDAG_TAU_START": "tau_sink_start",
    "SVIDAG_TAU_END": "tau_sink_end",
    "SVIDAG_TAU_ANNEAL_FRAC": "tau_anneal_frac",
    "SVIDAG_ST_WARMUP": "st_warmup_frac",
    "SVIDAG_ETA_R_WARMUP": "eta_r_warmup_frac",
    "SVIDAG_SVGD_REP_RATIO": "svgd_repulsion_max_ratio",
    "SVIDAG_SVGD_REP_ANNEAL": "svgd_repulsion_anneal_frac",
    "SVIDAG_WARMSTART_FRAC": "particle_warmstart_frac",
    "SVIDAG_KL_THETA": "kl_theta_weight",
    "SVIDAG_PR_PREC_KAPPA": "PR_PREC_KAPPA",
}
_ENV_INT = {
    "SVIDAG_NUM_ITERS": "num_iters",
    "SVIDAG_BATCH_SIZE": "batch_size",
    "SVIDAG_N_PARTICLES": "n_particles",
    "SVIDAG_SINKHORN_ITERS": "sinkhorn_iters",
    "SVIDAG_HIDDEN_DIM": "hidden_dim",
    "SVIDAG_FLOW_BLOCKS": "flow_n_blocks",
}
_ENV_BOOL = {
    "SVIDAG_SCALE_INV": "sinkhorn_scale_invariant",
    "SVIDAG_ROW_ONLY": "node_cond_row_only",
}


def _apply_env_overrides(verbose: bool = True) -> None:
    applied = {}
    for env_key, attr in _ENV_FLOAT.items():
        if env_key in os.environ:
            val = float(os.environ[env_key])
            for a in (attr if isinstance(attr, tuple) else (attr,)):
                setattr(config, a, val)
            applied[env_key] = val
    for env_key, attr in _ENV_INT.items():
        if env_key in os.environ:
            val = int(os.environ[env_key])
            setattr(config, attr, val)
            applied[env_key] = val
    for env_key, attr in _ENV_BOOL.items():
        if env_key in os.environ:
            setattr(config, attr, bool(int(os.environ[env_key])))
            applied[env_key] = getattr(config, attr)
    if "SVIDAG_FLOW_TYPE" in os.environ:
        config.flow_type = os.environ["SVIDAG_FLOW_TYPE"]
        applied["SVIDAG_FLOW_TYPE"] = config.flow_type
    if "SVIDAG_FLOW_HIDDEN" in os.environ:
        widths = [int(w) for w in os.environ["SVIDAG_FLOW_HIDDEN"].split(",") if w.strip()]
        config.flow_hidden = widths * 2 if len(widths) == 1 else widths
        applied["SVIDAG_FLOW_HIDDEN"] = config.flow_hidden
    if verbose and applied:
        print(f"  [case1] env profile: {applied}", flush=True)


NUM_POSTERIOR_SAMPLES = 10000  # hard posterior DAG samples for classification

# ---------------------------------------------------------------------------
# Environment-driven hyperparameter profile (same convention as the case-4
# runner): ``run_case1.sh`` exports a handful of ``SVIDAG_*`` variables and we
# apply them to ``svidag.config`` before the model is built or ``train_step``
# is traced.  With nothing exported every value falls through to the committed
# default in ``config.py``.
# ---------------------------------------------------------------------------
_ENV_FLOAT = {
    "SVIDAG_LR": "lr",
    "SVIDAG_ETA_R": "eta_r",
    "SVIDAG_GRAD_CLIP": "grad_clip",
    "SVIDAG_PRIOR_R_SIGMA": "prior_r_sigma",
    "SVIDAG_T_B": ("T_B_start", "T_B_end"),
    "SVIDAG_TAU_START": "tau_sink_start",
    "SVIDAG_TAU_END": "tau_sink_end",
    "SVIDAG_TAU_ANNEAL_FRAC": "tau_anneal_frac",
    "SVIDAG_ST_WARMUP": "st_warmup_frac",
    "SVIDAG_ETA_R_WARMUP": "eta_r_warmup_frac",
    "SVIDAG_PARTICLE_CLIP": "particle_grad_clip",
    "SVIDAG_SVGD_REPULSION": "svgd_repulsion_weight",
    "SVIDAG_SVGD_REP_RATIO": "svgd_repulsion_max_ratio",
    "SVIDAG_PR_PREC_KAPPA": "PR_PREC_KAPPA",
}
_ENV_INT = {
    "SVIDAG_NUM_ITERS": "num_iters",
    "SVIDAG_BATCH_SIZE": "batch_size",
    "SVIDAG_N_PARTICLES": "n_particles",
    "SVIDAG_SINKHORN_ITERS": "sinkhorn_iters",
    "SVIDAG_HIDDEN_DIM": "hidden_dim",
    "SVIDAG_FLOW_BLOCKS": "flow_n_blocks",
}
_ENV_BOOL = {
    "SVIDAG_ROW_ONLY": "node_cond_row_only",
    "SVIDAG_SCALE_INV": "sinkhorn_scale_invariant",
}


def _apply_env_overrides():
    applied = {}
    for env_key, attr in _ENV_FLOAT.items():
        if env_key in os.environ:
            val = float(os.environ[env_key])
            for a in (attr if isinstance(attr, tuple) else (attr,)):
                setattr(config, a, val)
            applied[env_key] = val
    for env_key, attr in _ENV_INT.items():
        if env_key in os.environ:
            val = int(os.environ[env_key])
            setattr(config, attr, val)
            applied[env_key] = val
    for env_key, attr in _ENV_BOOL.items():
        if env_key in os.environ:
            setattr(config, attr, bool(int(os.environ[env_key])))
            applied[env_key] = getattr(config, attr)
    if "SVIDAG_FLOW_HIDDEN" in os.environ:
        widths = [int(w) for w in os.environ["SVIDAG_FLOW_HIDDEN"].split(",") if w.strip()]
        config.flow_hidden = widths * 2 if len(widths) == 1 else widths
        applied["SVIDAG_FLOW_HIDDEN"] = config.flow_hidden
    if "SVIDAG_PARTICLE_CLIP_MODE" in os.environ:
        config.particle_grad_clip_mode = os.environ["SVIDAG_PARTICLE_CLIP_MODE"]
        applied["SVIDAG_PARTICLE_CLIP_MODE"] = config.particle_grad_clip_mode
    if applied:
        print(f"  [svidag] env profile: {applied}", flush=True)

PATIENCE = 20_000              # early-stopping patience (iterations)


# ===================================================================
# Training
# ===================================================================
def train_model(scenario, dataset):
    """
    Train SVIDAG for one prior scenario on the given dataset.

    Returns the best TrainState (by ELBO) together with the prior matrices
    needed for posterior sampling.
    """
    key = jrand.PRNGKey(config.seed)

    # Domain-informed prior  ➜  Logistic-Beta parameters
    p_prior = to_device(
        get_prior_matrix(scenario, dataset.node_names,
                         dataset.true_adj_np, dataset.num_nodes)
    )
    alpha_mat, beta_mat = compute_alpha_beta_from_prior(p_prior)
    alpha_mat = to_device(alpha_mat)
    beta_mat = to_device(beta_mat)

    # Model + optimiser initialisation
    key, init_key = jrand.split(key)
    _model, state = make_model_and_state(
        init_key, dataset.train_data, p_prior, dataset.num_nodes,
        fixed_noise_scales=dataset.noise_scales,
    )

    # Early-stopping bookkeeping.  The relaxation warm-up is excluded: while
    # ``st_weight`` < 1 the likelihood is evaluated on the SOFT adjacency, which
    # fits strictly better than the hard DAG, so every warm-up iterate outscores
    # everything after it and "best ELBO" would always land inside the warm-up.
    best_elbo = -np.inf
    best_state = _clone_train_state(state)
    no_improve = 0
    stopped_early = False
    warm_end = int(float(config.st_warmup_frac) * config.num_iters)

    t0 = time.time()
    for it in range(1, config.num_iters + 1):
        key, kb, ks = jrand.split(key, 3)
        idx = jrand.randint(kb, (config.batch_size,), 0, dataset.dataset_size)
        batch = dataset.train_data[idx]

        state, _, aux = train_step_donated(
            state, batch, ks, it, config.num_iters,
            alpha_mat, beta_mat, dataset.dataset_size,
            config.ELBO_MC_SAMPLES,
        )
        state, _warmstarted = maybe_warmstart(
            state, ks, it, config.num_iters, config.T_B_end
        )
        if _warmstarted:
            print(f"    warm-started SVGD particles at iter {it}")

        if it > warm_end:
            cur_elbo = float(aux["elbo"])
            if cur_elbo > best_elbo:
                best_elbo = cur_elbo
                best_state = _clone_train_state(state)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f"    Early stopping at iter {it}")
                    state = best_state
                    stopped_early = True
                    break

        if it % config.print_every == 0:
            elapsed = time.time() - t0
            t0 = time.time()
            print(
                f"    iter {it:6d}/{config.num_iters} | "
                f"ELBO {aux['elbo']:.3f} | ELL {aux['ell']:.3f} | "
                f"KLγ {aux['kl_gamma']:.3f} | {elapsed:.1f}s"
            )

    # Keep the final state unless early stopping actually fired: the
    # single-batch ELBO is noisy enough that its running maximum is close
    # to an arbitrary iterate.
    return (best_state if stopped_early else state), alpha_mat, beta_mat


# ===================================================================
# Posterior classification
# ===================================================================
def classify_posterior_samples(state, alpha_mat, beta_mat, dataset,
                               num_samples=NUM_POSTERIOR_SAMPLES):
    """
    Draw hard posterior DAG samples A = B ⊙ M(r) and classify each one.

    Convention:  A[i,j] = 1  means  j → i.

    For a 2-node graph, acyclicity-by-construction leaves exactly three
    possible structures:
        A[1,0]=1, A[0,1]=0  ➜  x1 → x2
        A[0,1]=1, A[1,0]=0  ➜  x2 → x1   (x1 ← x2)
        A[0,1]=0, A[1,0]=0  ➜  independent (x1 ⊥ x2)
    A bidirectional sample is impossible (Theorem 3.1); asserted below.
    """
    key = jrand.PRNGKey(42)

    A_samples = sample_hard_adj(
        state.apply_fn, state.params, state.particles,
        key, config.T_B_end, config.tau_sink_end,
        alpha_mat, beta_mat,
        num_samples=num_samples,
        train_data=dataset.train_data,
        distinct_particles=True,
    )

    A_bin = to_numpy(A_samples).astype(np.int32)
    n = A_bin.shape[0]

    a10 = A_bin[:, 1, 0]          # 1 ⟹ edge x1 → x2 present
    a01 = A_bin[:, 0, 1]          # 1 ⟹ edge x2 → x1 present

    n_bidirectional = int(np.sum((a10 == 1) & (a01 == 1)))
    assert n_bidirectional == 0, (
        f"{n_bidirectional} bidirectional samples found — the hard "
        "construction A = B ⊙ M(r) guarantees acyclicity, so this "
        "indicates a bug in the sampler."
    )

    n_x1_to_x2   = int(np.sum((a10 == 1) & (a01 == 0)))
    n_x2_to_x1   = int(np.sum((a01 == 1) & (a10 == 0)))
    n_independent = int(np.sum((a10 == 0) & (a01 == 0)))

    return {
        "x1_to_x2":   round(100.0 * n_x1_to_x2   / n, 2),
        "x2_to_x1":   round(100.0 * n_x2_to_x1   / n, 2),
        "independent": round(100.0 * n_independent / n, 2),
    }


# ===================================================================
# LaTeX table generation
# ===================================================================
def build_latex_table(results):
    """
    Build a self-contained LaTeX table from the results dict.

    The table has 9 data columns (3 scenarios × 3 structures) and
    two data blocks (linear, nonlinear).
    """
    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Effect of domain-informed prior on posterior structure inference "
                 r"(synthetic 2-node graph). Each cell shows the percentage of hard posterior "
                 r"DAG samples classified as the corresponding graph structure.}")
    lines.append(r"\label{tab:domain_prior_effect}")
    lines.append(r"\newcolumntype{Y}{>{\centering\arraybackslash}X}")
    lines.append(r"\begin{tabularx}{\linewidth}{|*{9}{Y|}}")
    lines.append(r"\hline")

    # ── Header row 1: scenario names (each spans 3 sub-columns) ──
    scenario_cells = []
    for sc in SCENARIOS:
        scenario_cells.append(
            r"\multicolumn{3}{c|}{" + SCENARIO_LABELS[sc] + r"}"
        )
    # Fix first cell to include leading |
    scenario_cells[0] = scenario_cells[0].replace(
        r"\multicolumn{3}{c|}", r"\multicolumn{3}{|c|}", 1
    )
    lines.append(" & ".join(scenario_cells) + r" \\ \hline")

    # ── Header row 2: structure labels under each scenario ──
    dir_labels = r"$\to$ & $\gets$ & $\perp\!\!\!\perp$"
    lines.append(" & ".join([dir_labels] * len(SCENARIOS)) + r" \\ \hline")

    # ── Data rows ──
    def make_data_row(label, row_data):
        """One label row + one data row for a dataset type."""
        cells = []
        for sc in SCENARIOS:
            r = row_data[sc]
            cells.append(f"{r['x1_to_x2']:.1f}")
            cells.append(f"{r['x2_to_x1']:.1f}")
            cells.append(f"{r['independent']:.1f}")
        n_cols = 3 * len(SCENARIOS)
        return (
            r"\multicolumn{" + str(n_cols) + r"}{|c|}{\textbf{" + label + r"}} \\ \hline" + "\n"
            + " & ".join(cells) + r" \\ \hline"
        )

    lines.append(make_data_row(
        "Linear synthetic data (DAG not identifiable from data)",
        results["linear"],
    ))
    lines.append(make_data_row(
        "Nonlinear synthetic data (DAG identifiable from data)",
        results["nonlinear"],
    ))

    lines.append(r"\end{tabularx}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ===================================================================
# Figure data (for domain_prior_svidag.pdf)
# ===================================================================
def write_figure_csv(results, path):
    """Long-form CSV with one row per (dataset, scenario, structure)."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "scenario", "structure", "percentage"])
        for gen_label in ["linear", "nonlinear"]:
            for sc in SCENARIOS:
                r = results[gen_label][sc]
                writer.writerow([gen_label, SCENARIO_LABELS[sc], "x1_to_x2", r["x1_to_x2"]])
                writer.writerow([gen_label, SCENARIO_LABELS[sc], "x2_to_x1", r["x2_to_x1"]])
                writer.writerow([gen_label, SCENARIO_LABELS[sc], "independent", r["independent"]])


# ===================================================================
# Console summary
# ===================================================================
def print_summary_table(results):
    """Pretty-print results to the terminal."""
    header = (
        f"{'':18s} | {'x1→x2':>7s} | {'x1←x2':>7s} | {'x1⊥x2':>7s}"
    )
    sep = "-" * len(header)
    for gen_label in ["linear", "nonlinear"]:
        print(f"\n  {gen_label.upper()} DATA")
        print(f"  {sep}")
        print(f"  {header}")
        print(f"  {sep}")
        for sc in SCENARIOS:
            r = results[gen_label][sc]
            print(
                f"  {SCENARIO_LABELS[sc]:18s} | "
                f"{r['x1_to_x2']:6.1f}% | "
                f"{r['x2_to_x1']:6.1f}% | "
                f"{r['independent']:6.1f}%"
            )
        print(f"  {sep}")


# ===================================================================
# Main
# ===================================================================
def main():
    _apply_env_overrides()
    os.makedirs(SCRIPT_DIR, exist_ok=True)

    # Standardise both generators.
    #
    # The linear generator is VARSORTABLE: it draws x1 ~ N(0,1) and
    # x2 = x1 + N(0,1), so Var(x1) = 1 < Var(x2) = 2 and the causal direction
    # leaks through the marginal scale (Reisach et al., 2021). The two
    # orientations are Markov-equivalent -- measured on this data the variance
    # products are identical to 4 decimals (0.9944 either way) -- but the
    # reverse fit needs a 2.2x smaller slope, and KL(q(theta) || N(0,1))
    # charges for weight magnitude, so the ELBO prefers one orientation for
    # reasons that have nothing to do with causality. That artifact, not the
    # prior, decided the orientation here. Standardising removes it and leaves
    # the domain-informed prior as the only tie-breaker, which is exactly the
    # regime this experiment is meant to demonstrate. Every other case already
    # standardises (StandardScaler in each case's common.py).
    generators = [
        ("linear",    two_node_generator_linear,    True),
        ("nonlinear", two_node_generator_nonlinear, True),
    ]

    all_results = {}

    for gen_label, gen_fn, normalize in generators:
        print(f"\n{'=' * 70}")
        print(f"  Dataset: {gen_label}")
        print(f"{'=' * 70}")

        dataset = generate_synthetic_dataset(
            generator_fn=gen_fn, normalize=normalize,
        )
        all_results[gen_label] = {}

        for scenario in SCENARIOS:
            print(f"\n  ── Scenario: {scenario} ──")
            state, alpha_mat, beta_mat = train_model(scenario, dataset)
            pcts = classify_posterior_samples(
                state, alpha_mat, beta_mat, dataset, NUM_POSTERIOR_SAMPLES,
            )
            all_results[gen_label][scenario] = pcts

            print(
                f"  ➜ x1→x2: {pcts['x1_to_x2']:.1f}%  "
                f"x1←x2: {pcts['x2_to_x1']:.1f}%  "
                f"⊥: {pcts['independent']:.1f}%"
            )

    # ── Save JSON results ─────────────────────────────────────────
    json_path = os.path.join(SCRIPT_DIR, "case_1_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved JSON  → {json_path}")

    # ── Save LaTeX table ──────────────────────────────────────────
    latex_str = build_latex_table(all_results)
    tex_path = os.path.join(SCRIPT_DIR, "case_1_table.tex")
    with open(tex_path, "w") as f:
        f.write(latex_str)
    print(f"Saved LaTeX → {tex_path}")

    # ── Save figure data ──────────────────────────────────────────
    csv_path = os.path.join(SCRIPT_DIR, "case_1_figure_data.csv")
    write_figure_csv(all_results, csv_path)
    print(f"Saved CSV   → {csv_path}")

    # ── Console summary ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print_summary_table(all_results)

    print("\n" + "=" * 70)
    print("  LaTeX table (case_1_table.tex)")
    print("=" * 70)
    print(latex_str)


if __name__ == "__main__":
    main()
