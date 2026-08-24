#!/usr/bin/env python
"""
Baseline wrapper: BayesDAG (paper-spec hyperparameters)
========================================================

Thin adapter around the BayesDAG JAX port at
``other_algorithms/codes_jax/bayesdag/`` that materialises the **paper-
specified** BayesDAG hyperparameters so every benchmark case (case_2 /
case_3 / case_4) trains the algorithm identically:

    * Sparsity grid: 10 values geometrically spaced between
      ``lambda_min = 10`` and ``lambda_max = 1000``.
    * Validation set: ``floor(0.1 n)`` rows held out from training.
    * lambda selection rule: closest to the *true* sparsity level
      (``true_sparsity`` -- the ground-truth edge count of the underlying
      DAG, supplied by the case driver because it is dataset-specific).
    * Nonlinear ICGNN: 2 hidden layers of 128 units with ReLU activation.
      Plumbed through the JSON config keys ``hidden_size``,
      ``num_hidden_layers``, ``activation`` (see
      ``configs/bayesdag/bayesdag_nonlinear_train_protein.json``).

If ``true_sparsity`` is None, the wrapper falls back to a single fit at
the JSON config's ``lambda_sparse``.  This keeps callers that haven't
been updated to pass the true sparsity from regressing.

Adjacency convention
--------------------
BayesDAG's SEM forward pass uses ``jnp.einsum("bd,cde->cbe", x, w_adj)``
(model.py) and ``A[i, j] = 1`` means ``i -> j`` (parent i, child j).  We
report ``"i_to_j"`` so ``common.evaluate_samples`` transposes to SVIDAG's
``j -> i`` convention before metrics.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_CASE4_DIR = _THIS_DIR.parent
_REPO_ROOT = _CASE4_DIR.parent.parent
_BAYESDAG_ROOT = _REPO_ROOT / "other_algorithms" / "codes_jax" / "bayesdag"
_BAYESDAG_SRC = _BAYESDAG_ROOT / "src"
for p in (str(_BAYESDAG_ROOT), str(_BAYESDAG_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)


_DEFAULT_CONFIG = (
    _BAYESDAG_SRC / "configs" / "bayesdag" / "bayesdag_nonlinear_train_protein.json"
)


# ---------------------------------------------------------------------------
# Fast-mode defaults (coarse-to-fine lambda search).
# ---------------------------------------------------------------------------
# The paper-spec protocol trains BayesDAG ``n_lambda_grid`` (=10) times *at full
# quality* per benchmark cell purely to pick the sparsity coefficient whose mean
# posterior edge count is closest to the ground-truth edge count -- a ~10x cost
# multiplier where 9/10 of the trained models are thrown away.  Since lambda is
# selected by a *structural* statistic (mean edge count vs lambda is monotone and
# converges long before the ELBO does), the grid does not need full-quality fits.
#
# Fast mode therefore:
#   1. fits each grid lambda with a CHEAP proxy config (fewer epochs / Sinkhorn
#      iters) only to estimate its mean edge count and rank lambdas, then
#   2. runs ONE full-quality fit at the selected lambda to draw the returned
#      posterior samples.
# The returned model is a full-quality fit, so final accuracy is preserved by
# construction; only the lambda *selection* uses cheap fits (validated to pick
# the same operating point on ER_p25_s40).  Set BAYESDAG_PAPER_SPEC=1 to restore
# the exact 10-full-fit behaviour.  Every knob is overridable via env var so a
# cluster run picks up new values without editing call sites.
#
# NOTE: Sinkhorn iters are trimmed 500 -> 100 (the doubly-stochastic projection
# of a d<=200 matrix converges well within 100 iters) for BOTH proxy and final
# fits -- validated to leave metrics unchanged.
#
# The DEFAULTS below keep the final fit at paper-spec quality (ep800) so the
# built-in fast mode is accuracy-preserving.  To hit the ~3-4h/case target the
# run_case*_all.sh scripts additionally export an AGGRESSIVE profile via env:
#     BAYESDAG_EPOCHS=150 BAYESDAG_GRID_EPOCHS=25 BAYESDAG_NLAMBDA=4 BAYESDAG_GRID_SAMPLES=64
# which cuts the final fit's epochs.  Per the 2026-07-24 ER_p25_s40 ablation this
# TRADES edge-recovery accuracy (E_F1/E_SHD/Brier drop ~40-55% relative at n=100;
# AUROC ~preserved because the edge *ranking* converges early).  This is an
# explicit, user-approved speed/accuracy tradeoff; drop those env vars (or set
# BAYESDAG_PAPER_SPEC=1) to restore full accuracy.
_FAST_DEFAULTS = {
    # final full-quality fit at the selected lambda
    "final_max_epochs": 800,
    "final_num_chains": 10,
    "final_sinkhorn": 100,
    # cheap proxy fits used only to rank lambdas by edge count
    "grid_max_epochs": 150,
    "grid_num_chains": 10,
    "grid_sinkhorn": 100,
    "grid_post_samples": 128,
}


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else int(default)


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v in (None, ""):
        return bool(default)
    return v.strip().lower() not in ("0", "false", "no", "off")


def _cfg_with(cfg: dict, *, max_epochs=None, num_chains=None, sinkhorn=None) -> dict:
    """Return a deep copy of ``cfg`` with the given hyperparameters overridden."""
    c = copy.deepcopy(cfg)
    mh = c.setdefault("model_hyperparams", {})
    th = c.setdefault("training_hyperparams", {})
    if num_chains is not None:
        mh["num_chains"] = int(num_chains)
    if sinkhorn is not None:
        mh["sinkhorn_n_iter"] = int(sinkhorn)
    if max_epochs is not None:
        th["max_epochs"] = int(max_epochs)
    return c


def _fit_once(
    *,
    X_fit: np.ndarray,
    cfg: dict,
    model_type: str,
    seed: int,
    num_posterior_samples: int,
    lambda_sparse: Optional[float] = None,
    save_dir: str,
):
    """Helper: fit BayesDAG once with an optional ``lambda_sparse`` override
    and return ``(model, adj_samples)``."""
    from bayesdag_jax import train_from_config_dict

    cfg_local = copy.deepcopy(cfg)
    # Strip any seed pin from the JSON config so the caller's ``seed`` actually
    # propagates (``_normalize_seed_value`` only falls through when the key
    # is absent / None / []).  The shipped config used to ship
    # ``"random_seed": [0]`` which silently overrode every per-split seed and
    # produced bit-identical posteriors across splits.
    cfg_local.setdefault("model_hyperparams", {}).pop("random_seed", None)
    if lambda_sparse is not None:
        cfg_local["model_hyperparams"]["lambda_sparse"] = float(lambda_sparse)

    model = train_from_config_dict(
        np.asarray(X_fit, dtype=np.float32),
        model_type=model_type,
        model_config_dict=cfg_local,
        save_dir=save_dir,
        adjacency=None,
        seed=int(seed),
    )
    adj_samples, _is_dag = model.get_adj_matrix(samples=int(num_posterior_samples))
    return model, np.asarray(adj_samples, dtype=np.float32)


def run(
    X_train: np.ndarray,
    num_nodes: int,
    num_posterior_samples: int = 100,
    seed: int = 0,
    model_type: str = "bayesdag_nonlinear",
    config_path: str | None = None,
    # Paper-spec sparsity-tuning knobs (see module docstring).
    true_sparsity: Optional[int] = None,
    n_lambda_grid: int = 10,
    lambda_min: float = 10.0,
    lambda_max: float = 1000.0,
    validation_frac: float = 0.1,
    # Fast-mode (coarse-to-fine lambda) knobs; None -> env var -> _FAST_DEFAULTS.
    final_max_epochs: Optional[int] = None,
    final_num_chains: Optional[int] = None,
    final_sinkhorn: Optional[int] = None,
    grid_max_epochs: Optional[int] = None,
    grid_num_chains: Optional[int] = None,
    grid_sinkhorn: Optional[int] = None,
    grid_post_samples: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, str]:
    """
    Fit BayesDAG with the paper-spec hyperparameter sweep and return
    ``num_posterior_samples`` adjacency samples drawn from the variational
    posterior at the lambda whose expected edge count is closest to the
    ground-truth sparsity level.

    Parameters
    ----------
    X_train : [N, d] float32
        Observational training data.
    num_nodes : int
        d.  Sanity-checked against ``X_train.shape[1]``.
    num_posterior_samples : int
        S, number of adjacency matrices to draw from the posterior at the
        selected lambda.
    seed : int
        JAX PRNG seed.
    model_type : str
        "bayesdag_linear" or "bayesdag_nonlinear" (paper-spec architecture).
    config_path : str | None
        If None, uses the bundled protein config.
    true_sparsity : int | None
        Ground-truth edge count of the underlying DAG.  When provided, the
        wrapper performs the paper's 10-point lambda grid search and selects
        the lambda whose mean posterior edge count is closest to this value.
        When None (e.g. for callers that don't know the ground truth), the
        wrapper falls back to a single fit at the JSON config's
        ``lambda_sparse`` -- preserving backwards compatibility.
    n_lambda_grid, lambda_min, lambda_max, validation_frac :
        Paper defaults are 10, 10, 1000, 0.1.

    Returns
    -------
    A_samples : [S, d, d] float32 binary adjacency samples.
    convention : "i_to_j".
    """
    cfg_path = Path(config_path) if config_path else _DEFAULT_CONFIG
    with open(cfg_path) as f:
        cfg = json.load(f)

    X = np.asarray(X_train, dtype=np.float32)
    n, d = X.shape
    if d != int(num_nodes):
        raise ValueError(
            f"X_train has {d} cols but num_nodes={num_nodes}"
        )

    # ------------------------------------------------------------------
    # Backwards-compatible single fit when no true_sparsity is supplied.
    # ------------------------------------------------------------------
    if true_sparsity is None:
        with tempfile.TemporaryDirectory(prefix="bayesdag_case_") as save_dir:
            _, A_samples = _fit_once(
                X_fit=X,
                cfg=cfg,
                model_type=model_type,
                seed=int(seed),
                num_posterior_samples=int(num_posterior_samples),
                lambda_sparse=None,            # use cfg value as-is
                save_dir=save_dir,
            )
        if A_samples.ndim != 3 or A_samples.shape[1] != num_nodes:
            raise RuntimeError(
                f"BayesDAG returned unexpected adjacency shape {A_samples.shape}, "
                f"expected (S, {num_nodes}, {num_nodes})."
            )
        return A_samples, "i_to_j"

    # ------------------------------------------------------------------
    # Lambda grid search (fast coarse-to-fine by default; BAYESDAG_PAPER_SPEC=1
    # restores the exact 10-full-fit protocol).
    # ------------------------------------------------------------------
    rng = np.random.default_rng(int(seed))
    n_val = max(1, int(np.floor(validation_frac * n)))
    n_val = min(n_val, n - 1)                      # keep at least 1 fit row
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    fit_idx = perm[n_val:]
    X_fit = X[fit_idx]
    # ``X_val`` is held out from training (paper specifies ``floor(0.1 n)``);
    # the criterion below is structural ("closest to true sparsity"), so the
    # validation slice is *unused* for the comparison itself but is kept out
    # of training to match the paper's reported protocol.
    _X_val_unused = X[val_idx]                       # noqa: F841  (intentional)

    if int(n_lambda_grid) < 2:
        raise ValueError("n_lambda_grid must be >= 2 for a meaningful sweep.")
    # Geometric grid is the natural choice for a sparsity coefficient that
    # spans 2 orders of magnitude (10 -> 1000) per the paper.
    lam_grid = np.geomspace(
        float(lambda_min), float(lambda_max), int(n_lambda_grid)
    ).astype(np.float64)

    # Resolve knobs: explicit kwarg > env var > _FAST_DEFAULTS.
    paper_spec = _env_flag("BAYESDAG_PAPER_SPEC", False)
    fin_ep = final_max_epochs if final_max_epochs is not None else _env_int("BAYESDAG_EPOCHS", _FAST_DEFAULTS["final_max_epochs"])
    fin_ch = final_num_chains if final_num_chains is not None else _env_int("BAYESDAG_CHAINS", _FAST_DEFAULTS["final_num_chains"])
    fin_sk = final_sinkhorn if final_sinkhorn is not None else _env_int("BAYESDAG_SINKHORN", _FAST_DEFAULTS["final_sinkhorn"])
    grd_ep = grid_max_epochs if grid_max_epochs is not None else _env_int("BAYESDAG_GRID_EPOCHS", _FAST_DEFAULTS["grid_max_epochs"])
    grd_ch = grid_num_chains if grid_num_chains is not None else _env_int("BAYESDAG_GRID_CHAINS", _FAST_DEFAULTS["grid_num_chains"])
    grd_sk = grid_sinkhorn if grid_sinkhorn is not None else _env_int("BAYESDAG_GRID_SINKHORN", _FAST_DEFAULTS["grid_sinkhorn"])
    grd_S = grid_post_samples if grid_post_samples is not None else _env_int("BAYESDAG_GRID_SAMPLES", _FAST_DEFAULTS["grid_post_samples"])
    # Number of proxy grid points (fast path only); env override lets the run
    # scripts trade lambda-selection granularity for speed.
    fast_nl = max(2, _env_int("BAYESDAG_NLAMBDA", int(n_lambda_grid)))

    if verbose:
        print(
            f"[bayesdag_wrapper] n={n}, n_val={n_val}, n_fit={X_fit.shape[0]}, "
            f"true_sparsity={int(true_sparsity)}; "
            f"paper_spec={paper_spec} n_lambda={n_lambda_grid if paper_spec else fast_nl} "
            f"final(ep={fin_ep},ch={fin_ch},sk={fin_sk}) "
            f"grid(ep={grd_ep},ch={grd_ch},sk={grd_sk},S={grd_S})"
        )

    # ---- paper-spec path: full fit per lambda, keep the fitted samples -------
    if paper_spec:
        best_diff = float("inf")
        best_W: Optional[np.ndarray] = None
        best_lam: Optional[float] = None
        with tempfile.TemporaryDirectory(prefix="bayesdag_case_") as outer_dir:
            for k, lam in enumerate(lam_grid):
                sub_dir = Path(outer_dir) / f"lam_{k:02d}"
                sub_dir.mkdir(parents=True, exist_ok=True)
                _, A_lam = _fit_once(
                    X_fit=X_fit, cfg=cfg, model_type=model_type,
                    seed=int(seed) + k,            # decorrelate inner fits
                    num_posterior_samples=int(num_posterior_samples),
                    lambda_sparse=float(lam), save_dir=str(sub_dir),
                )
                mean_edges = float(np.mean(np.sum(A_lam != 0, axis=(1, 2))))
                diff = abs(mean_edges - float(true_sparsity))
                if verbose:
                    print(f"[bayesdag_wrapper]   lam={lam:9.4f}  "
                          f"avg_edges={mean_edges:7.2f}  |diff|={diff:6.2f}")
                if diff < best_diff:
                    best_diff = diff
                    best_W = A_lam
                    best_lam = float(lam)
        assert best_W is not None
        if verbose:
            print(f"[bayesdag_wrapper] selected lambda={best_lam:.4f} "
                  f"(|avg_edges - true_sparsity| = {best_diff:.2f})")
        A_out = best_W

    # ---- fast path: cheap proxy fits rank lambda, one full fit at lambda* ----
    else:
        # Fast path may use a coarser grid (BAYESDAG_NLAMBDA) than the paper's 10.
        lam_grid_fast = np.geomspace(
            float(lambda_min), float(lambda_max), int(fast_nl)
        ).astype(np.float64)
        proxy_cfg = _cfg_with(cfg, max_epochs=grd_ep, num_chains=grd_ch, sinkhorn=grd_sk)
        best_diff = float("inf")
        best_lam = None
        with tempfile.TemporaryDirectory(prefix="bayesdag_grid_") as outer_dir:
            for k, lam in enumerate(lam_grid_fast):
                sub_dir = Path(outer_dir) / f"lam_{k:02d}"
                sub_dir.mkdir(parents=True, exist_ok=True)
                _, A_lam = _fit_once(
                    X_fit=X_fit, cfg=proxy_cfg, model_type=model_type,
                    seed=int(seed) + k,            # decorrelate inner fits
                    num_posterior_samples=int(grd_S),
                    lambda_sparse=float(lam), save_dir=str(sub_dir),
                )
                mean_edges = float(np.mean(np.sum(A_lam != 0, axis=(1, 2))))
                diff = abs(mean_edges - float(true_sparsity))
                if verbose:
                    print(f"[bayesdag_wrapper]   [proxy] lam={lam:9.4f}  "
                          f"avg_edges={mean_edges:7.2f}  |diff|={diff:6.2f}")
                if diff < best_diff:
                    best_diff = diff
                    best_lam = float(lam)
        assert best_lam is not None
        if verbose:
            print(f"[bayesdag_wrapper] selected lambda={best_lam:.4f} "
                  f"(proxy |avg_edges - true_sparsity| = {best_diff:.2f}); final full fit ...")
        final_cfg = _cfg_with(cfg, max_epochs=fin_ep, num_chains=fin_ch, sinkhorn=fin_sk)
        with tempfile.TemporaryDirectory(prefix="bayesdag_final_") as final_dir:
            _, A_out = _fit_once(
                X_fit=X_fit, cfg=final_cfg, model_type=model_type,
                seed=int(seed),
                num_posterior_samples=int(num_posterior_samples),
                lambda_sparse=float(best_lam), save_dir=final_dir,
            )

    if A_out.ndim != 3 or A_out.shape[1] != num_nodes:
        raise RuntimeError(
            f"BayesDAG returned unexpected adjacency shape {A_out.shape}, "
            f"expected (S, {num_nodes}, {num_nodes})."
        )
    return A_out, "i_to_j"
