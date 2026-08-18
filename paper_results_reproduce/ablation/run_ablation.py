#!/usr/bin/env python
"""
Run one variant of the SVI-DAG component ablation.

    python run_ablation.py --variant full            --seeds 0-9
    python run_ablation.py --variant no_flow_no_svgd --seeds 0-9 --p-nodes 20 --s-edges 40

Nonlinear ER graphs, MLP SEM; reports Brier / E-SHD / E-F1 / AUROC per seed.
Graph size, edge count, training-set size and iteration budget are all CLI
knobs so the same driver serves both configuration sweeps and the final run.

Environment: source profiles/case3.env first (the sbatch script does).  This
driver then applies the per-variant overrides BEFORE any training:

    SVIDAG_FLOW_TYPE  = meanfield       for no_flow / no_flow_no_svgd
    SVIDAG_PRIOR_P0   = s/(p(p-1))      for prior-informed variants
                        (unset for no_prior, which gets the flat 0.5 matrix)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent

STUDY = "nonlinear"   # kept in the payload for result-file compatibility


def _parse_seeds(spec: str):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(",")]


def _configure_variant_env(variant: str, p_nodes: int, s_edges: int) -> str:
    """Set flow/prior env for this variant; return the scenario string."""
    # Flow: meanfield for the flow-ablated variants, profile value otherwise.
    if variant in ("no_flow", "no_flow_no_svgd"):
        os.environ["SVIDAG_FLOW_TYPE"] = "meanfield"
    # else: leave whatever profiles/case3.env exported (nsf_coupling).

    if variant == "no_prior":
        # Flat noninformative 0.5: scenario "noninformative" with NO p0
        # override (case_3's _build_prior only builds the sparsity prior when
        # SVIDAG_PRIOR_P0 is set).
        os.environ.pop("SVIDAG_PRIOR_P0", None)
        return "noninformative"

    # Sparsity-matched domain prior on every ordered pair, computed from the
    # run's own graph so the prior always encodes the right density.
    p0 = s_edges / (p_nodes * (p_nodes - 1))
    os.environ["SVIDAG_PRIOR_P0"] = f"{p0:.6f}"
    return "noninformative"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--p-nodes", type=int, default=10)
    ap.add_argument("--s-edges", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--posterior-samples", type=int, default=1000)
    ap.add_argument("--num-iters", type=int,
                    default=int(os.environ.get("SVIDAG_NUM_ITERS", "1500")))
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    scenario = _configure_variant_env(args.variant, args.p_nodes, args.s_edges)

    # Import AFTER the env is final: ablation_lib installs the meanfield shim
    # and drags in jax/case_3 modules whose config reads happen at fit time,
    # but keeping every env decision above the imports removes ordering traps.
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    import ablation_lib as lib
    import common as c3          # case_3's common (path set by ablation_lib)

    if args.variant not in lib.VARIANTS:
        raise SystemExit(f"unknown variant {args.variant!r}; pick from {lib.VARIANTS}")

    scen_obj = c3.GraphScenario(
        label=f"ablation_ER_p{args.p_nodes}_s{args.s_edges}_nonlinear",
        p=args.p_nodes, s=args.s_edges, sem_type="nonlinear",
    )

    seeds = _parse_seeds(args.seeds)
    out_path = Path(args.out) if args.out else (
        _THIS_DIR / "results" / f"ablation_{STUDY}_{args.variant}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ablation] variant={args.variant} scenario={scenario}"
          f" flow={os.environ.get('SVIDAG_FLOW_TYPE', '(profile)')}"
          f" p0={os.environ.get('SVIDAG_PRIOR_P0', '(none)')}")
    print(f"[ablation] p={args.p_nodes} s={args.s_edges} n={args.n_train}"
          f" S={args.posterior_samples} iters={args.num_iters} seeds={seeds}")

    # Resume support: mit_preemptable requeues restart the process from the
    # top, so pick up whatever seeds the previous incarnation already wrote.
    rows = []
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            # The config fingerprint must match exactly -- resuming across a
            # changed graph size / budget would silently mix incomparable rows.
            fingerprint = ("variant", "p", "s", "n_train",
                           "posterior_samples", "num_iters", "metrics_version")
            ours = (args.variant, args.p_nodes, args.s_edges, args.n_train,
                    args.posterior_samples, args.num_iters, 3)
            if tuple(prev.get(k) for k in fingerprint) == ours:
                rows = [r for r in prev.get("rows", []) if r.get("seed") in seeds]
                if rows:
                    print(f"[ablation] resuming: {sorted(r['seed'] for r in rows)} already done")
        except (json.JSONDecodeError, OSError):
            pass
    done_seeds = {r["seed"] for r in rows}

    for seed in seeds:
        if seed in done_seeds:
            continue
        t0 = time.time()
        ds = c3.generate_dataset(scen_obj, num_samples=args.n_train, replicate=seed)
        # Hold out the last 20% for predictive scoring.  Rows are iid draws
        # from the SEM, so a tail split is an unbiased holdout; the scaler is
        # fit on the TRAINING rows only (no leakage into the test density).
        n_tr = int(round(args.n_train * 0.8))
        scaler = StandardScaler().fit(ds.X[:n_tr])
        X_scaled = scaler.transform(ds.X[:n_tr]).astype(np.float32)
        X_test = scaler.transform(ds.X[n_tr:]).astype(np.float32)

        A, extras = lib.fit_and_sample(
            variant=args.variant, X_scaled=X_scaled, true_adj=ds.true_adj,
            node_names=ds.node_names, scenario=scenario, cell_index=seed,
            num_posterior_samples=args.posterior_samples,
            num_iters=args.num_iters, seed=0,
            verbose=(args.verbose and seed == seeds[0]), X_test=X_test,
        )

        row = {"seed": seed}
        row.update(c3.evaluate_samples(A, ds, source_convention="j_to_i"))
        row.update(extras)
        row["time_sec"] = time.time() - t0
        rows.append(row)

        print(f"[ablation]   seed {seed}: E_SHD={row['E_SHD']:.2f}"
              f" E_F1={row['E_F1']:.3f} AUROC={row['AUROC']:.3f}"
              f" Brier={row['Brier']:.4f}"
              f" predLL={row.get('pred_ll', float('nan')):.2f}"
              f"  ({row['time_sec']:.0f}s)",
              flush=True)

        # Rewrite after every seed so a preempted job loses one seed, not all.
        payload = {
            "metrics_version": 3,
            "study": STUDY, "variant": args.variant, "scenario": scenario,
            # Retained as a literal so the payload still matches the schema of
            # the committed result JSONs, which all record "sparsity".
            "prior_mode": "sparsity", "p": args.p_nodes, "s": args.s_edges,
            "n_train": args.n_train, "posterior_samples": args.posterior_samples,
            "num_iters": args.num_iters,
            "flow_type": os.environ.get("SVIDAG_FLOW_TYPE", "(profile)"),
            "prior_p0": os.environ.get("SVIDAG_PRIOR_P0"),
            "posterior_bias_intercept":
                os.environ.get("SVIDAG_POSTERIOR_BIAS_INTERCEPT", "(profile)"),
            "rows": rows,
        }
        out_path.write_text(json.dumps(payload, indent=2))

    print(f"[ablation] wrote {out_path}  ({len(rows)} seeds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
