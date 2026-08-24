#!/usr/bin/env python
"""
Reproduce the table's MEC-cov companion study.

    python run_mec_study.py --variants full,no_flow,no_prior --seeds 0-9
    python run_mec_study.py --variants no_svgd,no_flow_no_svgd --seeds 0-9

Linear-Gaussian ER graphs at p=10, s=10, n=1000, 1500 iterations, S=1000
posterior samples, with the case-3 profile's sampling-time logit bias LEFT AT
ITS DEFAULT (-1.0): do NOT export SVIDAG_POSTERIOR_BIAS_INTERCEPT=0 for this
study -- removing the bias was measured to drive exact-member coverage to
0.000 for every variant.

Writes results/mec_study/ablation_linear_mec_<variant>.json, the files
make_table.py reads for the MEC-cov column.  Seed derivation, dataset labels
and fitting go through the same ablation_lib.fit_and_sample as the main
study, so reruns reproduce the committed values up to GPU nondeterminism.

NOTE: the Gaussian-r variants (no_svgd, no_flow_no_svgd) must run on CPU
nodes -- their train step stalls XLA:GPU compilation indefinitely (see
ablation_lib._make_gauss_step).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent

P, S_EDGES, N_TRAIN, NUM_ITERS, PSAMP = 10, 10, 1000, 1500, 1000


def _parse_seeds(spec: str):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(",")]


def _variant_env(variant: str) -> None:
    """Per-variant flow/prior environment (sparsity prior mode)."""
    os.environ["SVIDAG_FLOW_TYPE"] = (
        "meanfield" if variant in ("no_flow", "no_flow_no_svgd") else "nsf_coupling")
    if variant == "no_prior":
        os.environ.pop("SVIDAG_PRIOR_P0", None)
    else:
        os.environ["SVIDAG_PRIOR_P0"] = f"{S_EDGES / (P * (P - 1)):.6f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True,
                    help="comma-separated subset of the 5 variants")
    ap.add_argument("--seeds", default="0-9")
    args = ap.parse_args()

    import numpy as np
    from sklearn.preprocessing import StandardScaler

    import ablation_lib as lib
    import common as c3
    import mec

    out_dir = _THIS_DIR / "results" / "mec_study"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_seeds(args.seeds)

    for variant in args.variants.split(","):
        if variant not in lib.VARIANTS:
            raise SystemExit(f"unknown variant {variant!r}")
        _variant_env(variant)
        out_path = out_dir / f"ablation_linear_mec_{variant}.json"

        rows = []
        if out_path.exists():
            try:
                prev = json.loads(out_path.read_text())
                if (prev.get("p"), prev.get("s"), prev.get("n_train"),
                        prev.get("num_iters")) == (P, S_EDGES, N_TRAIN, NUM_ITERS):
                    rows = [r for r in prev.get("rows", []) if r.get("seed") in seeds]
            except (json.JSONDecodeError, OSError):
                pass
        done = {r["seed"] for r in rows}
        print(f"[mec-study] {variant}: seeds {seeds}, {len(done)} already done",
              flush=True)

        for seed in seeds:
            if seed in done:
                continue
            t0 = time.time()
            sc = c3.GraphScenario(
                label=f"ablation_ER_p{P}_s{S_EDGES}_linear",
                p=P, s=S_EDGES, sem_type="linear")
            ds = c3.generate_dataset(sc, num_samples=N_TRAIN, replicate=seed)
            Xs = StandardScaler().fit_transform(ds.X).astype(np.float32)

            A, _extras = lib.fit_and_sample(
                variant=variant, X_scaled=Xs, true_adj=ds.true_adj,
                node_names=ds.node_names, scenario="noninformative",
                cell_index=seed, num_posterior_samples=PSAMP,
                num_iters=NUM_ITERS, seed=0, verbose=False)

            row = {"seed": seed}
            row.update(c3.evaluate_samples(A, ds, source_convention="j_to_i"))
            row.update(mec.mec_coverage(np.asarray(A), ds.true_adj))
            row["time_sec"] = time.time() - t0
            rows.append(row)
            print(f"[mec-study]   {variant} seed {seed}: |MEC|={row['mec_size']:.0f}"
                  f" cov={row['mec_cov']:.2f} mass={row['mass_in_mec']:.4f}"
                  f"  ({row['time_sec']:.0f}s)", flush=True)

            payload = {
                "study": "linear_mec", "variant": variant,
                "p": P, "s": S_EDGES, "n_train": N_TRAIN,
                "posterior_samples": PSAMP, "num_iters": NUM_ITERS,
                "posterior_bias_intercept":
                    os.environ.get("SVIDAG_POSTERIOR_BIAS_INTERCEPT", "(profile -1.0)"),
                "rows": rows,
            }
            out_path.write_text(json.dumps(payload, indent=2))
        print(f"[mec-study] wrote {out_path} ({len(rows)} seeds)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
