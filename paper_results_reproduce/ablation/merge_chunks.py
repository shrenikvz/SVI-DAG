#!/usr/bin/env python
"""Merge per-chunk result JSONs (results/chunks/<variant>_s*.json) into the
canonical results/ablation_nonlinear_<variant>.json the table reads.
Chunks exist because seed-parallel Slurm jobs must not share one output file.
Refuses to merge chunks whose config fingerprints disagree."""
import json, glob, sys
from pathlib import Path
R = Path(__file__).resolve().parent / "results"
FP = ("variant", "p", "s", "n_train", "posterior_samples", "num_iters", "metrics_version")
for variant in ("no_svgd", "no_flow_no_svgd"):
    chunks = sorted(glob.glob(str(R / "chunks" / f"{variant}_s*.json")))
    if not chunks:
        continue
    payloads = [json.loads(Path(c).read_text()) for c in chunks]
    fps = {tuple(p.get(k) for k in FP) for p in payloads}
    if len(fps) != 1:
        sys.exit(f"chunk fingerprints disagree for {variant}: {fps}")
    rows = {r["seed"]: r for p in payloads for r in p["rows"]}
    merged = dict(payloads[0], rows=[rows[s] for s in sorted(rows)])
    (R / f"ablation_nonlinear_{variant}.json").write_text(json.dumps(merged, indent=2))
    print(f"merged {variant}: {sorted(rows)} from {len(chunks)} chunks")
