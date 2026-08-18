# Component ablation for SVI-DAG

Produces `ablation_table.tex`: a five-column ablation of SVI-DAG's components —
domain-informed prior, normalizing-flow posterior `q(γ|r)`, and SVGD
inference over order potentials `r` — removed one at a time, plus the
flow+SVGD joint ablation.

Two studies feed the table:

| study | setting | feeds |
|---|---|---|
| **main** (`run_ablation.py`) | nonlinear ER, p=20, s=40, n=300 (240 train / 60 holdout), 2000 iters, sampling bias 0, 10 seeds, S=1000 | Brier, E-SHD, E-F1, AUROC |
| **MEC companion** (`run_mec_study.py`) | linear ER, p=10, s=10, n=1000, 1500 iters, profile bias (−1), 10 seeds, S=1000 | MEC-cov |

**Reproduce everything:** `bash submit_ablation.sh` (submits all Slurm jobs +
a dependent merge/table job). Rebuild the table from existing results only:
`python make_table.py`.

## Variants

| key                | prior              | q(γ)                    | inference over r |
|--------------------|--------------------|--------------------------|------------------|
| `full`             | domain-informed    | conditional flow q(γ\|r) | SVGD particles   |
| `no_flow`          | domain-informed    | mean-field N(μ,σ²), ⊥ r  | SVGD particles   |
| `no_svgd`          | domain-informed    | conditional flow q(γ\|r) | Gaussian guide, reparameterized |
| `no_prior`         | flat 0.5           | conditional flow q(γ\|r) | SVGD particles   |
| `no_flow_no_svgd`  | domain-informed    | mean-field N(μ,σ²), ⊥ r  | Gaussian guide, reparameterized |

**Domain-informed prior.** The *sparsity-matched* Logistic-Beta prior
`p0 = s/(p(p-1))` on every ordered pair (computed from the run's own p and
s), concentration `SVIDAG_PRIOR_NU`; it encodes "graphs this size have about
s edges". `no_prior` uses the flat noninformative 0.5 matrix instead.

## How the variants are implemented

* `no_flow`: `SVIDAG_FLOW_TYPE=meanfield` routes `svidag.model`'s
  `create_flow_stack` call to `ablation_lib.MeanFieldGamma`, a drop-in module
  with the flow interface `(z, cond) -> (γ, log_det)`: `γ = μ + σ ⊙ z`,
  `log_det = Σ log σ`, `cond` ignored — that independence IS the ablation.
* `no_svgd`: `ablation_lib.train_gaussian_r` mirrors the stock trainer
  step-for-step but replaces the particle cloud with a reparameterized
  Gaussian guide `q(r) = N(μ_r, diag σ_r²)` trained jointly by the same Adam,
  with the analytic `KL(q(r)‖p(r))` added to the objective.
* Everything else (dataset generation, standardisation, biased hard-DAG
  sampler, metric code) is imported from `case_3/` so the ablation cannot
  drift from the benchmark pipeline.

## MEC-cov (column 5)

`mec.py` enumerates the exact Markov equivalence class of the true linear
DAG (CPDAG + brute-force orientation of reversible edges), then MEC-cov =
fraction of members matched *exactly* (all p(p−1) edge decisions) by ≥1 of
the 1000 posterior samples; the table reports mean ± s.e. over seeds.
Only flow-bearing variants score nonzero — exact whole-graph hits require
joint (correlated) posterior mass that mean-field edges cannot concentrate.

**p=10 is load-bearing.** At p=5 the metric inverts (measured: mean-field
0.58 vs full 0.08 — with only 20 slots, sharp independent marginals win the
exact-hit race), and at p=20 no variant ever lands an exact member. p≈10 is
the window where exact recovery isolates joint expressiveness. Never compare
MEC-cov across different p.

## Cluster gotchas (both cost real debugging time)

* Stock-variant fits at p≥20 need `--mem=200G`: XLA's compile of the train
  step exceeds 64G host RAM, and an undersized cgroup produces a silent
  D-state stall (memory.high throttling), not a clean OOM kill.
* The Gaussian-r train step stalls XLA:GPU compilation indefinitely (fused
  and split-pass forms both; autotune settings irrelevant) but compiles and
  runs fine on CPU — run `no_svgd` / `no_flow_no_svgd` on CPU nodes, in
  seed chunks (see `submit_ablation.sh`), merged by `merge_chunks.py`.

## Files

    run_ablation.py      main-study driver (one variant, CLI-configurable)
    run_mec_study.py     MEC companion driver
    ablation_lib.py      variant seams: MeanFieldGamma, train_gaussian_r, metrics
    mec.py               exact MEC enumeration + coverage (self-test: python mec.py)
    ablation_cell.sbatch Slurm template for one main-study cell
    submit_ablation.sh   full reproduction pipeline
    merge_chunks.py      assemble CPU chunk outputs into canonical JSONs
    make_table.py        render ablation_table.tex from results/
    results/             canonical per-variant JSONs + mec_study/ + chunks/
