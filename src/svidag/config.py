'''
Configuration module for the SVIDAG.

Central location for all hyperparameters and experimental settings.
Modify these values to adjust model architecture, training dynamics,
and experimental scenarios.

Author: Shrenik Zinage
'''

import os

# =============================================================================
# Training Configuration
# =============================================================================
seed = 0                # Random seed for reproducibility
num_iters = 60000       # Total training iterations
batch_size = 64        # Mini-batch size for SGD
print_every = 100       # Logging frequency (iterations)
lr = 1e-4               # Adam learning rate for model parameters
grad_clip = 0.1         # Global gradient norm clipping

# NOTE (implementation details beyond the paper's Algorithm 1, kept for
# numerical robustness): global grad-norm clipping (above), edge-logit
# clipping to ±15 before the sigmoid, SVGD score/update nan_to_num + clip
# ±10 and particle clamp ±5 (train.py), softplus σ floors (+1e-6), ELBO
# early stopping with best-state restore (runner / case scripts), and
# mean-centering of the order potentials r (shift-invariant under the
# Sinkhorn construction; affects only the flow's conditioning input).

# =============================================================================
# Observation Noise Settings
# =============================================================================
# Reference observation noise scale. When likelihood-noise learning is enabled,
# this is the prior center (and intuitive scale knob) in normalized space.
# When disabled, it is used as the fixed Gaussian likelihood noise.
obs_noise_scale = 0.1
learn_likelihood_noise = True
prior_log_noise_sigma = 2.0

obs_noise_scales = None  # Set automatically at dataset creation time (do not edit)

# =============================================================================
# Data Configuration
# =============================================================================
n_test_samples = 100     # Number of samples held out for testing

# =============================================================================
# Bayesian Neural Network (Node Models) Settings
# =============================================================================
hidden_dim = 32         # Hidden layer width for node MLPs
prior_theta_mu = 0.0    # Prior mean for network weights
prior_theta_sigma = 1.0 # Prior std for network weights (controls regularization)

# =============================================================================
# Normalizing Flow Configuration (for q(γ|r))
# =============================================================================
flow_hidden = None      # Auto-set per fit to [num_nodes, num_nodes] in
                        # ``svidag.train.make_model_and_state``.  This value
                        # is kept as a placeholder fallback only -- the
                        # benchmarks always override it based on the actual
                        # graph size.
flow_n_blocks = 5       # Number of flow transformation blocks

# Flow architecture selection
# Options: 'maf' (Masked Autoregressive Flow - affine transforms)
#          'nsf' (Neural Spline Flow - autoregressive splines, most flexible)
#          'nsf_coupling' (Neural Spline Flow - coupling architecture, faster)
flow_type = 'nsf_coupling'   # fast coupling variant (used across all cases)

# Neural Spline Flow specific settings
nsf_num_bins = 8        # Spline bins (more = more expressive, slower)
nsf_tail_bound = 5.0    # Domain boundary; identity transform outside [-B, B] (paper: 5)

# =============================================================================
# Temperature Annealing (for differentiable relaxations)
# =============================================================================
# Concrete relaxation temperature for edge sampling (lower = sharper)
T_B_start = 0.3         # Initial temperature
T_B_end = 0.3          # Final temperature (can anneal from high to low)

# Sinkhorn temperature for soft permutation P_τ(r) (lower = closer to hard perm)
tau_sink_start = 0.1
tau_sink_end = 0.1

# Fraction of training over which tau_sink travels from start to end; it is
# held at ``tau_sink_end`` afterwards.  1.0 reproduces the original schedule
# (anneal across the whole run).
#
# This exists so a soft-mask warm-up can actually END. With a full-length
# linear anneal, tau is still ~12 at 40% of the run -- M_tau is nearly constant
# in r there, so the ordering gradient is ~0 and the search only gets the last
# few hundred iterations. Setting this below st_warmup_frac's complement gives
# the ordering search a properly sharp mask for most of the run.
tau_anneal_frac = 1.0

# Straight-through gradient estimator: use hard threshold in forward pass,
# but pass gradients through soft relaxation
straight_through_mask = True

# Fraction of training spent warming the straight-through estimator up.
#
# A_ST = A_relaxed + w * sg(A_hard - A_relaxed) with w annealed 0 -> 1 over the
# first ``st_warmup_frac`` of the run (w = 1 thereafter, i.e. the usual ST).
# Note the gradient is IDENTICAL for every w -- the stop-gradient term carries
# none -- so this anneals only the *value* the likelihood is evaluated at.
#
# Why it matters: with w = 1 from step 0 the likelihood only ever sees a hard
# DAG, so the edge logits can only be learned for pairs the current (random)
# ordering happens to permit, and the ordering can only be improved using edge
# logits that were never trained. Starting at w = 0 with a soft, near-uniform
# M_tau lets every node see every candidate parent, so gamma learns the parent
# sets first; by the time the mask hardens the ordering gradient has a trained
# gamma to act on. Pair with a tau_sink anneal (soft -> sharp).
st_warmup_frac = 0.0

# Re-seed the SVGD particles from the learned edge structure once, at this
# fraction of training (None = never, the original behaviour).
#
# The natural point is the end of the relaxation warm-up. By then the flow has
# learned which ordered pairs are worth an edge -- but it learned that under a
# soft mask, so the belief is generally cyclic and the particles are still at
# their random initialisation. Ranking nodes by (out-mass - in-mass) of the
# learned edge marginals turns that belief into a topological order, which is a
# far better starting point for the ordering search than N(0, sigma^2) noise.
# Particles are re-seeded around it with ``particle_warmstart_jitter`` of noise
# so SVGD keeps a spread of orderings to explore.
particle_warmstart_frac = None
particle_warmstart_jitter = 0.5

# ELBO-weighted resampling of the SVGD particles (an SMC-style selection move).
#
#   particle_resample_every -- resample every N iterations once the relaxation
#                              warm-up has finished (None = never).
#   particle_resample_temp  -- softmax temperature on the per-particle objective,
#                              in units of the objective's own spread across the
#                              cloud, so it does not need retuning per dataset.
#   particle_resample_jitter-- Gaussian noise (x prior_r_sigma) added after
#                              resampling so duplicated particles separate again.
#
# Why this and not a gradient fix: the landscape probe shows the objective
# RANKS orderings correctly once the structural equations have warmed up (the
# oracle sits at the 77th percentile of random orderings), but SVGD's score is
# too noisy and too local to climb that ranking -- Kendall tau stays ~0 and all
# K particles end on K distinct orderings. Resampling turns the ranking the
# objective already provides into selection pressure. Crucially it only ever
# reuses r values ALREADY in the cloud, so unlike ``particle_warmstart_frac``
# (which invents a fresh r from the edge marginals, and measurably hurt) it
# never pushes the flow's conditioning input out of distribution.
particle_resample_every = None
particle_resample_temp = 1.0
particle_resample_jitter = 0.1

# =============================================================================
# Sinkhorn Normalization Parameters
# =============================================================================
sinkhorn_iters = 300    # Iterations for doubly-stochastic projection
sinkhorn_eps = 1e-9     # Numerical stability constant
# Run the doubly-stochastic projection in log space (logsumexp) instead of
# probability space.  The probability-space iteration underflows to exact
# zeros once the order potentials grow (|S0/tau| ~ 250*||r|| at m=25,
# tau=0.1) and cannot recover, which silently destroys M_tau -- and with it
# the only gradient path from the likelihood to the edge logits.
# Set False to restore the original (numerically fragile) behaviour.
sinkhorn_log_space = True

# Normalise the order potentials to unit RMS before dividing by tau inside
# ``build_P_tau``.  The hard mask M(r) depends only on the *ranks* of r, so it
# is invariant to a positive rescaling; the relaxation M_tau is not.  Without
# normalisation the effective temperature is tau/rms(r) rather than tau, so as
# the particles drift outwards during training the relaxation silently sharpens
# and its gradient decays like 1/||r|| -- the ordering signal fades for reasons
# unrelated to the objective, and tau stops being a meaningful knob.  Enabling
# this leaves every hard quantity bit-identical.
sinkhorn_scale_invariant = False

# =============================================================================
# SVGD Settings (for posterior over order potentials r)
# =============================================================================
n_particles = 50        # Number of SVGD particles approximating p(r|data)
eta_r = 1e-3            # Step size for SVGD particle updates
# Prior std for r: r ~ N(0, prior_r_sigma² I).
#
# NOTE: the induced prior over node *orderings* is uniform for ANY sigma -- the
# ordering depends only on the ranks of r, and an isotropic Gaussian assigns
# equal mass to all m! rank patterns.  sigma therefore encodes no structural
# belief at all; it only sets how strongly particles are pulled toward the
# origin, which is exactly the region where the ordering is a coin flip
# (all |r_i| comparable to the sampling noise).  sigma=0.1 makes the prior
# score -r/sigma^2 = -100 r, which saturates the particle-gradient clip at
# |r|=0.1 and pins the particles in that undecided region.  A wider prior is
# structurally equivalent and lets the likelihood commit to an ordering.
prior_r_sigma = 1.0

# Hold the SVGD particles still for the first this-fraction of training
# (0.0 = never hold, the original behaviour).
#
# While the per-node structural equations are still at their initialisation,
# EVERY edge makes the likelihood worse, so the ordering score prefers whichever
# ordering DISALLOWS the edges the flow currently wants -- and the model then
# co-adapts to that ordering and stays there.  Measured on the 2-node case 1:
# a CORRECT prior drove P(x1 before x2) DOWN (0.45 -> 0.35) when the prior wants
# it UP.  The ordering gradient only means something once the node models can
# actually use a parent, so do not spend it before then.
eta_r_warmup_frac = 0.0

# Particle-gradient clipping mode for the SVGD score and update.
#   "elementwise" -- clip each coordinate to +/- particle_grad_clip (original).
#   "norm"        -- rescale each particle's gradient so its L2 norm is at most
#                    particle_grad_clip, preserving direction.
#
# The sum-form ELBO scales the likelihood by N/|B| (105x on Sachs kfold), so
# every coordinate of the raw particle score is far outside +/-10 and the
# elementwise clip collapses the score to a pure sign vector.  The ordering
# signal lives in the *relative* magnitudes across coordinates -- which
# coordinate should move most -- and elementwise clipping destroys exactly
# that.  Norm clipping bounds the step without changing its direction.
particle_grad_clip_mode = "norm"
particle_grad_clip = 10.0

# SVGD repulsion controls.
#
# ``compute_svgd_update`` sets the RBF bandwidth by the median heuristic,
# h = median(d^2)/log(K+1).  As the cloud agrees on an ordering the pairwise
# feature distances go to zero, h goes to zero with them, and the repulsive
# term grad_k = -(2/h) * k * diff DIVERGES -- measured |repulsion| 0.47 -> 19.7
# at m=2 as the spread shrinks 1.0 -> 0.01.  The attractive term meanwhile is
# hard-bounded (particle_grad_clip), so whenever the particles are about to
# reach consensus the repulsion wins by construction and pushes them apart
# again.  That is why every Sachs run ends with all K particles on K DISTINCT
# orderings and why case 1 never becomes decisive about direction.
#
#   svgd_repulsion_weight    -- plain scale on the repulsive term.
#   svgd_repulsion_max_ratio -- cap |repulsion| at this multiple of |attraction|
#                               per particle (None = uncapped, original).
# Capping by the attraction norm keeps the two terms in the same units, so the
# balance no longer depends on how tightly the cloud happens to be packed.
svgd_repulsion_weight = 1.0
svgd_repulsion_max_ratio = None

# Linearly decay the repulsion weight to zero over this fraction of training
# (None = never decay, the original behaviour).
#
# SVGD's repulsion encodes the posterior's spread, but with n = 6720 Sachs rows
# the true posterior over ORDERINGS is sharp, so a cloud that never agrees is
# under-converged inference rather than honest uncertainty.  It also caps the
# metrics mechanically: with K distinct orderings a true edge j->i is permitted
# in only the fraction of particles where j precedes i, so p(true edge) cannot
# exceed roughly that fraction (measured p_true ~ 0.25-0.35 learned vs 0.53-0.72
# with a single oracle ordering).  Annealing the repulsion away lets the cloud
# concentrate once the warm-up has made the ordering landscape informative.
svgd_repulsion_anneal_frac = None

# =============================================================================
# Structural-equation guide conditioning
# =============================================================================
# What the per-node parameter hypernetwork H_phi sees.
#
#   False -- every node's H_phi receives the FULL m x m adjacency (original).
#   True  -- node i's H_phi receives only row i of A, i.e. its own parent set.
#
# theta_i parameterises f_i : pa(i) -> x_i, so row i is exactly the part of A
# that theta_i may depend on; conditioning on the whole matrix lets node i's
# weights react to edges among other nodes entirely. That matters for the
# gradient, not just the fit: with full conditioning, dELL/dA_kl picks up a
# contribution from EVERY node's hypernetwork, so the clean per-edge signal
# ("does masking j into node i reduce node i's residual?") arrives buried in
# m-1 unrelated hypernetwork derivatives. Row conditioning also shrinks the
# hypernetwork input from m^2 to m, which matters because H_phi funnels it
# through a 16-unit bottleneck.
node_cond_row_only = False

# =============================================================================
# Domain-Informed Prior (Logistic-Beta) Settings
# =============================================================================
# Concentration ν = NU_MIN + PR_PREC_KAPPA * |p - 0.5|^PR_PREC_GAMMA
# Higher κ and γ = stronger prior influence when p is far from 0.5
NU_MIN = 1.0            # Minimum concentration (prevents very weak priors)
PR_PREC_KAPPA = 500.0    # Scaling factor for prior concentration
PR_PREC_GAMMA = 2.0     # Exponent controlling concentration growth

# =============================================================================
# ELBO Estimation Settings
# =============================================================================
# Optimise the PER-DATUM ELBO (mean batch log-lik minus KL/N) rather than the
# sum form (N/|B| * sum log-lik minus KL).  The two have the same maximiser,
# but the sum form's gradient grows like N, which the fixed grad-norm clip and
# the +/-10 particle-score clip then normalise away -- making the whole update
# n-independent and the metrics flat in sample size.  The per-datum form keeps
# gradients O(1) at every n and makes the prior's per-observation weight decay
# as 1/N, which is the correct Bayesian n-dependence.
elbo_per_datum = False   # measured WORSE and flatter in n than the sum form; see nsweep 11380918

# Weight on the parameter-space KL, KL(q(theta|A) || p(theta)).
# Measured term balance at n=10: ELL = -129, KL_gamma = 1567, KL_theta = 3126 --
# the data term is 4% of the objective.  ELL is the ONLY term that grows with
# n (it scales as N), so the likelihood does not overtake the priors until
# n ~ 400, which is precisely why the case-2 metrics are flat from n=10 to
# n=316 and only move at n=1000.  KL_theta is a weight-space regulariser that
# says nothing about graph structure, yet it is the single largest term.
# Downweighting it moves the crossover to smaller n.
kl_theta_weight = 1.0
# elbo_mc_samples = 20    # MC samples per particle for ELBO gradient estimation
# elbo_mc_samples = 10    # MC samples per particle for ELBO gradient estimation
elbo_mc_samples = 1     # Minimal MC: a single reparameterized sample is an unbiased
                        # ELBO-gradient estimator, and the ELBO is already averaged over
                        # n_particles=50 SVGD particles, so 1 sample/particle approximates
                        # it well while ~10x reducing per-step model forward passes.
ELBO_MC_SAMPLES = max(1, int(elbo_mc_samples))

# =============================================================================
# Experimental Scenarios
# =============================================================================
# Each scenario represents a different domain-informed prior setting
scenarios = [
    "noninformative",    # Uniform 0.5 prior (no domain knowledge)
    "strong_correct",    # High confidence, correct edges (0.99/0.01)
    "strong_incorrect",  # High confidence, wrong direction (adversarial)
]

# =============================================================================
# Posterior Sampling Configuration
# =============================================================================
# Posterior samples are hard DAGs A = B ⊙ M(r) drawn from the generative
# construction directly (binary and acyclic by construction) — no relaxed-
# adjacency thresholding is involved.
posterior_samples_A = 2000  # Samples for adjacency statistics (mean, std)
dag_prob_samples = 1000     # Samples for DAG/CPDAG probability estimation

# =============================================================================
# Output Settings
# =============================================================================
plot_dir = "plots_svidag"
os.makedirs(plot_dir, exist_ok=True)

# =============================================================================
# Benchmark Graph Settings (Erdos-Renyi / Scale-Free)
# =============================================================================
# Use with generate_benchmark_dataset() for standardized DAG learning benchmarks.

# Graph structure:
bench_num_nodes = 10         # Number of nodes m in the benchmark graph
bench_graph_type = 'er'      # Graph model: 'er' (Erdos-Renyi) or 'sf' (Scale-Free)
bench_edge_prob = 0.3        # Edge probability p for ER graphs (expected edges ≈ m*(m-1)/2 * p)
bench_sf_num_edges = 2       # Edges per new node k for SF (Barabasi-Albert) graphs

# Structural equation model:
bench_sem_type = 'linear'    # SEM type: 'linear', 'nonlinear'/'mlp', or 'mim'
bench_noise_scale = 1.0      # Noise standard deviation s in the SEM
bench_weight_range = (0.3, 0.7)  # Edge weight magnitude range [w_min, w_max] (linear SEM). Paper spec: U([-0.7,-0.3] U [0.3,0.7])

# Data generation:
bench_num_samples = 1000     # Total samples to generate (split into train + test)
bench_rng_seed = 0           # Random seed for graph + data generation
