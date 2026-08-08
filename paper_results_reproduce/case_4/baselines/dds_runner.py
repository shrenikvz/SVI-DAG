#!/usr/bin/env python
"""
DDS (VI-DP-DAG) trainer -- dataloader-free wrapper around the autoencoder path.
===============================================================================

The "DDS" row in the paper refers to VI-DP-DAG (Charpentier et al.),
specifically the *autoencoder* pipeline in
``vi-dp-dag/src/probabilistic_dag_model/train_probabilistic_dag_autoencoder.py``
-- NOT ``train_dag.py``.  ``train_dag.train`` minimises L2 between the
sampled DAG and a ground-truth adjacency; it never consumes observational
data and is therefore not the published observational-data algorithm.  The
autoencoder pipeline (``ProbabilisticDAGAutoencoder`` + the masked-AE
likelihood) is the published observational-data algorithm.

This module re-uses VI-DP-DAG's own ``train_autoencoder`` training loop
and ``ArrayDataLoader`` *verbatim*; we just construct the loaders from a
numpy ``X_train`` rather than via ``DAGDataset.sachs`` / ``get_dag_dataset``.

Adjacency convention
--------------------
``ProbabilisticDAG.sample`` returns ``dag_adj = permutation_inv @ mask @
permutation`` where ``mask`` is upper-triangular.  After the permutation
conjugation, ``dag_adj[i, j] = 1`` means ``i -> j``.  We return ``"i_to_j"``
(independent of ``pd_order_type``: both orderings apply the same permutation
conjugation).

Convergence fix (2026-07-23): corrected sparsity regularizer + config
--------------------------------------------------------------------
With this module's original defaults (``regr=0``, ``pd_order_type="sinkhorn"``)
DDS returned a *near-complete* DAG at every sample size -- E[SHD] pinned at
~#possible-edges (≈190 for p=25), E[F1]≈0.22, flat in n -- while AUROC still
crept up with n.  Two problems, both fixed here:

1. The sparsity term is absent, and the vendored one is *backwards*.  The
   published ELBO adds ``regr * KLDivLoss(edge_log_params, prior_p)`` only when
   ``regr>0`` (the paper default is 0, so there is no sparsity pressure at all,
   and edge probs sit at their ``uniform(0,1)`` init, ``sigmoid≈0.62`` -> dense).
   Worse, ``torch.nn.KLDivLoss`` treats its first argument as *log*-probabilities
   but ``edge_log_params`` is a *logit* (edge prob = ``sigmoid(logit)``): the term
   reduces to a constant push that *raises* the logits and DENSIFIES the graph
   (measured: ``regr`` 0→1 took edges/sample 187→257 on ER_p25).  We replace it
   (via an instance-level override of ``_elbo_loss_with_params``; the vendored
   source is left untouched) with a proper Bernoulli KL between the edge
   posterior ``Bernoulli(sigmoid(logit))`` and a sparse prior
   ``Bernoulli(prior_p)``, averaged over off-diagonal entries.  This pushes
   unsupported edges toward ``prior_p`` while reconstruction holds genuine edges
   up, so the thresholded graph is sparse.

2. Config aligned with the authors' own notebook (``02-VI-DP-DAG.ipynb``):
   ``pd_order_type="topk"`` (SoftSort ranking; also avoids the removed
   ``jnp.logsumexp`` in the sinkhorn path), ``pd_lr=1e-2``,
   ``ma_hidden_dims=(16,16,16)``, ``max_epochs=100``, and ``regr=0.1``,
   ``prior_p=0.01`` (the sparsity strength; tuned on the ER_p25 grid).

Result on ER_p25_s40 (E[SHD], corrected vs original):
   n= 10   163  vs ~190      n=316    65  vs ~193
   n=100    97  vs ~193      n=1000  (decreasing)
with AUROC rising 0.56→0.85→0.96 for n=10/100/316.  E[SHD] now *decreases*
with n as expected of a converging estimator.
"""

from __future__ import annotations

import functools
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


_THIS_DIR = Path(__file__).resolve().parent
_CASE4_DIR = _THIS_DIR.parent
_REPO_ROOT = _CASE4_DIR.parent.parent
_VIDPDAG_ROOT = _REPO_ROOT / "other_algorithms" / "codes_jax" / "vi-dp-dag"
# ``train_probabilistic_dag_autoencoder`` uses absolute imports starting with
# ``src.`` -- so the root of sys.path must be the package parent.
if str(_VIDPDAG_ROOT) not in sys.path:
    sys.path.insert(0, str(_VIDPDAG_ROOT))


@functools.lru_cache(maxsize=1)
def _fast_model_class():
    """
    ``ProbabilisticDAGAutoencoder`` with the eager-dispatch overhead removed.

    Built lazily (the vendored package needs ``sys.path`` set up first) and
    cached, so the class object -- and therefore the ``jax.jit`` cache keyed on
    it -- is created once per process.

    Two fixes, both of which leave the vendored source untouched:

    1.  **Corrected sparsity regularizer.**  Previously installed on the
        instance with ``types.MethodType``; now a plain method override, which
        is what ``jax.jit`` can trace through cleanly.  Rationale unchanged --
        see fix item 1 in the module docstring.

    2.  **Jitted train/eval kernels.**  The vendored training loop runs
        ``update_mask(); model(X); step()`` per iteration, and every one of
        those ran in eager JAX, dispatching each primitive from Python.  For
        this model that is brutal: ``MaskedAutoencoder.apply`` loops over
        ``input_dim`` separate MLPs in Python (25 nodes x 4 layers = 100
        matmuls per forward), so a single step dispatched several hundred
        tiny kernels, and the same work was done THREE times --
        ``update_mask`` sampled a DAG whose only consumer is the forward in
        ``model(X)``, whose result ``step()`` then discards because its
        ``loss_fn`` re-derives both the mask and the loss to differentiate
        them.  Measured at ~0.45 s/iteration for p=25, roughly 6-20x the
        per-step cost of every other baseline in the benchmark and the entire
        reason DDS's wall clock stood out; because the loop runs
        ``max_epochs * ceil(0.8n / batch_size)`` iterations, that constant
        multiplied straight into the n-scaling.

        The redundant passes are dropped and the surviving one is compiled.
        The RNG stream is untouched: ``update_mask`` still draws its key from
        ``probabilistic_dag.next_key()`` in the same order, and ``step`` still
        re-derives the mask from that key so the permutation stays part of the
        differentiated graph.  Same algorithm, same updates, same sequence of
        random draws -- only the dispatch changes (XLA may fuse and reassociate
        float ops, so results agree to floating-point tolerance rather than
        bit-for-bit).

    Every path that is NOT the training loop -- ``pd_initial_adj`` set, a
    non-deterministic mask requested in eval, ``step`` called outside the
    training flow -- falls back to the vendored implementation.
    """
    import jax
    import jax.numpy as jnp

    from src.jax_utils import adam_update
    from src.probabilistic_dag_model.probabilistic_dag_autoencoder import (
        ProbabilisticDAGAutoencoder,
    )

    class _FastDDSAutoencoder(ProbabilisticDAGAutoencoder):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            ma, pd = self.mask_autoencoder, self.probabilistic_dag

            @jax.jit
            def _train_step(ma_params, pd_params, ma_state, pd_state, batch, mask_key):
                def loss_fn(ma_p, pd_p):
                    mask = pd.sample(params=pd_p, key=mask_key)
                    X_pred = ma.apply(batch, params=ma_p, mask=mask)
                    return self._elbo_loss_with_params(X_pred, batch, pd_p)

                loss, (ma_grads, pd_grads) = jax.value_and_grad(loss_fn, argnums=(0, 1))(
                    ma_params, pd_params,
                )
                ma_params, ma_state = adam_update(ma_params, ma_grads, ma_state, lr=ma.lr)
                pd_params, pd_state = adam_update(pd_params, pd_grads, pd_state, lr=pd.lr)
                return ma_params, pd_params, ma_state, pd_state, loss

            @jax.jit
            def _eval_forward(ma_params, pd_params, batch, threshold):
                mask = pd.get_threshold_mask(threshold, params=pd_params)
                X_pred = ma.apply(batch, params=ma_params, mask=mask)
                return X_pred, self._elbo_loss_with_params(X_pred, batch, pd_params)

            self._train_step_jit = _train_step
            self._eval_forward_jit = _eval_forward

        # -- fix item 1: proper Bernoulli KL to a sparse prior --------------
        def _elbo_loss_with_params(self, X_pred, X, pd_params):
            recon = jnp.mean((X_pred - X) ** 2)
            if self.regr > 0:
                logits = pd_params["edge_log_params"]
                d_ = logits.shape[0]
                q = jnp.clip(jax.nn.sigmoid(logits), 1e-7, 1.0 - 1e-7)
                p = jnp.asarray(self.prior_p, dtype=q.dtype)
                kl = (q * (jnp.log(q) - jnp.log(p))
                      + (1.0 - q) * (jnp.log1p(-q) - jnp.log1p(-p)))
                offdiag = 1.0 - jnp.eye(d_, dtype=kl.dtype)
                recon = recon + self.regr * (jnp.sum(kl * offdiag) / jnp.sum(offdiag))
            return recon

        # -- fix item 2: skip the two redundant passes, compile the third ---
        def update_mask(self, type=None, threshold=0.5):
            if type is not None or not self.training:
                return super().update_mask(type=type, threshold=threshold)
            # Training path: `step` re-derives the mask from pd_params under
            # this key, and nothing on this path reads
            # `mask_autoencoder.mask`, so materialising it here is dead work.
            # The key is drawn exactly where the vendored code draws it.
            self._last_mask_type = None
            self._last_threshold = threshold
            self._last_mask_key = self.probabilistic_dag.next_key()

        def forward(self, X, compute_loss=True):
            X = jnp.asarray(X, dtype=jnp.float32)
            self._last_batch = X
            if self.training:
                # `train_autoencoder` discards this return value and `step`
                # recomputes the loss from scratch, so there is nothing to do
                # but record the batch.
                return None
            if self._last_mask_type != "deterministic":
                return super().forward(X, compute_loss=compute_loss)
            X_pred, loss = self._eval_forward_jit(
                self.mask_autoencoder.params,
                self.probabilistic_dag.params,
                X,
                self._last_threshold,
            )
            if compute_loss:
                self.grad_loss = loss
            return X_pred

        # `__call__` is resolved on the type, so the base class's binding to
        # its own `forward` has to be re-pointed here explicitly.
        __call__ = forward

        def step(self):
            if self._last_batch is None:
                raise RuntimeError(
                    "forward must be called before step, matching the original "
                    "training flow."
                )
            if self.pd_initial_adj is not None or self._last_mask_type is not None:
                return super().step()  # not the training path

            ma, pd = self.mask_autoencoder, self.probabilistic_dag
            (ma.params, pd.params, ma.optimizer_state, pd.optimizer_state,
             loss) = self._train_step_jit(
                ma.params, pd.params, ma.optimizer_state, pd.optimizer_state,
                self._last_batch, self._last_mask_key,
            )
            self.grad_loss = loss

    return _FastDDSAutoencoder


def fit_dds(
    X_train: np.ndarray,
    num_nodes: int,
    num_posterior_samples: int = 100,
    seed: int = 0,
    # Autoencoder / DAG hyper-parameters (names match run_probabilistic_dag_autoencoder.py).
    # Defaults follow the authors' 02-VI-DP-DAG notebook + the convergence fix
    # (see module docstring): topk ordering, pd_lr=1e-2, hidden=(16,16,16),
    # 100 epochs, and a *corrected* sparsity regularizer weighted by regr=0.1.
    max_epochs: int = 100,
    patience: int = 20,
    batch_size: int = 64,
    ma_hidden_dims: Tuple[int, ...] = (16, 16, 16),
    ma_architecture: str = "linear",
    ma_lr: float = 1e-3,
    ma_fast: bool = False,
    pd_temperature: float = 1.0,
    pd_hard: bool = True,
    pd_order_type: str = "topk",
    pd_noise_factor: float = 1.0,
    pd_lr: float = 1e-2,
    loss: str = "ELBO",
    regr: float = 0.1,
    prior_p: float = 0.01,
    frequency: int = 2,
    verbose: bool = False,
) -> np.ndarray:
    """
    Fit VI-DP-DAG (autoencoder path) on observational data and draw binary
    DAG samples from the variational posterior over permutations + edge
    probabilities.

    Parameters
    ----------
    X_train : [N, d] float32
        Observational data (already standardised; the original code
        standardises using train-split statistics inside ``_make_splits``).
    num_nodes : int
        d.
    num_posterior_samples : int
        S samples to draw via ``ProbabilisticDAG.sample``.
    seed : int
        RNG seed (numpy + JAX).
    max_epochs, patience, batch_size, ma_*, pd_*, loss, regr, prior_p, frequency :
        VI-DP-DAG hyper-parameters.  Default values are those used in
        ``run_probabilistic_dag_autoencoder.py`` for the Sachs run (linear
        autoencoder, Sinkhorn ordering, hard sampling).

    Returns
    -------
    A_samples : [S, d, d] float32 binary adjacency samples (``i_to_j``).
    """
    # Lazy imports.
    import jax

    from src.jax_utils import ArrayDataLoader
    from src.probabilistic_dag_model.train_probabilistic_dag_autoencoder import (
        train_autoencoder,
    )

    X = np.asarray(X_train, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X_train must be [N, d]; got {X.shape}")
    N, d = X.shape
    if d != int(num_nodes):
        raise ValueError(f"num_nodes={num_nodes} != X_train.shape[1]={d}")

    # Mimic _make_splits' train/val carving (60/20/20) -- training only uses
    # train & val loaders (test is ignored by train_autoencoder).
    rng = np.random.RandomState(int(seed))
    indices = np.arange(N, dtype=np.int32)
    rng.shuffle(indices)
    n_train = int(N * 0.8)  # 80% train, 20% val (test not needed here)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_loader = ArrayDataLoader(
        X, indices=train_idx,
        batch_size=batch_size, shuffle=True, seed=int(seed),
    )
    val_loader = ArrayDataLoader(
        X, indices=val_idx,
        batch_size=1024, shuffle=False, seed=int(seed) + 1,
    )

    # The corrected sparsity regularizer (fix item 1) and the jitted train /
    # eval kernels (fix item 2) both live on this subclass -- see
    # ``_fast_model_class``.  The vendored source is untouched.
    model = _fast_model_class()(
        input_dim=d, output_dim=d,
        loss=loss, regr=regr, prior_p=prior_p, seed=int(seed),
        ma_hidden_dims=ma_hidden_dims, ma_architecture=ma_architecture,
        ma_lr=ma_lr, ma_fast=ma_fast,
        pd_temperature=pd_temperature, pd_hard=pd_hard,
        pd_order_type=pd_order_type, pd_noise_factor=pd_noise_factor,
        pd_lr=pd_lr,
    )

    # train_autoencoder will save_checkpoint(model_path, ...) periodically.
    # Pipe to a throwaway temp dir.
    with tempfile.TemporaryDirectory(prefix="vidpdag_case4_") as model_dir:
        model_path = str(Path(model_dir) / "model.ckpt")
        train_autoencoder(
            model,
            train_loader,
            val_loader,
            max_epochs=int(max_epochs),
            frequency=int(frequency),
            patience=int(patience),
            model_path=model_path,
            full_config_dict=None,
        )

        # ----- posterior sampling -----
        # Each call to ProbabilisticDAG.sample draws one DAG.
        keys = jax.random.split(jax.random.PRNGKey(int(seed) + 999), int(num_posterior_samples))
        samples = []
        for k in keys:
            A = model.probabilistic_dag.sample(key=k)
            samples.append(np.asarray(A))
        return np.stack(samples, axis=0).astype(np.float32)
