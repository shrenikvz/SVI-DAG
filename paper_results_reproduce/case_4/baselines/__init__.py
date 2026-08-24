"""Case 4 baseline wrappers. Each module exports ``run(...)`` that returns
a tuple ``(A_samples [S, d, d], convention: str)`` where ``convention`` is
one of ``"j_to_i"`` (SVIDAG-compatible) or ``"i_to_j"`` (standard).

All wrappers take the same standardised input signature so ``run_case4.py``
can dispatch them uniformly.  See ``baselines/prodag_wrapper.py`` for the
reference implementation.
"""

# ---------------------------------------------------------------------------
# JAX compatibility shim: ``jax.numpy.logsumexp`` was removed in JAX 0.4.26+,
# but two vendored baselines (BayesDAG, VI-DP-DAG/DDS) still reference it.
# Re-expose it from ``jax.scipy.special`` so those imports succeed without
# editing the algorithm source.  Idempotent on older JAX where the attribute
# already exists.
# ---------------------------------------------------------------------------
try:
    import jax.numpy as _jnp
    if not hasattr(_jnp, "logsumexp"):
        from jax.scipy.special import logsumexp as _logsumexp
        _jnp.logsumexp = _logsumexp
except Exception:  # pragma: no cover
    pass
