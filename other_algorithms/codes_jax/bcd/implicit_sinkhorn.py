import jax.numpy as jnp
from ott.geometry.geometry import Geometry

# -----------------------------------------------------------------------------
# Compatibility shim: the functional ``ott.core.sinkhorn.sinkhorn`` API that
# this module was originally written against was removed in ott-jax >= 0.3.
# Newer ott versions expose a class-based API under
# ``ott.solvers.linear.sinkhorn.Sinkhorn``.  We try the legacy import first so
# this file keeps working on older envs, and fall back to a thin wrapper that
# replicates the (log_u, log_v, *_) 5-tuple return convention the rest of BCD
# unpacks below.
# -----------------------------------------------------------------------------
try:
    # Legacy API (ott-jax <= 0.2.x).
    from ott.core.sinkhorn import sinkhorn as _legacy_sinkhorn  # type: ignore

    def _sinkhorn(gg, *, a, b, threshold, parallel_dual_updates, use_danskin,
                  implicit_differentiation, max_iterations):
        return _legacy_sinkhorn(
            gg,
            a=a,
            b=b,
            threshold=threshold,
            parallel_dual_updates=parallel_dual_updates,
            jit=False,
            use_danskin=use_danskin,
            implicit_differentiation=implicit_differentiation,
            max_iterations=max_iterations,
        )

except ImportError:
    # Modern API (ott-jax >= 0.3, including 0.6.x).
    from ott.problems.linear.linear_problem import LinearProblem
    from ott.solvers.linear.sinkhorn import Sinkhorn
    from ott.solvers.linear.implicit_differentiation import ImplicitDiff

    # ---------------------------------------------------------------------
    # NOTE on numerical stability
    # ---------------------------------------------------------------------
    # BCD's Sinkhorn is a *balanced* OT problem (a == b == 1), which per
    # the ott docs routinely produces a rank-deficient linear operator
    # during implicit differentiation.  The default lineax-based solver
    # in ott >= 0.3 responds by emitting NaNs and raising
    # ``_EquinoxRuntimeError: A linear solver returned non-finite ...``.
    # Falling back to unrolled backprop (``implicit_diff=None``) instead
    # trips ott's own ``fixed_point_loop.fixpoint_iter_bwd``, which tries
    # to add float0 cotangents to float cotangents and fails with
    # ``TypeError: Called add with a float0 array``.
    #
    # The documented remedy is to keep the default lineax-based
    # ``ImplicitDiff.solver`` (so we dodge a separate ``UnboundLocalError``
    # upstream bug in ott 0.6.0 that triggers whenever a custom ``solver``
    # callable is supplied) and add the ridge regularisers that stabilise
    # the balanced / rank-deficient case.  ``ridge_kernel`` promotes a
    # zero-sum solution (required when tau_a == tau_b == 1.0) and
    # ``ridge_identity`` regularises the linear operator when rows/columns
    # of the cost matrix are near-collinear (common for permutation-logit
    # inputs).  These values match the ones originally carried in this
    # file's commented
    # ``linear_solve_kwargs = {'ridge_identity': 1e-2, 'ridge_kernel': 1e-2}``.
    # Lineax's ``solve_lineax`` accepts both as keyword arguments and
    # passes them through to the underlying linear solver.
    # ---------------------------------------------------------------------
    _IMPLICIT_DIFF = ImplicitDiff(
        solver_kwargs={"ridge_kernel": 1e-2, "ridge_identity": 1e-2},
    )

    def _sinkhorn(gg, *, a, b, threshold, parallel_dual_updates, use_danskin,
                  implicit_differentiation, max_iterations):
        prob = LinearProblem(gg, a=a, b=b)
        solver = Sinkhorn(
            threshold=threshold,
            max_iterations=max_iterations,
            parallel_dual_updates=parallel_dual_updates,
            use_danskin=use_danskin,
            implicit_diff=_IMPLICIT_DIFF if implicit_differentiation else None,
        )
        out = solver(prob)
        # Match the legacy 5-tuple unpacking: (f, g, errors, reg_cost, converged)
        # In lse-mode (the default), ``f`` and ``g`` are the log-space dual
        # potentials -- i.e. the ``log_u``/``log_v`` the callers expect.
        return (out.f, out.g, out.errors, out.reg_ot_cost, out.converged)


def while_uv_sinkhorn(X, tol):
    gg = Geometry(-X, epsilon=1)
    dim = len(X)
    # linear_solve_kwargs = {'ridge_identity': 1e-2, 'ridge_kernel': 1e-2}
    log_u, log_v, _, _, _ = _sinkhorn(
        gg,
        a=jnp.ones(dim),
        b=jnp.ones(dim),
        threshold=tol,
        parallel_dual_updates=False,
        use_danskin=True,
        implicit_differentiation=True,
        max_iterations=20_000,
    )
    u = gg.scaling_from_potential(log_u)
    v = gg.scaling_from_potential(log_v)
    return gg.transport_from_scalings(u, v)


def while_uv_sinkhorn_debug(X, tol):
    gg = Geometry(-X, epsilon=1)
    dim = len(X)
    # linear_solve_kwargs = {'ridge_identity': 1e-2, 'ridge_kernel': 1e-2}
    log_u, log_v, _, errors, _ = _sinkhorn(
        gg,
        a=jnp.ones(dim),
        b=jnp.ones(dim),
        threshold=tol,
        parallel_dual_updates=False,
        use_danskin=True,
        implicit_differentiation=True,
        max_iterations=20_000,
    )
    u = gg.scaling_from_potential(log_u)
    v = gg.scaling_from_potential(log_v)
    return gg.transport_from_scalings(u, v), errors
