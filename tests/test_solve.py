import numpy as np

from curvsteer.solve import OFF_TARGET, TOL, off_target, solve_alpha


def test_solver_hits_a_quadratic_target():
    """cost(alpha) = alpha^2 has the exact answer sqrt(target)."""
    alpha, cost = solve_alpha(lambda a: a * a, target=0.25, sgn=1.0, seed=1.0)
    assert abs(cost - 0.25) <= TOL * 0.25
    assert np.isclose(abs(alpha), 0.5, rtol=0.05)


def test_solver_handles_a_negative_sign():
    alpha, _ = solve_alpha(lambda a: a * a, target=0.25, sgn=-1.0, seed=1.0)
    assert alpha < 0


def test_solver_survives_a_cost_that_dips_below_baseline():
    """An alpha small enough to be harmless sometimes scores a hair under
    baseline. The solver must not raise a negative base to a fractional power."""
    def cost(a):
        return a * a - 1e-6
    alpha, _ = solve_alpha(cost, target=0.25, sgn=1.0, seed=1e-4)
    assert np.isfinite(alpha) and not isinstance(alpha, complex)


def test_off_target_threshold_is_one_constant():
    assert off_target(1.0 + OFF_TARGET * 1.01, 1.0)
    assert not off_target(1.0 + OFF_TARGET * 0.99, 1.0)
