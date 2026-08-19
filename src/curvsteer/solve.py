"""Matched-cost comparison.

Cost is measured, not predicted. For each arm and each target cost, alpha is
solved against the true capability cross-entropy until the achieved cost equals
the target. This is what makes the comparison meaningful: a direction that
damages the model in a way the curvature estimate failed to anticipate is caught
by the solver and throttled, so an over-confident estimate cannot win by
accident.
"""
from __future__ import annotations

import numpy as np

#: Accept a solve within half a per cent of its target.
TOL = 0.005

#: A swept point whose achieved cost misses its target by more than this
#: fraction is excluded from every comparison. Defined once: this threshold was
#: previously written separately for the printed warning and for the exclusion
#: filter, which is one run classifiable two ways.
OFF_TARGET = 0.15


def off_target(achieved: float, target: float) -> bool:
    return abs(achieved - target) / max(target, 1e-12) > OFF_TARGET


def solve_alpha(cost_fn, target: float, sgn: float, seed: float,
                max_evals: int = 18):
    """Alpha whose measured capability cost equals `target`.

    Bisection needed about 32 evaluations per sign because it threw away the one
    thing known about the curve: cost is quadratic in alpha near zero. A log-log
    secant that keeps a bracket lands in a handful and cannot do worse, and the
    achieved cost is measured and reported either way.

    `cost_fn(alpha) -> float` must return measured cost above baseline.
    Returns `(alpha, achieved_cost)`.
    """
    cache: dict[float, float] = {}

    def cost(al):
        al = float(np.real(al))   # a complex alpha must never reach the model
        if al not in cache:
            cache[al] = cost_fn(al)
        return cache[al]

    def step(c):
        return 2.0 if c <= 0 else float(np.clip((target / c) ** 0.5, 0.2, 5.0))

    lo, clo = 0.0, 0.0            # cost(0) is 0 by construction
    hi = chi = None
    x = sgn * max(abs(seed), 1e-6)
    for _ in range(max_evals):
        c = cost(x)
        if c < target:
            if abs(x) > abs(lo):
                lo, clo = x, c
        else:
            if hi is None or abs(x) < abs(hi):
                hi, chi = x, c
        # Stop on the point just evaluated, whichever side of the target it fell.
        # Testing only the upper bracket kept solving after a step had already
        # landed inside tolerance from below.
        if abs(c - target) <= TOL * target:
            break
        if hi is None:                      # no upper bound yet, climb
            x = x * max(step(c), 1.2)
        elif lo == 0.0:                     # no lower bound yet, descend
            x = hi * min(step(chi), 0.9)
        else:
            # clo can land at or below zero: an alpha small enough to be harmless
            # sometimes scores a hair under baseline. A negative base to a
            # fractional power is complex, and that alpha propagates into the
            # residual, so fall back to bisection rather than extrapolating from
            # a cost that carries no slope.
            if clo <= 0:
                x = sgn * 0.5 * (abs(lo) + abs(hi))
            else:
                p = (np.log(chi / clo) / np.log(abs(hi) / abs(lo))
                     if abs(hi) != abs(lo) else 2.0)
                if not np.isfinite(p) or p <= 0.2:
                    p = 2.0
                x = sgn * abs(lo) * (target / clo) ** (1.0 / p)
            inner = 0.02 * (abs(hi) - abs(lo))
            if not (abs(lo) + inner < abs(x) < abs(hi) - inner) or x in cache:
                x = sgn * 0.5 * (abs(lo) + abs(hi))
        if abs(x) > 1e7:
            break
    return min(cache.items(), key=lambda t: abs(t[1] - target) / max(target, 1e-9))
