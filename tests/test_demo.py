import numpy as np
import pytest

from curvsteer.demo import RANKS, run


def _best(rows, arm):
    return max(rows[r][arm] for r in RANKS)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_curvature_normalisation_wins_at_every_seed(seed):
    """The direction of the effect, which is what replicates. The margin is not
    asserted here because it is not a constant; see the anisotropy test."""
    rows = run(d=32, seed=seed)
    assert _best(rows, "M*") > _best(rows, "raw gradient")


def test_the_margin_grows_with_anisotropy():
    """The mechanism. Curvature normalisation is worth whatever the curvature
    spread is worth, so a flatter spectrum must shrink the advantage."""
    flat = run(d=32, seed=0, decay=0.99)
    steep = run(d=32, seed=0, decay=0.90)
    r_flat = _best(flat, "M*") / _best(flat, "raw gradient")
    r_steep = _best(steep, "M*") / _best(steep, "raw gradient")
    assert r_steep > r_flat


def test_curvature_alone_is_not_enough():
    """Enormous headroom without knowing where the behaviour is does not steer."""
    rows = run(d=32, seed=3)
    assert _best(rows, "M*") > _best(rows, "curvature only")


def test_random_is_the_floor():
    rows = run(d=32, seed=3)
    assert rows["random"]["M*"] < min(rows[r]["M*"] for r in RANKS)
