import numpy as np
import pytest

from curvsteer.edits import bias_edit, frob1, m_star, rank_trunc


def test_frob1_is_unit_norm():
    M = np.random.default_rng(0).standard_normal((16, 16))
    assert np.isclose(np.linalg.norm(frob1(M)), 1.0)


def test_rank_trunc_has_the_rank_it_claims():
    M = np.random.default_rng(1).standard_normal((16, 16))
    for r in (1, 2, 4, 8):
        assert np.linalg.matrix_rank(rank_trunc(M, r)) == r


def test_rank_zero_is_refused_not_silently_reinterpreted():
    """rank 0 is a different object, a bias edit, so asking rank_trunc for it is
    a caller error rather than something to guess at."""
    M = np.eye(4)
    with pytest.raises(ValueError):
        rank_trunc(M, 0)


def test_bias_edit_is_a_unit_vector():
    v = np.array([3.0, 4.0])
    assert np.allclose(bias_edit(v), [0.6, 0.8])


def test_m_star_reduces_to_the_gradient_under_isotropy():
    """With A = G = I the curvature normalisation has nothing to do, so M* must
    be the raw gradient up to the ridge. This pins the degenerate path."""
    d = 8
    g = np.random.default_rng(2).standard_normal((d, d))
    M = m_star(g, np.eye(d), np.eye(d))
    assert np.allclose(frob1(M), frob1(g), atol=1e-6)
