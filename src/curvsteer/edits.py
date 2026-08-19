"""The edit objects.

An activation steering vector adds a constant to the residual stream. This
package's object is the general edit at a site,

    h -> h + alpha * (M h),      ||M||_F = 1,  rank(M) = r

which is what editing every matrix that reads the residual there does:
`W_read -> W_read (I + alpha M)`. Setting `r = 0` recovers the steering vector,
so rank is a free parameter the activation framing does not expose.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

#: Ridge on A and G before inversion, relative to each matrix's mean eigenvalue.
#: A hyperparameter, not a result: it is an input to the method and may be a
#: literal here. Both A and G are estimated from finitely many token gradients
#: and then inverted, which amplifies their least reliable directions.
LAMBDA_REL = 1e-3


def frob1(M: np.ndarray) -> np.ndarray:
    """Unit Frobenius norm, so alpha is the only amplitude in the comparison."""
    n = np.linalg.norm(M)
    return M / n if n > 1e-12 else M


def rank_trunc(M: np.ndarray, r: int) -> np.ndarray:
    """Best rank-r approximation, renormalised.

    r == 0 returns a vector, not a matrix. That is the degenerate case on
    purpose: rank 0 is the unconditional bias edit, i.e. an activation steering
    vector, and `Site` dispatches on dimensionality. `tests/test_edits.py` pins
    the equivalence.
    """
    if r <= 0:
        raise ValueError("rank 0 is a bias edit; build it with bias_edit()")
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    return frob1((U[:, :r] * S[:r]) @ Vt[:r])


def bias_edit(v: np.ndarray) -> np.ndarray:
    """Unit-norm rank-0 edit. `h -> h + alpha*v`, the activation steering vector."""
    v = np.asarray(v, dtype=np.float64)
    return v / max(np.linalg.norm(v), 1e-12)


def _ridge(X: np.ndarray) -> np.ndarray:
    d = X.shape[0]
    return X + LAMBDA_REL * np.trace(X) / d * np.eye(d)


def m_star_jax(mean_g, A, G):
    """`M*` under JAX, for use inside a jitted or gradient-taking context.

    Kept beside the NumPy path rather than replacing it because the sweep solves
    `M*` once on host and then sweeps, while a differentiable pipeline needs it
    as a traceable function. Both call the same two ridged solves, and
    `tests/test_edits.py` pins them to each other.
    """
    d = A.shape[0]
    Ar = A + LAMBDA_REL * jnp.trace(A) / d * jnp.eye(d)
    Gr = G + LAMBDA_REL * jnp.trace(G) / d * jnp.eye(d)
    return jnp.linalg.solve(Gr, mean_g) @ jnp.linalg.inv(Ar)


def m_star(mean_g: np.ndarray, A: np.ndarray, G: np.ndarray) -> np.ndarray:
    """The curvature-normalised edit, `M* = G^-1 mean_g A^-1`.

    Moving behaviour by `alpha * <M, mean_g>` while paying capability
    `(alpha^2/2) vec(M)^T H_c vec(M)` gives `M* = H_c^-1 mean_g` at fixed cost.
    Under the Kronecker approximation `H_c ~= A (x) G`, and using
    `vec(delta h^T) = h (x) delta`, that is the expression above. Both inverses
    are `d_model x d_model` and exact; there is no iterative solver and no
    dictionary.
    """
    return np.linalg.solve(_ridge(G), mean_g) @ np.linalg.inv(_ridge(A))


def curvature_only(A: np.ndarray, G: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A direction chosen for low curvature alone, ignoring where behaviour is.

    The ablation that separates the two ingredients. It has enormous headroom
    and, in the reported sweep, produces less behaviour change than `M*` at the
    same measured cost, which is what rules out amplitude as the mechanism.
    """
    d = A.shape[0]
    R = rng.standard_normal((d, d))
    return frob1(np.linalg.solve(_ridge(G), R) @ np.linalg.inv(_ridge(A)))


def random_rank1(d: int, seed: int) -> np.ndarray:
    """The floor. Any arm not clearly above this is measuring nothing."""
    g = np.random.default_rng(seed)
    return frob1(np.outer(g.standard_normal(d), g.standard_normal(d)))
