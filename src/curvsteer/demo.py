"""A no-setup demonstration of why curvature normalisation buys what it buys.

Runs in about a second on CPU with no model, no data and no download. This is
the *algebra*, not the language-model result: on a quadratic model the optimal
edit and its cost are both available in closed form, so the two ingredients can
be separated exactly rather than measured. The measured result on
`google/gemma-4-12B` is in the README, and reproducing it needs a GPU.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .edits import curvature_only, m_star, rank_trunc, random_rank1

RANKS = (1, 2, 4, 8)


def _anisotropic(d: int, rng, decay: float) -> np.ndarray:
    """A covariance with a realistic spectrum: a few loud directions, a tail."""
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return (Q * (decay ** np.arange(d))) @ Q.T


def quadratic_cost(M: np.ndarray, A: np.ndarray, G: np.ndarray) -> float:
    """vec(M)^T (A (x) G) vec(M), which is tr(M^T G M A)."""
    return float(np.trace(M.T @ G @ M @ A))


def shift_at_cost(M: np.ndarray, mean_g: np.ndarray, A: np.ndarray,
                  G: np.ndarray, target: float) -> float:
    """Behaviour moved by the largest alpha whose quadratic cost is `target`.

    Magnitude, because alpha's sign is free: the real sweep solves both signs per
    arm and keeps whichever moves behaviour further, so an arm is not penalised
    for pointing the other way along its own axis.
    """
    q = quadratic_cost(M, A, G)
    if q <= 0:
        return 0.0
    alpha = np.sqrt(2.0 * target / q)
    return float(abs(alpha * np.sum(M * mean_g)))


def run(d: int = 64, target: float = 0.05, seed: int = 0,
        decay: float = 0.93) -> dict:
    """`decay` sets how anisotropic the curvature is. At decay = 1 the model is
    isotropic, curvature normalisation has nothing to do, and `M*` collapses onto
    the raw gradient. The advantage is a function of that spread, not a constant.
    """
    rng = np.random.default_rng(seed)
    A = _anisotropic(d, rng, decay)
    G = _anisotropic(d, rng, decay - 0.03)
    mean_g = rng.standard_normal((d, d)) / np.sqrt(d)

    Mstar = m_star(mean_g, A, G)
    curv = curvature_only(A, G, rng)
    rows = {}
    for r in RANKS:
        rows[r] = {
            "M*": shift_at_cost(rank_trunc(Mstar, r), mean_g, A, G, target),
            "raw gradient": shift_at_cost(rank_trunc(mean_g, r), mean_g, A, G, target),
            "curvature only": shift_at_cost(rank_trunc(curv, r), mean_g, A, G, target),
        }
    rows["random"] = {
        "M*": shift_at_cost(random_rank1(d, 7), mean_g, A, G, target),
    }
    return rows


def main() -> int:
    rows = run()
    print("Behaviour moved at matched quadratic cost (higher is more steering).")
    print("Synthetic quadratic model, d=64. Not the language-model result.\n")
    print(f"{'rank':>5} | {'M*':>10} | {'raw gradient':>13} | {'curvature only':>15}")
    print("-" * 54)
    for r in RANKS:
        v = rows[r]
        print(f"{r:>5} | {v['M*']:>10.4f} | {v['raw gradient']:>13.4f} | "
              f"{v['curvature only']:>15.4f}")
    print("-" * 54)
    print(f"{'random':>5} | {rows['random']['M*']:>10.4f} |")
    best_star = max(rows[r]["M*"] for r in RANKS)
    best_raw = max(rows[r]["raw gradient"] for r in RANKS)
    best_curv = max(rows[r]["curvature only"] for r in RANKS)
    print(f"\nM* over raw gradient:   {best_star / best_raw:5.1f}x")
    print(f"M* over curvature only: {best_star / max(best_curv, 1e-9):5.1f}x")
    print("\nCurvature-only has enormous headroom and does not know where the "
          "behaviour is.\nHeadroom alone is not the mechanism.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
