"""The residual-stream site: the intervention, and the Kronecker factors.

There is no hook here, and that is the point. In a framework where a frozen
model's residual carries no autodiff edge, capturing it means attaching a hook
and manufacturing an edge that does not otherwise exist. Here the residual is an
argument to a function, so the intervention is threaded through the forward and
`jax.grad` gives `delta = dL/dh` directly. An entire class of hook-ordering and
tensor-aliasing bug does not arise.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


def apply_edit(h: jnp.ndarray, M: jnp.ndarray | None, alpha: float) -> jnp.ndarray:
    """`h -> h + alpha * (M h)`, or the rank-0 bias edit when `M` is a vector.

    Dispatch is on dimensionality, and rank 0 is a genuinely different object:
    an unconditional bias edit is what an activation steering vector is, and it
    is the restricted case the rank-r framing generalises.
    """
    if M is None or alpha == 0.0:
        return h
    if M.ndim == 1:
        return h + alpha * M                    # rank 0: unconditional
    return h + alpha * (h @ M.T)                # rank r: conditional on h


@dataclass
class Factors:
    """The three quantities every arm is built from.

        A       = E[h h^T]            over capability tokens
        G       = E[delta delta^T]    over capability tokens
        mean_g  = E[delta h^T]        over behaviour pairs
        mean_dg = E[delta]            the rank-0 behaviour gradient

    Accumulated as sums and normalised once, so a partial run is resumable and
    the divisor is written down in one place instead of at every call site.
    """
    h_sum: jnp.ndarray | None = None
    d_sum: jnp.ndarray | None = None
    g_sum: jnp.ndarray | None = None
    gv_sum: jnp.ndarray | None = None
    n_tok: int = 0
    n_ex: int = 0

    def add_capability(self, h: jnp.ndarray, delta: jnp.ndarray) -> None:
        flat = h.reshape(-1, h.shape[-1]).astype(jnp.float32)
        g = delta.reshape(-1, delta.shape[-1]).astype(jnp.float32)
        self.h_sum = flat.T @ flat if self.h_sum is None else self.h_sum + flat.T @ flat
        self.d_sum = g.T @ g if self.d_sum is None else self.d_sum + g.T @ g
        self.n_tok += flat.shape[0]

    def add_behaviour(self, h: jnp.ndarray, delta: jnp.ndarray) -> None:
        flat = h.reshape(-1, h.shape[-1]).astype(jnp.float32)
        g = delta.reshape(-1, delta.shape[-1]).astype(jnp.float32)
        self.g_sum = g.T @ flat if self.g_sum is None else self.g_sum + g.T @ flat
        gv = g.sum(0)
        self.gv_sum = gv if self.gv_sum is None else self.gv_sum + gv
        self.n_ex += 1

    def finish(self):
        import numpy as np
        return (np.asarray(self.g_sum) / max(self.n_ex, 1),
                np.asarray(self.h_sum) / max(self.n_tok, 1),
                np.asarray(self.d_sum) / max(self.n_tok, 1),
                np.asarray(self.gv_sum) / max(self.n_ex, 1))


def residual_grad(loss_fn, h: jnp.ndarray):
    """`(loss, dL/dh)` at the site. This is the whole capture mechanism."""
    return jax.value_and_grad(loss_fn)(h)
