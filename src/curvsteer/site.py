"""The residual-stream site: captures the Kronecker factors, applies the edit.

One hook does both jobs so the statistics and the intervention cannot drift onto
different tensors.
"""
from __future__ import annotations

import torch


def get_blocks(model, attr: str):
    obj = model
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def hidden_size(cfg_obj) -> int:
    for name in ("hidden_size", "n_embd", "d_model"):
        if getattr(cfg_obj, name, None):
            return int(getattr(cfg_obj, name))
    inner = getattr(cfg_obj, "text_config", None)
    if inner is not None:
        return hidden_size(inner)
    raise AttributeError("cannot find hidden size on this config")


class Site:
    """Captures `h` and its gradient at one block, and applies the rank-r edit.

    Accumulates, in one pass each:
      A       = E[h h^T]              from capability text
      G       = E[delta delta^T]      from capability text
      mean_g  = E[delta h^T]          from behaviour pairs
      mean_dg = E[delta]              the rank-0 behaviour gradient
    """

    def __init__(self, blocks, block: int, d_model: int, device, dtype):
        self.M = None
        self.scale = 0.0
        self.capture = False
        # Every parameter is frozen and the input is token ids, so the residual
        # carries no grad_fn and cannot be hooked. Adding a zero leaf that does
        # require grad creates the edge without changing any value.
        self.zero = torch.zeros(d_model, device=device, dtype=dtype,
                                requires_grad=True)
        self.h_sum = self.d_sum = self.g_sum = self.gv_sum = None
        self.n_tok = 0
        # with_kwargs: a decoder layer may receive hidden_states positionally or
        # by keyword depending on the transformers release, and a positional-only
        # hook sees an empty tuple in the second case.
        blocks[block].register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, mod, inp, kw):
        if inp:
            h = inp[0]
        elif "hidden_states" in kw:
            h = kw["hidden_states"]
        else:
            raise RuntimeError("hook cannot find hidden_states at the site")
        if self.M is None:
            out = h
        elif self.M.dim() == 1:
            out = h + self.scale * self.M            # rank 0: unconditional bias
        else:
            out = h + self.scale * (h @ self.M.T)    # rank r: conditional
        if self.capture:
            flat = h.detach().reshape(-1, h.shape[-1]).float()
            self.h_sum = flat.T @ flat if self.h_sum is None else self.h_sum + flat.T @ flat
            self.n_tok += flat.shape[0]
            # Bind this call's h into the hook. A behaviour example runs two
            # forwards (positive and negative) before either backward, so a
            # single attribute would be overwritten and the shapes would not
            # match the gradient that eventually arrives.
            h_det = h.detach()
            out = out + self.zero
            out.register_hook(lambda grad, hd=h_det: self._grab(grad, hd))
        if inp:
            return (out,) + inp[1:], kw
        kw = dict(kw)
        kw["hidden_states"] = out
        return inp, kw

    def _grab(self, grad, h_det):
        g = grad.detach().reshape(-1, grad.shape[-1]).float()
        hh = h_det.reshape(-1, grad.shape[-1]).float()
        self.d_sum = g.T @ g if self.d_sum is None else self.d_sum + g.T @ g
        self.g_sum = g.T @ hh if self.g_sum is None else self.g_sum + g.T @ hh
        gv = g.sum(0)
        self.gv_sum = gv if self.gv_sum is None else self.gv_sum + gv
        return grad

    def reset(self):
        self.h_sum = self.d_sum = self.g_sum = self.gv_sum = None
        self.n_tok = 0

    def set(self, M, s):
        self.M, self.scale = M, s

    def clear(self):
        self.M, self.scale = None, 0.0
