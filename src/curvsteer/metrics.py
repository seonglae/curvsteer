"""The two measurements, each defined once.

Behaviour `L_b` is the mean per-token NLL of positive reviews minus that of
negative ones. Capability is mean token cross-entropy on held-out raw text.

Both are raw-text NLL, which is a base-model task. An instruct checkpoint scored
this way can land worse than uniform over its own vocabulary while answering
correctly through its chat template, which is why the sweep refuses a reference
above `--max-base-ce` rather than reporting deltas from nonsense.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def token_nll(logits: jnp.ndarray, ids: jnp.ndarray, keep: jnp.ndarray) -> jnp.ndarray:
    """Per-example mean NLL over the kept positions.

    `keep` carries both the padding mask and the prompt exclusion, so rows never
    interact and the prompt is never scored.
    """
    lp = jax.nn.log_softmax(logits[:, :-1].astype(jnp.float32), -1)
    tok = -jnp.take_along_axis(lp, ids[:, 1:, None], axis=-1)[..., 0]
    k = keep[:, 1:]
    return (tok * k).sum(1) / jnp.clip(k.sum(1), 1, None)


class Metrics:
    def __init__(self, model, tokenizer, prompt: str = "Review: ",
                 cont_len: int = 48):
        self.model, self.tk = model, tokenizer
        self.prompt, self.cont_len = prompt, cont_len
        self.plen = len(tokenizer.encode(prompt))
        # One shape for every behaviour batch. A word of review text can be
        # several tokens, so the ceiling is generous rather than exact; `keep`
        # makes the slack free.
        self.width = self.plen + 2 * cont_len + 8

    def encode(self, texts: list[str], width: int | None = None):
        """Pad to a **fixed** width, not the batch maximum.

        A batch-dependent width gives every batch a new shape, and each new
        shape recompiles the whole 48-layer forward. `keep` already excludes
        padding from the score, so the extra columns cost arithmetic and change
        no number.
        """
        seqs = [self.tk.encode(self.prompt + " ".join(t.split()[:self.cont_len]))
                for t in texts]
        n = width or self.width
        ids = np.zeros((len(seqs), n), dtype=np.int32)
        keep = np.zeros((len(seqs), n), dtype=np.float32)
        for i, s in enumerate(seqs):
            s = s[:n]
            ids[i, :len(s)] = s
            keep[i, self.plen:len(s)] = 1.0    # score the continuation only
        return jnp.asarray(ids), jnp.asarray(keep)

    def nll(self, texts, M=None, alpha: float = 0.0) -> jnp.ndarray:
        ids, keep = self.encode(texts)
        return token_nll(self.model.logits(ids, M, alpha), ids, keep)

    def behaviour_from_site(self, h_pos, h_neg, ids_p, keep_p, ids_n, keep_n,
                            M=None, alpha: float = 0.0) -> jnp.ndarray:
        """`L_b` computed from cached residuals, so a gradient with respect to
        the residual is a gradient of exactly the reported quantity."""
        p = token_nll(self.model.from_site(h_pos, M, alpha), ids_p, keep_p)
        n = token_nll(self.model.from_site(h_neg, M, alpha), ids_n, keep_n)
        return (p - n).mean()

    def capability(self, chunks, M=None, alpha: float = 0.0, bs: int = 8) -> float:
        tot = cnt = 0.0
        for i in range(0, chunks.shape[0], bs):
            b = chunks[i:i + bs]
            keep = jnp.ones(b.shape, jnp.float32)
            v = token_nll(self.model.logits(b, M, alpha), b, keep)
            tot += float(v.sum())
            cnt += b.shape[0]
        return tot / cnt

    def behaviour_per(self, pos, neg, idx, M=None, alpha: float = 0.0,
                      bs: int = 8) -> np.ndarray:
        """Per-example behaviour, batched. Positive and negative are encoded
        separately so each group's padding is independent of the other."""
        out = []
        for i in range(0, len(idx), bs):
            ch = idx[i:i + bs]
            p = self.nll([pos[j] for j in ch], M, alpha)
            n = self.nll([neg[j] for j in ch], M, alpha)
            out.append(np.asarray(p - n))
        return np.concatenate(out) if out else np.zeros(0)
