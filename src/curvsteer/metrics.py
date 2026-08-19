"""The two measurements, each defined once.

Behaviour `L_b` is the mean per-token NLL of positive reviews minus that of
negative ones. Capability is mean token cross-entropy on held-out raw text.

Both are raw-text NLL, which is a base-model task. An instruct checkpoint scored
this way can land worse than uniform over its own vocabulary while answering
correctly through its chat template; `max_base_ce` in the sweep exists to refuse
that case rather than produce a table of deltas from nonsense.
"""
from __future__ import annotations

import numpy as np
import torch


class Metrics:
    def __init__(self, model, tokenizer, prompt: str = "Review: ",
                 cont_len: int = 48, device: str = "cuda"):
        self.model, self.tk, self.device = model, tokenizer, device
        self.prompt, self.cont_len = prompt, cont_len
        self.plen = len(tokenizer(prompt)["input_ids"])

    def encode(self, texts: list[str]):
        s = [self.prompt + " ".join(t.split()[:self.cont_len]) for t in texts]
        b = self.tk(s, return_tensors="pt", padding=True, truncation=True,
                    max_length=self.plen + self.cont_len + 8)
        return b["input_ids"].to(self.device), b["attention_mask"].to(self.device)

    def nll_per(self, ids, mask):
        """Per-example mean NLL over the continuation only.

        Padding is on the right and the mask carries it, so rows do not interact.
        """
        logits = self.model(ids, attention_mask=mask).logits[:, :-1].float()
        lp = torch.log_softmax(logits, -1)
        tokl = -lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        keep = mask[:, 1:].clone()
        keep[:, :self.plen - 1] = 0
        return (tokl * keep).sum(1) / keep.sum(1).clamp(min=1)

    def nll(self, ids, mask):
        return self.nll_per(ids, mask).mean()

    def capability(self, data, bs: int = 16) -> float:
        t = n = 0.0
        with torch.no_grad():
            for i in range(0, data.shape[0], bs):
                b = data[i:i + bs]
                t += self.model(b, labels=b).loss.item() * b.shape[0]
                n += b.shape[0]
        return t / n

    def behaviour_per(self, pos, neg, idx, bs: int = 16) -> np.ndarray:
        """Per-example behaviour, batched.

        At 12B the one-at-a-time loop is hundreds of forwards per swept point and
        dominates the run. Positive and negative are encoded separately so each
        group's padding is independent of the other. `sweep --verify-batch`
        checks this against the one-at-a-time path.
        """
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), bs):
                ch = idx[i:i + bs]
                p = self.nll_per(*self.encode([pos[j] for j in ch]))
                n = self.nll_per(*self.encode([neg[j] for j in ch]))
                out.append((p - n).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)
