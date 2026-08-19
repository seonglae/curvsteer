"""The dictionary arms.

Only constructible where a sparse-autoencoder release exists for the exact
checkpoint. That constraint is the argument rather than an inconvenience: this
method needs no dictionary, so letting dictionary availability pick the model
pins a dependency-free method to whatever checkpoint someone else happened to
train on.

Two selection rules, because a dictionary baseline chosen badly is a straw man:

  by correlation  the feature whose activation correlates most with the label,
                  which is how a practitioner picks a steering feature
  by gradient     the feature whose own rank-1 edit has the largest first-order
                  effect, `d_j^T mean_g e_j`, which is the same objective `M*`
                  optimises. This is the strong version of the baseline.
"""
from __future__ import annotations

import numpy as np

from .edits import frob1


def load_dictionary(cfg):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(cfg.sae_repo, cfg.sae_file)
    if cfg.sae_file.endswith(".npz"):
        z = np.load(path)
        return (z["W_enc"], z["b_enc"], z["W_dec"], z["b_dec"],
                z["threshold"] if "threshold" in z else None)
    from safetensors.torch import load_file
    sf = load_file(path)
    return (sf["W_enc"].float().numpy(), sf["b_enc"].float().numpy(),
            sf["W_dec"].float().numpy(), sf["b_dec"].float().numpy(), None)


def dictionary_arms(cfg, model, blocks, met, pos, neg, fit_idx, mean_g, d, ranks):
    W_enc, b_enc, W_dec, b_dec, thr = load_dictionary(cfg)
    F = W_enc.shape[1]
    cap: dict = {}
    handle = blocks[cfg.block].register_forward_pre_hook(
        lambda m, i: cap.__setitem__("h", i[0]))
    acts, y = [], []
    for texts, lab in ((pos, 1.0), (neg, -1.0)):
        for i in fit_idx:
            ii, mm = met.encode([texts[i]])
            model(ii, attention_mask=mm)
            act = cap["h"][0][met.plen:].float().cpu().numpy()
            pre = (act - b_dec) @ W_enc + b_enc
            fa = np.maximum(pre, 0.0) if thr is None else np.where(pre > thr, pre, 0.0)
            acts.append(fa.mean(0))
            y.append(lab)
    handle.remove()

    f, y = np.stack(acts), np.asarray(y)
    sd = f.std(0)
    corr = np.where(sd > 1e-8,
                    ((f - f.mean(0)) * (y - y.mean())[:, None]).mean(0)
                    / (sd * y.std() + 1e-12), 0.0)
    # A correlation this far above the noise floor for a random feature is what
    # separates "the dictionary found the behaviour" from "some feature had to
    # come first". Printed so a reader can check the baseline was not vacuous.
    noise = 4.1 / np.sqrt(len(y))
    print(f"  dictionary: max|r| {np.abs(corr).max():.3f} vs noise {noise:.3f}, "
          f"active {int((f > 0).any(0).sum())}/{F}", flush=True)

    align = np.einsum("fd,de,fe->f", W_dec, mean_g, W_enc.T)
    order_c, order_g = np.argsort(-np.abs(corr)), np.argsort(-np.abs(align))

    def top_r(order, r, sign_from):
        M = np.zeros((d, d))
        for j in order[:r]:
            M += -np.sign(sign_from[j]) * np.outer(W_dec[j], W_enc[:, j])
        return frob1(M)

    out = {}
    for r in ranks:
        out[f"dict_corr_r{r}"] = top_r(order_c, r, corr)
        out[f"dict_grad_r{r}"] = top_r(order_g, r, align)
    return out
