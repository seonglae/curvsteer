"""Data, and the split rule.

Directions are fitted on a hash-split fit half of the behaviour pairs and every
number is reported on the held-out half. The split is a hash of the text, not an
index, so it is stable under reordering and resampling.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch


def fit_split(text: str) -> bool:
    """True if this pair belongs to the fit half. Deterministic, order-free."""
    return hashlib.sha1(text.encode()).digest()[0] < 128


def imdb_pairs(n_pairs: int, seed: int = 0):
    from datasets import load_dataset
    imdb = load_dataset("imdb")["train"]
    pos = [t for t, l in zip(imdb["text"], imdb["label"]) if l == 1]
    neg = [t for t, l in zip(imdb["text"], imdb["label"]) if l == 0]
    rng = np.random.default_rng(seed)
    pos = [pos[i] for i in rng.permutation(len(pos))[:n_pairs]]
    neg = [neg[i] for i in rng.permutation(len(neg))[:n_pairs]]
    is_fit = np.array([fit_split(p + n) for p, n in zip(pos, neg)])
    return pos, neg, np.where(is_fit)[0], np.where(~is_fit)[0]


def wikitext_chunks(tokenizer, n_cap: int, device: str, chunk: int = 127):
    """Two disjoint halves: A and G are estimated on the first, capability scored
    on the second, so the curvature estimate never sees the text it is judged on.

    A BOS token is prepended where the tokenizer has one. Gemma scores an
    unprefixed chunk far worse than it deserves; on one checkpoint this alone
    moved WikiText cross-entropy from 13.92 to 7.32.
    """
    from datasets import load_dataset
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    txt = "\n\n".join(t for t in wt["text"] if t.strip())
    ids = torch.tensor(tokenizer(txt)["input_ids"])[:2 * n_cap * chunk].view(-1, chunk)
    if tokenizer.bos_token_id is not None:
        bos = torch.full((ids.shape[0], 1), tokenizer.bos_token_id, dtype=ids.dtype)
        ids = torch.cat([bos, ids], 1)
    return ids[:n_cap].to(device), ids[n_cap:2 * n_cap].to(device)
