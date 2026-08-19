"""The gate: does the hand-written JAX forward reproduce the reference model?

A forward written by hand from a config and a weight file is wrong until it is
shown otherwise, and every way it can be wrong runs without complaining. So
nothing downstream is allowed to start until the unsteered capability
cross-entropy on WikiText matches the number the reference implementation
produced on the same corpus, chunking and BOS handling.

Usage:
    python tools/gate.py --expect 2.6302
"""
from __future__ import annotations

import argparse
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="gemma4")
    ap.add_argument("--expect", type=float, default=2.6302)
    ap.add_argument("--tol", type=float, default=0.01,
                    help="absolute CE tolerance; the reference is quoted to 4dp")
    ap.add_argument("--n-cap", type=int, default=96)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--gen", action="store_true",
                    help="also greedily continue a prompt, which catches a "
                         "forward that is numerically close but structurally wrong")
    a = ap.parse_args()

    import jax.numpy as jnp
    from tokenizers import Tokenizer

    from curvsteer.config import CONFIGS
    from curvsteer.data import wikitext_chunks
    from curvsteer.metrics import token_nll
    from curvsteer.model import load

    cfg = CONFIGS[a.config]
    print(f"gate: {cfg.model}, site {cfg.block}, expecting CE {a.expect:.4f}",
          flush=True)

    t0 = time.time()
    tk = _tokenizer(cfg.model)
    model = load(cfg.model, cfg.block, cfg.dtype)
    print(f"  loaded in {time.time() - t0:.0f}s, d_model={model.d_model}, "
          f"{model.n_layers} layers", flush=True)
    kinds = {}
    for s in model.specs:
        kinds[s.kind] = kinds.get(s.kind, 0) + 1
    print(f"  layer types: {kinds}", flush=True)

    _, cev = wikitext_chunks(tk, a.n_cap)
    print(f"  capability corpus: {cev.shape[0]} chunks of {cev.shape[1]}",
          flush=True)

    t0 = time.time()
    tot = cnt = 0.0
    for i in range(0, cev.shape[0], a.bs):
        b = cev[i:i + a.bs]
        keep = jnp.ones(b.shape, jnp.float32)
        v = token_nll(model.logits(b), b, keep)
        tot += float(v.sum())
        cnt += b.shape[0]
        if i == 0:
            print(f"  first batch in {time.time() - t0:.0f}s", flush=True)
    ce = tot / cnt

    if a.gen:
        print(f"  greedy: {_greedy(model, tk, 'The capital of France is', 10)!r}",
              flush=True)

    gap = abs(ce - a.expect)
    print(f"\n  unsteered CE = {ce:.4f}   reference {a.expect:.4f}   "
          f"gap {gap:+.4f}", flush=True)
    ok = gap <= a.tol
    print("  PASS" if ok else f"  FAIL: gap {gap:.4f} exceeds tol {a.tol}",
          flush=True)
    return 0 if ok else 1


def _tokenizer(model_name: str):
    """The Rust tokenizer, read straight from the checkpoint.

    Deliberately not `transformers.AutoTokenizer`: this package is JAX-only and
    should not pull a torch-first library in to split strings. `tokenizers`
    applies the checkpoint's own post-processor, so the BOS handling is the
    checkpoint's rather than something reimplemented here.
    """
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download(model_name, "tokenizer.json"))

    class _T:
        bos_token_id = None

        def encode(self, s: str):
            return tok.encode(s).ids

        def decode(self, ids):
            return tok.decode(list(ids))

    t = _T()
    # Always expose BOS. The post-processor prepends one to a whole encoded
    # string, but the capability corpus is chunked *after* encoding, so all but
    # the first chunk would start mid-document with no BOS. Gemma scores an
    # unprefixed chunk far worse than it deserves, which is a 1.2 nat error here
    # and reads as a broken forward rather than a corpus bug.
    t.bos_token_id = tok.token_to_id("<bos>")
    probe = t.encode("hello")
    print(f"  tokenizer: bos={t.bos_token_id}, "
          f"whole-string encode prepends it={probe[:1] == [t.bos_token_id]}",
          flush=True)
    return t


def _greedy(model, tk, prompt: str, n: int) -> str:
    import jax.numpy as jnp
    import numpy as np
    ids = list(tk.encode(prompt))
    for _ in range(n):
        lg = model.logits(jnp.asarray([ids]))
        ids.append(int(np.asarray(lg[0, -1]).argmax()))
    return tk.decode(ids)


if __name__ == "__main__":
    raise SystemExit(main())
