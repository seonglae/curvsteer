"""The tokenizer, in one place.

`tokenizers` is the Rust library the checkpoint's own `tokenizer.json` was
written for. It is used here rather than `transformers.AutoTokenizer` because
this package is JAX-only and should not pull a torch-first library in to split
strings, and rather than a hand-rolled vocabulary reader because the
checkpoint's post-processor is the checkpoint's business.
"""
from __future__ import annotations


class Tok:
    """Just enough of a tokenizer for this experiment.

    `bos_token_id` is always exposed, even though the post-processor already
    prepends one to a whole encoded string. The capability corpus is chunked
    *after* encoding, so every chunk but the first would otherwise start
    mid-document with no BOS. Gemma scores an unprefixed chunk far worse than it
    deserves: on this checkpoint that single omission moves WikiText
    cross-entropy from 2.63 to 3.85, which reads as a broken forward.
    """

    def __init__(self, backend, bos_token_id: int | None):
        self._t = backend
        self.bos_token_id = bos_token_id

    def encode(self, s: str) -> list[int]:
        return self._t.encode(s).ids

    def decode(self, ids) -> str:
        return self._t.decode(list(ids))


def load_tokenizer(model_name: str, bos_token: str = "<bos>") -> Tok:
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    backend = Tokenizer.from_file(hf_hub_download(model_name, "tokenizer.json"))
    return Tok(backend, backend.token_to_id(bos_token))
