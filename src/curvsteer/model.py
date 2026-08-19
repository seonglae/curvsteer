"""Loading a Gemma checkpoint in JAX and running a forward that can be edited
mid-stack.

The published weights are safetensors, which `safetensors.numpy` reads straight
to arrays, so the forward below is written here rather than delegated. That is
not a fallback: the official JAX library reads orbax checkpoints and this
checkpoint has no orbax form, so a direct read is the only path that exists.
It also keeps the promise the README makes, since the intervention is threaded
through a forward whose every step is visible instead of into someone's wrapper.

The Gemma 4 text tower is not a generic decoder and the differences are not
cosmetic. Each is written down at the point it is implemented, because every one
of them runs fine when wrong and silently reproduces nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from .site import apply_edit


@dataclass
class LayerSpec:
    """Per-layer geometry. Gemma 4 alternates two attention types that differ in
    head width, KV count, rope base and whether a value projection exists."""
    kind: str
    head_dim: int
    n_kv: int
    theta: float
    partial: float | None
    window: int | None
    k_eq_v: bool


@dataclass
class Loaded:
    """A model reduced to what the experiment needs.

    `residual_at_site` returns the residual at the site so a gradient can be
    taken with respect to it, and `from_site` resumes the stack with the edit
    applied. Splitting the forward this way is what makes `jax.grad` give
    `delta = dL/dh` without a hook.
    """
    params: dict
    d_model: int
    n_layers: int
    block: int
    embed: callable
    layer: callable
    head: callable
    name: str = ""
    specs: list = field(default_factory=list)
    _jit: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        # Compiled once per (shape, edit-arity). Without this the 48 layers run
        # as several thousand separate dispatches per batch and the sweep does
        # not finish. `alpha` is traced, not static, so re-solving it inside the
        # matched-cost bisection does not trigger a recompile.
        def stack(p, h, lo, hi):
            for i in range(lo, hi):
                h = self.layer(p, i, h)
            return h

        self._jit["pre"] = jax.jit(
            lambda p, t: stack(p, self.embed(p, t), 0, self.block))
        self._jit["post"] = jax.jit(
            lambda p, h: self.head(p, stack(p, h, self.block, self.n_layers)))
        self._jit["post_edit"] = jax.jit(
            lambda p, h, M, al: self.head(
                p, stack(p, apply_edit(h, M, al), self.block, self.n_layers)))

    def set_block(self, block: int) -> None:
        """Move the site and rebuild the compiled halves.

        The split point is baked into the traced graphs, so assigning `.block`
        alone would leave the old compilation in place and silently keep
        measuring the old site.
        """
        if not 0 <= block < self.n_layers:
            raise ValueError(f"site {block} is outside a {self.n_layers}-layer stack")
        self.block = block
        self.__post_init__()

    def residual_at_site(self, tokens: jnp.ndarray) -> jnp.ndarray:
        return self._jit["pre"](self.params, tokens)

    def from_site(self, h: jnp.ndarray, M=None, alpha: float = 0.0) -> jnp.ndarray:
        if M is None:
            return self._jit["post"](self.params, h)
        return self._jit["post_edit"](self.params, h, M, jnp.asarray(alpha, jnp.float32))

    def logits(self, tokens: jnp.ndarray, M=None, alpha: float = 0.0) -> jnp.ndarray:
        return self.from_site(self.residual_at_site(tokens), M, alpha)


def load(model_name: str, block: int, dtype: str = "bfloat16") -> Loaded:
    return _load_direct(model_name, block, dtype)


def _inv_freq(head_dim: int, theta: float, partial: float | None) -> np.ndarray:
    """Rope inverse frequencies.

    The `proportional` variant rotates only the first `partial` fraction of the
    head and pads the rest with **zero** frequencies, so those channels get
    `cos = 1, sin = 0` and pass through unrotated. Truncating the head instead
    would change its width and is the obvious wrong reading.
    """
    if partial is None:
        return 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    rope_angles = int(partial * head_dim // 2)
    rot = 1.0 / (theta ** (np.arange(0, 2 * rope_angles, 2, dtype=np.float64) / head_dim))
    nope = head_dim // 2 - rope_angles
    return np.concatenate([rot, np.zeros(nope, dtype=np.float64)]) if nope > 0 else rot


def _rope(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """`(x * cos) + (rotate_half(x) * sin)`; x is (B, T, H, D), cos/sin are (T, D)."""
    d = x.shape[-1] // 2
    half = jnp.concatenate([-x[..., d:], x[..., :d]], axis=-1)
    return x * cos[None, :, None, :] + half * sin[None, :, None, :]


def _load_direct(model_name: str, block: int, dtype: str) -> Loaded:
    import glob
    import json
    import os

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = snapshot_download(model_name, allow_patterns=["*.safetensors", "*.json"])
    raw = json.load(open(os.path.join(path, "config.json")))
    cfg = raw.get("text_config", raw)

    d_model = int(cfg["hidden_size"])
    n_layers = int(cfg["num_hidden_layers"])
    n_heads = int(cfg["num_attention_heads"])
    eps = float(cfg["rms_norm_eps"])
    softcap = cfg.get("final_logit_softcapping")
    window = int(cfg.get("sliding_window", 0)) or None
    layer_types = list(cfg["layer_types"])
    rope_params = cfg["rope_parameters"]
    k_eq_v_cfg = bool(cfg.get("attention_k_eq_v", False))

    # Two hard paths in the reference implementation are dead for this
    # checkpoint and are asserted rather than implemented, so a checkpoint that
    # turns them on fails loudly instead of being silently mis-run.
    if int(cfg.get("num_kv_shared_layers", 0)) != 0:
        raise NotImplementedError("this checkpoint shares KV across layers")
    if bool(cfg.get("use_double_wide_mlp", False)):
        raise NotImplementedError("this checkpoint uses a double-wide MLP")
    if not 0 <= block < n_layers:
        raise ValueError(f"site {block} is outside a {n_layers}-layer stack")

    def spec_for(kind: str) -> LayerSpec:
        rp = rope_params[kind]
        sliding = kind == "sliding_attention"
        # head_dim and KV count differ per type; global_head_dim names the wide one.
        hd = int(cfg["head_dim"]) if sliding else int(cfg["global_head_dim"])
        nkv = (int(cfg["num_key_value_heads"]) if sliding
               else int(cfg["num_global_key_value_heads"]))
        return LayerSpec(
            kind=kind, head_dim=hd, n_kv=nkv,
            theta=float(rp["rope_theta"]),
            partial=rp.get("partial_rotary_factor"),
            window=window if sliding else None,
            # `use_alternative_attention = attention_k_eq_v and not is_sliding`:
            # only the full layers drop v_proj and reuse the keys as values.
            k_eq_v=k_eq_v_cfg and not sliding,
        )

    specs = [spec_for(t) for t in layer_types]
    kinds = {s.kind: s for s in specs}
    inv_freq = {k: jnp.asarray(_inv_freq(s.head_dim, s.theta, s.partial), jnp.float32)
                for k, s in kinds.items()}

    jd = getattr(jnp, dtype)
    params: dict[str, jnp.ndarray] = {}
    n_skipped = 0
    # Read one tensor at a time. Materialising the whole file as numpy before
    # converting would hold the checkpoint twice at peak.
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, framework="numpy") as sf:
            for k in sf.keys():
                # Text tower only. The vision and audio towers are never reached
                # by a token-only forward and are a large share of the file.
                if (".vision" in k or ".audio" in k
                        or "embed_vision" in k or "embed_audio" in k):
                    n_skipped += 1
                    continue
                params[k] = jnp.asarray(sf.get_tensor(k), dtype=jd)
    print(f"  loaded {len(params)} text tensors, skipped {n_skipped} vision/audio",
          flush=True)

    P = "model.language_model"

    def rms(x, w=None):
        """`x * (mean(x^2) + eps)^-0.5`, then `* weight` when the norm is scaled.

        Note `* weight`, not Gemma 2's `* (1 + weight)`: these weights are
        initialised to ones, and the off-by-one runs fine and reproduces nothing.
        """
        v = x.astype(jnp.float32)
        ms = jnp.mean(v * v, axis=-1, keepdims=True) + eps
        n = v * (ms ** -0.5)
        if w is not None:
            n = n * w.astype(jnp.float32)
        return n.astype(x.dtype)

    def attention(p, i, x, s: LayerSpec):
        g = lambda n: p[f"{P}.layers.{i}.{n}"]
        B, T, _ = x.shape
        hd = s.head_dim
        pos = jnp.arange(T, dtype=jnp.float32)
        freqs = pos[:, None] * inv_freq[s.kind][None, :]
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        cos, sin = jnp.cos(emb), jnp.sin(emb)

        q = (x @ g("self_attn.q_proj.weight").T).reshape(B, T, n_heads, hd)
        q = _rope(rms(q, g("self_attn.q_norm.weight")), cos, sin)

        k_raw = (x @ g("self_attn.k_proj.weight").T).reshape(B, T, s.n_kv, hd)
        k = _rope(rms(k_raw, g("self_attn.k_norm.weight")), cos, sin)

        # When v_proj is absent the values are the *raw* k_proj output, taken
        # before k_norm and given no rope. Reusing the normed or roped keys here
        # is the same shape and a different model.
        v_in = k_raw if s.k_eq_v else (
            x @ g("self_attn.v_proj.weight").T).reshape(B, T, s.n_kv, hd)
        v = rms(v_in)          # v_norm has with_scale=False: no learned gain

        if s.n_kv != n_heads:
            rep = n_heads // s.n_kv
            k = jnp.repeat(k, rep, axis=2)
            v = jnp.repeat(v, rep, axis=2)

        # scaling is 1.0. There is no 1/sqrt(head_dim) here; q_norm does that
        # job, and dividing again gives a plausible-looking wrong model.
        att = jnp.einsum("bqhd,bkhd->bhqk", q, k).astype(jnp.float32)
        qi = jnp.arange(T)[:, None]
        ki = jnp.arange(T)[None, :]
        allowed = ki <= qi
        if s.window:
            allowed = allowed & (ki > qi - s.window)
        att = jnp.where(allowed[None, None], att, -jnp.inf)
        att = jax.nn.softmax(att, axis=-1).astype(x.dtype)
        o = jnp.einsum("bhqk,bkhd->bqhd", att, v).reshape(B, T, n_heads * hd)
        return o @ g("self_attn.o_proj.weight").T

    def layer(p, i, h):
        """Gemma-2 sandwich, plus a scalar on the whole layer output.

        post_attention_layernorm normalises the *attention output* before the
        residual add, not the stream before the MLP.
        """
        g = lambda n: p[f"{P}.layers.{i}.{n}"]
        s = specs[i]
        r = h
        a = attention(p, i, rms(h, g("input_layernorm.weight")), s)
        h = r + rms(a, g("post_attention_layernorm.weight"))

        r = h
        y = rms(h, g("pre_feedforward_layernorm.weight"))
        act = jax.nn.gelu(y @ g("mlp.gate_proj.weight").T, approximate=True)
        m = (act * (y @ g("mlp.up_proj.weight").T)) @ g("mlp.down_proj.weight").T
        h = r + rms(m, g("post_feedforward_layernorm.weight"))

        # Applied to the entire layer output, after both residual adds.
        return h * g("layer_scalar").astype(h.dtype)

    embed_scale = jnp.asarray(d_model ** 0.5, dtype=jd)

    def embed(p, tokens):
        return p[f"{P}.embed_tokens.weight"][tokens] * embed_scale

    def head_fn(p, h):
        h = rms(h, p[f"{P}.norm.weight"])
        logits = (h @ p[f"{P}.embed_tokens.weight"].T).astype(jnp.float32)
        if softcap:                       # tanh soft-cap, tied embedding head
            c = float(softcap)
            logits = jnp.tanh(logits / c) * c
        return logits

    return Loaded(params, d_model, n_layers, block, embed, layer, head_fn,
                  model_name, specs)
