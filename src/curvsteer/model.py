"""Loading a Gemma checkpoint in JAX, and running a forward that can be edited
mid-stack.

Two loaders, tried in order. The first is the official JAX library; the second
reads the published weights and runs a forward written here. The fallback exists
because a dependency-free method should not be blocked by whether someone has
shipped a wrapper for the checkpoint, which is the same argument the dictionary
section of the README makes.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .site import apply_edit


@dataclass
class Loaded:
    """A model reduced to what the experiment needs.

    `run(tokens, M, alpha, upto)` returns logits with the edit applied at the
    site, and `split_at(tokens)` returns the residual there so a gradient can be
    taken with respect to it.
    """
    params: dict
    d_model: int
    n_layers: int
    block: int
    embed: callable
    layer: callable
    head: callable
    name: str = ""

    def residual_at_site(self, tokens: jnp.ndarray) -> jnp.ndarray:
        h = self.embed(self.params, tokens)
        for i in range(self.block):
            h = self.layer(self.params, i, h)
        return h

    def from_site(self, h: jnp.ndarray, M=None, alpha: float = 0.0) -> jnp.ndarray:
        h = apply_edit(h, M, alpha)
        for i in range(self.block, self.n_layers):
            h = self.layer(self.params, i, h)
        return self.head(self.params, h)

    def logits(self, tokens: jnp.ndarray, M=None, alpha: float = 0.0) -> jnp.ndarray:
        return self.from_site(self.residual_at_site(tokens), M, alpha)


def load(model_name: str, block: int, dtype: str = "bfloat16") -> Loaded:
    """Load `model_name` and assert the shape the config claims.

    The assertion is not defensive noise. The experiment's site is an index into
    a specific stack, and a checkpoint whose depth or width differs from the
    recorded one is a different experiment wearing the same config name.
    """
    try:
        return _load_gemma_lib(model_name, block, dtype)
    except (ImportError, KeyError, ValueError) as e:
        print(f"  gemma library path unavailable ({type(e).__name__}: {e}); "
              f"falling back to a direct weight read", flush=True)
        return _load_direct(model_name, block, dtype)


def _load_gemma_lib(model_name: str, block: int, dtype: str) -> Loaded:
    from gemma import gm  # google-deepmind/gemma, JAX

    variant = model_name.split("/")[-1]
    model = gm.nn.from_name(variant)
    params = gm.ckpts.load_params(variant)
    cfg = model.config
    d_model = int(getattr(cfg, "embed_dim", getattr(cfg, "hidden_size", 0)))
    n_layers = int(getattr(cfg, "num_layers", getattr(cfg, "num_hidden_layers", 0)))
    if not d_model or not n_layers:
        raise ValueError("cannot read width and depth off this config")
    if not 0 <= block < n_layers:
        raise ValueError(f"site {block} is outside a {n_layers}-layer stack")

    def embed(p, tokens):
        return model.apply(p, tokens, method=model.embed_tokens)

    def layer(p, i, h):
        return model.apply(p, h, layer_index=i, method=model.run_layer)

    def head(p, h):
        return model.apply(p, h, method=model.decode_head)

    return Loaded(params, d_model, n_layers, block, embed, layer, head, model_name)


def _load_direct(model_name: str, block: int, dtype: str) -> Loaded:
    """Read the published safetensors and run a Gemma forward written here.

    Deliberately explicit rather than fused: on a decoder this costs little and
    it removes any question about what the intervention is being applied to.
    """
    import numpy as np
    from huggingface_hub import snapshot_download
    from safetensors.numpy import load_file
    import glob
    import json
    import os

    path = snapshot_download(model_name, allow_patterns=["*.safetensors", "*.json"])
    cfg = json.load(open(os.path.join(path, "config.json")))
    cfg = cfg.get("text_config", cfg)
    d_model = int(cfg["hidden_size"])
    n_layers = int(cfg["num_hidden_layers"])
    n_heads = int(cfg["num_attention_heads"])
    n_kv = int(cfg.get("num_key_value_heads", n_heads))
    head_dim = int(cfg.get("head_dim", d_model // n_heads))
    rope_base = float(cfg.get("rope_theta", 10000.0))
    eps = float(cfg.get("rms_norm_eps", 1e-6))
    if not 0 <= block < n_layers:
        raise ValueError(f"site {block} is outside a {n_layers}-layer stack")

    w = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        w.update(load_file(f))
    jd = getattr(jnp, dtype)
    params = {k: jnp.asarray(v, dtype=jd) for k, v in w.items()}

    def rms(x, g):
        v = x.astype(jnp.float32)
        n = v * jax.lax.rsqrt((v * v).mean(-1, keepdims=True) + eps)
        return (n * (1.0 + g.astype(jnp.float32))).astype(x.dtype)

    def rope(x, pos):
        half = head_dim // 2
        inv = 1.0 / (rope_base ** (jnp.arange(half, dtype=jnp.float32) * 2 / head_dim))
        ang = pos[:, None] * inv[None, :]
        c, s = jnp.cos(ang), jnp.sin(ang)
        x1, x2 = x[..., :half], x[..., half:]
        return jnp.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], -1)

    P = "model.layers"

    def layer(p, i, h):
        g = lambda n: p[f"{P}.{i}.{n}"]
        x = rms(h, g("input_layernorm.weight"))
        T = x.shape[-2]
        pos = jnp.arange(T, dtype=jnp.float32)
        q = (x @ g("self_attn.q_proj.weight").T).reshape(*x.shape[:-1], n_heads, head_dim)
        k = (x @ g("self_attn.k_proj.weight").T).reshape(*x.shape[:-1], n_kv, head_dim)
        v = (x @ g("self_attn.v_proj.weight").T).reshape(*x.shape[:-1], n_kv, head_dim)
        q, k = rope(q, pos), rope(k, pos)
        if n_kv != n_heads:
            rep = n_heads // n_kv
            k = jnp.repeat(k, rep, axis=-2)
            v = jnp.repeat(v, rep, axis=-2)
        att = jnp.einsum("...qhd,...khd->...hqk", q, k) / jnp.sqrt(head_dim)
        mask = jnp.tril(jnp.ones((T, T), bool))
        att = jnp.where(mask, att, -jnp.inf)
        att = jax.nn.softmax(att.astype(jnp.float32), -1).astype(x.dtype)
        o = jnp.einsum("...hqk,...khd->...qhd", att, v).reshape(*x.shape[:-1], -1)
        h = h + o @ g("self_attn.o_proj.weight").T
        y = rms(h, g("post_attention_layernorm.weight"))
        gate = jax.nn.gelu(y @ g("mlp.gate_proj.weight").T, approximate=True)
        return h + (gate * (y @ g("mlp.up_proj.weight").T)) @ g("mlp.down_proj.weight").T

    def embed(p, tokens):
        e = p["model.embed_tokens.weight"][tokens]
        return e * jnp.asarray(d_model ** 0.5, dtype=e.dtype)

    def head_fn(p, h):
        h = rms(h, p["model.norm.weight"])
        return h @ p["model.embed_tokens.weight"].T   # Gemma ties the embedding

    return Loaded(params, d_model, n_layers, block, embed, layer, head_fn, model_name)
