"""The measured result: matched-cost sweep over arms, ranks and cost budgets.

Three stages, each cached to disk so a dropped session costs the stage in flight
rather than the whole run.

  1. Kronecker factors. `mean_g` on behaviour pairs, `A` and `G` on capability
     text. In JAX the residual at the site is an ordinary intermediate value, so
     each is one `jax.grad` with respect to it rather than a hooked backward.
  2. Arms. Every edit matrix, built from stage 1 and nothing else.
  3. Sweep. For each arm and each target cost, solve alpha against the *measured*
     capability cross-entropy, then score behaviour on the held-out half.
"""
from __future__ import annotations

import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from .config import CONFIGS
from .data import imdb_pairs, wikitext_chunks
from .edits import (bias_edit, curvature_only, m_star, random_rank1, rank_trunc)
from .metrics import Metrics
from .model import load as load_model
from .site import Factors
from .solve import off_target, solve_alpha

RANKS = (1, 2, 4, 8)
COSTS = (0.05, 0.10, 0.20, 0.50)


def _tokenizer(cfg):
    try:
        from gemma import gm
        return gm.text.Tokenizer.from_name(cfg.model.split("/")[-1])
    except Exception:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(cfg.model)
        if not hasattr(tk, "bos_id"):
            tk.bos_id = tk.bos_token_id
        return tk


def run_sweep(a) -> int:
    cfg = CONFIGS[a.config]
    os.makedirs(a.out_dir, exist_ok=True)
    P = lambda n: os.path.join(a.out_dir, f"{a.config}_{n}")

    model = load_model(cfg.model, cfg.block, cfg.dtype)
    tk = _tokenizer(cfg)
    met = Metrics(model, tk, cont_len=a.cont_len)
    d = model.d_model
    print(f"{cfg.model} d_model={d}, {model.n_layers} layers, "
          f"rank-r edit at block {cfg.block}", flush=True)

    pos, neg, fit_idx, test_idx = imdb_pairs(a.n_pairs)
    cfit, cev = wikitext_chunks(tk, a.n_cap)
    print(f"pairs: {len(fit_idx)} fit / {len(test_idx)} test", flush=True)

    # ---- stage 1 ----------------------------------------------------------
    if not os.path.exists(P("kfac.npz")):
        fac = Factors()
        for j, i in enumerate(fit_idx):
            ids_p, keep_p = met.encode([pos[i]])
            ids_n, keep_n = met.encode([neg[i]])
            h_p = model.residual_at_site(ids_p)
            h_n = model.residual_at_site(ids_n)

            def lb(hp, hn):
                return met.behaviour_from_site(hp, hn, ids_p, keep_p, ids_n, keep_n)

            _, (gp, gn) = jax.value_and_grad(lb, argnums=(0, 1))(h_p, h_n)
            fac.add_behaviour(h_p, gp)
            fac.add_behaviour(h_n, gn)
            if j % 32 == 0:
                print(f"  behaviour grad {j}/{len(fit_idx)}", flush=True)

        for i in range(cfit.shape[0]):
            b = cfit[i:i + 1]
            h = model.residual_at_site(b)

            def ce(hh):
                from .metrics import token_nll
                return token_nll(model.from_site(hh), b,
                                 jnp.ones(b.shape, jnp.float32)).mean()

            _, g = jax.value_and_grad(ce)(h)
            fac.add_capability(h, g)
        mean_g, A, G, mean_dg = fac.finish()
        np.savez_compressed(P("kfac.npz"), mean_g=mean_g, A=A, G=G, mean_dg=mean_dg)
        print("stage 1 done", flush=True)
    z = np.load(P("kfac.npz"))
    mean_g, A, G, mean_dg = z["mean_g"], z["A"], z["G"], z["mean_dg"]

    # ---- stage 2 ----------------------------------------------------------
    Mstar = m_star(mean_g, A, G)
    curv = curvature_only(A, G, np.random.default_rng(11))
    mats: dict[str, np.ndarray] = {}
    for r in RANKS:
        mats[f"Mstar_r{r}"] = rank_trunc(Mstar, r)
        mats[f"mean_g_r{r}"] = rank_trunc(mean_g, r)
        mats[f"curvonly_r{r}"] = rank_trunc(curv, r)
    # Rank 0. An activation steering vector is an unconditional bias edit, which
    # is exactly the restricted case the rank-r framing generalises, so the
    # rank-0 to rank-1 step is the claim itself rather than a control.
    mats["bias_meang_r0"] = bias_edit(mean_dg)
    mats["bias_normalised_r0"] = bias_edit(
        np.linalg.solve(G + 1e-3 * np.trace(G) / d * np.eye(d), mean_dg))
    for k in range(a.n_random):
        mats[f"random_r1_{k}"] = random_rank1(d, 200 + k)
    if cfg.sae_repo:
        from .sae import dictionary_arms
        mats.update(dictionary_arms(cfg, model, met, pos, neg, fit_idx, mean_g, d, RANKS))
    else:
        print("  no dictionary release for this checkpoint: the dictionary arms "
              "are not constructible here, which is the reason for running here",
              flush=True)
    print(f"stage 2 done: {len(mats)} matrices", flush=True)

    # ---- stage 3 ----------------------------------------------------------
    base_cap = met.capability(cev)
    base_beh = float(met.behaviour_per(pos, neg, test_idx, bs=a.beh_bs).mean())
    print(f"unsteered: CE {base_cap:.4f}, test behaviour {base_beh:+.4f}\n", flush=True)
    # A sweep on a broken reference produces a full table of plausible numbers
    # that are all deltas from nonsense, and it would run to completion. Refuse.
    if base_cap > a.max_base_ce:
        print(f"ABORT: unsteered CE {base_cap:.4f} exceeds --max-base-ce "
              f"{a.max_base_ce}. The model is not modelling this corpus, so no "
              f"matched-cost comparison built on it means anything.", flush=True)
        return 2

    res = json.load(open(P("sweep.json"))) if os.path.exists(P("sweep.json")) else {}
    seed_mag: dict[str, float] = {}
    for name, Mnp in mats.items():
        M = jnp.asarray(Mnp, dtype=getattr(jnp, cfg.dtype))
        res.setdefault(name, {})
        for target in COSTS:
            if str(target) in res[name]:
                continue
            best = None
            for sgn in (1.0, -1.0):
                alpha, cost = solve_alpha(
                    lambda al: met.capability(cev, M, al) - base_cap,
                    target, sgn, seed_mag.get(name, 1.0))
                shift = float(met.behaviour_per(pos, neg, test_idx, M, alpha,
                                                bs=a.beh_bs).mean()) - base_beh
                if best is None or shift < best["shift"]:
                    best = dict(alpha=alpha, cost=cost, shift=shift)
            seed_mag[name] = abs(best["alpha"])
            best["off_target"] = off_target(best["cost"], target)
            res[name][str(target)] = best
            flag = "  OFF-TARGET" if best["off_target"] else ""
            print(f"  {name:>22} @ {target:.2f}: shift {best['shift']:+.4f} "
                  f"alpha {best['alpha']:+.3g} actual {best['cost']:.3f}{flag}",
                  flush=True)
            json.dump(res, open(P("sweep.json"), "w"), indent=2)
    print(f"\nwrote {P('sweep.json')}")
    return 0


def run_layer_scan(a) -> int:
    """Which site, in one pass over every block.

    A post-hoc diagnostic, not a registered test. The reported site was chosen by
    analogy with a dictionary release someone else trained for an unrelated
    reason, and this is the tool that would replace that choice with a measured
    one.
    """
    cfg = CONFIGS[a.config]
    os.makedirs(a.out_dir, exist_ok=True)
    model = load_model(cfg.model, cfg.block, cfg.dtype)
    tk = _tokenizer(cfg)
    met = Metrics(model, tk)
    pos, neg, fit_idx, _ = imdb_pairs(getattr(a, "n_pairs", 64))

    rows = []
    for blk in range(model.n_layers):
        model.block = blk
        fac = Factors()
        for i in fit_idx[:16]:
            ids_p, keep_p = met.encode([pos[i]])
            ids_n, keep_n = met.encode([neg[i]])
            h_p, h_n = model.residual_at_site(ids_p), model.residual_at_site(ids_n)

            def lb(hp, hn):
                return met.behaviour_from_site(hp, hn, ids_p, keep_p, ids_n, keep_n)

            _, (gp, _) = jax.value_and_grad(lb, argnums=(0, 1))(h_p, h_n)
            fac.add_behaviour(h_p, gp)
        g, _, _, _ = fac.finish()
        rows.append(dict(block=blk, grad_norm=float(np.linalg.norm(g))))
        print(f"  block {blk:>3}: |mean_g| {rows[-1]['grad_norm']:.4g}", flush=True)
    out = os.path.join(a.out_dir, f"{a.config}_layer_scan.json")
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    return 0
