"""The measured result: matched-cost sweep over arms, ranks and cost budgets.

Three stages, each cached to disk so a dropped session costs the stage in flight
rather than the whole run.

  1. Kronecker factors. `mean_g` on behaviour pairs, `A` and `G` on capability
     text, all in one hooked pass each.
  2. Arms. Every edit matrix, built from stage 1 and nothing else.
  3. Sweep. For each arm and each target cost, solve alpha against the *measured*
     capability cross-entropy, then score behaviour on the held-out half.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .config import CONFIGS
from .data import imdb_pairs, wikitext_chunks
from .edits import (bias_edit, curvature_only, m_star, random_rank1,
                    rank_trunc)
from .metrics import Metrics
from .site import Site, get_blocks, hidden_size
from .solve import off_target, solve_alpha

RANKS = (1, 2, 4, 8)
COSTS = (0.05, 0.10, 0.20, 0.50)


def _load(cfg, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tk = AutoTokenizer.from_pretrained(cfg.model)
    tk.padding_side = "right"
    if tk.pad_token is None:
        tk.pad_token = tk.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model, torch_dtype=getattr(torch, cfg.dtype))
    except (ValueError, KeyError) as e:
        # Multimodal checkpoints are not AutoModelForCausalLM. Fall back rather
        # than hard-code one wrapper class, whose name moves between releases.
        print(f"  AutoModelForCausalLM refused ({type(e).__name__}), trying AutoModel")
        from transformers import AutoModel
        model = AutoModel.from_pretrained(cfg.model,
                                          torch_dtype=getattr(torch, cfg.dtype))
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tk


def run_sweep(a) -> int:
    cfg = CONFIGS[a.config]
    os.makedirs(a.out_dir, exist_ok=True)
    P = lambda n: os.path.join(a.out_dir, f"{a.config}_{n}")

    model, tk = _load(cfg, a.device)
    d = hidden_size(model.config)
    blocks = get_blocks(model, cfg.blocks_attr)
    site = Site(blocks, cfg.block, d, a.device, getattr(torch, cfg.dtype))
    met = Metrics(model, tk, cont_len=a.cont_len, device=a.device)
    print(f"{cfg.model} d_model={d}, rank-r edit at block {cfg.block}", flush=True)

    pos, neg, fit_idx, test_idx = imdb_pairs(a.n_pairs)
    cfit, cev = wikitext_chunks(tk, a.n_cap, a.device)
    print(f"pairs: {len(fit_idx)} fit / {len(test_idx)} test", flush=True)

    if a.verify_batch:
        sub = test_idx[:24]
        one = np.asarray([float((met.nll(*met.encode([pos[i]]))
                                 - met.nll(*met.encode([neg[i]]))).item())
                          for i in sub])
        many = met.behaviour_per(pos, neg, sub, bs=a.beh_bs)
        dmax = float(np.abs(one - many).max())
        print(f"batched vs one-at-a-time over {len(sub)}: max|diff| {dmax:.3e}")
        print("PASS" if dmax < 1e-3 else "FAIL")
        return 0 if dmax < 1e-3 else 1

    # ---- stage 1 ----------------------------------------------------------
    if not os.path.exists(P("kfac.npz")):
        site.capture = True
        site.reset()
        for j, i in enumerate(fit_idx):
            model.zero_grad(set_to_none=True)
            (met.nll(*met.encode([pos[i]])) - met.nll(*met.encode([neg[i]]))).backward()
            if j % 32 == 0:
                print(f"  behaviour grad {j}/{len(fit_idx)}", flush=True)
        mean_g = (site.g_sum / len(fit_idx)).cpu().numpy()
        mean_dg = (site.gv_sum / len(fit_idx)).cpu().numpy()
        site.reset()
        for i in range(cfit.shape[0]):
            model.zero_grad(set_to_none=True)
            b = cfit[i:i + 1]
            model(b, labels=b).loss.backward()
        A = (site.h_sum / site.n_tok).cpu().numpy()
        G = (site.d_sum / cfit.shape[0]).cpu().numpy()
        site.capture = False
        np.savez_compressed(P("kfac.npz"), mean_g=mean_g, A=A, G=G, mean_dg=mean_dg)
        print("stage 1 done", flush=True)
    z = np.load(P("kfac.npz"))
    mean_g, A, G, mean_dg = z["mean_g"], z["A"], z["G"], z["mean_dg"]
    torch.set_grad_enabled(False)

    # ---- stage 2 ----------------------------------------------------------
    Mstar = m_star(mean_g, A, G)
    curv = curvature_only(A, G, np.random.default_rng(11))
    mats: dict[str, np.ndarray] = {}
    for r in RANKS:
        mats[f"Mstar_r{r}"] = rank_trunc(Mstar, r)
        mats[f"mean_g_r{r}"] = rank_trunc(mean_g, r)
        mats[f"curvonly_r{r}"] = rank_trunc(curv, r)
    # Rank 0. An activation steering vector is an unconditional bias edit, which
    # is exactly the restricted case the parameter-space framing generalises, so
    # the rank-0 to rank-1 step is the claim itself rather than a control.
    mats["bias_meang_r0"] = bias_edit(mean_dg)
    mats["bias_normalised_r0"] = bias_edit(
        np.linalg.solve(G + 1e-3 * np.trace(G) / d * np.eye(d), mean_dg))
    for k in range(a.n_random):
        mats[f"random_r1_{k}"] = random_rank1(d, 200 + k)
    if cfg.sae_repo:
        from .sae import dictionary_arms
        mats.update(dictionary_arms(cfg, model, blocks, met, pos, neg, fit_idx,
                                    mean_g, d, RANKS))
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
        M = torch.tensor(Mnp, device=a.device, dtype=getattr(torch, cfg.dtype))
        res.setdefault(name, {})
        for target in COSTS:
            if str(target) in res[name]:
                continue
            best = None
            for sgn in (1.0, -1.0):
                alpha, cost = solve_alpha(
                    lambda al: (site.set(M, al), met.capability(cev))[1] - base_cap,
                    target, sgn, seed_mag.get(name, 1.0))
                site.set(M, alpha)
                shift = float(met.behaviour_per(pos, neg, test_idx,
                                                bs=a.beh_bs).mean()) - base_beh
                if best is None or shift < best["shift"]:
                    best = dict(alpha=alpha, cost=cost, shift=shift)
            site.clear()
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
    """Which site, in one backward pass over every block.

    A post-hoc diagnostic, not a registered test. The reported site was chosen by
    analogy with a dictionary release someone else trained for an unrelated
    reason, and this is the tool that would replace that choice with a measured
    one.
    """
    cfg = CONFIGS[a.config]
    os.makedirs(a.out_dir, exist_ok=True)
    model, tk = _load(cfg, a.device)
    d = hidden_size(model.config)
    blocks = get_blocks(model, cfg.blocks_attr)
    sites = [Site(blocks, i, d, a.device, getattr(torch, cfg.dtype))
             for i in range(len(blocks))]
    met = Metrics(model, tk, device=a.device)
    pos, neg, fit_idx, _ = imdb_pairs(a.n_pairs if hasattr(a, "n_pairs") else 64)

    for s in sites:
        s.capture = True
        s.reset()
    for i in fit_idx[:32]:
        model.zero_grad(set_to_none=True)
        (met.nll(*met.encode([pos[i]])) - met.nll(*met.encode([neg[i]]))).backward()

    rows = []
    for i, s in enumerate(sites):
        if s.g_sum is None:
            continue
        g = (s.g_sum / 32).cpu().numpy()
        rows.append(dict(block=i, grad_norm=float(np.linalg.norm(g)),
                         trA=float(np.trace((s.h_sum / max(s.n_tok, 1)).cpu().numpy())),
                         trG=float(np.trace((s.d_sum / 32).cpu().numpy()))))
    out = os.path.join(a.out_dir, f"{a.config}_layer_scan.json")
    json.dump(rows, open(out, "w"), indent=2)
    for r in rows:
        print(f"  block {r['block']:>3}: |mean_g| {r['grad_norm']:.4g}  "
              f"trA {r['trA']:.4g}  trG {r['trG']:.4g}")
    print(f"\nwrote {out}")
    return 0
