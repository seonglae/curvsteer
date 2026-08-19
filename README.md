<h1 align="center">curvsteer</h1>

<p align="center">
  <b>Steering is a curvature problem, not a direction problem.</b><br>
  Rank-<i>r</i> <b>parameter-space</b> edits, curvature-normalised, with no dictionary.<br>
  <b>JAX end to end.</b>
</p>

<p align="center">
  <img alt="tests" src="https://github.com/seonglae/curvsteer/actions/workflows/ci.yml/badge.svg">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-3776ab">
  <img alt="jax" src="https://img.shields.io/badge/JAX-only-b45309">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-a93a20">
</p>

An activation steering vector adds a constant to the residual stream. That is a
**bias edit**, and bias is one parameter group among several. The general edit is

```
h  ->  h + alpha * (M h)      ||M||_F = 1,   rank(M) = r
```

which is exactly what editing every matrix that reads the residual there does:
`W_read -> W_read (I + alpha M)`. Setting `r = 0` recovers the steering vector, so
**rank is a free parameter the activation framing does not expose**.

Choosing `M` is then a constrained optimisation rather than a search for a
direction. Moving behaviour by `alpha <M, mean_g>` while paying capability
`(alpha^2/2) vec(M)^T H_c vec(M)` gives `M* = H_c^-1 mean_g`, and under the
Kronecker approximation `H_c ~= A (x) G`:

```
M* = G^-1 mean_g A^-1          A = E[h h^T],  G = E[delta delta^T]
```

Both inverses are `d_model x d_model` and exact. **No sparse autoencoder, no
dictionary, no feature labels, no pretrained interpretability artifact.** That is
what lets the experiment run on a current model at all: no dictionary release
exists for the checkpoint the headline is measured on.

## Result

`google/gemma-4-12B`, block 24 of 48, IMDB sentiment as behaviour and WikiText-2
cross-entropy as capability. Shift is the change in `L_b`; **more negative is more
steering**. Every `alpha` is solved against the *measured* cross-entropy until the
achieved cost equals the target, so an over-confident curvature estimate cannot
win by accident.

| cost (nats) | `M*` (rank) | raw gradient | rank-0 bias | curvature-only | random floor |
|---|---|---|---|---|---|
| 0.05 | **-0.3039** (r8) | -0.0356 | -0.1037 | -0.0697 | -0.0017 |
| 0.10 | **-0.3576** (r4) | -0.0701 | -0.1302 | -0.1004 | -0.0117 |
| 0.20 | **-0.3631** (r4) | -0.0574 | -0.2120 | -0.1130 | -0.0265 |
| 0.50 | -0.1817 (r8) | -0.1252 | **-0.2998** | **-0.3158** | -0.0359 |

**5 to 8 times the raw behaviour gradient at identical measured capability cost,
up to 0.2 nats.** Past that it degrades, and two simpler baselines overtake it.

### What does not work

This is in the README because you would find it in the sweep output anyway, and
because two of the three pre-registered comparisons came back mixed.

| pre-registered primary | outcome | ratios across the four costs |
|---|---|---|
| vs raw behaviour gradient | **confirmed 4/4** | 8.5x, 5.1x, 6.3x, 1.5x |
| vs rank-0 bias edit | **mixed 3/4** | 2.9x, 2.7x, 1.7x, **0.6x** |
| vs curvature-only | **mixed 3/4** | 4.4x, 3.6x, 3.2x, **0.6x** |

`M*` is the **only arm whose effect turns down at cost 0.50**. Every baseline rises
monotonically through it. This holds at every rank, and it is what loses both
mixed primaries. The derivation is first-order in behaviour and second-order in
capability, so it is valid in a small-amplitude regime; past that, multiplying the
residual by a fixed matrix damages the representation in a way that drags
behaviour back toward baseline, while adding a constant vector does not.

The available excuse, that 0.50 nats is a 19 per cent relative degradation and not
a budget anyone would use, is not taken: the four cost points were fixed before
any number existed precisely so that argument would be unavailable.

`M*` also carries roughly 3x the per-example variance of the raw gradient. It buys
a larger mean shift at the price of acting much less uniformly across inputs.

## Which ingredient does the work

The four arms differ by which of two ingredients they carry. At matched cost 0.05:

| edit | curvature-normalised | conditional | shift |
|---|---|---|---|
| bias, raw gradient | no | no | -0.0301 |
| rank-r, raw gradient | no | **yes** | -0.0353 |
| bias, normalised | **yes** | no | -0.1037 |
| `M*` rank 8 | **yes** | **yes** | **-0.3039** |

Curvature normalisation is worth about **3.4x on its own**. Conditionality is worth
**almost nothing without it** and 2.9x with it. The two compose to roughly 10x.

This contradicts the hypothesis the project started from, which was that the
restriction to bias edits was the mistake. It is not. The restriction that matters
is the missing curvature normalisation, and **rank only becomes worth anything once
the direction has been normalised**.

The reason is visible in how rank behaves per object. The raw behaviour gradient is
one direction: sixteen points across four ranks and four costs are identical to
three decimals, so a second component adds a component with nothing in it.

| object | rank 1 | rank 2 | rank 4 | rank 8 |
|---|---|---|---|---|
| `M*` | -0.0766 | -0.1031 | -0.2695 | **-0.3039** |
| raw gradient | -0.0353 | -0.0346 | -0.0350 | -0.0356 |
| curvature-only | -0.0489 | -0.0427 | -0.0414 | -0.0697 |

**Amplitude is not the mechanism.** The rank-1 curvature-only direction solves to
`alpha = 13911` against `M*` rank 8's `alpha = 37.9`, which is 367 times more
amplitude for the same measured capability damage, and it produces 6.2x *less*
behaviour change. Headroom alone is not sufficient; the direction has to know
where the behaviour is.

## Try it in a second

No model, no data, no GPU, no download:

```bash
pip install -e .
curvsteer demo
```

```
 rank |         M* |  raw gradient |  curvature only
------------------------------------------------------
    1 |    55.1743 |        3.4433 |          5.5944
    2 |    70.2264 |        4.1951 |          5.9915
    4 |    89.7891 |        5.9313 |          7.9504
    8 |   107.1416 |        8.0282 |          3.0155
------------------------------------------------------
random |     0.2221 |

M* over raw gradient:    13.3x
M* over curvature only:  13.5x
```

That is the **algebra**, on a synthetic quadratic model where the optimal edit and
its cost are both available in closed form. It is not the language-model
measurement, and the command says so when it runs.

## Reproducing the measured result

```bash
pip install -e ".[sweep]"
curvsteer sweep --config gemma4 --n-pairs 256 --n-cap 96
```

The model is loaded through the official JAX Gemma library where it covers the
checkpoint, and otherwise by reading the published weights and running a forward
written in `model.py`. The fallback is not a workaround: a method that needs no
dictionary should not be blocked by whether someone has shipped a wrapper for the
checkpoint, which is the same argument the dictionary section below makes.

| claim | command |
|---|---|
| the demo separation, closed form | `curvsteer demo` |
| the headline table, no dictionary exists here | `curvsteer sweep --config gemma4` |
| the dictionary comparison, where one does exist | `curvsteer sweep --config gemma2` |
| which site, rather than an inherited one | `curvsteer layer-scan --config gemma4` |

## Design notes

For a reader judging the code rather than the result, these are the places where
something could have gone silently wrong, and what was done about it.

**There is no hook, and that is the point.** The residual at the site is an
ordinary intermediate value: `model.residual_at_site(tokens)` returns it and
`model.from_site(h, M, alpha)` continues the stack from it. `delta = dL/dh` is
then one `jax.grad` with respect to that value, and the same `h` object feeds
both the statistics and the intervention, so they cannot end up on different
tensors.

This is worth stating because the framework choice removed a bug class rather
than moving it. In a define-by-run framework the same experiment needs a forward
hook, and on a frozen model whose input is token ids the residual carries no
autodiff edge at all, so capturing it means adding a zero leaf that requires grad
purely to manufacture the edge. Splitting the forward at the site makes that
unnecessary. The one real cost is that the model's layer stack has to be
addressable rather than opaque, which is what `model.py` provides.

**Cost is measured, never predicted.** `solve.py` solves `alpha` against the true
cross-entropy. A direction that damages the model in a way the curvature estimate
failed to anticipate is caught and throttled. The off-target threshold is **one
constant**, used by both the printed warning and the exclusion filter; it was
previously written separately in the two places, which is one run classifiable two
ways.

**Rank 0 is a different object and is refused, not guessed at.**
`rank_trunc(M, 0)` raises rather than quietly returning something matrix-shaped.
The bias edit has its own constructor and `Site` dispatches on dimensionality.
`tests/test_edits.py` pins the degenerate case: under `A = G = I` the curvature
normalisation has nothing to do and `M*` must equal the raw gradient.

**The dictionary baseline is built in its strong form.** `sae.py` selects features
two ways: by label correlation, which is how a practitioner picks a steering
feature, and by first-order gradient alignment `d_j^T mean_g e_j`, which is the
same objective `M*` optimises. It also prints the best correlation against a noise
floor, so a reader can check the baseline was not vacuous.

**A broken reference is refused rather than swept.** An instruct checkpoint scored
on raw text reached WikiText cross-entropy 13.92 here, worse than uniform over its
own 262k vocabulary, and would have run to completion producing a full table of
deltas from nonsense. `--max-base-ce` aborts instead.

**Guards added after the corresponding failure**, each kept because it caught
something: `--verify-batch` checks the batched behaviour evaluation against the
one-at-a-time path; a cost landing a hair below baseline once made the solver
raise a negative number to a fractional power and send a complex `alpha` into the
model; and every swept point reports its achieved cost so an off-target point
cannot silently decide a comparison.

## Limitations

- One behaviour, one corpus, one site. IMDB sentiment, WikiText capability, block
  24. Whether this transfers to refusal, factuality or style is untested.
- **The site was inherited, not chosen.** Block 24 is the midpoint by analogy with
  a smaller model's block 12, which was the layer a dictionary release happened to
  be trained on. `curvsteer layer-scan` is the tool that would replace that with a
  measured choice; it is a post-hoc diagnostic and was written before the
  registered verdict was known.
- Capability is proxied by WikiText while behaviour is IMDB. An edit that wrecked
  review-like text while sparing WikiText would have its cost understated.
- `H_c` is estimated from about 12k token gradients in 3840 dimensions and then
  inverted, which amplifies its least reliable directions. The matched-cost
  protocol is the defence, not a claim that the estimate is good.
- Rank 8 is the top of the grid. The true optimum could lie beyond it.
- The `M*` versus `mean_g` standard errors are unpaired, though both are scored on
  the same 122 held-out examples, so the true separation is understated.

## Why the headline is not the smaller model

The same design ran first on `google/gemma-2-2b-it`, where a dictionary exists.
There `M*` beat the best of 16384 dictionary features at all four costs (22.6x,
12.7x, 6.5x, 2.7x) and the raw gradient at all four (12.4x, 9.5x, 7.6x, 6.2x).

That run is **not** the headline, and the reason is itself a finding about method.
The model was chosen because a dictionary existed for it, which inverts the
argument: this method needs no dictionary, so letting dictionary availability pick
the model pinned a dependency-free method to a two-generation-old checkpoint and
forced a base-dictionary-on-instruct-model mismatch. It is retained as
`--config gemma2` because it carries the only dictionary comparison in the package.

Two things replicate across the two models and one does not. The curvature
normalisation advantage replicates. The rank dissociation replicates and is
stronger on the larger model. The **optimal rank moves**, from 2 to 4-8, which is
consistent with 2304 against 3840 hidden units but is a two-point observation, not
a scaling law.

## Citation

```bibtex
@software{cho2026curvsteer,
  author  = {Cho, Seonglae},
  title   = {curvsteer: curvature-normalised rank-r parameter-space steering},
  year    = {2026},
  url     = {https://github.com/seonglae/curvsteer}
}
```

## License

MIT.
