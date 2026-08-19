"""curvsteer command line.

    curvsteer demo                     no model, no data, about a second
    curvsteer sweep --config gemma4    the measured result, needs a GPU
    curvsteer layer-scan --config ...  which site, in one backward pass
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="curvsteer", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("demo", help="closed-form separation of the two ingredients")

    s = sub.add_parser("sweep", help="matched-cost sweep on a real model")
    s.add_argument("--config", default="gpt2")
    s.add_argument("--device", default="cuda")
    s.add_argument("--n-pairs", type=int, default=256)
    s.add_argument("--n-cap", type=int, default=96)
    s.add_argument("--cont-len", type=int, default=48)
    s.add_argument("--n-random", type=int, default=1)
    s.add_argument("--beh-bs", type=int, default=16)
    s.add_argument("--out-dir", default="results")
    s.add_argument("--max-base-ce", type=float, default=6.0,
                   help="refuse to sweep if the unsteered model cannot model the "
                        "capability corpus at all")
    s.add_argument("--verify-batch", action="store_true",
                   help="check the batched behaviour path against one-at-a-time "
                        "and exit")

    l = sub.add_parser("layer-scan", help="post-hoc diagnostic over every block")
    l.add_argument("--config", default="gpt2")
    l.add_argument("--device", default="cuda")
    l.add_argument("--out-dir", default="results")

    a = ap.parse_args(argv)
    if a.cmd == "demo":
        from .demo import main as demo_main
        return demo_main()
    if a.cmd == "sweep":
        from .sweep import run_sweep
        return run_sweep(a)
    if a.cmd == "layer-scan":
        from .sweep import run_layer_scan
        return run_layer_scan(a)
    return 1


if __name__ == "__main__":
    sys.exit(main())
