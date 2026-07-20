#!/usr/bin/env python3
"""CLI for the generic v2 loop (WP5+). Examples:

    python run_loop2.py --benchmark silicon_boundary --run-dir runs/v2_si01
    python run_loop2.py --benchmark si_eos --run-dir runs/v2_eos
    python run_loop2.py --benchmark si_elastic --run-dir runs/v2_cij --seed 3
    python run_loop2.py --benchmark si_eos --run-dir runs/x --corruption unit_slip --no-gates
"""
import argparse
import json

from core.loop2 import run_loop_v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="silicon_boundary")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gates", action="store_true",
                    help="ablation: gates computed but not enforced")
    ap.add_argument("--corruption", default=None,
                    help="named corruption to inject (harness experiments)")
    args = ap.parse_args()

    out = run_loop_v2(args.benchmark, args.run_dir, seed=args.seed,
                      enforce_gates=not args.no_gates,
                      corruption=args.corruption)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
