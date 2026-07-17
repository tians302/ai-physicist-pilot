#!/usr/bin/env python3
"""Run one closed-loop rediscovery experiment.

Examples:
    python run_loop.py --benchmark silicon_boundary --run-dir runs/si_01
    python run_loop.py --benchmark silicon_boundary --run-dir runs/ablate_01 --no-gates
    ANTHROPIC_API_KEY=... python run_loop.py --benchmark silicon_boundary --run-dir runs/llm_01
"""
import argparse
import json

from core.loop import run_loop


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="silicon_boundary")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scripted", action="store_true",
                   help="force deterministic scripted mode even if an API key is set")
    p.add_argument("--no-gates", action="store_true",
                   help="ablation mode: gates are recorded but not enforced")
    args = p.parse_args()

    summary = run_loop(args.benchmark, args.run_dir, seed=args.seed,
                       scripted=args.scripted, enforce_gates=not args.no_gates)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
