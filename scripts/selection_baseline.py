#!/usr/bin/env python3
"""WP11: LLM-value experiment-selection baseline harness.

Task (pre-registered design): under a fixed budget of K experiment slots,
a selector chooses a sequence from the registered benchmark menu. Arms:

  random    — uniform choices (seeded)
  grid      — fixed round-robin over the menu
  scripted  — greedy heuristic: maximize property-family coverage first,
              then cheapest-per-new-capability
  llm       — the LLM proposes the next choice from the menu, constrained
              to menu items (requires an API key; the harness validates
              every choice — an off-menu reply falls back to `grid`)

Metrics per arm (means over seeds where stochastic):
  family_coverage      — distinct property families / total families
  capability_coverage  — distinct capabilities exercised / K
  cost_per_family      — total estimated cost / families covered

The measured per-benchmark costs come from the sandbox smoke runs and
are recalibrated on CARC before the pre-registered comparison.
"""
import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from registry import default_registry                               # noqa: E402

# measured sandbox walltimes (s) per benchmark, 2026-07-19; recalibrate on CARC
COST_S = {"silicon_boundary": 2.0, "si_eos": 3.0, "si_elastic": 6.0,
          "si_expansion": 25.0, "lj_diffusion": 8.0, "lj_kappa_gk": 18.0}

BENCH_TO_CAP = {"silicon_boundary": "si_kappa_callaway",
                "si_eos": "si_eos_sw_lammps",
                "si_elastic": "si_elastic_sw_lammps",
                "si_expansion": "si_expansion_sw_lammps",
                "lj_diffusion": "lj_diffusion_lammps",
                "lj_kappa_gk": "lj_kappa_gk_lammps"}


def _families():
    reg = default_registry()
    return {b: reg.get(c).property_family for b, c in BENCH_TO_CAP.items()}


def select_random(menu, k, rng):
    return [rng.choice(menu) for _ in range(k)]


def select_grid(menu, k, rng=None):
    return [menu[i % len(menu)] for i in range(k)]


def select_scripted(menu, k, rng=None):
    fam = _families()
    chosen, seen_fam, seen_cap = [], set(), set()
    for _ in range(k):
        def key(b):
            new_fam = fam[b] not in seen_fam
            new_cap = b not in seen_cap
            return (not new_fam, not new_cap, COST_S[b])
        pick = sorted(menu, key=key)[0]
        chosen.append(pick)
        seen_fam.add(fam[pick])
        seen_cap.add(pick)
    return chosen


def select_llm(menu, k, rng):
    """Constrained LLM selection. Every reply is validated against the
    menu (registry-authoritative, same principle as routing); invalid
    replies fall back to the grid choice for that slot."""
    from core.llm import LLM
    llm = LLM()
    if llm.mode == "scripted":
        raise RuntimeError(
            "LLM arm needs an API key (OPENAI_API_KEY or ANTHROPIC_API_KEY); "
            "run the other arms now and this arm after WP11 key setup")
    fam = _families()
    chosen = []
    for i in range(k):
        reply = llm.backend.ask(
            "You are selecting simulation experiments to maximize coverage "
            "of distinct physical property families under a fixed budget.\n"
            f"Menu (benchmark: family, est. cost s): "
            f"{[(b, fam[b], COST_S[b]) for b in menu]}\n"
            f"Already chosen: {chosen}\n"
            "Reply with exactly one benchmark name from the menu.",
            max_tokens=20).strip()
        chosen.append(reply if reply in menu else select_grid(menu, k)[i])
    return chosen


SELECTORS = {"random": select_random, "grid": select_grid,
             "scripted": select_scripted, "llm": select_llm}


def evaluate(seq):
    fam = _families()
    families = {fam[b] for b in seq}
    caps = set(seq)
    cost = sum(COST_S[b] for b in seq)
    return {"sequence": seq,
            "family_coverage": len(families) / len(set(fam.values())),
            "capability_coverage": len(caps) / len(seq),
            "cost_s": cost,
            "cost_per_family_s": cost / len(families)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--arms", nargs="+",
                    default=["random", "grid", "scripted"])
    ap.add_argument("--out", default="reports/selection_baseline")
    args = ap.parse_args()

    menu = sorted(COST_S)
    results = {}
    for arm in args.arms:
        evals = []
        for s in range(args.seeds):
            rng = random.Random(s)
            seq = SELECTORS[arm](menu, args.k, rng)
            evals.append(evaluate(seq))
        agg = {m: sum(e[m] for e in evals) / len(evals)
               for m in ("family_coverage", "capability_coverage",
                         "cost_s", "cost_per_family_s")}
        results[arm] = {"mean": agg, "runs": evals}
        print(f"{arm:9s} coverage={agg['family_coverage']:.2f} "
              f"caps={agg['capability_coverage']:.2f} "
              f"cost/family={agg['cost_per_family_s']:.1f}s")

    out = Path(__file__).resolve().parent.parent / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "selection_metrics.json").write_text(json.dumps(
        {"generated_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
         "k": args.k, "n_seeds": args.seeds, "results": results,
         "note": "llm arm pending API key; costs recalibrated on CARC "
                 "before the pre-registered comparison"}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
