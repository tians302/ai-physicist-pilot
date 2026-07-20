#!/usr/bin/env python3
"""WP10: multi-seed battery with gates-on/off ablation.

Arms per benchmark (PREREGISTRATION_DRAFT.md §5):
  clean/gates-on      -> recovery rate
  clean/gates-off     -> control
  corrupted/gates-on  -> unflagged-false-claim rate (expected ~0)
  corrupted/gates-off -> unflagged-false-claim rate (HEADLINE comparison)

Corruption for seed i is CORRUPTIONS[kind][i mod K] (deterministic).
An 'unflagged false claim' = corrupted run whose conclusion is
valid=True AND benchmark success=True.

Usage:
  python scripts/run_battery.py --benchmarks silicon_boundary --seeds 30 \\
      --out reports/battery/full
Sandbox pilot uses small --seeds; CARC runs the pre-registered N=30.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.v2 import REGISTRY_V2                               # noqa: E402
from core.loop2 import run_loop_v2                                  # noqa: E402
from corruptions import CORRUPTIONS                                 # noqa: E402
from scripts.stats import fmt_ci, wilson                            # noqa: E402


def run_arm(bench_name, kind, seeds, gates_on, corrupted, workdir):
    corr_names = sorted(CORRUPTIONS[kind])
    results = []
    for s in seeds:
        corruption = corr_names[s % len(corr_names)] if corrupted else None
        tag = f"{'c' if corrupted else 'x'}{'on' if gates_on else 'off'}_s{s}"
        out = run_loop_v2(bench_name, workdir / tag, seed=s,
                          enforce_gates=gates_on, corruption=corruption)
        c = out.get("conclusion", {})
        results.append({
            "seed": s, "corruption": corruption, "status": out["status"],
            "gate_passed": out.get("gate_passed"),
            "valid": c.get("valid"), "score": out.get("score"),
            "unflagged_false_claim": bool(corrupted and c.get("valid")
                                          and out.get("score")),
            "recovered": bool(not corrupted and out["status"] == "completed"
                              and out.get("gate_passed") and out.get("score")),
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", default=["silicon_boundary"])
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--out", default="reports/battery/pilot")
    ap.add_argument("--runs-dir", default=None,
                    help="where per-run artifacts go (default <out>/runs)")
    ap.add_argument("--arm", default=None,
                    choices=["clean_on", "clean_off", "corr_on", "corr_off"],
                    help="run ONE arm and cache it (for chunked execution); "
                         "omit to run/aggregate all arms")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    out = root / args.out
    runs = Path(args.runs_dir) if args.runs_dir else out / "runs"
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    arm_specs = {"clean_on": (True, False), "clean_off": (False, False),
                 "corr_on": (True, True), "corr_off": (False, True)}

    if args.arm:                       # chunked mode: one arm, cache, exit
        for bname in args.benchmarks:
            kind = REGISTRY_V2[bname].kind
            gates_on, corrupted = arm_specs[args.arm]
            res = run_arm(bname, kind, seeds, gates_on, corrupted,
                          runs / bname / args.arm)
            cache = out / f"arm_{bname}_{args.arm}.json"
            cache.write_text(json.dumps(res, indent=2))
            print(f"cached {cache.name} ({len(res)} runs)")
        return

    all_metrics = {}
    for bname in args.benchmarks:
        kind = REGISTRY_V2[bname].kind
        arms = {}
        for arm, (gates_on, corrupted) in arm_specs.items():
            cache = out / f"arm_{bname}_{arm}.json"
            if cache.exists():
                arms[arm] = json.loads(cache.read_text())
            else:
                arms[arm] = run_arm(bname, kind, seeds, gates_on, corrupted,
                                    runs / bname / arm)

        n = len(seeds)
        rec = sum(r["recovered"] for r in arms["clean_on"])
        ufc_on = sum(r["unflagged_false_claim"] for r in arms["corr_on"])
        ufc_off = sum(r["unflagged_false_claim"] for r in arms["corr_off"])
        all_metrics[bname] = {
            "n_seeds": n,
            "recovery": {"k": rec, "n": n, "wilson95": wilson(rec, n)},
            "unflagged_false_claims_gates_on":
                {"k": ufc_on, "n": n, "wilson95": wilson(ufc_on, n)},
            "unflagged_false_claims_gates_off":
                {"k": ufc_off, "n": n, "wilson95": wilson(ufc_off, n)},
            "arms": arms,
        }
        print(f"{bname}: recovery {fmt_ci(rec, n)} | UFC on "
              f"{fmt_ci(ufc_on, n)} | UFC off {fmt_ci(ufc_off, n)}")

    payload = {"generated_utc": datetime.now(timezone.utc).isoformat(
                   timespec="seconds"),
               "seeds": seeds, "metrics": all_metrics,
               "note": "Headline claims require the pre-registered N on "
                       "CARC after the WP7 freeze; smaller runs are pilots."}
    (out / "battery_metrics.json").write_text(json.dumps(payload, indent=2))

    md = ["# Battery metrics", f"*{payload['generated_utc']} — N = "
          f"{len(seeds)} seeds per arm.*", "",
          "| Benchmark | Recovery | Unflagged false claims (gates ON) | "
          "Unflagged false claims (gates OFF) |", "|---|---|---|---|"]
    for b, m in all_metrics.items():
        md.append(f"| {b} | {fmt_ci(m['recovery']['k'], m['recovery']['n'])} |"
                  f" {fmt_ci(m['unflagged_false_claims_gates_on']['k'], len(seeds))} |"
                  f" {fmt_ci(m['unflagged_false_claims_gates_off']['k'], len(seeds))} |")
    (out / "battery_report.md").write_text("\n".join(md) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
