#!/usr/bin/env python3
"""WP12: cross-model sensitivity — the same EOS + elastic protocols run
under two independently parameterized Si potentials (SW 1985, Tersoff
1988). The spread bounds how model-conditional each observable is; a
cost ledger is accumulated from RawRun walltimes.

Run: AIPHYS_LMP_BIN=<lmp> python scripts/cross_model.py [--out DIR]
"""
import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyzers import analyze_elastic, analyze_eos                  # noqa: E402
from contracts import ResourceBudget                                # noqa: E402
from contracts.lammps import LammpsElasticPlan, LammpsEosPlan       # noqa: E402
from engines.lammps_adapter import LammpsAdapter                    # noqa: E402
from gates.elastic import run_elastic_gates                         # noqa: E402
from gates.eos import run_eos_gates                                 # noqa: E402

MODELS = ["sw_si_native_1985", "tersoff_si_native_1988"]


def run_model(adapter, model_key, workdir):
    out = {"model": model_key}
    eos_plan = LammpsEosPlan(
        plan_id=f"xm_eos_{model_key}", capability_id="si_eos_sw_lammps",
        hypothesis="Cross-model EOS comparison run (WP12).",
        model_key=model_key,
        resource_budget=ResourceBudget(max_walltime_s=300))
    run = adapter.execute(eos_plan, workdir / "eos")
    raw = json.loads((workdir / "eos" / "raw_outputs.json").read_text())
    _, diag = analyze_eos(raw, run)
    out["eos"] = {"gates_passed": run_eos_gates(diag)["passed"],
                  "B0_GPa": diag["B0_GPa"], "V0_A3": diag["V0_A3"],
                  "a0_A": diag["V0_A3"] ** (1 / 3),
                  "B0_prime": diag["B0_prime"],
                  "walltime_s": run.resource_usage["walltime_s"]}

    ela_plan = LammpsElasticPlan(
        plan_id=f"xm_cij_{model_key}", capability_id="si_elastic_sw_lammps",
        hypothesis="Cross-model elastic comparison run (WP12).",
        model_key=model_key,
        resource_budget=ResourceBudget(max_walltime_s=300))
    run = adapter.execute(ela_plan, workdir / "cij")
    raw = json.loads((workdir / "cij" / "raw_outputs.json").read_text())
    _, diag = analyze_elastic(raw, run)
    out["elastic"] = {"gates_passed": run_elastic_gates(diag)["passed"],
                      "C11_GPa": diag["C11_GPa"], "C12_GPa": diag["C12_GPa"],
                      "C44_GPa": diag["C44_GPa"],
                      "walltime_s": run.resource_usage["walltime_s"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/cross_model")
    args = ap.parse_args()

    adapter = LammpsAdapter(binary=os.environ.get("AIPHYS_LMP_BIN", "lmp"))
    results = []
    with tempfile.TemporaryDirectory() as td:
        for m in MODELS:
            results.append(run_model(adapter, m, Path(td) / m))

    def spread(path1, path2, key):
        a = results[0][path1][key] if path1 else results[0][key]
        b = results[1][path1][key] if path1 else results[1][key]
        return abs(a - b) / (0.5 * (abs(a) + abs(b)))

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": results,
        "relative_spread": {
            "a0": spread("eos", None, "a0_A"),
            "B0": spread("eos", None, "B0_GPa"),
            "C11": spread("elastic", None, "C11_GPa"),
            "C12": spread("elastic", None, "C12_GPa"),
            "C44": spread("elastic", None, "C44_GPa"),
        },
        "cost_ledger_s": {r["model"]: r["eos"]["walltime_s"]
                          + r["elastic"]["walltime_s"] for r in results},
        "note": "Same protocols, same gates, two independent potentials. "
                "Spread = |a-b| / mean(|a|,|b|); observables with large "
                "spread are strongly model-conditional and must be "
                "reported as such (PLAN.md).",
    }

    out = Path(__file__).resolve().parent.parent / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "cross_model.json").write_text(json.dumps(summary, indent=2))

    rows = ["# Cross-model sensitivity (WP12)", "",
            f"*{summary['generated_utc']}*", "",
            "| Observable | SW 1985 | Tersoff 1988 | rel. spread |",
            "|---|---|---|---|"]
    e0, e1 = results[0]["eos"], results[1]["eos"]
    c0, c1 = results[0]["elastic"], results[1]["elastic"]
    rows += [
        f"| a0 (A) | {e0['a0_A']:.4f} | {e1['a0_A']:.4f} | {summary['relative_spread']['a0']:.3f} |",
        f"| B0 (GPa) | {e0['B0_GPa']:.2f} | {e1['B0_GPa']:.2f} | {summary['relative_spread']['B0']:.3f} |",
        f"| C11 (GPa) | {c0['C11_GPa']:.2f} | {c1['C11_GPa']:.2f} | {summary['relative_spread']['C11']:.3f} |",
        f"| C12 (GPa) | {c0['C12_GPa']:.2f} | {c1['C12_GPa']:.2f} | {summary['relative_spread']['C12']:.3f} |",
        f"| C44 (GPa) | {c0['C44_GPa']:.2f} | {c1['C44_GPa']:.2f} | {summary['relative_spread']['C44']:.3f} |",
        "", f"Gates passed: SW eos/elastic = {e0['gates_passed']}/{c0['gates_passed']}, "
            f"Tersoff = {e1['gates_passed']}/{c1['gates_passed']}",
        f"Cost ledger (s): {summary['cost_ledger_s']}"]
    (out / "cross_model.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows[4:]))


if __name__ == "__main__":
    main()
