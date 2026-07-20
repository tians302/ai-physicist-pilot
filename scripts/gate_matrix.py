#!/usr/bin/env python3
"""WP9: gate sensitivity/selectivity matrix. For every registered
(capability kind, corruption) pair, apply the corruption to clean
diagnostics and record which gates fail. Outputs JSON + markdown to
reports/gate_matrix/.

  sensitivity  = intended gate fails when its corruption is applied
  selectivity  = no OTHER gate fails (collateral count = 0)

Run: python scripts/gate_matrix.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corruptions import CORRUPTIONS, INTENDED_GATE                  # noqa: E402
from corruptions.clean import CLEAN_FACTORIES                       # noqa: E402
from gates import run_gates                                         # noqa: E402
from gates.elastic import run_elastic_gates                         # noqa: E402
from gates.eos import run_eos_gates                                 # noqa: E402
from gates.expansion import run_expansion_gates                     # noqa: E402
from gates.transport import run_diffusion_gates, run_gk_gates       # noqa: E402

RUNNERS = {"callaway": run_gates, "eos": run_eos_gates,
           "elastic": run_elastic_gates, "expansion": run_expansion_gates,
           "diffusion": run_diffusion_gates, "gk": run_gk_gates}


def main():
    rows = []
    for kind, table in CORRUPTIONS.items():
        clean = CLEAN_FACTORIES[kind]()
        base = RUNNERS[kind](clean)
        assert base["passed"], f"clean {kind} diagnostics must pass gates"
        for cname, fn in table.items():
            report = RUNNERS[kind](fn(clean))
            failed = [g["name"] for g in report["gates"] if not g["passed"]]
            intended = INTENDED_GATE[(kind, cname)]
            rows.append({
                "kind": kind, "corruption": cname,
                "intended_gate": intended,
                "detected_by_intended": intended in failed,
                "failed_gates": failed,
                "collateral_gates": [g for g in failed if g != intended],
            })

    n = len(rows)
    sens = sum(r["detected_by_intended"] for r in rows)
    sel = sum(not r["collateral_gates"] for r in rows)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_pairs": n,
        "sensitivity": sens / n,
        "selectivity": sel / n,
        "rows": rows,
    }

    out = Path(__file__).resolve().parent.parent / "reports" / "gate_matrix"
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate_matrix.json").write_text(json.dumps(summary, indent=2))

    md = ["# Gate sensitivity/selectivity matrix (WP9)", "",
          f"*Generated {summary['generated_utc']} — {n} corruption/gate "
          f"pairs. Sensitivity (intended gate fires): "
          f"**{sens}/{n}**. Selectivity (no collateral): **{sel}/{n}**.*",
          "", "| Kind | Corruption | Intended gate | Detected | Collateral |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['kind']} | {r['corruption']} | {r['intended_gate']} |"
                  f" {'YES' if r['detected_by_intended'] else '**MISS**'} |"
                  f" {', '.join(r['collateral_gates']) or '—'} |")
    (out / "gate_matrix.md").write_text("\n".join(md) + "\n")

    print(f"sensitivity {sens}/{n}, selectivity {sel}/{n} -> {out}")
    return 0 if sens == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
