"""Generic research-note renderer for the v2 loop (WP5/WP6). Works for
any capability: renders observations (scalar/curve/tensor), gate and
score tables, model-conditional caveats from the manifest, and full
provenance. No property-specific prose."""
import platform
from datetime import datetime, timezone


def _obs_rows(bundle):
    rows = []
    for o in bundle.observations:
        v = o.value
        cond = ", ".join(f"{k}={c.value:g} {c.unit}"
                         for k, c in o.conditions.items())
        if v.kind == "scalar":
            sig = f" ± {v.sigma:.3g}" if v.sigma is not None else ""
            rows.append(f"| {o.name} | scalar | {v.value:.6g}{sig} {v.unit} | {cond} |")
        elif v.kind == "curve":
            rows.append(f"| {o.name} | curve | {len(v.x)} pts, x [{min(v.x):g},"
                        f" {max(v.x):g}] {v.x_unit}, y in {v.y_unit} | {cond} |")
        else:
            n = len(v.value)
            rows.append(f"| {o.name} | tensor | {n}x{len(v.value[0])} {v.unit},"
                        f" diag {v.value[0][0]:.4g}.. | {cond} |")
    return "\n".join(rows)


def _gate_rows(report):
    return "\n".join(f"| {g['name']} | {'PASS' if g['passed'] else 'FAIL'} | "
                     f"{g['detail']} |" for g in report["gates"])


def _score_rows(report):
    return "\n".join(f"| {c['name']} | {'PASS' if c['passed'] else 'FAIL'} | "
                     f"{c['detail']} |" for c in report["checks"])


def write_note_v2(run_dir, bench, manifest, plan, raw_run, bundle,
                  gate_report, score_report, narrative, conclusion, seed):
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    flag = ""
    if not gate_report["passed"]:
        flag += ("\n> **GATE FAILURE — conclusions below are NOT valid "
                 "claims.** Retained for the record only.\n")
    if not conclusion["gates_enforced"]:
        flag += "\n> **Gates were NOT enforced (ablation mode).**\n"
    if conclusion.get("corruption_injected"):
        flag += (f"\n> **Harness experiment: corruption "
                 f"'{conclusion['corruption_injected']}' was injected.**\n")

    info = "\n".join(f"- {ln}" for ln in score_report.get("info_only", []))
    caveats = "\n".join(f"- {m}" for m in
                        manifest.physical_model.known_failure_modes)
    notes = "\n".join(f"- {n}" for n in bundle.notes)
    plan_id = getattr(plan, "plan_id", "?")

    note = f"""# Research note: {plan_id}
{flag}
*Generated {ts} · loop v2 · capability `{manifest.capability_id}` v{manifest.version} · engine `{manifest.engine}` · seed {seed}*

## Hypothesis

{bench.hypothesis}

## Capability (trusted manifest)

Property **{manifest.property_name}** ({manifest.property_family}) via protocol
`{manifest.protocol}` on engine `{manifest.engine}`; physical model
**{manifest.physical_model.name}** ({manifest.physical_model.version},
source: {manifest.physical_model.source}). Analyzer
`{bundle.analyzer}` v{bundle.analyzer_version}.

**Model-conditional caveats (from manifest):**

{caveats}

## Observations (analyzer output, unit-checked)

| Name | Kind | Value | Conditions |
|---|---|---|---|
{_obs_rows(bundle)}

**Analyzer notes:**

{notes}

## Validity gates (v{gate_report['version']}, run before narration)

| Gate | Verdict | Detail |
|---|---|---|
{_gate_rows(gate_report)}

## Answer-key score (harness-computed)

| Check | Verdict | Detail |
|---|---|---|
{_score_rows(score_report)}

**Benchmark success: {score_report['success']}**

**Info-only (never scored):**

{info if info else '- none'}

## Narrative

{narrative}

## Provenance

- Request/route: [`request.json`](request.json), [`route_decision.json`](route_decision.json)
- Plan: [`plan.json`](plan.json) (sha256 `{raw_run.plan_sha256[:16]}...`)
- RawRun: [`rawrun.json`](rawrun.json) — {len(raw_run.artifacts)} checksummed artifacts, engine `{raw_run.environment.engine_version}`
- Bundle/gates/score: [`bundle.json`](bundle.json), [`gate_report.json`](gate_report.json), [`score_report.json`](score_report.json)
- Python {platform.python_version()}, seed {seed}
"""
    (run_dir / "note.md").write_text(note)
    return note
