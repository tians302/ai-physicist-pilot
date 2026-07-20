"""WP5: the generic 12-stage loop (PLAN.md target architecture).

load-capabilities -> hypothesize -> route -> plan -> validate-plan ->
execute -> analyze -> gate-check -> score -> interpret ->
iterate-or-conclude -> write-note

Discipline (unchanged from v1, now generic):
  * only registered capabilities; unsupported hypotheses END the run as
    `unsupported_hypothesis` (a measured behavior, not an error);
  * typed plans validated before execution;
  * engines produce RawRun; fixed analyzers produce ObservationBundles;
  * gates run on analyzer diagnostics BEFORE any LLM sees results;
  * the harness computes scores against answer keys; LLM narrates only
    gated facts; gate failure voids the conclusion regardless of text.

`corruption` names an entry in corruptions.CORRUPTIONS[kind]; it is the
single injection point (post-analysis, pre-gates) used by the WP9 matrix
and the WP10 gates-on/off ablation battery.
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from analyzers.callaway import analyze_callaway
from analyzers.elastic import analyze_elastic
from analyzers.eos import analyze_eos
from analyzers.expansion import analyze_expansion
from analyzers.transport import analyze_diffusion, analyze_gk
from benchmarks.v2 import get_benchmark_v2
from core.note2 import write_note_v2
from corruptions import CORRUPTIONS
from engines.callaway_adapter import CallawayAdapter
from engines.lammps_adapter import LammpsAdapter
from gates import run_gates
from gates.elastic import run_elastic_gates
from gates.eos import run_eos_gates
from gates.expansion import run_expansion_gates
from gates.transport import run_diffusion_gates, run_gk_gates
from registry import default_registry
from registry.router import route

LOOP_VERSION = "2.0"

_ANALYZERS = {"callaway": analyze_callaway, "eos": analyze_eos,
              "elastic": analyze_elastic, "expansion": analyze_expansion,
              "diffusion": analyze_diffusion, "gk": analyze_gk}
_GATE_RUNNERS = {"callaway": run_gates, "eos": run_eos_gates,
                 "elastic": run_elastic_gates,
                 "expansion": run_expansion_gates,
                 "diffusion": run_diffusion_gates, "gk": run_gk_gates}


def _executor(engine):
    if engine == "callaway_rta_si":
        return CallawayAdapter()
    if engine == "lammps":
        return LammpsAdapter(binary=os.environ.get("AIPHYS_LMP_BIN", "lmp"))
    raise ValueError(f"engine {engine!r} has no registered executor")


def run_loop_v2(benchmark_name, run_dir, seed=0, enforce_gates=True,
                corruption=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    trace = []

    def stage(name, **info):
        trace.append({"stage": name, "t": time.time(),
                      "utc": datetime.now(timezone.utc).isoformat(
                          timespec="seconds"), **info})

    def dump(obj, fname):
        (run_dir / fname).write_text(
            obj if isinstance(obj, str) else json.dumps(obj, indent=2))

    def finish(status, **extra):
        dump({"status": status, "loop_version": LOOP_VERSION,
              "trace": trace}, "loop_trace.json")
        return {"status": status, "run_dir": str(run_dir), **extra}

    bench = get_benchmark_v2(benchmark_name)
    registry = default_registry()

    # 1. load-capabilities
    stage("load_capabilities", capabilities=registry.ids())

    # 2. hypothesize (scripted registered hypothesis; LLM mode via WP11)
    hypothesis = bench.hypothesis
    stage("hypothesize", hypothesis=hypothesis)

    # 3. route (deterministic; registry authoritative)
    request = bench.request(hypothesis, f"{benchmark_name}_seed{seed}")
    decision = route(request, registry)
    dump(request.model_dump(mode="json"), "request.json")
    dump(decision.model_dump(mode="json"), "route_decision.json")
    stage("route", outcome=decision.outcome, capability=decision.capability_id)
    if not decision.accepted:
        return finish("unsupported_hypothesis", reasons=decision.reasons)

    manifest = registry.get(decision.capability_id)

    # 4-5. plan + validate (typed construction IS schema validation;
    # legacy Callaway plan uses its own validator)
    try:
        plan = bench.make_plan(hypothesis, decision,
                               f"{benchmark_name}_seed{seed}", seed)
        if hasattr(plan, "model_dump_json"):
            dump(plan.model_dump(mode="json"), "plan.json")
        else:
            from plans import validate_plan
            errors = validate_plan(plan)
            if errors:
                raise ValueError("; ".join(errors))
            dump(plan.to_dict(), "plan.json")
    except Exception as e:
        stage("validate_plan", errors=str(e))
        return finish("aborted_invalid_plan", errors=str(e))
    stage("validate_plan", errors=[])

    # 6. execute (allowlisted engine; RawRun evidence record)
    raw_run = _executor(manifest.engine).execute(plan, run_dir / "raw",
                                                 seed=seed)
    dump(raw_run.model_dump(mode="json"), "rawrun.json")
    stage("execute", engine=manifest.engine, exit_status=raw_run.exit_status)
    if raw_run.exit_status != "completed":
        return finish(f"engine_{raw_run.exit_status}",
                      conclusion={"valid": False, "hypothesis_supported": None})

    # 7. analyze (fixed, versioned)
    raw_outputs = json.loads((run_dir / "raw" / "raw_outputs.json").read_text())
    bundle, diag = _ANALYZERS[bench.kind](raw_outputs, raw_run)
    dump(bundle.model_dump(mode="json"), "bundle.json")
    stage("analyze", analyzer=bundle.analyzer, n_obs=len(bundle.observations))

    # (corruption injection point — WP9/WP10 harness experiments only)
    if corruption is not None:
        diag = CORRUPTIONS[bench.kind][corruption](diag)
        stage("inject_corruption", corruption_name=corruption)

    # 8. gate-check — BEFORE any narration
    gate_report = _GATE_RUNNERS[bench.kind](diag)
    dump(gate_report, "gate_report.json")
    stage("gate_check", passed=gate_report["passed"], enforced=enforce_gates)

    # 9. score (harness only)
    score_report = bench.score(bundle, diag)
    dump(score_report, "score_report.json")
    stage("score", success=score_report["success"])

    # 10. interpret (narration of gated facts only; scripted here)
    if gate_report["passed"] or not enforce_gates:
        narrative = ("Scripted narration: all reported facts are "
                     "harness-computed. See observation, gate, and score "
                     "tables; claims are conditional on the physical model "
                     "declared in the capability manifest.")
    else:
        narrative = ("Gate failure: no scientific narrative was generated. "
                     "See gate report.")
    stage("interpret")

    # 11. iterate-or-conclude (single-path in Phase 1B; search deferred)
    valid = gate_report["passed"] if enforce_gates else True
    conclusion = {
        "hypothesis_supported": bool(score_report["success"]) if valid else None,
        "valid": valid, "gates_enforced": enforce_gates,
        "corruption_injected": corruption,
    }
    stage("conclude", **conclusion)

    # 12. write-note (generic renderer)
    write_note_v2(run_dir, bench, manifest, plan, raw_run, bundle,
                  gate_report, score_report, narrative, conclusion, seed)
    stage("write_note")

    return finish("completed", conclusion=conclusion,
                  gate_passed=gate_report["passed"],
                  score=score_report["success"])
