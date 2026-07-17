"""The closed loop: a FIXED state machine. The agent fills content, never order.

Stages:
  hypothesize -> plan -> validate_plan -> execute -> gate_check
  -> interpret -> conclude -> write_note

Discipline properties enforced here:
  * plans are schema-validated before execution (invalid plan aborts);
  * engines come only from the allowlist registry;
  * gates run BEFORE the LLM sees any result;
  * the answer-key score is computed by the harness, never by the LLM;
  * a gate failure makes the run's conclusion "invalid" regardless of
    what any narrative says.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from plans import validate_plan
from engines import get_engine
from gates import run_gates
from benchmarks import get_benchmark
from core.llm import LLM
from core.note import write_note


def run_loop(benchmark_name, run_dir, seed=0, scripted=False, enforce_gates=True):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    llm = LLM(scripted=scripted)
    trace = []

    def stage(name, **info):
        trace.append({"stage": name, "t": time.time(),
                      "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **info})

    benchmark = get_benchmark(benchmark_name)

    # 1. hypothesize
    hypothesis = llm.hypothesize(benchmark)
    stage("hypothesize", mode=llm.mode, hypothesis=hypothesis)

    # 2. plan (typed)
    plan_id = f"{benchmark_name}_seed{seed}"
    plan = benchmark.make_plan(hypothesis, plan_id)
    (run_dir / "plan.json").write_text(json.dumps(plan.to_dict(), indent=2))
    stage("plan", plan_id=plan_id)

    # 3. validate plan (guardrails)
    errors = validate_plan(plan)
    stage("validate_plan", errors=errors)
    if errors:
        _dump(run_dir, trace, status="aborted_invalid_plan")
        return {"status": "aborted_invalid_plan", "errors": errors}

    # 4. execute (allowlisted engine only)
    results = get_engine(plan.engine).run(plan, rng)
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    stage("execute", engine=plan.engine)

    # 5. gate check — BEFORE any LLM judgment
    gate_report = run_gates(results)
    (run_dir / "gate_report.json").write_text(json.dumps(gate_report, indent=2))
    stage("gate_check", passed=gate_report["passed"], enforced=enforce_gates)

    # 6. interpret — harness scores the answer key; LLM only narrates gated facts
    score_report = benchmark.score(results)
    (run_dir / "score_report.json").write_text(json.dumps(score_report, indent=2))
    if gate_report["passed"] or not enforce_gates:
        narrative = llm.narrate(hypothesis, gate_report, score_report, results["fit"])
    else:
        narrative = ("Gate failure: no scientific narrative was generated. "
                     "See gate report for which checks failed.")
    stage("interpret", benchmark_success=score_report["success"])

    # 7. conclude
    valid = gate_report["passed"] if enforce_gates else True
    conclusion = {
        "hypothesis_supported": bool(score_report["success"]) if valid else None,
        "valid": valid,
        "gates_enforced": enforce_gates,
    }
    stage("conclude", **conclusion)

    # 8. write note
    write_note(run_dir, plan, results, gate_report, score_report, narrative,
               llm.mode, seed, enforce_gates)
    stage("write_note", path=str(run_dir / "note.md"))

    _dump(run_dir, trace, status="completed")
    return {"status": "completed", "conclusion": conclusion,
            "gate_passed": gate_report["passed"], "score": score_report["success"],
            "run_dir": str(run_dir)}


def _dump(run_dir, trace, status):
    (run_dir / "loop_trace.json").write_text(
        json.dumps({"status": status, "trace": trace}, indent=2))
