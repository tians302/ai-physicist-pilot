"""WP5/WP6 tests: v2 loop parity with v1 Callaway, negative suite, and
(with a real binary) LAMMPS benchmarks through the full loop.
"""
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import core.loop2 as loop2
from core.loop import run_loop as run_loop_v1
from core.loop2 import run_loop_v2
from registry.router import HypothesisRequest

_BIN = os.environ.get("AIPHYS_LMP_BIN", "lmp")
needs_lammps = pytest.mark.skipif(shutil.which(_BIN) is None,
                                  reason=f"no LAMMPS binary ({_BIN!r})")


def test_callaway_parity_v1_v2(tmp_path):
    """WP5 acceptance: migration leaves Callaway numbers unchanged."""
    r1 = run_loop_v1("silicon_boundary", tmp_path / "v1", seed=0, scripted=True)
    r2 = run_loop_v2("silicon_boundary", tmp_path / "v2", seed=0)
    assert r1["status"] == r2["status"] == "completed"
    assert r1["gate_passed"] and r2["gate_passed"]
    res1 = json.loads((tmp_path / "v1" / "results.json").read_text())
    res2 = json.loads((tmp_path / "v2" / "raw" / "raw_outputs.json").read_text())["callaway"]
    for k in ("chi2_dof_calibration", "chi2_dof_holdout"):
        assert res1[k] == pytest.approx(res2[k], rel=1e-12)
    for p in ("A", "B"):
        assert res1["fit"][p]["value"] == pytest.approx(
            res2["fit"][p]["value"], rel=1e-12)
    assert (tmp_path / "v2" / "note.md").exists()
    assert (tmp_path / "v2" / "bundle.json").exists()


def test_unsupported_hypothesis_ends_run(tmp_path, monkeypatch):
    class BandGapBench:
        name, kind = "bandgap_stub", "eos"
        hypothesis = "Compressive strain widens the silicon band gap."

        def request(self, hypothesis, rid):
            return HypothesisRequest(request_id=rid, hypothesis=hypothesis,
                                     property_name="band_gap", elements=["Si"])

    monkeypatch.setitem(loop2.__dict__, "get_benchmark_v2",
                        lambda name: BandGapBench())
    out = run_loop_v2("bandgap_stub", tmp_path)
    assert out["status"] == "unsupported_hypothesis"
    assert out["reasons"]
    d = json.loads((tmp_path / "route_decision.json").read_text())
    assert d["outcome"] == "unsupported_hypothesis"


def test_invalid_plan_aborts(tmp_path, monkeypatch):
    from benchmarks.v2 import SiEosV2
    from contracts import ResourceBudget
    from contracts.lammps import LammpsEosPlan

    class BadPlanBench(SiEosV2):
        def make_plan(self, hypothesis, decision, plan_id, seed):
            return LammpsEosPlan(       # scale_min >= scale_max: schema abort
                plan_id=plan_id, capability_id=decision.capability_id,
                hypothesis=hypothesis, model_key="sw_si_native_1985",
                scale_min=0.99, scale_max=0.995,   # violates span (max>=1.001)
                resource_budget=ResourceBudget(max_walltime_s=60))

    monkeypatch.setitem(loop2.__dict__, "get_benchmark_v2",
                        lambda name: BadPlanBench())
    out = run_loop_v2("si_eos", tmp_path)
    assert out["status"] == "aborted_invalid_plan"


def test_engine_failure_is_measured_not_hidden(tmp_path, monkeypatch):
    import engines.callaway_adapter as ca

    class Boom:
        def run(self, plan, rng):
            raise RuntimeError("engine exploded")

    monkeypatch.setattr(ca, "CallawayEngine", Boom)
    out = run_loop_v2("silicon_boundary", tmp_path)
    assert out["status"] == "engine_failed"
    assert out["conclusion"]["valid"] is False


def test_corruption_voids_conclusion(tmp_path):
    out = run_loop_v2("silicon_boundary", tmp_path, corruption="bad_fit")
    assert out["status"] == "completed" and out["gate_passed"] is False
    assert out["conclusion"]["valid"] is False
    assert out["conclusion"]["hypothesis_supported"] is None
    assert "GATE FAILURE" in (tmp_path / "note.md").read_text()


def test_gates_off_ablation_lets_false_claim_through(tmp_path):
    """The WP10 headline mechanism: gates-off + corruption -> a 'valid'
    conclusion that gates-on would have voided (an unflagged false claim)."""
    out = run_loop_v2("silicon_boundary", tmp_path, corruption="bad_fit",
                      enforce_gates=False)
    assert out["status"] == "completed" and out["gate_passed"] is False
    assert out["conclusion"]["valid"] is True          # ablation mode
    assert "NOT enforced" in (tmp_path / "note.md").read_text()


@needs_lammps
def test_eos_benchmark_through_loop_and_reproducible(tmp_path):
    out1 = run_loop_v2("si_eos", tmp_path / "a", seed=0)
    out2 = run_loop_v2("si_eos", tmp_path / "b", seed=0)
    assert out1["status"] == out2["status"] == "completed"
    assert out1["gate_passed"] and out1["score"]
    b1 = json.loads((tmp_path / "a" / "bundle.json").read_text())
    b2 = json.loads((tmp_path / "b" / "bundle.json").read_text())
    v1 = {o["name"]: o["value"].get("value") for o in b1["observations"]}
    v2 = {o["name"]: o["value"].get("value") for o in b2["observations"]}
    assert v1["B0"] == pytest.approx(v2["B0"], rel=1e-10)


@needs_lammps
def test_elastic_benchmark_through_loop(tmp_path):
    out = run_loop_v2("si_elastic", tmp_path, seed=0)
    assert out["status"] == "completed"
    assert out["gate_passed"] and out["score"]
    note = (tmp_path / "note.md").read_text()
    assert "born_stability" in note and "INFO-ONLY" in note


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "pytest", __file__, "-q"]))
