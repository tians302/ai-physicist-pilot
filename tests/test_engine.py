"""Engine sanity tests — run with: python tests/test_engine.py (from repo root)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from plans import ExperimentPlan, validate_plan
from engines.callaway import CallawayEngine, kappa_model


def test_kappa_magnitude():
    # With literature-scale params, kappa(300 K) should be O(100) W/m-K
    k = kappa_model(300.0, 1e-3, 1.32e-45, 1.73e-19)
    assert 50 < k < 500, k
    print(f"  kappa(300K) = {k:.0f} W/m-K  OK")


def test_plan_guardrails():
    good = ExperimentPlan(plan_id="t", hypothesis="Boundary scattering suppresses kappa at low T.")
    assert validate_plan(good) == []
    bad = ExperimentPlan(plan_id="t", hypothesis="Boundary scattering suppresses kappa at low T.",
                         baseline_boundary_length_m=1.0, engine="my_evil_engine")
    errs = validate_plan(bad)
    assert len(errs) == 2, errs
    print(f"  guardrails caught: {errs}  OK")


def test_fit_quality():
    eng = CallawayEngine()
    plan = ExperimentPlan(plan_id="t", hypothesis="Boundary scattering suppresses kappa at low T.")
    res = eng.run(plan, np.random.default_rng(0))
    assert res["chi2_dof_calibration"] < 10, res["chi2_dof_calibration"]
    assert res["fit"]["A"]["value"] > 0 and res["fit"]["B"]["value"] > 0
    print(f"  fit: chi2/dof cal = {res['chi2_dof_calibration']:.2f}, "
          f"holdout = {res['chi2_dof_holdout']:.2f}, "
          f"A = {res['fit']['A']['value']:.2e}, B = {res['fit']['B']['value']:.2e}  OK")


if __name__ == "__main__":
    for t in [test_kappa_magnitude, test_plan_guardrails, test_fit_quality]:
        print(t.__name__)
        t()
    print("ENGINE TESTS PASSED")
