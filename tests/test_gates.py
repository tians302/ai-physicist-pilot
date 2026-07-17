"""Gate tests: good results pass; corrupted results fail the right gates.

Run with: python tests/test_gates.py (from repo root). No API key needed.
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from plans import ExperimentPlan
from engines.callaway import CallawayEngine
from gates import run_gates


def _verdicts(report):
    return {g["name"]: g["passed"] for g in report["gates"]}


def main():
    eng = CallawayEngine()
    plan = ExperimentPlan(plan_id="t", hypothesis="Boundary scattering suppresses kappa at low T.")
    good = eng.run(plan, np.random.default_rng(0))

    report = run_gates(good)
    print("GOOD results:")
    for g in report["gates"]:
        print(f"  {g['name']:22s} {'PASS' if g['passed'] else 'FAIL'}  {g['detail']}")
    assert report["passed"], "good results must pass all gates"

    # Corruption 1: terrible fit
    bad = copy.deepcopy(good)
    bad["chi2_dof_calibration"] = 240.0
    v = _verdicts(run_gates(bad))
    assert not v["goodness_of_fit"] and v["residual_structure"]

    # Corruption 2: structured residuals
    bad = copy.deepcopy(good)
    bad["residuals_calibration"] = list(np.linspace(-3, 3, len(bad["residuals_calibration"])))
    v = _verdicts(run_gates(bad))
    assert not v["residual_structure"]

    # Corruption 3: missing uncertainties
    bad = copy.deepcopy(good)
    bad["fit"]["A"]["sigma"] = 0.0
    v = _verdicts(run_gates(bad))
    assert not v["uncertainties"]

    # Corruption 4: unstable bootstrap
    bad = copy.deepcopy(good)
    bad["bootstrap_cv"]["B"] = 2.5
    v = _verdicts(run_gates(bad))
    assert not v["bootstrap_stability"]

    # Corruption 5: holdout disagreement (overfitting signature)
    bad = copy.deepcopy(good)
    bad["chi2_dof_holdout"] = 40.0
    v = _verdicts(run_gates(bad))
    assert not v["holdout_validation"]

    # Corruption 6: NaN in curves
    bad = copy.deepcopy(good)
    bad["curves"]["kappa_baseline"][3] = float("nan")
    v = _verdicts(run_gates(bad))
    assert not v["finite_values"]

    print("All 6 corruptions caught by the intended gate.")
    print("GATE TESTS PASSED")


if __name__ == "__main__":
    main()
