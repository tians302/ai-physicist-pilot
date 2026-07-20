"""Real-engine smoke (auto-skipped when no LAMMPS binary is available).
The full battery lives in scripts/validate_lammps.py; this keeps a
minimal version inside `pytest` for laptop/CARC envs.

Binary resolution: $AIPHYS_LMP_BIN, else `lmp` on PATH.
"""
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from analyzers import analyze_elastic, analyze_eos
from contracts import ResourceBudget
from contracts.lammps import (LammpsElasticPlan, LammpsEosPlan,
                              LammpsRelaxPlan)
from engines.lammps_adapter import LammpsAdapter
from engines.lammps_adapter.fixtures import (SW_SI_MECHANICAL_REGRESSION,
                                             SW_SI_REGRESSION)
from gates.elastic import run_elastic_gates
from gates.eos import run_eos_gates

_BIN = os.environ.get("AIPHYS_LMP_BIN", "lmp")
needs_lammps = pytest.mark.skipif(shutil.which(_BIN) is None,
                                  reason=f"no LAMMPS binary ({_BIN!r})")


@needs_lammps
def test_real_sw_relax_matches_regression_fixtures(tmp_path):
    adapter = LammpsAdapter(binary=_BIN)
    plan = LammpsRelaxPlan(
        plan_id="real_smoke", capability_id="validation_smoke",
        hypothesis="SW silicon relaxes to the published lattice constant.",
        model_key="sw_si_native_1985",
        resource_budget=ResourceBudget(max_walltime_s=120))
    run = adapter.execute(plan, tmp_path / "run")
    assert run.exit_status == "completed"
    raw = json.loads((tmp_path / "run" / "raw_outputs.json").read_text())["relax"]
    fa, fe = SW_SI_REGRESSION["a0_A"], SW_SI_REGRESSION["ecoh_eV_per_atom"]
    assert abs(raw["lx_A"] - fa["value"]) <= fa["tol"]
    assert abs(raw["pe_eV"] / raw["natoms"] - fe["value"]) <= fe["tol"]


@needs_lammps
def test_real_eos_pipeline_gates_and_regression(tmp_path):
    adapter = LammpsAdapter(binary=_BIN)
    plan = LammpsEosPlan(
        plan_id="real_eos", capability_id="si_eos_sw_lammps",
        hypothesis="SW-Si E(V) near equilibrium follows Birch-Murnaghan.",
        model_key="sw_si_native_1985",
        resource_budget=ResourceBudget(max_walltime_s=300))
    run = adapter.execute(plan, tmp_path / "eos")
    assert run.exit_status == "completed"
    raw = json.loads((tmp_path / "eos" / "raw_outputs.json").read_text())
    _, diag = analyze_eos(raw, run)
    assert run_eos_gates(diag)["passed"], diag
    f = SW_SI_MECHANICAL_REGRESSION["B0_GPa"]
    assert abs(diag["B0_GPa"] - f["value"]) <= f["tol"]


@needs_lammps
def test_real_elastic_pipeline_gates_and_regression(tmp_path):
    adapter = LammpsAdapter(binary=_BIN)
    plan = LammpsElasticPlan(
        plan_id="real_cij", capability_id="si_elastic_sw_lammps",
        hypothesis="SW-Si elastic tensor is cubic and mechanically stable.",
        model_key="sw_si_native_1985",
        resource_budget=ResourceBudget(max_walltime_s=300))
    run = adapter.execute(plan, tmp_path / "cij")
    assert run.exit_status == "completed"
    raw = json.loads((tmp_path / "cij" / "raw_outputs.json").read_text())
    assert raw["conventions"]["ion_relaxation"] is True
    _, diag = analyze_elastic(raw, run)
    assert run_elastic_gates(diag)["passed"], diag
    for key, dkey in [("C11_GPa", "C11_GPa"), ("C12_GPa", "C12_GPa"),
                      ("C44_relaxed_GPa", "C44_GPa")]:
        f = SW_SI_MECHANICAL_REGRESSION[key]
        assert abs(diag[dkey] - f["value"]) <= f["tol"], (key, diag[dkey])
