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

from contracts import ResourceBudget
from contracts.lammps import LammpsRelaxPlan
from engines.lammps_adapter import LammpsAdapter
from engines.lammps_adapter.fixtures import SW_SI_REGRESSION

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
