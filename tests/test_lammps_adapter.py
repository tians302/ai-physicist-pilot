"""WP3 adapter tests — run WITHOUT LAMMPS. A fake `lmp` binary exercises
the isolation/fail-closed paths; template, structure, and pin checks are
pure. Real-engine smoke lives in scripts/validate_lammps.py (+ optional
tests/test_lammps_real.py, skipped when no binary is present).
"""
import json
import os
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from contracts import RawRun, ResourceBudget
from contracts.lammps import LammpsEosPlan, LammpsRelaxPlan
from engines.lammps_adapter import LammpsAdapter
from engines.lammps_adapter.models import (MODELS, UnpinnedModelError,
                                           get_model, sha256_file)
from engines.lammps_adapter.templates import (TemplateError, parse_sentinels,
                                              render, _PLACEHOLDER, TEMPLATES)
from engines.lammps_adapter import structures as st

FAKE_LMP = r'''#!/usr/bin/env python3
import os, shutil, sys, time
from pathlib import Path
mode = os.environ.get("FAKE_LMP_MODE", "ok")
if mode == "hang":
    time.sleep(60)
if mode == "fail":
    sys.exit(3)
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
text = Path(args["-in"]).read_text()
lines = ["AIPHYS_RESULT natoms 8", "AIPHYS_RESULT pe_eV -34.6912",
         "AIPHYS_RESULT vol_A3 160.19", "AIPHYS_RESULT lx_A 5.4310",
         "AIPHYS_RESULT pxx_bar -0.001"]
if "run 0" in text:
    lines += ["AIPHYS_RESULT %s_bar -0.001" % k
              for k in ("pyy", "pzz", "pxy", "pxz", "pyz")]
if mode != "no_sentinels":
    Path(args.get("-log", "log.lammps")).write_text("\n".join(lines) + "\n")
else:
    Path(args.get("-log", "log.lammps")).write_text("normal log noise\n")
for ln in text.splitlines():
    if ln.startswith("write_data"):
        shutil.copy("structure.data", ln.split()[1])
sys.exit(0)
'''


@pytest.fixture
def fake_adapter(tmp_path, monkeypatch):
    fake = tmp_path / "fake_lmp.py"
    fake.write_text(FAKE_LMP)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("FAKE_LMP_MODE", "ok")
    return LammpsAdapter(binary=str(fake)), tmp_path


def _relax_plan(**kw):
    d = dict(plan_id="t_relax", capability_id="si_relax_lammps",
             hypothesis="SW silicon relaxes to the diamond structure.",
             model_key="sw_si_native_1985",
             resource_budget=ResourceBudget(max_walltime_s=60))
    d.update(kw)
    return LammpsRelaxPlan(**d)


# ------------------------------------------------------------- templates

def test_templates_have_no_stray_placeholders_after_render():
    import re
    for name in TEMPLATES:
        vals = {"data_file": "structure.data"}
        if name == "relax_v1":
            vals |= {"etol": 0.0, "ftol": 1e-8, "maxiter": 100,
                     "maxeval": 1000, "out_data": "relaxed.data"}
        out = render(name, vals, {"init_block": "units metal",
                                  "interaction_block": "pair_style sw"})
        # LAMMPS ${var} references are legitimate; anything else in braces is
        # an unfilled placeholder and must not survive rendering
        stripped = re.sub(r"\$\{[a-z0-9_]+\}", "", out)
        assert _PLACEHOLDER.findall(stripped) == []
        assert "${" in out                       # LAMMPS variables intact


def test_render_fails_closed():
    blocks = {"init_block": "units metal", "interaction_block": "pair_style sw"}
    with pytest.raises(TemplateError):           # missing keys
        render("relax_v1", {"data_file": "d"}, blocks)
    with pytest.raises(TemplateError):           # extra key
        render("static_v1", {"data_file": "d", "sneaky": 1}, blocks)
    with pytest.raises(TemplateError):           # injection attempt
        render("static_v1", {"data_file": "d; shell rm -rf /"}, blocks)
    with pytest.raises(TemplateError):           # plan may not set trusted keys
        render("static_v1", {"data_file": "d", "init_block": "evil"}, blocks)
    with pytest.raises(TemplateError):
        render("nope_v9", {}, blocks)


def test_parse_sentinels():
    s = parse_sentinels("noise\nAIPHYS_RESULT pe_eV -34.69\nAIPHYS_RESULT tag ok\n")
    assert s == {"pe_eV": -34.69, "tag": "ok"}


# ------------------------------------------------------------- structures

def test_structure_builders_and_lineage():
    atoms, lin = st.diamond_si(5.43, supercell=2)
    assert len(atoms) == 64 and lin["parent_sha256"] is None
    scaled, lin2 = st.scale_volume(atoms, 1.06, lin)
    assert scaled.get_volume() / atoms.get_volume() == pytest.approx(1.06)
    assert lin2["parent_sha256"] == lin["sha256"] != lin2["sha256"]
    F = st.strain_matrix(1, 0.01)
    strained, lin3 = st.apply_deformation(atoms, F, lin)
    assert strained.get_cell().array[0, 0] == pytest.approx(
        atoms.get_cell().array[0, 0] * 1.01)
    Fs = st.strain_matrix(6, 0.01)               # xy shear, gamma convention
    assert Fs[0, 1] == pytest.approx(0.005)
    with pytest.raises(ValueError):
        st.strain_matrix(7, 0.01)
    assert "Masses" in st.to_lammps_data(atoms)


def test_pinned_sw_file_checksum():
    m = MODELS["sw_si_native_1985"]
    assert sha256_file(m.file_path) == m.file_sha256, \
        "Si.sw changed without re-pinning (update models.py deliberately)"


def test_unverified_models_refused_by_default():
    with pytest.raises(UnpinnedModelError):
        get_model("sw_si_kim")
    assert get_model("sw_si_kim", allow_unverified=True).mode == "kim"
    with pytest.raises(UnpinnedModelError):
        get_model("not_a_model", allow_unverified=True)


# ---------------------------------------------------------- adapter runs

def test_relax_completes_with_checksummed_artifacts(fake_adapter):
    adapter, tmp = fake_adapter
    run = adapter.execute(_relax_plan(), tmp / "run1")
    assert isinstance(run, RawRun) and run.exit_status == "completed"
    paths = {a.path for a in run.artifacts}
    assert {"raw_outputs.json", "relax/input.in", "relax/log.lammps",
            "relax/structure.data"} <= paths
    assert all(len(a.sha256) == 64 for a in run.artifacts)
    raw = json.loads((tmp / "run1" / "raw_outputs.json").read_text())
    assert raw["relax"]["pe_eV"] == pytest.approx(-34.6912)
    RawRun.model_validate_json(run.model_dump_json())   # contract round-trip


def test_engine_failure_fails_closed(fake_adapter, monkeypatch):
    adapter, tmp = fake_adapter
    monkeypatch.setenv("FAKE_LMP_MODE", "fail")
    run = adapter.execute(_relax_plan(), tmp / "run2")
    assert run.exit_status == "failed"


def test_timeout_fails_closed(fake_adapter, monkeypatch):
    adapter, tmp = fake_adapter
    monkeypatch.setenv("FAKE_LMP_MODE", "hang")
    run = adapter.execute(
        _relax_plan(resource_budget=ResourceBudget(max_walltime_s=2)),
        tmp / "run3")
    assert run.exit_status == "timeout"


def test_missing_sentinels_fail_closed(fake_adapter, monkeypatch):
    adapter, tmp = fake_adapter
    monkeypatch.setenv("FAKE_LMP_MODE", "no_sentinels")
    run = adapter.execute(_relax_plan(), tmp / "run4")
    assert run.exit_status == "failed"


def test_missing_binary_fails_closed(tmp_path):
    adapter = LammpsAdapter(binary=str(tmp_path / "no_such_lmp"))
    run = adapter.execute(_relax_plan(), tmp_path / "run5")
    assert run.exit_status == "failed" and not adapter.available()


def test_unverified_model_refused_at_execute(fake_adapter):
    adapter, tmp = fake_adapter
    run = adapter.execute(_relax_plan(model_key="sw_si_kim"), tmp / "run6")
    assert run.exit_status == "failed"
    err = (tmp / "run6" / "error.txt").read_text()
    assert "not pin-verified" in err


def test_eos_chain_lineage_and_points(fake_adapter):
    adapter, tmp = fake_adapter
    plan = LammpsEosPlan(
        plan_id="t_eos", capability_id="si_eos_lammps",
        hypothesis="SW-Si energy-volume curve is convex near equilibrium.",
        model_key="sw_si_native_1985", n_volumes=5,
        resource_budget=ResourceBudget(max_walltime_s=120))
    run = adapter.execute(plan, tmp / "run7")
    assert run.exit_status == "completed"
    raw = json.loads((tmp / "run7" / "raw_outputs.json").read_text())
    pts = raw["eos_points"]
    assert len(pts) == 5
    scales = [p["volume_scale"] for p in pts]
    assert scales == sorted(scales)
    relaxed_sha = raw["structure_lineage"][-1]["sha256"]
    assert all(p["lineage"]["parent_sha256"] == relaxed_sha for p in pts)
    assert {f"eos_{i:02d}/log.lammps" for i in range(5)} \
        <= {a.path for a in run.artifacts}


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "pytest", __file__, "-q"]))
