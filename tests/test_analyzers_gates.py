"""WP4 acceptance: synthetic-data recovery + targeted corruptions each
caught by the INTENDED gate. Run with: pytest tests/ -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from analyzers.eos import analyze_eos, bm3_energy
from analyzers.elastic import analyze_elastic, _BAR_TO_GPA
from contracts import ObservationBundle, convert
from gates.eos import run_eos_gates
from gates.elastic import run_elastic_gates

RUN = SimpleNamespace(run_id="synthetic_run", capability_id="synthetic_cap")

# ------------------------------------------------------- synthetic factories

V_REF, E0_TRUE, B0_TRUE_EVA3, B0P_TRUE = 160.2, -34.69, 0.6, 4.2  # per 8-atom cell


def eos_raw(scales=None, v0_true=V_REF, energy_scale=1.0, wiggle=0.0):
    scales = np.linspace(0.94, 1.06, 11) if scales is None else np.asarray(scales)
    V = V_REF * scales
    E = bm3_energy(V, E0_TRUE, v0_true, B0_TRUE_EVA3, B0P_TRUE)
    if wiggle:
        E = E + wiggle * (E.max() - E.min()) * np.sin(200.0 * scales)
    E = E * energy_scale
    return {"eos_points": [
        {"volume_scale": float(s),
         "sentinels": {"vol_A3": float(v), "pe_eV": float(e), "natoms": 8}}
        for s, v, e in zip(scales, V, E)]}


C_TRUE = np.zeros((6, 6))
C_TRUE[:3, :3] = 64.0
np.fill_diagonal(C_TRUE[:3, :3], 166.0)
for i in (3, 4, 5):
    C_TRUE[i, i] = 80.0


def elastic_raw(C=None, offset_GPa=0.0, nonlin_GPa=0.0, pressure_unit_slip=1.0,
                col_scale=None):
    C = C_TRUE if C is None else np.asarray(C, dtype=float)
    amps = [0.0025, 0.005]
    pts = []
    for j in range(1, 7):
        col = C[:, j - 1] * (col_scale[j - 1] if col_scale is not None else 1.0)
        for a in [-x for x in amps[::-1]] + amps:
            sigma = col * a + offset_GPa + nonlin_GPa * a * a
            p_bar = -sigma / _BAR_TO_GPA * pressure_unit_slip
            keys = ("pxx_bar", "pyy_bar", "pzz_bar", "pyz_bar", "pxz_bar", "pxy_bar")
            pts.append({"voigt": j, "amplitude": float(a),
                        "F": np.eye(3).tolist(),
                        "sentinels": {"natoms": 8, "pe_eV": -34.69,
                                      "vol_A3": 160.2,
                                      **{k: float(v) for k, v in zip(keys, p_bar)}}})
    return {"strain_points": pts}


def _fails(report, name):
    v = {g["name"]: g["passed"] for g in report["gates"]}
    assert not report["passed"] and not v[name], (name, v)
    return v


# ------------------------------------------------------------------ EOS

def test_eos_recovery_and_gates_pass():
    bundle, diag = analyze_eos(eos_raw(), RUN)
    assert isinstance(bundle, ObservationBundle)
    B0_expect = convert(B0_TRUE_EVA3, "eV/Ang^3", "GPa")
    assert diag["B0_GPa"] == pytest.approx(B0_expect, rel=1e-3)
    assert diag["V0_A3"] == pytest.approx(V_REF, rel=1e-4)
    assert diag["B0_prime"] == pytest.approx(B0P_TRUE, rel=0.02)
    report = run_eos_gates(diag)
    assert report["passed"], report
    ObservationBundle.model_validate_json(bundle.model_dump_json())


def test_eos_corruption_bad_units_caught_by_magnitude():
    _, diag = analyze_eos(eos_raw(energy_scale=160.2), RUN)   # eV/A^3-like slip
    v = _fails(run_eos_gates(diag), "magnitude_plausibility")
    assert v["fit_residuals"] and v["fit_form_sensitivity"]   # targeted, no cascade


def test_eos_corruption_unconverged_relax_caught_by_interior():
    _, diag = analyze_eos(eos_raw(v0_true=V_REF * 1.10), RUN)
    _fails(run_eos_gates(diag), "minimum_interior")


def test_eos_corruption_poor_coverage_caught():
    _, diag = analyze_eos(eos_raw(scales=np.linspace(0.995, 1.06, 8)), RUN)
    _fails(run_eos_gates(diag), "volume_coverage")


def test_eos_corruption_wiggle_caught_by_residuals():
    _, diag = analyze_eos(eos_raw(wiggle=0.02), RUN)
    _fails(run_eos_gates(diag), "fit_residuals")


# -------------------------------------------------------------- elastic

def test_elastic_recovery_and_gates_pass():
    bundle, diag = analyze_elastic(elastic_raw(), RUN)
    assert diag["C11_GPa"] == pytest.approx(166.0, rel=1e-6)
    assert diag["C12_GPa"] == pytest.approx(64.0, rel=1e-6)
    assert diag["C44_GPa"] == pytest.approx(80.0, rel=1e-6)
    report = run_elastic_gates(diag)
    assert report["passed"], report
    tensor = [o for o in bundle.observations if o.name == "elastic_tensor"][0]
    assert tensor.value.kind == "tensor" and tensor.value.unit == "GPa"


def test_elastic_corruption_bad_units_caught_by_magnitude():
    # engine reporting GPa where bar is expected: everything x1e-4
    _, diag = analyze_elastic(elastic_raw(pressure_unit_slip=1e-4), RUN)
    v = _fails(run_elastic_gates(diag), "magnitude_plausibility")
    assert v["linearity"] and v["tensor_symmetry"]            # targeted


def test_elastic_corruption_nonlinearity_caught():
    _, diag = analyze_elastic(elastic_raw(nonlin_GPa=5000.0), RUN)
    _fails(run_elastic_gates(diag), "linearity")


def test_elastic_corruption_unconverged_relax_caught_by_reference_stress():
    _, diag = analyze_elastic(elastic_raw(offset_GPa=0.2), RUN)
    v = _fails(run_elastic_gates(diag), "reference_state_stress")
    assert v["linearity"] and v["mechanical_stability"]       # targeted


def test_elastic_corruption_broken_symmetry_caught():
    _, diag = analyze_elastic(elastic_raw(col_scale=[1.3, 1, 1, 1, 1, 1]), RUN)
    _fails(run_elastic_gates(diag), "tensor_symmetry")


def test_elastic_corruption_unstable_tensor_caught():
    C = C_TRUE.copy()
    C[:3, :3] = 80.0
    np.fill_diagonal(C[:3, :3], 60.0)          # C11 < C12: Born violation
    _, diag = analyze_elastic(elastic_raw(C=C), RUN)
    v = _fails(run_elastic_gates(diag), "mechanical_stability")
    assert v["tensor_symmetry"] and v["magnitude_plausibility"]  # targeted


def test_bar_to_gpa_constant():
    assert _BAR_TO_GPA == pytest.approx(1e-4)


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "pytest", __file__, "-q"]))
