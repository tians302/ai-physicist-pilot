"""WP2 acceptance: planted unsupported hypotheses are rejected before
planning — 100% on the fixture set — and supported ones route correctly.
Run with: pytest tests/ -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from contracts import ScalarValue
from registry import (CapabilityRegistry, DuplicateCapabilityError,
                      default_registry)
from registry.builtin import callaway_capability
from registry.router import HypothesisRequest, RoutingLog, route

K = lambda v: ScalarValue(value=v, unit="K")


def _req(rid, prop="thermal_conductivity_curve", elements=("Si",),
         conditions=None, engine=None, preferred=None,
         hyp="Boundary scattering suppresses kappa at low temperature."):
    return HypothesisRequest(request_id=rid, hypothesis=hyp,
                             property_name=prop, elements=list(elements),
                             conditions=conditions or {}, engine=engine,
                             preferred_capability_id=preferred)


# Planted fixture battery. Label = genuinely supported by the registry?
SUPPORTED = [
    _req("s1", conditions={"T_min_K": K(50.0), "T_max_K": K(500.0)}),
    _req("s2", prop="kappa_vs_T",   # alias + non-SI condition unit (um -> m)
         conditions={"intervention_boundary_length_m":
                     ScalarValue(value=10.0, unit="um")}),
    _req("s3", prop="thermal_conductivity", engine="callaway_rta_si"),
    _req("s4", prop="equation_of_state", engine="lammps",
         conditions={"a0_A": ScalarValue(value=0.543, unit="nm")},
         hyp="Diamond-Si E(V) follows Birch-Murnaghan near equilibrium."),
    _req("s5", prop="Cij",
         hyp="Silicon's elastic tensor is cubic and mechanically stable."),
]
EXPECTED_CAP = {"s1": "si_kappa_callaway", "s2": "si_kappa_callaway",
                "s3": "si_kappa_callaway", "s4": "si_eos_sw_lammps",
                "s5": "si_elastic_sw_lammps"}
UNSUPPORTED = [
    _req("u1", prop="band_gap",
         hyp="Compressive strain widens the silicon band gap."),
    _req("u2", elements=("Ge",),
         hyp="Boundary scattering suppresses germanium kappa at low T."),
    _req("u3", conditions={"T_max_K": K(5000.0)},
         hyp="Umklapp scattering saturates kappa above 2000 K in silicon."),
    _req("u4", engine="lammps",
         hyp="NEMD reproduces the boundary-scattering suppression of kappa."),
    _req("u5", prop="viscosity", elements=("H", "O"),
         hyp="Water viscosity decreases with temperature along an isobar."),
    _req("u6", conditions={"T_max_K": ScalarValue(value=0.05, unit="eV")},
         hyp="Thermal kappa of silicon depends on temperature scale in eV."),
    _req("u7", conditions={"pressure_GPa": ScalarValue(value=1.0, unit="GPa")},
         hyp="Pressure enhances silicon kappa in the Callaway picture."),
    _req("u8", elements=(),
         hyp="Some unspecified material shows kappa suppression at low T."),
]


def test_acceptance_100pct_rejection_and_metrics():
    reg = default_registry()
    log = RoutingLog()
    labels = {}
    for r in SUPPORTED:
        d = log.record(route(r, reg))
        labels[r.request_id] = True
        assert d.accepted and d.outcome == "routed", (r.request_id, d.reasons)
        assert d.capability_id == EXPECTED_CAP[r.request_id]
    for r in UNSUPPORTED:
        d = log.record(route(r, reg))
        labels[r.request_id] = False
        assert not d.accepted and d.outcome == "unsupported_hypothesis", r.request_id
        assert d.capability_id is None
        assert d.reasons, "rejections must carry reasons"

    m = log.rejection_metrics(labels)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["accuracy"] == 1.0
    assert log.summary() == {"n": 13, "routed": 5, "rejected": 8}


def _any_reason(req, reg, needle):
    return any(needle in r for r in route(req, reg).reasons)


def test_rejection_reasons_are_specific():
    reg = default_registry()
    assert _any_reason(UNSUPPORTED[0], reg, "not offered")
    assert _any_reason(UNSUPPORTED[1], reg, "outside model coverage")
    assert _any_reason(UNSUPPORTED[2], reg, "outside")
    assert _any_reason(UNSUPPORTED[3], reg, "engine")
    assert _any_reason(UNSUPPORTED[5], reg, "dimension mismatch")
    assert _any_reason(UNSUPPORTED[6], reg, "fail-closed")
    assert _any_reason(UNSUPPORTED[7], reg, "declares no elements")


def test_determinism_and_tie_break():
    reg = default_registry()
    r = SUPPORTED[0]
    assert route(r, reg) == route(r, reg)          # pure function

    # second matching capability: deterministic alphabetical tie-break
    clone = callaway_capability().model_copy(update={"capability_id": "aa_clone"})
    reg2 = default_registry()
    reg2.register(clone)
    d = route(r, reg2)
    assert d.candidates == ["aa_clone", "si_kappa_callaway"]
    assert d.capability_id == "aa_clone"


def test_preference_selects_but_never_expands():
    reg = default_registry()
    clone = callaway_capability().model_copy(update={"capability_id": "aa_clone"})
    reg.register(clone)

    # preference among candidates: honored
    d = route(_req("p1", preferred="si_kappa_callaway"), reg)
    assert d.capability_id == "si_kappa_callaway" and d.preference_honored is True

    # preference outside candidates: overridden, not expanded
    d = route(_req("p2", preferred="lammps_eos_magic"), reg)
    assert d.capability_id == "aa_clone" and d.preference_honored is False

    # preference cannot resurrect a rejected request
    d = route(_req("p3", prop="band_gap", preferred="si_kappa_callaway",
                   hyp="Compressive strain widens the silicon band gap."), reg)
    assert not d.accepted


def test_registry_basics():
    reg = default_registry()
    assert len(reg) == 6 and "si_kappa_callaway" in reg
    assert "si_eos_sw_lammps" in reg and "si_elastic_sw_lammps" in reg
    assert {"si_expansion_sw_lammps", "lj_diffusion_lammps",
            "lj_kappa_gk_lammps"} <= set(reg.ids())
    assert [m.capability_id for m in reg.supports_property("kappa_vs_T")] \
        == ["si_kappa_callaway"]
    with pytest.raises(DuplicateCapabilityError):
        reg.register(callaway_capability())
    empty = CapabilityRegistry()
    assert route(SUPPORTED[0], empty).outcome == "unsupported_hypothesis"


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "pytest", __file__, "-q"]))
