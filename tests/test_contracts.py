"""WP1 acceptance tests: schema round-trip + dimensional checks for
scalar/curve/tensor cases. Run with: pytest tests/ -q
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from pydantic import ValidationError

from contracts import (BaseExperimentPlan, CapabilityManifest, CurveValue,
                       DimensionError, EnvironmentLock, GateRef, Observation,
                       ObservationBundle, OutputSpec, PhysicalModelCard,
                       RawRun, ResourceBudget, ScalarValue, TensorValue,
                       ArtifactRef, convert, dim_of, same_dimension,
                       to_canonical)

SHA = "0" * 64


def _roundtrip(model):
    restored = type(model).model_validate_json(model.model_dump_json())
    assert restored == model
    return restored


# ------------------------------------------------------------------ units

def test_unit_conversions():
    assert convert(1.0, "GPa", "Pa") == pytest.approx(1e9)
    assert convert(1.0, "eV", "J") == pytest.approx(1.602176634e-19)
    assert convert(1.0, "Ang", "nm") == pytest.approx(0.1)
    assert convert(300.0, "K", "K") == 300.0
    si, name = to_canonical(1.0, "eV/Ang^3")   # energy density -> Pa
    assert si == pytest.approx(1.602176634e-19 / 1e-30)
    assert dim_of(name) == dim_of("Pa")


def test_dimensional_equivalence():
    assert same_dimension("W/m/K", "W/K/m")
    assert same_dimension("J/s/m/K", "W/m/K")
    assert not same_dimension("W/m/K", "W/m")
    assert dim_of("1/K") == tuple(-x for x in dim_of("K"))


def test_bad_units_rejected():
    with pytest.raises(DimensionError):
        dim_of("parsecs")
    with pytest.raises(DimensionError):
        dim_of("m^one")
    with pytest.raises(DimensionError):
        convert(1.0, "eV", "K")  # dimension mismatch


# ----------------------------------------------------------------- values

def test_scalar_roundtrip():
    a0 = ScalarValue(value=5.431, sigma=0.001, unit="Ang")
    _roundtrip(a0)


def test_curve_roundtrip_and_validation():
    kappa = CurveValue(x=[100.0, 200.0, 300.0], y=[884.0, 264.0, 148.0],
                       y_sigma=[44.0, 13.0, 7.4], x_unit="K", y_unit="W/m/K")
    _roundtrip(kappa)
    with pytest.raises(ValidationError):
        CurveValue(x=[1.0, 2.0], y=[1.0], x_unit="K", y_unit="W/m/K")
    with pytest.raises(ValidationError):
        CurveValue(x=[1.0, 2.0], y=[1.0, 2.0], x_unit="K", y_unit="furlongs")


def test_tensor_roundtrip_and_validation():
    cij = [[166.0, 64.0, 64.0, 0, 0, 0],
           [64.0, 166.0, 64.0, 0, 0, 0],
           [64.0, 64.0, 166.0, 0, 0, 0],
           [0, 0, 0, 80.0, 0, 0],
           [0, 0, 0, 0, 80.0, 0],
           [0, 0, 0, 0, 0, 80.0]]
    t = TensorValue(value=cij, unit="GPa")
    _roundtrip(t)
    with pytest.raises(ValidationError):
        TensorValue(value=[[1.0, 2.0], [3.0]], unit="GPa")  # ragged


# -------------------------------------------------------------- contracts

def _manifest():
    return CapabilityManifest(
        capability_id="si_eos_sw", version="0.1",
        property_name="equation_of_state", property_family="structural",
        engine="lammps", protocol="eos_volume_sweep_v1",
        physical_model=PhysicalModelCard(
            name="SW_StillingerWeber_1985_Si", version="pin-at-WP3",
            source="openkim:TBD", elements=["Si"], phases=["diamond"],
            known_failure_modes=["high-T melt structure unreliable"]),
        plan_type="EosPlan", analyzers=["birch_murnaghan_fit_v1"],
        gates=[GateRef(name="volume_coverage", version="0.1",
                       params={"min_span_frac": 0.06})],
        outputs=[OutputSpec(name="E_V_curve", kind="curve", unit="eV",
                            conditions=["V"]),
                 OutputSpec(name="B0", kind="scalar", unit="GPa")],
        validity_domain={"strain": (-0.05, 0.05)})


def test_manifest_roundtrip_and_invariants():
    m = _roundtrip(_manifest())
    assert m.validity_domain["strain"] == (-0.05, 0.05)
    with pytest.raises(ValidationError):   # gates are mandatory
        CapabilityManifest(**{**_manifest().model_dump(), "gates": []})
    with pytest.raises(ValidationError):   # family is a closed set
        CapabilityManifest(**{**_manifest().model_dump(),
                              "property_family": "astrology"})


def test_plan_roundtrip_and_guardrails():
    plan = BaseExperimentPlan(
        plan_id="p1", capability_id="si_eos_sw", engine="lammps",
        hypothesis="Diamond-Si EOS near equilibrium is well fit by Birch-Murnaghan.",
        parameters={"n_volumes": 11},
        resource_budget=ResourceBudget(max_walltime_s=600))
    _roundtrip(plan)
    with pytest.raises(ValidationError):   # trivial hypothesis
        BaseExperimentPlan(plan_id="p", capability_id="c", engine="e",
                           hypothesis="short",
                           resource_budget=ResourceBudget(max_walltime_s=60))
    with pytest.raises(ValidationError):   # unknown field forbidden
        BaseExperimentPlan(plan_id="p", capability_id="c", engine="e",
                           hypothesis="A sufficiently long hypothesis here.",
                           resource_budget=ResourceBudget(max_walltime_s=60),
                           sneaky_extra_knob=1.0)


def test_rawrun_roundtrip_and_invariants():
    t0 = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 7, 19, 12, 5, tzinfo=timezone.utc)
    run = RawRun(run_id="r1", plan_id="p1", plan_sha256=SHA,
                 capability_id="si_eos_sw", engine="lammps", seed=0,
                 started_utc=t0, finished_utc=t1, exit_status="completed",
                 artifacts=[ArtifactRef(path="log.lammps", sha256=SHA, kind="log")],
                 environment=EnvironmentLock(python="3.10.12", platform="linux"),
                 resource_usage={"walltime_s": 300.0})
    _roundtrip(run)
    with pytest.raises(ValidationError):   # completed run must have artifacts
        RawRun(**{**run.model_dump(), "artifacts": []})
    with pytest.raises(ValidationError):   # time ordering
        RawRun(**{**run.model_dump(), "finished_utc": datetime(
            2026, 7, 19, 11, 0, tzinfo=timezone.utc)})


def test_observation_bundle_scalar_curve_tensor():
    bundle = ObservationBundle(
        bundle_id="b1", run_id="r1", capability_id="si_eos_sw",
        analyzer="birch_murnaghan_fit_v1", analyzer_version="0.1",
        observations=[
            Observation(name="lattice_constant",
                        value=ScalarValue(value=5.431, sigma=0.002, unit="Ang"),
                        conditions={"T": ScalarValue(value=0.0, unit="K")}),
            Observation(name="kappa_vs_T",
                        value=CurveValue(x=[100.0, 300.0], y=[884.0, 148.0],
                                         x_unit="K", y_unit="W/m/K")),
            Observation(name="elastic_tensor",
                        value=TensorValue(value=[[166.0, 64.0], [64.0, 166.0]],
                                          unit="GPa")),
        ])
    restored = _roundtrip(bundle)
    kinds = [o.value.kind for o in restored.observations]
    assert kinds == ["scalar", "curve", "tensor"]
    with pytest.raises(ValidationError):   # empty bundles are not observations
        ObservationBundle(bundle_id="b", run_id="r", capability_id="c",
                          analyzer="a", analyzer_version="0.1", observations=[])


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "pytest", __file__, "-q"]))
