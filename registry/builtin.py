"""Built-in capability manifests. Currently one: the Phase-1A Callaway
engine, described honestly (effective parameters, qualitative L-claims).

The validity domain is derived from plans.BOUNDS so the manifest cannot
drift from the enforced guardrails.
"""
from contracts import (Bound, CapabilityManifest, GateRef, OutputSpec,
                       PhysicalModelCard)
from gates import CONFIG, GATES_VERSION
from plans import BOUNDS


def _domain_from_bounds() -> dict[str, Bound]:
    unit_by_suffix = {"_K": "K", "_m": "m"}
    out = {}
    for name, (lo, hi) in BOUNDS.items():
        unit = next((u for suf, u in unit_by_suffix.items() if name.endswith(suf)), None)
        if unit is None:
            raise ValueError(f"cannot infer unit for guardrail {name!r}")
        out[name] = Bound(lo=lo, hi=hi, unit=unit)
    return out


def callaway_capability() -> CapabilityManifest:
    return CapabilityManifest(
        capability_id="si_kappa_callaway",
        version="1.0",
        property_name="thermal_conductivity_curve",
        property_aliases=["kappa_vs_T", "thermal_conductivity"],
        property_family="thermal-analytic",
        engine="callaway_rta_si",
        protocol="callaway_rta_fit_v1",
        physical_model=PhysicalModelCard(
            name="Callaway-RTA-Si",
            version="1.0",
            source="engines/callaway.py",
            elements=["Si"],
            phases=["diamond"],
            temperature_range_K=(10.0, 1500.0),
            per_atom_quantities_valid=False,
            known_failure_modes=[
                "single Debye branch; no N-process correction",
                "gray boundary term: no specularity/Casimir geometry factor",
                "A, B are effective fitted parameters, not transferable",
                "intervention-L magnitudes qualitative only (see report §8.5)",
            ]),
        plan_type="ExperimentPlan",          # legacy plan; migrates at WP5
        analyzers=["callaway_fit_v1"],       # fit+stats currently inside engine
        gates=[GateRef(name=n, version=GATES_VERSION, params=p) for n, p in [
            ("metrics_exist", {}),
            ("finite_values", {}),
            ("goodness_of_fit", {"chi2_dof_max": CONFIG["chi2_dof_max"]}),
            ("residual_structure", {"lag1_autocorr_max": CONFIG["lag1_autocorr_max"]}),
            ("uncertainties", {}),
            ("bootstrap_stability", {"bootstrap_cv_max": CONFIG["bootstrap_cv_max"],
                                     "min_bootstrap_ok": CONFIG["min_bootstrap_ok"]}),
            ("holdout_validation", {"holdout_chi2_dof_max": CONFIG["holdout_chi2_dof_max"]}),
        ]],
        outputs=[
            OutputSpec(name="kappa_baseline", kind="curve", unit="W/m/K",
                       conditions=["T"]),
            OutputSpec(name="kappa_intervention", kind="curve", unit="W/m/K",
                       conditions=["T"]),
            OutputSpec(name="A", kind="scalar", unit="s^3"),
            OutputSpec(name="B", kind="scalar", unit="s/K"),
        ],
        validity_domain=_domain_from_bounds(),
        provenance={
            "reference_data": "Glassbrenner & Slack 1964 via Touloukian 1970 "
                              "(approximate transcription; re-digitization pending)",
            "implemented": "2026-07-17",
        })
