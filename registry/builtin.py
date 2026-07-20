"""Built-in capability manifests. Currently one: the Phase-1A Callaway
engine, described honestly (effective parameters, qualitative L-claims).

The validity domain is derived from plans.BOUNDS so the manifest cannot
drift from the enforced guardrails.
"""
from contracts import (Bound, CapabilityManifest, GateRef, OutputSpec,
                       PhysicalModelCard)
from gates import CONFIG, GATES_VERSION
from gates.elastic import CONFIG as ELASTIC_CONFIG
from gates.elastic import ELASTIC_GATES_VERSION
from gates.eos import CONFIG as EOS_CONFIG
from gates.eos import EOS_GATES_VERSION
from gates.expansion import CONFIG as EXPANSION_CONFIG
from gates.expansion import EXPANSION_GATES_VERSION
from gates.transport import (DIFFUSION_CONFIG, GK_CONFIG,
                             TRANSPORT_GATES_VERSION)
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


def _sw_native_card() -> PhysicalModelCard:
    return PhysicalModelCard(
        name="SW_StillingerWeber_1985_Si_native",
        version="1985/in-repo",
        source="engines/lammps_adapter/potentials/Si.sw (PRB 31, 5262)",
        checksum="3876f9574a7f8f8b5c2a6519f250fe888854c8b436da53a2a293194aedbb156c",
        elements=["Si"], phases=["diamond"],
        temperature_range_K=(0.0, 1500.0),
        per_atom_quantities_valid=True,
        known_failure_modes=[
            "classical potential: no electronic effects",
            "fitted to melting/cohesion: transferability to defects limited",
            "same-model values are regression fixtures, never physics keys",
        ])


def _gate_refs(config: dict, version: str) -> list[GateRef]:
    return [GateRef(name=k, version=version,
                    params={} if not isinstance(v, (int, float))
                    else {k: float(v)})
            for k, v in config.items()]


def si_eos_capability() -> CapabilityManifest:
    return CapabilityManifest(
        capability_id="si_eos_sw_lammps",
        version="1.0",
        property_name="equation_of_state",
        property_aliases=["eos", "bulk_modulus", "B0"],
        property_family="structural",
        engine="lammps",
        protocol="eos_sweep_v1",
        physical_model=_sw_native_card(),
        plan_type="LammpsEosPlan",
        analyzers=["birch_murnaghan_fit"],
        gates=_gate_refs(EOS_CONFIG, EOS_GATES_VERSION),
        outputs=[
            OutputSpec(name="V0_per_cell", kind="scalar", unit="Ang^3"),
            OutputSpec(name="B0", kind="scalar", unit="GPa"),
            OutputSpec(name="B0_prime", kind="scalar", unit="1"),
            OutputSpec(name="E_V_curve", kind="curve", unit="eV",
                       conditions=["V"]),
        ],
        validity_domain={
            "a0_A": Bound(lo=4.0, hi=7.0, unit="Ang"),
            "scale_min": Bound(lo=0.85, hi=0.999, unit="1"),
            "scale_max": Bound(lo=1.001, hi=1.15, unit="1"),
        },
        provenance={"implemented": "2026-07-19", "work_package": "WP4"})


def si_elastic_capability() -> CapabilityManifest:
    return CapabilityManifest(
        capability_id="si_elastic_sw_lammps",
        version="1.0",
        property_name="elastic_tensor",
        property_aliases=["elastic_constants", "Cij", "stiffness_tensor"],
        property_family="mechanical",
        engine="lammps",
        protocol="finite_strain_v1",
        physical_model=_sw_native_card(),
        plan_type="LammpsElasticPlan",
        analyzers=["linear_stress_strain_fit"],
        gates=_gate_refs(ELASTIC_CONFIG, ELASTIC_GATES_VERSION),
        outputs=[
            OutputSpec(name="elastic_tensor", kind="tensor", unit="GPa"),
            OutputSpec(name="C11", kind="scalar", unit="GPa"),
            OutputSpec(name="C12", kind="scalar", unit="GPa"),
            OutputSpec(name="C44", kind="scalar", unit="GPa"),
        ],
        validity_domain={
            "a0_A": Bound(lo=4.0, hi=7.0, unit="Ang"),
            "max_strain": Bound(lo=0.0005, hi=0.02, unit="1"),
        },
        provenance={"implemented": "2026-07-19", "work_package": "WP4",
                    "note": "relax_ions=True default (diamond shear DOF)"})


def _lj_card() -> PhysicalModelCard:
    return PhysicalModelCard(
        name="LJ_argon_reduced", version="1.0",
        source="pair lj/cut 2.5, eps=sigma=m=kB=1 (reduced units)",
        elements=["Ar"], phases=["liquid", "fcc"],
        per_atom_quantities_valid=True,
        known_failure_modes=[
            "reduced-unit model system: quantitative argon comparison "
            "requires unit mapping + cutoff corrections",
            "truncated at 2.5 sigma without tail corrections",
        ])


def si_expansion_capability() -> CapabilityManifest:
    return CapabilityManifest(
        capability_id="si_expansion_sw_lammps", version="1.0",
        property_name="thermal_expansion",
        property_aliases=["alpha", "lattice_expansion"],
        property_family="thermal",
        engine="lammps", protocol="npt_lattice_v1",
        physical_model=_sw_native_card(),
        plan_type="SiExpansionPlan",
        analyzers=["npt_lattice_expansion_fit"],
        gates=_gate_refs(EXPANSION_CONFIG, EXPANSION_GATES_VERSION),
        outputs=[OutputSpec(name="alpha_linear", kind="scalar", unit="1/K"),
                 OutputSpec(name="a_vs_T", kind="curve", unit="Ang",
                            conditions=["T"])],
        validity_domain={"a0_A": Bound(lo=4.0, hi=7.0, unit="Ang"),
                         "T_K": Bound(lo=50.0, hi=1400.0, unit="K")},
        provenance={"implemented": "2026-07-19", "work_package": "WP8"})


def lj_diffusion_capability() -> CapabilityManifest:
    return CapabilityManifest(
        capability_id="lj_diffusion_lammps", version="1.0",
        property_name="self_diffusion_coefficient",
        property_aliases=["diffusion", "diffusivity", "D"],
        property_family="transport",
        engine="lammps", protocol="lj_msd_v1",
        physical_model=_lj_card(),
        plan_type="LjDiffusionPlan",
        analyzers=["msd_diffusivity_fit"],
        gates=_gate_refs(DIFFUSION_CONFIG, TRANSPORT_GATES_VERSION),
        outputs=[OutputSpec(name="D_star", kind="scalar", unit="1"),
                 OutputSpec(name="msd_vs_t", kind="curve", unit="1",
                            conditions=["t"])],
        validity_domain={"rho_star": Bound(lo=0.05, hi=1.2, unit="1"),
                         "T_star": Bound(lo=0.3, hi=3.0, unit="1")},
        provenance={"implemented": "2026-07-19", "work_package": "WP8"})


def lj_kappa_gk_capability() -> CapabilityManifest:
    return CapabilityManifest(
        capability_id="lj_kappa_gk_lammps", version="1.0",
        property_name="thermal_conductivity",
        property_aliases=["kappa_gk"],
        property_family="transport",
        engine="lammps", protocol="lj_gk_v1",
        physical_model=_lj_card(),
        plan_type="LjKappaGkPlan",
        analyzers=["green_kubo_kappa"],
        gates=_gate_refs(GK_CONFIG, TRANSPORT_GATES_VERSION),
        outputs=[OutputSpec(name="kappa_star", kind="scalar", unit="1")],
        validity_domain={"rho_star": Bound(lo=0.05, hi=1.2, unit="1"),
                         "T_star": Bound(lo=0.3, hi=3.0, unit="1")},
        provenance={"implemented": "2026-07-19", "work_package": "WP8",
                    "note": "convergence-gated; CARC-scale runs for stats"})


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
