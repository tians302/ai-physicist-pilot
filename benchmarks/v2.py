"""Benchmarks for the generic v2 loop (WP5/WP6). Each benchmark supplies
a registered hypothesis, a HypothesisRequest for the router, a typed plan
factory, and a harness scorer over analyzer diagnostics.

Scoring policy: pass/fail checks use only verified answer-key entries or
structural/qualitative keys; unverified experimental values appear as
info-only lines (never scored) — see benchmarks/answer_keys.py.
"""
import numpy as np
from scipy.stats import spearmanr

from benchmarks.answer_keys import provisional_info, verified_entry
from contracts import ResourceBudget, ScalarValue
from contracts.lammps import (LammpsElasticPlan, LammpsEosPlan,
                              LjDiffusionPlan, LjKappaGkPlan,
                              SiExpansionPlan)
from plans import ExperimentPlan
from registry.router import HypothesisRequest


class SiliconBoundaryV2:
    name = "silicon_boundary"
    kind = "callaway"
    hypothesis = ("Reducing the characteristic boundary length from 1 mm to "
                  "10 um suppresses silicon thermal conductivity, with a "
                  "larger relative effect at lower temperature.")

    def request(self, hypothesis, rid):
        return HypothesisRequest(
            request_id=rid, hypothesis=hypothesis,
            property_name="thermal_conductivity_curve", elements=["Si"],
            conditions={"T_min_K": ScalarValue(value=50.0, unit="K"),
                        "T_max_K": ScalarValue(value=500.0, unit="K")})

    def make_plan(self, hypothesis, decision, plan_id, seed):
        return ExperimentPlan(plan_id=plan_id, hypothesis=hypothesis)

    def score(self, bundle, diag):
        T = np.array(diag["curves"]["T_K"])
        kb = np.array(diag["curves"]["kappa_baseline"])
        ki = np.array(diag["curves"]["kappa_intervention"])
        s = 1.0 - ki / kb
        rho, _ = spearmanr(T, s)
        checks = [
            {"name": "suppressed_everywhere", "passed": bool(np.all(ki < kb)),
             "detail": f"min suppression = {s.min():.3f}"},
            {"name": "low_T_stronger",
             "passed": bool(s[0] > s[-1] and rho <= -0.9),
             "detail": f"{s[0]:.3f} @ {T[0]:.0f} K -> {s[-1]:.3f} @ "
                       f"{T[-1]:.0f} K, spearman = {rho:.3f}"},
        ]
        return _report(self.name, checks,
                       info=["claim_scope: qualitative_trend_only"])


class SiEosV2:
    name = "si_eos"
    kind = "eos"
    hypothesis = ("Diamond-silicon zero-temperature energy-volume response "
                  "near equilibrium is convex with an interior minimum and a "
                  "positive bulk modulus (Birch-Murnaghan form).")

    def request(self, hypothesis, rid):
        return HypothesisRequest(
            request_id=rid, hypothesis=hypothesis,
            property_name="equation_of_state", elements=["Si"])

    def make_plan(self, hypothesis, decision, plan_id, seed):
        return LammpsEosPlan(
            plan_id=plan_id, capability_id=decision.capability_id,
            hypothesis=hypothesis, model_key="sw_si_native_1985", seed=seed,
            resource_budget=ResourceBudget(max_walltime_s=300))

    def score(self, bundle, diag):
        checks = [
            {"name": "interior_minimum",
             "passed": diag["v0_interior_margin"] >= 0.1,
             "detail": f"margin {diag['v0_interior_margin']:.3f}"},
            {"name": "positive_bulk_modulus", "passed": diag["B0_GPa"] > 0,
             "detail": f"B0 = {diag['B0_GPa']:.2f} GPa"},
        ]
        info = [f"INFO-ONLY provisional experimental B0 ~ "
                f"{provisional_info('B0')} GPa (UNVERIFIED KEY; model-"
                f"conditional SW value expected to differ)"]
        return _report(self.name, checks, info=info)


class SiElasticV2:
    name = "si_elastic"
    kind = "elastic"
    hypothesis = ("The zero-temperature elastic tensor of diamond silicon "
                  "has cubic symmetry and satisfies the Born mechanical-"
                  "stability criteria.")

    def request(self, hypothesis, rid):
        return HypothesisRequest(
            request_id=rid, hypothesis=hypothesis,
            property_name="elastic_tensor", elements=["Si"])

    def make_plan(self, hypothesis, decision, plan_id, seed):
        return LammpsElasticPlan(
            plan_id=plan_id, capability_id=decision.capability_id,
            hypothesis=hypothesis, model_key="sw_si_native_1985", seed=seed,
            resource_budget=ResourceBudget(max_walltime_s=300))

    def score(self, bundle, diag):
        key = verified_entry("stability_relations")   # qualitative, verified
        b = diag["born_cubic"]
        checks = [
            {"name": "born_stability",
             "passed": (b["C11_minus_C12"] > 0 and b["C11_plus_2C12"] > 0
                        and b["C44"] > 0 and diag["min_eig_GPa"] > 0),
             "detail": f"key: {key['value']}; min eig = "
                       f"{diag['min_eig_GPa']:.1f} GPa"},
            {"name": "cubic_symmetry",
             "passed": (diag["asymmetry_rel"] <= 0.05
                        and diag["cubic_diag_dev_rel"] <= 0.05
                        and diag["offblock_rel"] <= 0.08),
             "detail": f"asym {diag['asymmetry_rel']:.3f}, "
                       f"diag dev {diag['cubic_diag_dev_rel']:.3f}"},
        ]
        info = [f"INFO-ONLY provisional experimental C11/C12/C44 ~ "
                f"{provisional_info('C11')}/{provisional_info('C12')}/"
                f"{provisional_info('C44')} GPa (UNVERIFIED KEY; SW is "
                f"model-conditional: measured {diag['C11_GPa']:.1f}/"
                f"{diag['C12_GPa']:.1f}/{diag['C44_GPa']:.1f})"]
        return _report(self.name, checks, info=info)


class SiExpansionV2:
    name = "si_expansion"
    kind = "expansion"
    hypothesis = ("The equilibrium lattice parameter of diamond silicon "
                  "increases monotonically with temperature between 300 and "
                  "900 K (positive linear thermal expansion).")

    def request(self, hypothesis, rid):
        return HypothesisRequest(
            request_id=rid, hypothesis=hypothesis,
            property_name="thermal_expansion", elements=["Si"])

    def make_plan(self, hypothesis, decision, plan_id, seed):
        return SiExpansionPlan(
            plan_id=plan_id, capability_id=decision.capability_id,
            hypothesis=hypothesis, model_key="sw_si_native_1985", seed=seed,
            resource_budget=ResourceBudget(max_walltime_s=600))

    def score(self, bundle, diag):
        T = np.array(diag["T_K"])
        a = np.array(diag["a_A"])
        rho, _ = spearmanr(T, a)
        checks = [
            {"name": "alpha_positive", "passed": diag["alpha_per_K"] > 0,
             "detail": f"alpha = {diag['alpha_per_K']:.3e} /K"},
            {"name": "a_monotonic_increasing", "passed": bool(rho >= 0.9),
             "detail": f"spearman(T, a) = {rho:.3f}"},
        ]
        info = ["INFO-ONLY: experimental Si alpha ~ 2.6e-6 /K near 300 K "
                "(UNVERIFIED ballpark; classical MD lacks quantum "
                "suppression at low T; model-conditional)"]
        return _report(self.name, checks, info=info)


class LjDiffusionV2:
    name = "lj_diffusion"
    kind = "diffusion"
    hypothesis = ("The Lennard-Jones liquid near its triple point "
                  "(rho*=0.8442, T*=0.722) shows a diffusive regime with a "
                  "positive self-diffusion coefficient.")

    def request(self, hypothesis, rid):
        return HypothesisRequest(
            request_id=rid, hypothesis=hypothesis,
            property_name="self_diffusion_coefficient", elements=["Ar"],
            conditions={"rho_star": ScalarValue(value=0.8442, unit="1"),
                        "T_star": ScalarValue(value=0.722, unit="1")})

    def make_plan(self, hypothesis, decision, plan_id, seed):
        return LjDiffusionPlan(
            plan_id=plan_id, capability_id=decision.capability_id,
            hypothesis=hypothesis, seed=seed,
            resource_budget=ResourceBudget(max_walltime_s=600))

    def score(self, bundle, diag):
        checks = [
            {"name": "diffusive_and_positive",
             "passed": diag["D_lj"] > 0 and diag["msd_final_lj"] >= 1.0,
             "detail": f"D* = {diag['D_lj']:.4f}, final MSD = "
                       f"{diag['msd_final_lj']:.2f}"},
        ]
        info = ["INFO-ONLY: literature D* ~ 0.03 near the LJ triple point "
                "(UNVERIFIED ballpark; DOI curation pending per WP8)"]
        return _report(self.name, checks, info=info)


class LjKappaGkV2:
    name = "lj_kappa_gk"
    kind = "gk"
    hypothesis = ("Green-Kubo integration of the heat-flux autocorrelation "
                  "for the Lennard-Jones liquid near its triple point yields "
                  "a positive, direction-consistent thermal conductivity.")

    def request(self, hypothesis, rid):
        return HypothesisRequest(
            request_id=rid, hypothesis=hypothesis,
            property_name="thermal_conductivity", elements=["Ar"],
            conditions={"rho_star": ScalarValue(value=0.8442, unit="1"),
                        "T_star": ScalarValue(value=0.722, unit="1")})

    def make_plan(self, hypothesis, decision, plan_id, seed):
        return LjKappaGkPlan(
            plan_id=plan_id, capability_id=decision.capability_id,
            hypothesis=hypothesis, seed=seed,
            resource_budget=ResourceBudget(max_walltime_s=900))

    def score(self, bundle, diag):
        checks = [
            {"name": "positive_kappa",
             "passed": diag["kappa_lj"] > 0 and diag["all_positive"],
             "detail": f"kappa* = {diag['kappa_lj']:.3f}, components "
                       f"{['%.3f' % k for k in diag['k_components_lj']]}"},
        ]
        info = ["INFO-ONLY: literature kappa* ~ 7 near the LJ triple point "
                "(UNVERIFIED ballpark; DOI curation pending per WP8)"]
        return _report(self.name, checks, info=info)


def _report(name, checks, info=None):
    return {"benchmark": name, "checks": checks,
            "success": all(c["passed"] for c in checks),
            "info_only": info or []}


REGISTRY_V2 = {b.name: b for b in
               [SiliconBoundaryV2(), SiEosV2(), SiElasticV2(),
                SiExpansionV2(), LjDiffusionV2(), LjKappaGkV2()]}


def get_benchmark_v2(name):
    return REGISTRY_V2[name]
