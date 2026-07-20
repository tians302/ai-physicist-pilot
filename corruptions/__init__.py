"""Named corruption registry (WP5/WP9/WP10). Each corruption is a pure
function diag -> diag' applied at the harness's injection point (AFTER
analysis, BEFORE gates), simulating a specific failure mode. Used by the
gate-matrix generator (WP9), the ablation battery (WP10), and tests.

Keyed by capability kind: 'callaway' | 'eos' | 'elastic'.
"""
import copy

import numpy as np


def _c(fn):
    def wrapped(diag):
        return fn(copy.deepcopy(diag))
    wrapped.__name__ = fn.__name__
    return wrapped


# ---------------------------------------------------------------- callaway

@_c
def cal_bad_fit(d):
    d["chi2_dof_calibration"] = 240.0
    return d


@_c
def cal_structured_residuals(d):
    n = len(d["residuals_calibration"])
    d["residuals_calibration"] = list(np.linspace(-3, 3, n))
    return d


@_c
def cal_missing_sigma(d):
    d["fit"]["A"]["sigma"] = 0.0
    return d


@_c
def cal_unstable_bootstrap(d):
    d["bootstrap_cv"]["B"] = 2.5
    return d


@_c
def cal_holdout_disagreement(d):
    d["chi2_dof_holdout"] = 40.0
    return d


@_c
def cal_nan_curve(d):
    d["curves"]["kappa_baseline"][3] = float("nan")
    return d


# --------------------------------------------------------------------- eos

@_c
def eos_unit_slip(d):
    d["B0_GPa"] *= 160.2                       # eV/A^3-vs-GPa style slip
    return d


@_c
def eos_edge_minimum(d):
    d["v0_interior_margin"] = 0.01             # unconverged relax signature
    return d


@_c
def eos_poor_coverage(d):
    d["scale_min"] = 0.995
    return d


@_c
def eos_noisy_fit(d):
    d["resid_over_span"] = 0.05
    return d


@_c
def eos_form_dependence(d):
    d["fit_form_rel_dB0"] = 0.5
    return d


# ----------------------------------------------------------------- elastic

@_c
def ela_unit_slip(d):
    for k in ("C11_GPa", "C12_GPa", "C44_GPa"):
        d[k] *= 1e-4                           # GPa-vs-bar style slip
    d["C_GPa"] = (np.array(d["C_GPa"]) * 1e-4).tolist()
    return d


@_c
def ela_nonlinear(d):
    d["linearity_max_rel"] = 0.30
    return d


@_c
def ela_unconverged_reference(d):
    d["reference_stress_max_GPa"] = 0.5
    return d


@_c
def ela_broken_symmetry(d):
    d["cubic_diag_dev_rel"] = 0.25
    return d


@_c
def ela_unstable(d):
    d["min_eig_GPa"] = -20.0
    d["born_cubic"]["C11_minus_C12"] = -20.0
    return d


@_c
def ela_window_drift(d):
    d["window_stability_max_rel"] = 0.4
    return d


# --------------------------------------------------------------- expansion

@_c
def exp_unit_slip(d):
    d["alpha_per_K"] *= 1e6                    # 1/K vs 1/uK style slip
    return d


@_c
def exp_unequilibrated(d):
    d["equilibration_drift_rel"] = 0.05
    return d


@_c
def exp_noisy(d):
    d["fit_resid_rel"] = 0.6
    return d


# --------------------------------------------------------------- diffusion

@_c
def dif_unit_slip(d):
    d["D_lj"] *= 1e4
    return d


@_c
def dif_subdiffusive(d):
    d["msd_final_lj"] = 0.05                   # never left the cage
    return d


@_c
def dif_noisy_fit(d):
    d["fit_resid_rel"] = 0.3
    return d


# ---------------------------------------------------------------------- gk

@_c
def gk_negative_component(d):
    d["k_components_lj"][0] = -abs(d["k_components_lj"][0])
    d["all_positive"] = False
    return d


@_c
def gk_unconverged(d):
    d["component_spread_rel"] = 2.0
    return d


@_c
def gk_unit_slip(d):
    d["kappa_lj"] *= 1e4
    return d


CORRUPTIONS = {
    "callaway": {
        "bad_fit": cal_bad_fit,
        "structured_residuals": cal_structured_residuals,
        "missing_sigma": cal_missing_sigma,
        "unstable_bootstrap": cal_unstable_bootstrap,
        "holdout_disagreement": cal_holdout_disagreement,
        "nan_curve": cal_nan_curve,
    },
    "eos": {
        "unit_slip": eos_unit_slip,
        "edge_minimum": eos_edge_minimum,
        "poor_coverage": eos_poor_coverage,
        "noisy_fit": eos_noisy_fit,
        "form_dependence": eos_form_dependence,
    },
    "elastic": {
        "unit_slip": ela_unit_slip,
        "nonlinear": ela_nonlinear,
        "unconverged_reference": ela_unconverged_reference,
        "broken_symmetry": ela_broken_symmetry,
        "unstable": ela_unstable,
        "window_drift": ela_window_drift,
    },
    "expansion": {
        "unit_slip": exp_unit_slip,
        "unequilibrated": exp_unequilibrated,
        "noisy": exp_noisy,
    },
    "diffusion": {
        "unit_slip": dif_unit_slip,
        "subdiffusive": dif_subdiffusive,
        "noisy_fit": dif_noisy_fit,
    },
    "gk": {
        "negative_component": gk_negative_component,
        "unconverged": gk_unconverged,
        "unit_slip": gk_unit_slip,
    },
}

# intended-gate map for WP9 selectivity scoring
INTENDED_GATE = {
    ("callaway", "bad_fit"): "goodness_of_fit",
    ("callaway", "structured_residuals"): "residual_structure",
    ("callaway", "missing_sigma"): "uncertainties",
    ("callaway", "unstable_bootstrap"): "bootstrap_stability",
    ("callaway", "holdout_disagreement"): "holdout_validation",
    ("callaway", "nan_curve"): "finite_values",
    ("eos", "unit_slip"): "magnitude_plausibility",
    ("eos", "edge_minimum"): "minimum_interior",
    ("eos", "poor_coverage"): "volume_coverage",
    ("eos", "noisy_fit"): "fit_residuals",
    ("eos", "form_dependence"): "fit_form_sensitivity",
    ("elastic", "unit_slip"): "magnitude_plausibility",
    ("elastic", "nonlinear"): "linearity",
    ("elastic", "unconverged_reference"): "reference_state_stress",
    ("elastic", "broken_symmetry"): "tensor_symmetry",
    ("elastic", "unstable"): "mechanical_stability",
    ("elastic", "window_drift"): "strain_window_stability",
    ("expansion", "unit_slip"): "magnitude_plausibility",
    ("expansion", "unequilibrated"): "equilibration",
    ("expansion", "noisy"): "fit_residuals",
    ("diffusion", "unit_slip"): "magnitude_plausibility",
    ("diffusion", "subdiffusive"): "diffusive_regime",
    ("diffusion", "noisy_fit"): "fit_residuals",
    ("gk", "negative_component"): "positive_components",
    ("gk", "unconverged"): "component_consistency",
    ("gk", "unit_slip"): "magnitude_plausibility",
}
