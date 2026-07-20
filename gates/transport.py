"""Transport property gates (WP8, LJ reduced units). Convergence gates
are deliberately strict for short runs: a Green-Kubo run that has not
converged FAILS component_consistency — measured, not hidden."""
import math

TRANSPORT_GATES_VERSION = "0.1"

DIFFUSION_CONFIG = {
    "min_checkpoints": 8,
    "diffusive_msd_min_lj": 1.0,      # MSD must exceed ~sigma^2: diffusive
    "fit_resid_max_rel": 0.05,
    "D_band_lj": (1e-5, 10.0),
}

GK_CONFIG = {
    "component_spread_max_rel": 0.5,
    "kappa_band_lj": (1e-3, 100.0),
}


def run_diffusion_gates(diag, config=None):
    cfg = dict(DIFFUSION_CONFIG, **(config or {}))
    checks = []

    def gate(name, passed, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    gate("checkpoints_exist",
         diag.get("n_checkpoints", 0) >= cfg["min_checkpoints"],
         f"n = {diag.get('n_checkpoints', 0)}")
    gate("finite_values", all(math.isfinite(x) for x in
                              [diag["D_lj"], diag["msd_final_lj"],
                               diag["fit_resid_rel"]]))
    gate("diffusive_regime",
         diag["msd_final_lj"] >= cfg["diffusive_msd_min_lj"],
         f"final MSD = {diag['msd_final_lj']:.3f} sigma^2 "
         f"(min {cfg['diffusive_msd_min_lj']})")
    gate("fit_residuals", diag["fit_resid_rel"] <= cfg["fit_resid_max_rel"],
         f"rel resid = {diag['fit_resid_rel']:.4f} (max {cfg['fit_resid_max_rel']})")
    lo, hi = cfg["D_band_lj"]
    gate("magnitude_plausibility", lo <= abs(diag["D_lj"]) <= hi,
         f"D* = {diag['D_lj']:.4f} (band [{lo:.0e}, {hi}])")

    return {"version": TRANSPORT_GATES_VERSION, "config": cfg,
            "passed": all(c["passed"] for c in checks), "gates": checks}


def run_gk_gates(diag, config=None):
    cfg = dict(GK_CONFIG, **(config or {}))
    checks = []

    def gate(name, passed, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    gate("finite_values", all(math.isfinite(k)
                              for k in diag["k_components_lj"]))
    gate("positive_components", diag["all_positive"],
         f"k = {['%.3f' % k for k in diag['k_components_lj']]}")
    gate("component_consistency",
         diag["component_spread_rel"] <= cfg["component_spread_max_rel"],
         f"spread = {diag['component_spread_rel']:.3f} "
         f"(max {cfg['component_spread_max_rel']}) — convergence proxy")
    lo, hi = cfg["kappa_band_lj"]
    gate("magnitude_plausibility", lo <= abs(diag["kappa_lj"]) <= hi,
         f"kappa* = {diag['kappa_lj']:.3f} (band [{lo:.0e}, {hi}])")

    return {"version": TRANSPORT_GATES_VERSION, "config": cfg,
            "passed": all(c["passed"] for c in checks), "gates": checks}
