"""Thermal-expansion property gates (WP8). Thresholds a priori; frozen at
WP7. Magnitude band catches unit slips (1/K vs 1/mK etc.), not agreement
with any answer key."""
import math

EXPANSION_GATES_VERSION = "0.1"

CONFIG = {
    "min_temps": 3,
    "equilibration_drift_max_rel": 5e-3,
    "fit_resid_max_rel": 0.20,          # MD noise on small cells
    "alpha_band_per_K": (1e-9, 1e-3),   # |alpha| unit-slip band
}


def run_expansion_gates(diag, config=None):
    cfg = dict(CONFIG, **(config or {}))
    checks = []

    def gate(name, passed, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    gate("temps_exist", diag.get("n_temps", 0) >= cfg["min_temps"],
         f"n = {diag.get('n_temps', 0)} (min {cfg['min_temps']})")
    gate("finite_values", all(math.isfinite(x) for x in
                              [diag["alpha_per_K"], diag["fit_resid_rel"],
                               diag["equilibration_drift_rel"]]))
    gate("equilibration",
         diag["equilibration_drift_rel"] <= cfg["equilibration_drift_max_rel"],
         f"half-vs-full drift = {diag['equilibration_drift_rel']:.2e} "
         f"(max {cfg['equilibration_drift_max_rel']:.0e})")
    gate("fit_residuals", diag["fit_resid_rel"] <= cfg["fit_resid_max_rel"],
         f"rel resid = {diag['fit_resid_rel']:.3f} (max {cfg['fit_resid_max_rel']})")
    lo, hi = cfg["alpha_band_per_K"]
    gate("magnitude_plausibility", lo <= abs(diag["alpha_per_K"]) <= hi,
         f"|alpha| = {abs(diag['alpha_per_K']):.2e} /K (band [{lo:.0e}, {hi:.0e}])")

    return {"version": EXPANSION_GATES_VERSION, "config": cfg,
            "passed": all(c["passed"] for c in checks), "gates": checks}
