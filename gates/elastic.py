"""Elastic property gates (WP4), run on analyzer diagnostics before any
LLM judgment. Thresholds documented a priori (frozen at WP7).

Corruption -> intended gate map (WP4 acceptance):
  bad units          -> magnitude_plausibility
  nonlinearity       -> linearity
  unconverged relax  -> reference_state_stress
  broken symmetry    -> tensor_symmetry
  unstable tensor    -> mechanical_stability
"""
import math

ELASTIC_GATES_VERSION = "0.1"

CONFIG = {
    "min_points": 12,                  # >= 2 amplitudes x 6 patterns
    "linearity_max_rel": 0.02,
    "window_stability_max_rel": 0.10,
    "reference_stress_max_GPa": 0.05,
    "asymmetry_max_rel": 0.05,
    "cubic_dev_max_rel": 0.05,
    "offblock_max_rel": 0.08,
    "C_diag_GPa_range": (1.0, 2000.0), # unit-slip band, not an answer key
}


def run_elastic_gates(diag, config=None):
    cfg = dict(CONFIG, **(config or {}))
    checks = []

    def gate(name, passed, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    gate("stress_data_exist", diag.get("n_points", 0) >= cfg["min_points"],
         f"n = {diag.get('n_points', 0)} (min {cfg['min_points']})")

    flat = [x for row in diag.get("C_GPa", [[float("nan")]]) for x in row]
    gate("finite_values", all(math.isfinite(x) for x in flat))

    gate("linearity", diag["linearity_max_rel"] <= cfg["linearity_max_rel"],
         f"max nonlinearity = {diag['linearity_max_rel']:.3f} "
         f"(max {cfg['linearity_max_rel']})")

    gate("strain_window_stability",
         diag["window_stability_max_rel"] <= cfg["window_stability_max_rel"],
         f"half-window dC/C = {diag['window_stability_max_rel']:.3f} "
         f"(max {cfg['window_stability_max_rel']})")

    gate("reference_state_stress",
         diag["reference_stress_max_GPa"] <= cfg["reference_stress_max_GPa"],
         f"max |sigma(0)| = {diag['reference_stress_max_GPa']:.4f} GPa "
         f"(max {cfg['reference_stress_max_GPa']}) — relax convergence check")

    sym_ok = (diag["asymmetry_rel"] <= cfg["asymmetry_max_rel"]
              and diag["cubic_diag_dev_rel"] <= cfg["cubic_dev_max_rel"]
              and diag["cubic_off_dev_rel"] <= cfg["cubic_dev_max_rel"]
              and diag["cubic_shear_dev_rel"] <= cfg["cubic_dev_max_rel"]
              and diag["offblock_rel"] <= cfg["offblock_max_rel"])
    gate("tensor_symmetry", sym_ok,
         f"asym {diag['asymmetry_rel']:.3f}, cubic dev "
         f"({diag['cubic_diag_dev_rel']:.3f}, {diag['cubic_off_dev_rel']:.3f}, "
         f"{diag['cubic_shear_dev_rel']:.3f}), off-block {diag['offblock_rel']:.3f}")

    born = diag["born_cubic"]
    stable = (diag["min_eig_GPa"] > 0.0 and born["C11_minus_C12"] > 0.0
              and born["C11_plus_2C12"] > 0.0 and born["C44"] > 0.0)
    gate("mechanical_stability", stable,
         f"min eig = {diag['min_eig_GPa']:.1f} GPa; C11-C12 = "
         f"{born['C11_minus_C12']:.1f}, C11+2C12 = {born['C11_plus_2C12']:.1f}, "
         f"C44 = {born['C44']:.1f}")

    lo, hi = cfg["C_diag_GPa_range"]
    gate("magnitude_plausibility",
         lo <= diag["C11_GPa"] <= hi and lo <= abs(diag["C44_GPa"]) <= hi,
         f"C11 = {diag['C11_GPa']:.1f}, C44 = {diag['C44_GPa']:.1f} GPa "
         f"(band [{lo}, {hi}]; unit-slip catcher)")

    return {"version": ELASTIC_GATES_VERSION, "config": cfg,
            "passed": all(c["passed"] for c in checks), "gates": checks}
