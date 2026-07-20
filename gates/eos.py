"""EOS property gates (WP4). Run on analyzer diagnostics BEFORE any LLM
judgment, in the same report shape as the common gates.

Thresholds are documented a priori; changes after the WP7 pre-registration
freeze require logged justification. `magnitude_plausibility` is a
deliberately order-of-magnitude band: it catches unit slips (eV/A^3 vs
GPa is x160; bar vs GPa is x1e4), NOT agreement with any answer key.
"""
import math

EOS_GATES_VERSION = "0.1"

CONFIG = {
    "min_points": 5,
    "coverage_low_max": 0.98,     # sampled scale range must reach below ...
    "coverage_high_min": 1.02,    # ... and above the reference volume
    "interior_margin_min": 0.10,  # V0 at least 10% of span from either edge
    "resid_over_span_max": 5e-3,
    "form_rel_dB0_max": 0.15,     # BM3 vs Murnaghan B0 shift
    "B0_GPa_range": (1.0, 1000.0),
}


def run_eos_gates(diag, config=None):
    cfg = dict(CONFIG, **(config or {}))
    checks = []

    def gate(name, passed, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    gate("points_exist", diag.get("n_points", 0) >= cfg["min_points"],
         f"n = {diag.get('n_points', 0)} (min {cfg['min_points']})")

    finite_keys = ("V0_A3", "B0_GPa", "B0_prime", "rms_resid_eV")
    gate("finite_values", all(math.isfinite(diag.get(k, float("nan")))
                              for k in finite_keys))

    gate("volume_coverage",
         diag["scale_min"] <= cfg["coverage_low_max"]
         and diag["scale_max"] >= cfg["coverage_high_min"],
         f"scales [{diag['scale_min']:.3f}, {diag['scale_max']:.3f}] "
         f"(need <= {cfg['coverage_low_max']} and >= {cfg['coverage_high_min']})")

    gate("minimum_interior",
         diag["v0_interior_margin"] >= cfg["interior_margin_min"],
         f"V0 margin = {diag['v0_interior_margin']:.3f} of span "
         f"(min {cfg['interior_margin_min']}) — V0 = {diag['V0_A3']:.2f} A^3")

    gate("fit_residuals", diag["resid_over_span"] <= cfg["resid_over_span_max"],
         f"rms resid / span = {diag['resid_over_span']:.2e} "
         f"(max {cfg['resid_over_span_max']:.0e})")

    gate("fit_form_sensitivity",
         diag["fit_form_rel_dB0"] <= cfg["form_rel_dB0_max"],
         f"|dB0|/B0 (BM3 vs Murnaghan) = {diag['fit_form_rel_dB0']:.3f} "
         f"(max {cfg['form_rel_dB0_max']})")

    lo, hi = cfg["B0_GPa_range"]
    gate("magnitude_plausibility", lo <= diag["B0_GPa"] <= hi,
         f"B0 = {diag['B0_GPa']:.1f} GPa (band [{lo}, {hi}]; unit-slip catcher)")

    return {"version": EOS_GATES_VERSION, "config": cfg,
            "passed": all(c["passed"] for c in checks), "gates": checks}
