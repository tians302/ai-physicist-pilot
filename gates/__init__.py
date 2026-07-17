"""Hard validity gates — run by the harness BEFORE any LLM judgment.

Design principle: the agent cannot grade its own homework. Gates are
human-written, versioned like code, and their verdicts are computed from
engine results only. An LLM never sees ungated results and never
overrides a gate.

Gate set (adapted from the ai-physicist pilot, plus holdout validation):
  1. metrics_exist        — required result fields present
  2. finite_values        — no NaN/inf in reported quantities
  3. goodness_of_fit      — chi2/dof on calibration within bound
  4. residual_structure   — |lag-1 autocorrelation| of residuals within bound
  5. uncertainties        — every fitted parameter has a finite, positive sigma
  6. bootstrap_stability  — bootstrap CV of each fitted parameter within bound
  7. holdout_validation   — chi2/dof on holdout data (never seen in fitting) within bound
"""
import math

GATES_VERSION = "1.0"

CONFIG = {
    "chi2_dof_max": 4.0,          # simplified model vs 5% data; documented, not tuned post hoc
    "lag1_autocorr_max": 0.5,
    "bootstrap_cv_max": 0.5,
    "holdout_chi2_dof_max": 6.0,  # looser: holdout has no fitted freedom
    "min_bootstrap_ok": 50,
}

REQUIRED_KEYS = ["fit", "chi2_dof_calibration", "chi2_dof_holdout",
                 "residuals_calibration", "curves", "bootstrap_cv"]


def _lag1_autocorr(r):
    n = len(r)
    if n < 3:
        return 0.0
    mean = sum(r) / n
    num = sum((r[i] - mean) * (r[i + 1] - mean) for i in range(n - 1))
    den = sum((x - mean) ** 2 for x in r)
    return num / den if den > 0 else 0.0


def run_gates(results, config=None):
    cfg = dict(CONFIG, **(config or {}))
    checks = []

    def gate(name, passed, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    missing = [k for k in REQUIRED_KEYS if k not in results]
    gate("metrics_exist", not missing, f"missing: {missing}" if missing else "")
    if missing:
        return {"version": GATES_VERSION, "passed": False, "gates": checks}

    flat = ([results["chi2_dof_calibration"], results["chi2_dof_holdout"]]
            + results["residuals_calibration"]
            + results["curves"]["kappa_baseline"]
            + results["curves"]["kappa_intervention"]
            + [p["value"] for p in results["fit"].values()])
    gate("finite_values", all(math.isfinite(x) for x in flat))

    chi2 = results["chi2_dof_calibration"]
    gate("goodness_of_fit", chi2 <= cfg["chi2_dof_max"],
         f"chi2/dof = {chi2:.2f} (max {cfg['chi2_dof_max']})")

    ac = _lag1_autocorr(results["residuals_calibration"])
    gate("residual_structure", abs(ac) <= cfg["lag1_autocorr_max"],
         f"lag-1 autocorr = {ac:.3f} (|.| max {cfg['lag1_autocorr_max']})")

    sig_ok = all(math.isfinite(p["sigma"]) and p["sigma"] > 0 for p in results["fit"].values())
    gate("uncertainties", sig_ok,
         "; ".join(f"{k}: sigma={p['sigma']:.3g}" for k, p in results["fit"].items()))

    cvs = results["bootstrap_cv"]
    cv_ok = (results.get("n_bootstrap_ok", 0) >= cfg["min_bootstrap_ok"]
             and all(math.isfinite(v) and v <= cfg["bootstrap_cv_max"] for v in cvs.values()))
    gate("bootstrap_stability", cv_ok,
         ", ".join(f"CV({k})={v:.3f}" for k, v in cvs.items())
         + f" (max {cfg['bootstrap_cv_max']}, n_ok={results.get('n_bootstrap_ok', 0)})")

    hchi2 = results["chi2_dof_holdout"]
    gate("holdout_validation", hchi2 <= cfg["holdout_chi2_dof_max"],
         f"holdout chi2/dof = {hchi2:.2f} (max {cfg['holdout_chi2_dof_max']})")

    return {"version": GATES_VERSION, "config": cfg,
            "passed": all(c["passed"] for c in checks), "gates": checks}
