"""Thermal-expansion analyzer (WP8): a(T) from NPT box averages ->
linear expansion coefficient alpha = (1/a_ref) da/dT.
"""
import numpy as np

from contracts import CurveValue, Observation, ObservationBundle, ScalarValue

ANALYZER_EXPANSION = "npt_lattice_expansion_fit"
EXPANSION_ANALYZER_VERSION = "0.1"


def analyze_expansion(raw_outputs: dict, raw_run) -> tuple[ObservationBundle, dict]:
    pts = raw_outputs["npt_points"]
    sc = raw_outputs.get("supercell", 1)
    T = np.array([p["T_K"] for p in pts])
    a = np.array([p["sentinels"]["lx_avg_A"] for p in pts]) / sc
    a_half = np.array([p["sentinels"]["lx_avg_half_A"] for p in pts]) / sc

    slope, intercept = np.polyfit(T, a, 1)
    a_ref = float(a[0])
    alpha = float(slope / a_ref)
    resid = np.polyval([slope, intercept], T) - a
    a_range = float(a.max() - a.min())

    diagnostics = {
        "n_temps": len(pts),
        "T_K": T.tolist(), "a_A": a.tolist(),
        "alpha_per_K": alpha,
        "a_ref_A": a_ref,
        "equilibration_drift_rel": float(np.max(np.abs(a - a_half) / a)),
        "fit_resid_rel": float(np.sqrt(np.mean(resid ** 2))
                               / max(a_range, 1e-12)),
        "spearman_like_monotonic": float(np.corrcoef(
            T, a)[0, 1]) if len(T) > 2 else float("nan"),
    }

    bundle = ObservationBundle(
        bundle_id=f"{raw_run.run_id}_expansion",
        run_id=raw_run.run_id, capability_id=raw_run.capability_id,
        analyzer=ANALYZER_EXPANSION,
        analyzer_version=EXPANSION_ANALYZER_VERSION,
        observations=[
            Observation(name="alpha_linear",
                        value=ScalarValue(value=alpha, unit="1/K"),
                        conditions={"T_min": ScalarValue(value=float(T[0]), unit="K"),
                                    "T_max": ScalarValue(value=float(T[-1]), unit="K")}),
            Observation(name="a_vs_T",
                        value=CurveValue(x=T.tolist(), y=a.tolist(),
                                         x_unit="K", y_unit="Ang")),
        ],
        notes=["alpha from linear fit of NPT time-averaged lattice "
               "parameter; classical MD (no quantum effects at low T)",
               "uncertainty from seed-to-seed scatter (Phase-2 battery), "
               "not reported per-run"])
    return bundle, diagnostics
