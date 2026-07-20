"""Callaway analyzer (WP5 migration). The legacy engine already computes
fit + statistics; this analyzer repackages them as a unit-checked
ObservationBundle and passes the results dict through as diagnostics for
the existing 7 gates (gates.run_gates), preserving Phase-1A behavior
exactly.
"""
from contracts import CurveValue, Observation, ObservationBundle, ScalarValue

ANALYZER_CALLAWAY = "callaway_fit_v1"
CALLAWAY_ANALYZER_VERSION = "1.0"


def analyze_callaway(raw_outputs: dict, raw_run) -> tuple[ObservationBundle, dict]:
    res = raw_outputs["callaway"]
    fit = res["fit"]
    curves = res["curves"]

    bundle = ObservationBundle(
        bundle_id=f"{raw_run.run_id}_bundle",
        run_id=raw_run.run_id, capability_id=raw_run.capability_id,
        analyzer=ANALYZER_CALLAWAY, analyzer_version=CALLAWAY_ANALYZER_VERSION,
        observations=[
            Observation(name="A_point_defect",
                        value=ScalarValue(value=fit["A"]["value"],
                                          sigma=fit["A"]["sigma"], unit="s^3")),
            Observation(name="B_umklapp",
                        value=ScalarValue(value=fit["B"]["value"],
                                          sigma=fit["B"]["sigma"], unit="s/K")),
            Observation(name="kappa_baseline",
                        value=CurveValue(x=curves["T_K"],
                                         y=curves["kappa_baseline"],
                                         x_unit="K", y_unit="W/m/K")),
            Observation(name="kappa_intervention",
                        value=CurveValue(x=curves["T_K"],
                                         y=curves["kappa_intervention"],
                                         x_unit="K", y_unit="W/m/K")),
        ],
        notes=[f"split_mode={res.get('split_mode')}",
               f"bootstrap={res.get('bootstrap_method')} "
               f"(n_ok={res.get('n_bootstrap_ok')})",
               "L is an effective parameter; intervention magnitudes are "
               "qualitative only (see plan caveats)"])
    # diagnostics: the raw results dict — exactly what gates v1.0 consume
    return bundle, res
