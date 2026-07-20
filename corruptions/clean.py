"""Clean (uncorrupted) diagnostics factories per capability kind — the
baseline inputs for the WP9 gate matrix and WP10 corrupted arms.

Callaway runs the real engine (fast, deterministic); EOS/elastic use the
same synthetic generators as the WP4 acceptance tests; WP8 kinds use
values measured in the 2026-07-19 sandbox smoke runs.
"""
import numpy as np

from analyzers.elastic import analyze_elastic, _BAR_TO_GPA
from analyzers.eos import analyze_eos, bm3_energy


class _Run:
    run_id, capability_id = "clean_factory", "clean_factory"


def _clean_callaway(seed=0):
    from engines.callaway import CallawayEngine
    from plans import ExperimentPlan
    plan = ExperimentPlan(plan_id="clean", hypothesis="Boundary scattering "
                          "suppresses kappa at low temperature.")
    return CallawayEngine().run(plan, np.random.default_rng(seed))


def _clean_eos(seed=0):
    scales = np.linspace(0.94, 1.06, 11)
    V = 160.2 * scales
    E = bm3_energy(V, -34.69, 160.2, 0.6, 4.2)
    raw = {"eos_points": [
        {"volume_scale": float(s),
         "sentinels": {"vol_A3": float(v), "pe_eV": float(e), "natoms": 8}}
        for s, v, e in zip(scales, V, E)]}
    return analyze_eos(raw, _Run())[1]


def _clean_elastic(seed=0):
    C = np.zeros((6, 6))
    C[:3, :3] = 64.0
    np.fill_diagonal(C[:3, :3], 166.0)
    for i in (3, 4, 5):
        C[i, i] = 80.0
    amps = [0.0025, 0.005]
    keys = ("pxx_bar", "pyy_bar", "pzz_bar", "pyz_bar", "pxz_bar", "pxy_bar")
    pts = []
    for j in range(1, 7):
        for a in [-x for x in amps[::-1]] + amps:
            sigma = C[:, j - 1] * a
            p_bar = -sigma / _BAR_TO_GPA
            pts.append({"voigt": j, "amplitude": float(a),
                        "F": np.eye(3).tolist(),
                        "sentinels": {"natoms": 8, "pe_eV": -34.69,
                                      "vol_A3": 160.2,
                                      **{k: float(v) for k, v
                                         in zip(keys, p_bar)}}})
    return analyze_elastic({"strain_points": pts}, _Run())[1]


def _clean_expansion(seed=0):
    return {"n_temps": 4, "T_K": [300, 500, 700, 900],
            "a_A": [5.435, 5.440, 5.446, 5.452], "alpha_per_K": 3.33e-6,
            "a_ref_A": 5.435, "equilibration_drift_rel": 8e-4,
            "fit_resid_rel": 0.05, "spearman_like_monotonic": 0.999}


def _clean_diffusion(seed=0):
    return {"n_checkpoints": 20, "t_lj": [], "msd_lj": [], "D_lj": 0.0283,
            "msd_final_lj": 5.6, "fit_resid_rel": 0.012,
            "fit_window_start_index": 8, "state_point": {}}


def _clean_gk(seed=0):
    return {"k_components_lj": [8.41, 6.86, 5.78], "kappa_lj": 7.02,
            "component_spread_rel": 0.154, "all_positive": True,
            "state_point": {}}


CLEAN_FACTORIES = {
    "callaway": _clean_callaway,
    "eos": _clean_eos,
    "elastic": _clean_elastic,
    "expansion": _clean_expansion,
    "diffusion": _clean_diffusion,
    "gk": _clean_gk,
}
