"""Transport analyzers (WP8, LJ reduced units): MSD -> diffusivity and
Green-Kubo heat-flux autocorrelation -> thermal conductivity.

Reduced (LJ) quantities are dimensionless by construction; observations
carry unit "1" with the convention recorded in notes.
"""
import numpy as np

from contracts import CurveValue, Observation, ObservationBundle, ScalarValue

ANALYZER_MSD = "msd_diffusivity_fit"
ANALYZER_GK = "green_kubo_kappa"
TRANSPORT_ANALYZER_VERSION = "0.1"


def analyze_diffusion(raw_outputs: dict, raw_run) -> tuple[ObservationBundle, dict]:
    sen = raw_outputs["msd"]
    n = int(sen["n_checkpoints"])
    spc = int(sen["steps_per_checkpoint"])
    dt = float(sen["dt_lj"])
    t = np.array([(i + 1) * spc * dt for i in range(n)])
    msd = np.array([sen[f"msd_{i + 1}"] for i in range(n)])

    # fit the late (diffusive) window: last 60% of checkpoints
    i0 = max(int(0.4 * n), 2)
    slope, intercept = np.polyfit(t[i0:], msd[i0:], 1)
    D = float(slope / 6.0)
    pred = np.polyval([slope, intercept], t[i0:])
    resid_rel = float(np.sqrt(np.mean((pred - msd[i0:]) ** 2))
                      / max(msd[i0:].max() - msd[i0:].min(), 1e-12))

    diagnostics = {
        "n_checkpoints": n,
        "t_lj": t.tolist(), "msd_lj": msd.tolist(),
        "D_lj": D,
        "msd_final_lj": float(msd[-1]),
        "fit_resid_rel": resid_rel,
        "fit_window_start_index": i0,
        "state_point": raw_outputs.get("state_point", {}),
    }

    bundle = ObservationBundle(
        bundle_id=f"{raw_run.run_id}_diffusion",
        run_id=raw_run.run_id, capability_id=raw_run.capability_id,
        analyzer=ANALYZER_MSD, analyzer_version=TRANSPORT_ANALYZER_VERSION,
        observations=[
            Observation(name="D_star",
                        value=ScalarValue(value=D, unit="1")),
            Observation(name="msd_vs_t",
                        value=CurveValue(x=t.tolist(), y=msd.tolist(),
                                         x_unit="1", y_unit="1")),
        ],
        notes=[f"LJ reduced units (sigma=eps=m=kB=1); state point "
               f"{raw_outputs.get('state_point')}",
               "D = slope(MSD)/6 over the late-time window"])
    return bundle, diagnostics


def analyze_gk(raw_outputs: dict, raw_run) -> tuple[ObservationBundle, dict]:
    sen = raw_outputs["gk"]
    ks = [float(sen["k11_lj"]), float(sen["k22_lj"]), float(sen["k33_lj"])]
    kmean = float(np.mean(ks))
    spread = float(np.std(ks) / abs(kmean)) if kmean else float("inf")

    diagnostics = {
        "k_components_lj": ks,
        "kappa_lj": kmean,
        "component_spread_rel": spread,
        "all_positive": bool(all(k > 0 for k in ks)),
        "state_point": raw_outputs.get("state_point", {}),
    }

    bundle = ObservationBundle(
        bundle_id=f"{raw_run.run_id}_gk",
        run_id=raw_run.run_id, capability_id=raw_run.capability_id,
        analyzer=ANALYZER_GK, analyzer_version=TRANSPORT_ANALYZER_VERSION,
        observations=[
            Observation(name="kappa_star",
                        value=ScalarValue(value=kmean,
                                          sigma=float(np.std(ks)),
                                          unit="1")),
        ],
        notes=[f"LJ reduced units; Green-Kubo trap() integral of HCACF; "
               f"state point {raw_outputs.get('state_point')}",
               "component spread is a convergence proxy; short runs fail "
               "the consistency gate honestly",
               "sigma = spread across the three Cartesian components"])
    return bundle, diagnostics
