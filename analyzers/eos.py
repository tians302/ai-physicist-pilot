"""Birch–Murnaghan EOS analyzer (WP4). Fits E(V) from an eos_sweep_v1
RawRun; reports V0, E0, B0, B0' with fit uncertainties and a fit-form
sensitivity diagnostic (BM3 vs Murnaghan). No validity judgment here —
gates/eos.py decides whether the fit is evidence.
"""
import numpy as np
from scipy.optimize import curve_fit

from contracts import (CurveValue, Observation, ObservationBundle,
                       ScalarValue, convert)

ANALYZER_EOS = "birch_murnaghan_fit"
EOS_ANALYZER_VERSION = "0.1"


def bm3_energy(V, E0, V0, B0, B0p):
    """3rd-order Birch–Murnaghan E(V); B0 in energy/volume units of inputs."""
    x = (V0 / V) ** (2.0 / 3.0)
    return E0 + 9.0 * V0 * B0 / 16.0 * ((x - 1.0) ** 3 * B0p
                                        + (x - 1.0) ** 2 * (6.0 - 4.0 * x))


def murnaghan_energy(V, E0, V0, B0, B0p):
    return (E0 + B0 * V / B0p * (((V0 / V) ** B0p) / (B0p - 1.0) + 1.0)
            - V0 * B0 / (B0p - 1.0))


def _fit(fn, V, E):
    p0 = (float(E.min()), float(V[np.argmin(E)]), 0.6, 4.0)
    popt, pcov = curve_fit(fn, V, E, p0=p0, maxfev=20000)
    return popt, np.sqrt(np.clip(np.diag(pcov), 0.0, None))


def analyze_eos(raw_outputs: dict, raw_run) -> tuple[ObservationBundle, dict]:
    pts = raw_outputs["eos_points"]
    V = np.array([p["sentinels"]["vol_A3"] for p in pts])
    E = np.array([p["sentinels"]["pe_eV"] for p in pts])
    natoms = int(pts[0]["sentinels"]["natoms"])
    scales = np.array([p["volume_scale"] for p in pts])

    (E0, V0, B0, B0p), sig = _fit(bm3_energy, V, E)
    (_, _, B0_m, _), _ = _fit(murnaghan_energy, V, E)

    resid = bm3_energy(V, E0, V0, B0, B0p) - E
    span = float(E.max() - E.min())
    B0_GPa = convert(B0, "eV/Ang^3", "GPa")
    B0_GPa_sigma = convert(float(sig[2]), "eV/Ang^3", "GPa")

    diagnostics = {
        "n_points": len(pts),
        "natoms": natoms,
        "scale_min": float(scales.min()),
        "scale_max": float(scales.max()),
        "V_min_A3": float(V.min()), "V_max_A3": float(V.max()),
        "V0_A3": float(V0), "E0_eV": float(E0),
        "B0_GPa": float(B0_GPa), "B0_prime": float(B0p),
        "B0_GPa_sigma": float(B0_GPa_sigma),
        "rms_resid_eV": float(np.sqrt(np.mean(resid ** 2))),
        "resid_over_span": float(np.sqrt(np.mean(resid ** 2)) / span) if span > 0 else float("inf"),
        "energy_span_eV": span,
        "v0_interior_margin": float(min(V0 - V.min(), V.max() - V0) / (V.max() - V.min())),
        "fit_form_rel_dB0": float(abs(B0 - B0_m) / abs(B0)) if B0 else float("inf"),
    }

    bundle = ObservationBundle(
        bundle_id=f"{raw_run.run_id}_eos",
        run_id=raw_run.run_id, capability_id=raw_run.capability_id,
        analyzer=ANALYZER_EOS, analyzer_version=EOS_ANALYZER_VERSION,
        observations=[
            Observation(name="V0_per_cell",
                        value=ScalarValue(value=float(V0), sigma=float(sig[1]),
                                          unit="Ang^3"),
                        conditions={"T": ScalarValue(value=0.0, unit="K")}),
            Observation(name="B0",
                        value=ScalarValue(value=float(B0_GPa),
                                          sigma=float(B0_GPa_sigma), unit="GPa"),
                        conditions={"T": ScalarValue(value=0.0, unit="K")}),
            Observation(name="B0_prime",
                        value=ScalarValue(value=float(B0p), sigma=float(sig[3]),
                                          unit="1")),
            Observation(name="E_V_curve",
                        value=CurveValue(x=V.tolist(), y=E.tolist(),
                                         x_unit="Ang^3", y_unit="eV")),
        ],
        notes=[f"fit-form sensitivity: |B0_BM3 - B0_Murnaghan|/B0 = "
               f"{diagnostics['fit_form_rel_dB0']:.3g}",
               "uncertainties are fit-covariance only; form spread reported "
               "separately (deterministic 0 K energies, no sampling noise)"])
    return bundle, diagnostics
