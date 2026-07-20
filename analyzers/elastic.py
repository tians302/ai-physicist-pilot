"""Elastic-tensor analyzer (WP4). Builds Cij (Voigt, GPa) from the
finite_strain_v1 stress–strain points by per-pattern linear fits.

Conventions (recorded by the adapter in raw_outputs):
  * strain amplitude a is ENGINEERING strain (for shears, a = gamma and
    the applied tensor strain is eps_ij = gamma/2), so slopes are
    directly C_ij in Voigt notation with engineering shear;
  * LAMMPS reports the pressure tensor in bar; stress sigma = -p,
    converted here through contracts.units (bar -> GPa).
"""
import numpy as np

from contracts import Observation, ObservationBundle, ScalarValue, TensorValue, convert

ANALYZER_ELASTIC = "linear_stress_strain_fit"
ELASTIC_ANALYZER_VERSION = "0.1"

_BAR_TO_GPA = convert(1.0, "bar", "GPa")
# Voigt order 1..6 = xx, yy, zz, yz, xz, xy
_SENTINEL_ORDER = ("pxx_bar", "pyy_bar", "pzz_bar", "pyz_bar", "pxz_bar", "pxy_bar")


def _stress_gpa(sentinels: dict) -> np.ndarray:
    return np.array([-sentinels[k] * _BAR_TO_GPA for k in _SENTINEL_ORDER])


def _slopes_and_intercepts(amps, stresses):
    """Per-component linear fit sigma_i(a); returns (slopes, intercepts,
    max deviation from linearity relative to the PATTERN's dominant stress
    scale). Normalizing per-pattern (not per-component) matters: components
    with only quadratic cross-talk (e.g. normal stress under shear) would
    otherwise show O(1) 'relative' nonlinearity on a negligible signal."""
    A = np.vstack([amps, np.ones_like(amps)]).T
    coef, *_ = np.linalg.lstsq(A, stresses, rcond=None)
    slopes, intercepts = coef[0], coef[1]
    pred = A @ coef
    pattern_scale = max(float(np.abs(stresses).max()), 1e-8)
    nonlin = np.abs(pred - stresses).max(axis=0) / pattern_scale
    return slopes, intercepts, nonlin


def analyze_elastic(raw_outputs: dict, raw_run) -> tuple[ObservationBundle, dict]:
    pts = raw_outputs["strain_points"]
    C = np.zeros((6, 6))
    C_half = np.zeros((6, 6))
    intercepts = np.zeros((6, 6))
    nonlin_max = 0.0

    for j in range(1, 7):
        col = [p for p in pts if p["voigt"] == j]
        amps = np.array([p["amplitude"] for p in col])
        stresses = np.vstack([_stress_gpa(p["sentinels"]) for p in col])
        order = np.argsort(amps)
        amps, stresses = amps[order], stresses[order]

        slopes, inter, nonlin = _slopes_and_intercepts(amps, stresses)
        C[:, j - 1], intercepts[:, j - 1] = slopes, inter
        nonlin_max = max(nonlin_max, float(nonlin.max()))

        half = np.abs(amps) <= (np.abs(amps).max() / 2.0 + 1e-15)
        if half.sum() >= 2:
            s2, _, _ = _slopes_and_intercepts(amps[half], stresses[half])
            C_half[:, j - 1] = s2
        else:
            C_half[:, j - 1] = slopes

    Cs = 0.5 * (C + C.T)                       # symmetrized for eigen test
    normC = np.linalg.norm(C) or 1.0
    diag3 = np.array([C[0, 0], C[1, 1], C[2, 2]])
    off3 = np.array([C[0, 1], C[0, 2], C[1, 2]])
    shear3 = np.array([C[3, 3], C[4, 4], C[5, 5]])
    offblock = np.abs(np.vstack([C[:3, 3:], C[3:, :3].T])).max()

    C11, C12, C44 = float(diag3.mean()), float(off3.mean()), float(shear3.mean())
    diagnostics = {
        "n_points": len(pts),
        "C_GPa": C.tolist(),
        "C11_GPa": C11, "C12_GPa": C12, "C44_GPa": C44,
        "linearity_max_rel": float(nonlin_max),
        "window_stability_max_rel": float(
            np.abs(C_half - C).max() / max(np.abs(C).max(), 1e-8)),
        "reference_stress_max_GPa": float(np.abs(intercepts).max()),
        "asymmetry_rel": float(np.linalg.norm(C - C.T) / normC),
        "cubic_diag_dev_rel": float(diag3.std() / max(abs(diag3.mean()), 1e-8)),
        "cubic_off_dev_rel": float(off3.std() / max(abs(off3.mean()), 1e-8)),
        "cubic_shear_dev_rel": float(shear3.std() / max(abs(shear3.mean()), 1e-8)),
        "offblock_rel": float(offblock / max(abs(C44), 1e-8)),
        "min_eig_GPa": float(np.linalg.eigvalsh(Cs).min()),
        "born_cubic": {"C11_minus_C12": C11 - C12,
                       "C11_plus_2C12": C11 + 2.0 * C12, "C44": C44},
    }

    bundle = ObservationBundle(
        bundle_id=f"{raw_run.run_id}_elastic",
        run_id=raw_run.run_id, capability_id=raw_run.capability_id,
        analyzer=ANALYZER_ELASTIC, analyzer_version=ELASTIC_ANALYZER_VERSION,
        observations=[
            Observation(name="elastic_tensor",
                        value=TensorValue(value=C.tolist(), unit="GPa"),
                        conditions={"T": ScalarValue(value=0.0, unit="K")}),
            Observation(name="C11", value=ScalarValue(
                value=C11, sigma=float(diag3.std()), unit="GPa")),
            Observation(name="C12", value=ScalarValue(
                value=C12, sigma=float(off3.std()), unit="GPa")),
            Observation(name="C44", value=ScalarValue(
                value=C44, sigma=float(shear3.std()), unit="GPa")),
        ],
        notes=["Voigt engineering-shear convention; sigma = -p (LAMMPS "
               "pressure tensor); cubic averages reported, full tensor kept",
               "sigmas are cubic-equivalence spreads, not statistical errors"])
    return bundle, diagnostics
