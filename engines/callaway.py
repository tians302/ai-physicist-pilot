"""Callaway-style RTA model of silicon thermal conductivity.

kappa(T) = kB/(2 pi^2 v) (kB T / hbar)^3  ∫_0^{Theta_D/T} tau(x,T) x^4 e^x/(e^x-1)^2 dx

with combined scattering rate
    tau^{-1} = v/L  +  A w^4  +  B w^2 T exp[-Theta_D/(3T)]
(boundary, isotope/point-defect, Umklapp). This is a simplified
Callaway/Holland-style model, not a first-principles prediction.

The engine is deterministic physics; the only agent-controllable inputs
arrive through a validated ExperimentPlan. Fitting calibrates only the
positive scalars A and B against the calibration split; physical
constants and the exponent structure are fixed.
"""
import csv
import numpy as np
from pathlib import Path
from scipy.integrate import quad
from scipy.optimize import least_squares

KB = 1.380649e-23      # J/K
HBAR = 1.0545718e-34   # J s
THETA_D = 645.0        # K, Debye temperature of Si
V_SOUND = 6400.0       # m/s, effective phonon velocity

N_BOOTSTRAP = 100


def kappa_model(T, L, A, B, theta_D=THETA_D, v=V_SOUND):
    """Thermal conductivity (W/m-K) at temperature T (K) for boundary length L (m)."""
    def integrand(x):
        w = x * KB * T / HBAR
        tau_inv = v / L + A * w**4 + B * w**2 * T * np.exp(-theta_D / (3.0 * T))
        return (1.0 / tau_inv) * x**4 * np.exp(x) / np.expm1(x) ** 2

    val, _ = quad(integrand, 1e-8, theta_D / T, limit=200)
    return KB / (2.0 * np.pi**2 * v) * (KB * T / HBAR) ** 3 * val


def kappa_curve(T_arr, L, A, B):
    return np.array([kappa_model(T, L, A, B) for T in T_arr])


def _fit_AB(T, kappa, sigma, L, x0=(-45.0, -19.0)):
    """Fit log10(A), log10(B) by weighted least squares. Positivity is built in."""
    def resid(p):
        return (kappa_curve(T, L, 10.0 ** p[0], 10.0 ** p[1]) - kappa) / sigma

    sol = least_squares(resid, x0=np.array(x0), method="lm", xtol=1e-10)
    return 10.0 ** sol.x[0], 10.0 ** sol.x[1], sol


def load_reference(data_path):
    rows = []
    with open(data_path) as f:
        for row in csv.DictReader(r for r in f if not r.startswith("#")):
            rows.append((float(row["T_K"]), float(row["kappa_W_mK"]), float(row["sigma_W_mK"])))
    rows.sort()
    return np.array(rows)


class CallawayEngine:
    """Allowlisted engine: run(plan, rng) -> results dict with uncertainties."""

    name = "callaway_rta_si"

    def __init__(self, data_path=None):
        if data_path is None:
            data_path = Path(__file__).resolve().parent.parent / "data" / "si_kappa_reference.csv"
        self.data = load_reference(data_path)

    def run(self, plan, rng):
        d = self.data
        mask = (d[:, 0] >= plan.T_min_K) & (d[:, 0] <= plan.T_max_K)
        d = d[mask]
        cal, hold = d[0::2], d[1::2]  # alternating split; agent never sees holdout
        L0 = plan.baseline_boundary_length_m
        L1 = plan.intervention_boundary_length_m

        A, B, sol = _fit_AB(cal[:, 0], cal[:, 1], cal[:, 2], L0)

        # Bootstrap over calibration points for parameter uncertainties
        boot = []
        n = len(cal)
        for _ in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, size=n)
            sub = cal[np.unique(idx)]
            if len(sub) < 4:
                continue
            try:
                Ai, Bi, _ = _fit_AB(sub[:, 0], sub[:, 1], sub[:, 2], L0,
                                    x0=(np.log10(A), np.log10(B)))
                boot.append((Ai, Bi))
            except Exception:
                continue
        boot = np.array(boot) if boot else np.empty((0, 2))

        def chi2_dof(pts):
            model = kappa_curve(pts[:, 0], L0, A, B)
            r = (model - pts[:, 1]) / pts[:, 2]
            dof = max(len(pts) - 2, 1)
            return float(np.sum(r**2) / dof), r

        chi2_cal, r_cal = chi2_dof(cal)
        chi2_hold, r_hold = chi2_dof(hold)

        T_grid = np.arange(plan.T_min_K, plan.T_max_K + 0.5 * plan.T_step_K, plan.T_step_K)
        results = {
            "engine": self.name,
            "fit": {
                "A": {"value": A, "sigma": float(np.std(boot[:, 0])) if len(boot) else float("nan"),
                      "units": "s^3"},
                "B": {"value": B, "sigma": float(np.std(boot[:, 1])) if len(boot) else float("nan"),
                      "units": "s/K"},
            },
            "chi2_dof_calibration": chi2_cal,
            "chi2_dof_holdout": chi2_hold,
            "residuals_calibration": r_cal.tolist(),
            "residuals_holdout": r_hold.tolist(),
            "n_bootstrap_ok": int(len(boot)),
            "bootstrap_cv": {
                "A": float(np.std(boot[:, 0]) / np.mean(boot[:, 0])) if len(boot) else float("nan"),
                "B": float(np.std(boot[:, 1]) / np.mean(boot[:, 1])) if len(boot) else float("nan"),
            },
            "curves": {
                "T_K": T_grid.tolist(),
                "kappa_baseline": kappa_curve(T_grid, L0, A, B).tolist(),
                "kappa_intervention": kappa_curve(T_grid, L1, A, B).tolist(),
            },
            "calibration_points": cal.tolist(),
            "holdout_points": hold.tolist(),
            "constants": {"theta_D_K": THETA_D, "v_m_s": V_SOUND,
                          "L_baseline_m": L0, "L_intervention_m": L1},
        }
        return results
