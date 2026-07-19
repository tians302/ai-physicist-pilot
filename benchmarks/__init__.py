"""Rediscovery benchmarks with answer keys scored by the HARNESS, not the LLM.

Each benchmark provides: a registered hypothesis (used verbatim in scripted
mode; a starting point in LLM mode), a default plan factory, and a scorer
that checks engine results against the hidden answer key.
"""
import numpy as np
from scipy.stats import spearmanr

from plans import ExperimentPlan


class SiliconBoundary:
    """Rediscover boundary-scattering suppression of kappa(T) in silicon.

    Answer key (from Callaway physics + Glassbrenner-Slack data):
      (1) reducing boundary length suppresses kappa at every temperature;
      (2) the RELATIVE suppression is larger at lower temperature
          (boundary scattering dominates when phonon-phonon scattering freezes out).
    """

    name = "silicon_boundary"
    registered_hypothesis = (
        "Reducing the characteristic boundary length from 1 mm to 10 um suppresses "
        "silicon thermal conductivity, with a larger relative effect at lower temperature."
    )

    def make_plan(self, hypothesis, plan_id):
        return ExperimentPlan(plan_id=plan_id, hypothesis=hypothesis)

    def score(self, results):
        T = np.array(results["curves"]["T_K"])
        kb = np.array(results["curves"]["kappa_baseline"])
        ki = np.array(results["curves"]["kappa_intervention"])
        suppression = 1.0 - ki / kb  # relative suppression, in (0,1) if claim (1) holds

        suppressed_everywhere = bool(np.all(ki < kb))
        rho, _ = spearmanr(T, suppression)
        low_T_stronger = bool(suppression[0] > suppression[-1] and rho <= -0.9)

        checks = [
            {"name": "suppressed_everywhere", "passed": suppressed_everywhere,
             "detail": f"min suppression = {suppression.min():.3f}"},
            {"name": "low_T_stronger", "passed": low_T_stronger,
             "detail": f"suppression {suppression[0]:.3f} @ {T[0]:.0f} K -> "
                       f"{suppression[-1]:.3f} @ {T[-1]:.0f} K, spearman(T, s) = {rho:.3f}"},
        ]
        return {
            "benchmark": self.name,
            "checks": checks,
            "success": all(c["passed"] for c in checks),
            # L is an effective model parameter: only the qualitative trend is
            # claimed; suppression magnitudes are not quantitative predictions.
            "claim_scope": "qualitative_trend_only",
            "summary_stats": {
                "suppression_low_T": float(suppression[0]),
                "suppression_high_T": float(suppression[-1]),
                "spearman_T_vs_suppression": float(rho),
            },
        }


REGISTRY = {SiliconBoundary.name: SiliconBoundary}


def get_benchmark(name):
    return REGISTRY[name]()
