"""LLM interface with a deterministic scripted fallback.

Scripted mode (no API key, or --scripted) keeps the loop fully
reproducible for tests and gate ablations. In LLM mode the model only
ever contributes *content* — a hypothesis string and a narrative over
gated, harness-computed statistics. It never sees ungated results,
never picks stage order, never writes code.
"""
import os

MODEL = "claude-sonnet-5"


class LLM:
    def __init__(self, scripted=False):
        self.client = None
        if not scripted and os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                self.client = anthropic.Anthropic()
            except ImportError:
                pass

    @property
    def mode(self):
        return "llm" if self.client else "scripted"

    def _ask(self, prompt, max_tokens=500):
        msg = self.client.messages.create(
            model=MODEL, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text.strip()

    def hypothesize(self, benchmark):
        if not self.client:
            return benchmark.registered_hypothesis
        return self._ask(
            "You are a physicist studying phonon thermal transport in crystalline "
            "silicon with a Callaway-style relaxation-time model (boundary, "
            "point-defect, and Umklapp scattering). Propose ONE falsifiable "
            "hypothesis about the effect of reducing the characteristic boundary "
            "length from 1 mm to 10 um on kappa(T) between 50 and 500 K, including "
            "how the effect should depend on temperature. One sentence.")

    def narrate(self, hypothesis, gate_report, score_report, fit):
        if not self.client:
            s = score_report["summary_stats"]
            verdict = "supported" if score_report["success"] else "not supported"
            return (
                f"All validity gates {'passed' if gate_report['passed'] else 'FAILED'}. "
                f"The intervention suppressed kappa at every temperature; the relative "
                f"suppression fell from {s['suppression_low_T']:.1%} at the low-T end to "
                f"{s['suppression_high_T']:.1%} at the high-T end "
                f"(Spearman rho = {s['spearman_T_vs_suppression']:.2f}), consistent with "
                f"boundary scattering dominating where phonon-phonon scattering freezes "
                f"out. The registered hypothesis is {verdict}."
            )
        return self._ask(
            "Write a 3-5 sentence results paragraph for a research note. "
            "Use ONLY these harness-computed facts; do not invent numbers.\n"
            f"Hypothesis: {hypothesis}\nGate report: {gate_report}\n"
            f"Score report: {score_report}\nFitted parameters: {fit}")
