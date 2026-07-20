"""Provider-agnostic LLM interface with a deterministic scripted
fallback (WP11).

Providers (in the order the phase plan prioritizes them):
  * "openai"     — needs OPENAI_API_KEY (+ `pip install openai`)
  * "anthropic"  — needs ANTHROPIC_API_KEY (+ `pip install anthropic`)
  * "scripted"   — deterministic, no network; default when no key exists

Selection: explicit `provider=` argument > AIPHYS_LLM_PROVIDER env var >
first provider with a usable key > scripted. Scripted mode keeps the
loop fully reproducible for tests and ablations.

In LLM mode the model only ever contributes *content* — a hypothesis
string and a narrative over gated, harness-computed statistics. It never
sees ungated results, never picks stage order, never writes code.
"""
import os

OPENAI_MODEL = os.environ.get("AIPHYS_OPENAI_MODEL", "gpt-4o")
ANTHROPIC_MODEL = os.environ.get("AIPHYS_ANTHROPIC_MODEL", "claude-sonnet-5")


class _ScriptedBackend:
    name = "scripted"

    def ask(self, prompt, max_tokens=500):
        raise RuntimeError("scripted backend has no free-form ask()")


class _OpenAIBackend:
    name = "openai"

    def __init__(self):
        import openai
        self.client = openai.OpenAI()

    def ask(self, prompt, max_tokens=500):
        r = self.client.chat.completions.create(
            model=OPENAI_MODEL, max_completion_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content.strip()


class _AnthropicBackend:
    name = "anthropic"

    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()

    def ask(self, prompt, max_tokens=500):
        r = self.client.messages.create(
            model=ANTHROPIC_MODEL, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}])
        return r.content[0].text.strip()


def _pick_backend(provider=None, scripted=False):
    if scripted:
        return _ScriptedBackend()
    provider = provider or os.environ.get("AIPHYS_LLM_PROVIDER")
    candidates = ([provider] if provider else ["openai", "anthropic"])
    for p in candidates:
        try:
            if p == "openai" and os.environ.get("OPENAI_API_KEY"):
                return _OpenAIBackend()
            if p == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
                return _AnthropicBackend()
            if p == "scripted":
                return _ScriptedBackend()
        except ImportError:
            continue
    return _ScriptedBackend()


class LLM:
    def __init__(self, scripted=False, provider=None):
        self.backend = _pick_backend(provider=provider, scripted=scripted)

    @property
    def mode(self):
        return ("scripted" if self.backend.name == "scripted"
                else f"llm/{self.backend.name}")

    def hypothesize(self, benchmark):
        if self.backend.name == "scripted":
            return benchmark.registered_hypothesis
        return self.backend.ask(
            "You are a physicist studying phonon thermal transport in "
            "crystalline silicon with a Callaway-style relaxation-time model "
            "(boundary, point-defect, and Umklapp scattering). Propose ONE "
            "falsifiable hypothesis about the effect of reducing the "
            "characteristic boundary length from 1 mm to 10 um on kappa(T) "
            "between 50 and 500 K, including how the effect should depend on "
            "temperature. One sentence.")

    def narrate(self, hypothesis, gate_report, score_report, fit):
        if self.backend.name == "scripted":
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
        return self.backend.ask(
            "Write a 3-5 sentence results paragraph for a research note. "
            "Use ONLY these harness-computed facts; do not invent numbers.\n"
            f"Hypothesis: {hypothesis}\nGate report: {gate_report}\n"
            f"Score report: {score_report}\nFitted parameters: {fit}")
