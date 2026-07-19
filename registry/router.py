"""WP2 deterministic router: hypothesis -> registered capability, or
`unsupported_hypothesis`. Pure function of (request, registry) — no LLM,
no clock, no randomness — so decisions are exactly reproducible.

Fail-closed rules (documented, deliberate):
  * a capability that declares elements requires the request to declare
    elements (subset match); an element-silent request is rejected;
  * any request condition whose key is not in the capability's validity
    domain rejects that capability (unknown knobs are not ignored);
  * condition values are unit-converted before comparison; a dimension
    mismatch is a rejection, not an exception.

The LLM may express `preferred_capability_id`; preference can only select
among validator-approved candidates, never expand them.
"""
from typing import Literal, Optional

from pydantic import Field

from contracts import DimensionError, ScalarValue, convert
from contracts.schemas import StrictModel as _Strict
from registry import CapabilityRegistry

ROUTER_VERSION = "1.0"


class HypothesisRequest(_Strict):
    request_id: str
    hypothesis: str = Field(min_length=10)
    property_name: str
    elements: list[str] = []
    conditions: dict[str, ScalarValue] = {}
    engine: Optional[str] = None
    preferred_capability_id: Optional[str] = None


class RouteDecision(_Strict):
    router_version: str = ROUTER_VERSION
    request_id: str
    accepted: bool
    outcome: Literal["routed", "unsupported_hypothesis"]
    capability_id: Optional[str] = None
    plan_type: Optional[str] = None
    candidates: list[str] = []
    preference_honored: Optional[bool] = None
    reasons: list[str] = []          # per-capability mismatch trail


def _mismatch(req: HypothesisRequest, m) -> Optional[str]:
    """First reason this manifest cannot serve the request, else None."""
    names = {m.property_name, *m.property_aliases}
    if req.property_name not in names:
        return f"property {req.property_name!r} not offered (offers {m.property_name!r})"
    if req.engine is not None and req.engine != m.engine:
        return f"engine {req.engine!r} requested but capability runs {m.engine!r}"
    card = m.physical_model
    if card.elements:
        if not req.elements:
            return ("capability is element-specific "
                    f"({'/'.join(card.elements)}); request declares no elements")
        extra = set(req.elements) - set(card.elements)
        if extra:
            return f"elements {sorted(extra)} outside model coverage {card.elements}"
    for key, cond in req.conditions.items():
        bound = m.validity_domain.get(key)
        if bound is None:
            return f"condition {key!r} not governed by this capability (fail-closed)"
        try:
            val = convert(cond.value, cond.unit, bound.unit)
        except DimensionError as e:
            return f"condition {key!r}: {e}"
        if not (bound.lo <= val <= bound.hi):
            return (f"condition {key!r} = {val:g} {bound.unit} outside "
                    f"[{bound.lo:g}, {bound.hi:g}] {bound.unit}")
    return None


def route(request: HypothesisRequest, registry: CapabilityRegistry) -> RouteDecision:
    reasons, candidates = [], []
    for m in registry.all():                       # sorted -> deterministic
        why = _mismatch(request, m)
        if why is None:
            candidates.append(m.capability_id)
        else:
            reasons.append(f"{m.capability_id}: {why}")

    if not candidates:
        return RouteDecision(request_id=request.request_id, accepted=False,
                             outcome="unsupported_hypothesis", reasons=reasons)

    pref = request.preferred_capability_id
    honored = pref in candidates if pref is not None else None
    chosen = pref if honored else candidates[0]
    return RouteDecision(request_id=request.request_id, accepted=True,
                         outcome="routed", capability_id=chosen,
                         plan_type=registry.get(chosen).plan_type,
                         candidates=candidates, preference_honored=honored,
                         reasons=reasons)


class RoutingLog:
    """Metrics hook: accumulate decisions; score rejection quality against
    labeled fixtures (Phase-2 metric: rejection precision/recall)."""

    def __init__(self):
        self.decisions: list[RouteDecision] = []

    def record(self, decision: RouteDecision) -> RouteDecision:
        self.decisions.append(decision)
        return decision

    def summary(self) -> dict:
        n = len(self.decisions)
        rejected = sum(not d.accepted for d in self.decisions)
        return {"n": n, "routed": n - rejected, "rejected": rejected}

    def rejection_metrics(self, supported_labels: dict[str, bool]) -> dict:
        """Positive class = 'correctly reject an unsupported hypothesis'.
        supported_labels: request_id -> True if genuinely supported."""
        tp = fp = tn = fn = 0
        for d in self.decisions:
            supported = supported_labels[d.request_id]
            if not d.accepted and not supported:
                tp += 1
            elif not d.accepted and supported:
                fp += 1
            elif d.accepted and supported:
                tn += 1
            else:
                fn += 1
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "precision": precision, "recall": recall,
                "accuracy": (tp + tn) / len(self.decisions) if self.decisions else float("nan")}
