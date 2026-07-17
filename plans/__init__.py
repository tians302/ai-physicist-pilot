"""Typed experiment plans with hard guardrails.

The LLM fills plan *content* (hypothesis, parameter choices within bounds).
It never chooses stage order and never writes code. Validation is enforced
by the harness before execution; an invalid plan aborts the loop.
"""
from dataclasses import dataclass, asdict, fields

# Hard physical guardrails (inclusive bounds)
BOUNDS = {
    "T_min_K": (10.0, 1500.0),
    "T_max_K": (10.0, 1500.0),
    "T_step_K": (1.0, 50.0),
    "baseline_boundary_length_m": (1e-8, 1e-2),
    "intervention_boundary_length_m": (1e-8, 1e-2),
}
ALLOWED_ENGINES = {"callaway_rta_si"}
ALLOWED_FREE_PARAMS = {"A", "B"}
ALLOWED_SPLITS = {"alternating"}
ALLOWED_DATASETS = {"si_kappa_reference"}


@dataclass
class ExperimentPlan:
    plan_id: str
    hypothesis: str
    engine: str = "callaway_rta_si"
    T_min_K: float = 50.0
    T_max_K: float = 500.0
    T_step_K: float = 5.0
    baseline_boundary_length_m: float = 1e-3   # 1 mm
    intervention_boundary_length_m: float = 1e-5  # 10 um
    free_params: tuple = ("A", "B")
    dataset: str = "si_kappa_reference"
    split: str = "alternating"

    def to_dict(self):
        d = asdict(self)
        d["free_params"] = list(self.free_params)
        return d


def validate_plan(plan: ExperimentPlan) -> list:
    """Return a list of guardrail violations (empty list = valid)."""
    errors = []
    if not isinstance(plan.hypothesis, str) or len(plan.hypothesis.strip()) < 10:
        errors.append("hypothesis must be a non-trivial statement")
    if plan.engine not in ALLOWED_ENGINES:
        errors.append(f"engine '{plan.engine}' not in allowlist {sorted(ALLOWED_ENGINES)}")
    if plan.dataset not in ALLOWED_DATASETS:
        errors.append(f"dataset '{plan.dataset}' not in allowlist {sorted(ALLOWED_DATASETS)}")
    if plan.split not in ALLOWED_SPLITS:
        errors.append(f"split '{plan.split}' not in allowlist {sorted(ALLOWED_SPLITS)}")
    for name, (lo, hi) in BOUNDS.items():
        val = getattr(plan, name)
        if not (isinstance(val, (int, float)) and lo <= val <= hi):
            errors.append(f"{name}={val!r} outside guardrail [{lo}, {hi}]")
    if plan.T_min_K >= plan.T_max_K:
        errors.append("T_min_K must be < T_max_K")
    if not set(plan.free_params) <= ALLOWED_FREE_PARAMS:
        errors.append(f"free_params {plan.free_params} not subset of {sorted(ALLOWED_FREE_PARAMS)}")
    if len(plan.free_params) == 0:
        errors.append("at least one free parameter required")
    return errors
