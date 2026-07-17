"""Allowlisted physics engines. One contract: run(plan, rng) -> results dict.

The registry is the allowlist — a plan naming any other engine fails
validation before execution. No model-generated code is ever run.
"""
from .callaway import CallawayEngine

REGISTRY = {
    CallawayEngine.name: CallawayEngine,
}


def get_engine(name):
    return REGISTRY[name]()
