"""WP1 core contracts: CapabilityManifest, BaseExperimentPlan, RawRun,
ObservationBundle, and typed physical values (scalar/curve/tensor).

Design rules (PLAN.md):
  * hypothesis, property, protocol, engine, physical model, analyzer, and
    benchmark are separate named things;
  * every physical quantity carries a unit validated by contracts.units;
  * plans are typed and schema-validated BEFORE execution;
  * raw engine output (RawRun) is separated from analyzed, unit-checked
    observations (ObservationBundle) — analyzers, not engines, produce
    claims-grade numbers;
  * everything round-trips through JSON for provenance.

These contracts are the target interfaces for WP2 (registry/router),
WP3 (LAMMPS adapter), and WP5 (Callaway migration). The existing Callaway
loop is intentionally untouched until WP5.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal, Optional, Union

from pydantic import (BaseModel, ConfigDict, Field, field_validator,
                      model_validator)

from contracts.units import DimensionError, dim_of

PROPERTY_FAMILIES = ("thermal-analytic", "structural", "mechanical",
                     "thermal", "transport")


def _check_unit(u: str) -> str:
    try:
        dim_of(u)
    except DimensionError as e:
        raise ValueError(str(e))
    return u


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


StrictModel = _Strict   # public alias for downstream contract models


# ---------------------------------------------------------------- values

class ScalarValue(_Strict):
    kind: Literal["scalar"] = "scalar"
    value: float
    sigma: Optional[float] = Field(default=None, ge=0)
    unit: str

    _u = field_validator("unit")(_check_unit)


class CurveValue(_Strict):
    kind: Literal["curve"] = "curve"
    x: list[float]
    y: list[float]
    y_sigma: Optional[list[float]] = None
    x_unit: str
    y_unit: str

    _xu = field_validator("x_unit")(_check_unit)
    _yu = field_validator("y_unit")(_check_unit)

    @model_validator(mode="after")
    def _lengths(self):
        if len(self.x) != len(self.y):
            raise ValueError(f"curve length mismatch: |x|={len(self.x)} |y|={len(self.y)}")
        if self.y_sigma is not None and len(self.y_sigma) != len(self.y):
            raise ValueError("y_sigma length must match y")
        if len(self.x) < 2:
            raise ValueError("curve needs at least 2 points")
        return self


class TensorValue(_Strict):
    kind: Literal["tensor"] = "tensor"
    value: list[list[float]]
    sigma: Optional[list[list[float]]] = None
    unit: str

    _u = field_validator("unit")(_check_unit)

    @model_validator(mode="after")
    def _rectangular(self):
        rows = self.value
        if not rows or not rows[0]:
            raise ValueError("tensor must be non-empty")
        w = len(rows[0])
        if any(len(r) != w for r in rows):
            raise ValueError("tensor rows must have equal length")
        if self.sigma is not None:
            if len(self.sigma) != len(rows) or any(len(r) != w for r in self.sigma):
                raise ValueError("sigma shape must match value shape")
        return self


PhysicalValue = Annotated[Union[ScalarValue, CurveValue, TensorValue],
                          Field(discriminator="kind")]


# ------------------------------------------------------ capability manifest

class PhysicalModelCard(_Strict):
    """Applicability/provenance card for an interatomic potential or
    analytic physical model. 'Which physics is this model allowed to
    claim?' lives here, not in prose."""
    name: str
    version: str
    source: str                      # e.g. OpenKIM model ID, DOI, or module path
    checksum: Optional[str] = None
    elements: list[str] = []
    phases: list[str] = []
    temperature_range_K: Optional[tuple[float, float]] = None
    per_atom_quantities_valid: Optional[bool] = None
    known_failure_modes: list[str] = []


class GateRef(_Strict):
    name: str
    version: str
    params: dict[str, float] = {}


class Bound(_Strict):
    """Closed interval with an explicit unit — validity-domain entries are
    dimensional, so routing/validation can convert before comparing."""
    lo: float
    hi: float
    unit: str

    _u = field_validator("unit")(_check_unit)

    @model_validator(mode="after")
    def _ordered(self):
        if self.lo > self.hi:
            raise ValueError(f"bound lo {self.lo} > hi {self.hi}")
        return self


class OutputSpec(_Strict):
    name: str
    kind: Literal["scalar", "curve", "tensor"]
    unit: str                        # canonical unit expected of the analyzer
    conditions: list[str] = []       # e.g. ["T"] for temperature-resolved outputs

    _u = field_validator("unit")(_check_unit)


class CapabilityManifest(_Strict):
    """One registered capability: property × protocol × engine × model,
    with its gates and outputs. The router may only match hypotheses to
    manifests; anything unmatched is rejected, never improvised."""
    capability_id: str
    version: str
    property_name: str
    property_aliases: list[str] = []
    property_family: Literal["thermal-analytic", "structural", "mechanical",
                             "thermal", "transport"]
    engine: str
    protocol: str
    physical_model: PhysicalModelCard
    plan_type: str                   # required BaseExperimentPlan subclass name
    analyzers: list[str]
    gates: list[GateRef]
    outputs: list[OutputSpec]
    validity_domain: dict[str, Bound] = {}  # dimensional plan-parameter bounds
    provenance: dict[str, str] = {}

    @model_validator(mode="after")
    def _nonempty(self):
        if not self.gates:
            raise ValueError("a capability must register at least one gate")
        if not self.outputs:
            raise ValueError("a capability must declare its outputs")
        return self


# ------------------------------------------------------------------ plans

class ResourceBudget(_Strict):
    max_walltime_s: float = Field(gt=0)
    max_processes: int = Field(default=1, ge=1)
    max_core_hours: Optional[float] = Field(default=None, gt=0)


class BaseExperimentPlan(_Strict):
    """Generic typed plan. Capability-specific plans subclass this and add
    typed fields; validators there enforce the manifest validity_domain.
    The LLM fills content; it never changes the schema."""
    plan_id: str = Field(min_length=1)
    capability_id: str
    hypothesis: str = Field(min_length=10)
    engine: str
    seed: int = 0
    parameters: dict[str, Union[float, int, str]] = {}
    resource_budget: ResourceBudget
    created_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc))


# --------------------------------------------------------------- raw runs

class ArtifactRef(_Strict):
    path: str                        # relative to the run directory
    sha256: str = Field(min_length=64, max_length=64)
    kind: Literal["log", "structure", "timeseries", "table", "figure",
                  "json", "other"] = "other"


class EnvironmentLock(_Strict):
    python: str
    platform: str
    engine_version: Optional[str] = None
    requirements_sha256: Optional[str] = None


class RawRun(_Strict):
    """What actually happened when an engine executed a plan. No physics
    interpretation here — only artifacts, checksums, status, environment."""
    run_id: str
    plan_id: str
    plan_sha256: str = Field(min_length=64, max_length=64)
    capability_id: str
    engine: str
    seed: int
    started_utc: datetime
    finished_utc: datetime
    exit_status: Literal["completed", "failed", "timeout", "aborted"]
    artifacts: list[ArtifactRef] = []
    environment: EnvironmentLock
    resource_usage: dict[str, float] = {}   # e.g. {"walltime_s": ..., "core_hours": ...}

    @model_validator(mode="after")
    def _times(self):
        if self.finished_utc < self.started_utc:
            raise ValueError("finished_utc before started_utc")
        if self.exit_status == "completed" and not self.artifacts:
            raise ValueError("a completed run must register artifacts")
        return self


# ------------------------------------------------------------ observations

class Observation(_Strict):
    """One named, unit-checked quantity extracted by an analyzer, with the
    conditions under which it holds (themselves unit-checked scalars)."""
    name: str
    value: PhysicalValue
    conditions: dict[str, ScalarValue] = {}
    method: Optional[str] = None


class ObservationBundle(_Strict):
    """Analyzer output for one RawRun: the only object gates and scoring
    ever read. Engines never produce these directly."""
    bundle_id: str
    run_id: str
    capability_id: str
    analyzer: str
    analyzer_version: str
    observations: list[Observation]
    notes: list[str] = []
    created_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _nonempty(self):
        if not self.observations:
            raise ValueError("an ObservationBundle must contain observations")
        return self
