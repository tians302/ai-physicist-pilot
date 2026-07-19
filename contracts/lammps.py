"""Typed LAMMPS experiment plans (WP3). Protocol schemas add only fields
they can validate physically (PLAN.md, typed plans).

The `model_key` references the pinned-model registry
(engines.lammps_adapter.models); the ADAPTER enforces the allowlist and
pin-verification at execute time, keeping contracts dependency-clean.
"""
from typing import Literal

from pydantic import Field, model_validator

from contracts.schemas import BaseExperimentPlan


class LammpsPlanBase(BaseExperimentPlan):
    engine: Literal["lammps"] = "lammps"
    model_key: str = Field(min_length=1)          # pinned-model registry key
    a0_A: float = Field(default=5.43, gt=4.0, lt=7.0)   # initial lattice guess
    supercell: int = Field(default=1, ge=1, le=4)       # conventional cells/axis
    # minimization controls (bounded; reviewed template fills them verbatim)
    etol: float = Field(default=0.0, ge=0.0, le=1e-4)
    ftol: float = Field(default=1e-8, ge=1e-12, le=1e-4)
    maxiter: int = Field(default=1000, ge=10, le=100000)
    maxeval: int = Field(default=10000, ge=10, le=1000000)


class LammpsRelaxPlan(LammpsPlanBase):
    protocol: Literal["relax_v1"] = "relax_v1"


class LammpsEosPlan(LammpsPlanBase):
    """Volume sweep: relax, then static energies at scaled volumes."""
    protocol: Literal["eos_sweep_v1"] = "eos_sweep_v1"
    scale_min: float = Field(default=0.94, ge=0.85, le=0.999)
    scale_max: float = Field(default=1.06, ge=1.001, le=1.15)
    n_volumes: int = Field(default=11, ge=5, le=21)

    @model_validator(mode="after")
    def _span(self):
        if self.scale_min >= self.scale_max:
            raise ValueError("scale_min must be < scale_max")
        return self


class LammpsElasticPlan(LammpsPlanBase):
    """Finite strain: relax, then static stress at symmetric +/- strains
    for fixed deformation patterns (3 uniaxial + 3 shear)."""
    protocol: Literal["finite_strain_v1"] = "finite_strain_v1"
    max_strain: float = Field(default=0.005, ge=0.0005, le=0.02)
    n_strains_per_side: int = Field(default=2, ge=1, le=5)


LAMMPS_PLAN_TYPES = {
    "LammpsRelaxPlan": LammpsRelaxPlan,
    "LammpsEosPlan": LammpsEosPlan,
    "LammpsElasticPlan": LammpsElasticPlan,
}
