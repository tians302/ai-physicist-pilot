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
    # diamond-structure shear response requires internal-ion relaxation at
    # fixed cell; False gives the affine (unrelaxed) tensor — comparison only
    relax_ions: bool = True


class SiExpansionPlan(LammpsPlanBase):
    """NPT lattice parameter vs temperature -> thermal expansion (WP8)."""
    protocol: Literal["npt_lattice_v1"] = "npt_lattice_v1"
    supercell: int = Field(default=2, ge=2, le=6)
    T_list_K: list[float] = Field(default=[300.0, 500.0, 700.0, 900.0],
                                  min_length=3, max_length=8)
    nsteps_equil: int = Field(default=4000, ge=1000, le=500000)
    nsteps_prod: int = Field(default=5000, ge=2000, le=2000000)

    @model_validator(mode="after")
    def _temps(self):
        if any(not (50.0 <= t <= 1400.0) for t in self.T_list_K):
            raise ValueError("temperatures must lie in [50, 1400] K")
        if sorted(self.T_list_K) != self.T_list_K:
            raise ValueError("T_list_K must be sorted ascending")
        # ave/time Nfreq=1000 requires production in multiples of 1000
        if self.nsteps_prod % 1000:
            raise ValueError("nsteps_prod must be a multiple of 1000")
        return self


class LjPlanBase(BaseExperimentPlan):
    """Lennard-Jones reduced-units protocols (WP8 transport ladder)."""
    engine: Literal["lammps"] = "lammps"
    model_key: str = "lj_argon"
    rho_star: float = Field(default=0.8442, ge=0.05, le=1.2)
    T_star: float = Field(default=0.722, ge=0.3, le=3.0)
    n_cells: int = Field(default=4, ge=3, le=10)   # fcc: 4*n^3 atoms
    nsteps_equil: int = Field(default=4000, ge=1000, le=500000)


class LjDiffusionPlan(LjPlanBase):
    protocol: Literal["lj_msd_v1"] = "lj_msd_v1"
    n_checkpoints: int = Field(default=20, ge=8, le=100)
    steps_per_checkpoint: int = Field(default=500, ge=100, le=20000)


class LjKappaGkPlan(LjPlanBase):
    protocol: Literal["lj_gk_v1"] = "lj_gk_v1"
    nsteps_prod: int = Field(default=50000, ge=10000, le=10000000)
    nevery: int = Field(default=10, ge=1, le=100)
    nrepeat: int = Field(default=500, ge=50, le=10000)

    @model_validator(mode="after")
    def _corr(self):
        # ave/correlate requires Nfreq = Nevery * Nrepeat dividing run length
        if self.nsteps_prod % (self.nevery * self.nrepeat):
            raise ValueError("nsteps_prod must be a multiple of nevery*nrepeat")
        return self


LAMMPS_PLAN_TYPES = {
    "LammpsRelaxPlan": LammpsRelaxPlan,
    "LammpsEosPlan": LammpsEosPlan,
    "LammpsElasticPlan": LammpsElasticPlan,
    "SiExpansionPlan": SiExpansionPlan,
    "LjDiffusionPlan": LjDiffusionPlan,
    "LjKappaGkPlan": LjKappaGkPlan,
}
