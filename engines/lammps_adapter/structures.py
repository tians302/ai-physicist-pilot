"""ASE structure builders with lineage (WP3).

PLAN.md safeguard: generated/transformed structures retain parent hashes
and a complete preparation trace. Every builder returns (atoms, lineage)
where lineage records the operation, parameters, and parent hash.
"""
import hashlib
import io

import numpy as np
from ase.build import bulk
from ase.io.lammpsdata import write_lammps_data


def atoms_sha256(atoms) -> str:
    """Deterministic hash of cell + species + scaled positions."""
    h = hashlib.sha256()
    h.update(np.round(atoms.get_cell().array, 10).tobytes())
    h.update(np.array(atoms.get_atomic_numbers()).tobytes())
    h.update(np.round(atoms.get_scaled_positions(wrap=True), 10).tobytes())
    return h.hexdigest()


def diamond_si(a0_A: float, supercell: int = 1):
    atoms = bulk("Si", "diamond", a=a0_A, cubic=True) * ((supercell,) * 3)
    lineage = {"op": "build_diamond_si", "a0_A": a0_A, "supercell": supercell,
               "natoms": len(atoms), "parent_sha256": None,
               "sha256": atoms_sha256(atoms)}
    return atoms, lineage


def scale_volume(atoms, scale: float, parent_lineage: dict):
    """Isotropic volume scaling: V -> scale * V (linear factor scale^(1/3))."""
    out = atoms.copy()
    lin = scale ** (1.0 / 3.0)
    out.set_cell(out.get_cell() * lin, scale_atoms=True)
    lineage = {"op": "scale_volume", "volume_scale": scale,
               "parent_sha256": parent_lineage["sha256"],
               "sha256": atoms_sha256(out)}
    return out, lineage


def apply_deformation(atoms, F: np.ndarray, parent_lineage: dict):
    """Apply deformation gradient F: cell rows r -> r @ F.T (atoms scaled)."""
    out = atoms.copy()
    out.set_cell(out.get_cell().array @ np.asarray(F).T, scale_atoms=True)
    lineage = {"op": "apply_deformation", "F": np.asarray(F).tolist(),
               "parent_sha256": parent_lineage["sha256"],
               "sha256": atoms_sha256(out)}
    return out, lineage


# fixed deformation patterns for finite_strain_v1 (Voigt order 1..6)
def strain_matrix(voigt_index: int, amplitude: float) -> np.ndarray:
    """Engineering-strain deformation gradient F = I + eps for one Voigt
    component. Shear amplitude is the ENGINEERING shear gamma (eps_ij =
    gamma/2); the convention is recorded in raw outputs for the analyzer."""
    F = np.eye(3)
    if voigt_index in (1, 2, 3):
        F[voigt_index - 1, voigt_index - 1] += amplitude
    elif voigt_index == 4:      # yz
        F[1, 2] += amplitude / 2.0
        F[2, 1] += amplitude / 2.0
    elif voigt_index == 5:      # xz
        F[0, 2] += amplitude / 2.0
        F[2, 0] += amplitude / 2.0
    elif voigt_index == 6:      # xy
        F[0, 1] += amplitude / 2.0
        F[1, 0] += amplitude / 2.0
    else:
        raise ValueError(f"voigt_index must be 1..6, got {voigt_index}")
    return F


def to_lammps_data(atoms) -> str:
    buf = io.StringIO()
    write_lammps_data(buf, atoms, masses=True, units="metal",
                      atom_style="atomic")
    return buf.getvalue()
