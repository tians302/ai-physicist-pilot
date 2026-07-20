"""Fixed, reviewed LAMMPS input templates (WP3). No LLM ever writes or
edits engine input; plans supply only numeric/keyword values for the
allowlisted placeholders below, and trusted multi-line blocks come solely
from the pinned-model registry (models.pair_blocks).

Sentinel convention: result lines are printed as
    AIPHYS_RESULT <key> <value>
and parsed by the adapter into raw outputs. Everything else is log noise.
"""
import re

# -------- reviewed templates (placeholders in {braces}) -------------------

RELAX_V1 = """\
# ai-physicist fixed template relax_v1 (reviewed 2026-07-19; do not edit per-run)
{init_block}
boundary p p p
atom_style atomic
read_data {data_file}
{interaction_block}
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
thermo 25
thermo_style custom step pe press vol lx ly lz
fix boxrelax all box/relax iso 0.0 vmax 0.001
min_style cg
minimize {etol} {ftol} {maxiter} {maxeval}
unfix boxrelax
minimize {etol} {ftol} {maxiter} {maxeval}
variable n equal atoms
variable pe equal pe
variable v equal vol
variable lxv equal lx
variable pxxv equal pxx
print "AIPHYS_RESULT natoms ${{n}}"
print "AIPHYS_RESULT pe_eV ${{pe}}"
print "AIPHYS_RESULT vol_A3 ${{v}}"
print "AIPHYS_RESULT lx_A ${{lxv}}"
print "AIPHYS_RESULT pxx_bar ${{pxxv}}"
write_data {out_data}
"""

STATIC_V1 = """\
# ai-physicist fixed template static_v1 (reviewed 2026-07-19; do not edit per-run)
{init_block}
boundary p p p
atom_style atomic
read_data {data_file}
{interaction_block}
neighbor 2.0 bin
thermo_style custom step pe press pxx pyy pzz pxy pxz pyz vol
run 0
variable n equal atoms
variable pe equal pe
variable v equal vol
variable sxx equal pxx
variable syy equal pyy
variable szz equal pzz
variable sxy equal pxy
variable sxz equal pxz
variable syz equal pyz
print "AIPHYS_RESULT natoms ${{n}}"
print "AIPHYS_RESULT pe_eV ${{pe}}"
print "AIPHYS_RESULT vol_A3 ${{v}}"
print "AIPHYS_RESULT pxx_bar ${{sxx}}"
print "AIPHYS_RESULT pyy_bar ${{syy}}"
print "AIPHYS_RESULT pzz_bar ${{szz}}"
print "AIPHYS_RESULT pxy_bar ${{sxy}}"
print "AIPHYS_RESULT pxz_bar ${{sxz}}"
print "AIPHYS_RESULT pyz_bar ${{syz}}"
"""

STATIC_RELAXED_IONS_V1 = """\
# ai-physicist fixed template static_relaxed_ions_v1 (reviewed 2026-07-19)
# Internal (ionic) relaxation at FIXED cell, then stress readout. Required
# for shear response of diamond-structure crystals (internal DOF).
{init_block}
boundary p p p
atom_style atomic
read_data {data_file}
{interaction_block}
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
thermo_style custom step pe press pxx pyy pzz pxy pxz pyz vol
min_style cg
minimize {etol} {ftol} {maxiter} {maxeval}
run 0
variable n equal atoms
variable pe equal pe
variable v equal vol
variable sxx equal pxx
variable syy equal pyy
variable szz equal pzz
variable sxy equal pxy
variable sxz equal pxz
variable syz equal pyz
print "AIPHYS_RESULT natoms ${{n}}"
print "AIPHYS_RESULT pe_eV ${{pe}}"
print "AIPHYS_RESULT vol_A3 ${{v}}"
print "AIPHYS_RESULT pxx_bar ${{sxx}}"
print "AIPHYS_RESULT pyy_bar ${{syy}}"
print "AIPHYS_RESULT pzz_bar ${{szz}}"
print "AIPHYS_RESULT pxy_bar ${{sxy}}"
print "AIPHYS_RESULT pxz_bar ${{sxz}}"
print "AIPHYS_RESULT pyz_bar ${{syz}}"
"""

NPT_LATTICE_V1 = """\
# ai-physicist fixed template npt_lattice_v1 (reviewed 2026-07-19)
# NPT MD at one temperature; running time-average of the box length,
# printed at half and full production for an equilibration-drift check.
{init_block}
boundary p p p
atom_style atomic
read_data {data_file}
{interaction_block}
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes
timestep 0.001
velocity all create {T_K} {vseed} mom yes rot yes
fix npt all npt temp {T_K} {T_K} 0.1 iso 0.0 0.0 1.0
thermo 500
thermo_style custom step temp press vol lx pe
run {nsteps_equil}
variable lxv equal lx
fix avg all ave/time 10 100 1000 v_lxv ave running
run {nsteps_prod}
variable lxh equal f_avg
print "AIPHYS_RESULT lx_avg_half_A ${{lxh}}"
run {nsteps_prod}
variable lxf equal f_avg
variable n equal atoms
print "AIPHYS_RESULT natoms ${{n}}"
print "AIPHYS_RESULT lx_avg_A ${{lxf}}"
"""

LJ_MSD_V1 = """\
# ai-physicist fixed template lj_msd_v1 (reviewed 2026-07-19)
# LJ liquid; NVT MSD time series via checkpoint loop -> diffusivity.
{init_block}
boundary p p p
atom_style atomic
lattice fcc {rho_star}
region box block 0 {n_cells} 0 {n_cells} 0 {n_cells}
create_box 1 box
create_atoms 1 box
mass 1 1.0
{interaction_block}
neighbor 0.3 bin
neigh_modify delay 0 every 1 check yes
timestep 0.005
velocity all create {T_star} {vseed} mom yes rot yes
fix nvt all nvt temp {T_star} {T_star} 0.5
thermo 1000
run {nsteps_equil}
reset_timestep 0
compute msd all msd com yes
thermo_style custom step temp c_msd[4]
variable m equal c_msd[4]
variable i loop {n_checkpoints}
label lp
run {steps_per_checkpoint} post no
print "AIPHYS_RESULT msd_${{i}} ${{m}}"
next i
jump SELF lp
variable n equal atoms
print "AIPHYS_RESULT natoms ${{n}}"
print "AIPHYS_RESULT n_checkpoints {n_checkpoints}"
print "AIPHYS_RESULT steps_per_checkpoint {steps_per_checkpoint}"
print "AIPHYS_RESULT dt_lj 0.005"
"""

LJ_GK_V1 = """\
# ai-physicist fixed template lj_gk_v1 (reviewed 2026-07-19)
# Green-Kubo thermal conductivity for the LJ liquid (canonical LAMMPS
# heat/flux + ave/correlate + trap() construction; kB = 1 in LJ units).
{init_block}
boundary p p p
atom_style atomic
lattice fcc {rho_star}
region box block 0 {n_cells} 0 {n_cells} 0 {n_cells}
create_box 1 box
create_atoms 1 box
mass 1 1.0
{interaction_block}
neighbor 0.3 bin
neigh_modify delay 0 every 1 check yes
timestep 0.005
velocity all create {T_star} {vseed} mom yes rot yes
fix nvt all nvt temp {T_star} {T_star} 0.5
thermo 1000
run {nsteps_equil}
reset_timestep 0
compute myKE all ke/atom
compute myPE all pe/atom
compute myStress all stress/atom NULL virial
compute flux all heat/flux myKE myPE myStress
variable Jx equal c_flux[1]/vol
variable Jy equal c_flux[2]/vol
variable Jz equal c_flux[3]/vol
fix JJ all ave/correlate {nevery} {nrepeat} {nfreq} c_flux[1] c_flux[2] c_flux[3] type auto file J0Jt.dat ave running
variable scale equal {nevery}*0.005/{T_star}/{T_star}/vol
variable k11 equal trap(f_JJ[3])*v_scale
variable k22 equal trap(f_JJ[4])*v_scale
variable k33 equal trap(f_JJ[5])*v_scale
run {nsteps_prod}
variable n equal atoms
print "AIPHYS_RESULT natoms ${{n}}"
print "AIPHYS_RESULT k11_lj ${{k11}}"
print "AIPHYS_RESULT k22_lj ${{k22}}"
print "AIPHYS_RESULT k33_lj ${{k33}}"
"""

TEMPLATES = {"relax_v1": RELAX_V1, "static_v1": STATIC_V1,
             "static_relaxed_ions_v1": STATIC_RELAXED_IONS_V1,
             "npt_lattice_v1": NPT_LATTICE_V1,
             "lj_msd_v1": LJ_MSD_V1, "lj_gk_v1": LJ_GK_V1}

# -------- strict renderer -------------------------------------------------

_PLACEHOLDER = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})")
_TRUSTED_KEYS = {"init_block", "interaction_block"}   # multi-line, registry-built
_SAFE_VALUE = re.compile(r"^[A-Za-z0-9_.\-/ ]+$")     # no newlines/quotes/$;&


class TemplateError(ValueError):
    pass


def render(template_name: str, values: dict, trusted_blocks: dict) -> str:
    """Fill a fixed template. Fail closed on: unknown template, missing or
    extra keys, unsafe characters in plan-supplied values."""
    if template_name not in TEMPLATES:
        raise TemplateError(f"unknown template {template_name!r}")
    tmpl = TEMPLATES[template_name]
    needed = set(_PLACEHOLDER.findall(tmpl))

    if set(trusted_blocks) - _TRUSTED_KEYS:
        raise TemplateError("unexpected trusted block keys")
    overlap = set(values) & _TRUSTED_KEYS
    if overlap:
        raise TemplateError(f"plan values may not set trusted keys {sorted(overlap)}")

    supplied = {**values, **trusted_blocks}
    missing = needed - set(supplied)
    extra = set(supplied) - needed
    if missing:
        raise TemplateError(f"missing placeholders: {sorted(missing)}")
    if extra:
        raise TemplateError(f"extra keys not in template: {sorted(extra)}")

    for k, v in values.items():
        s = str(v)
        if not _SAFE_VALUE.match(s):
            raise TemplateError(f"unsafe value for {k!r}: {s!r}")

    out = tmpl
    for k, v in supplied.items():
        out = out.replace("{%s}" % k, str(v))
    return out.replace("{{", "{").replace("}}", "}")


def parse_sentinels(log_text: str) -> dict:
    """Extract AIPHYS_RESULT key/value pairs from a LAMMPS log."""
    out = {}
    for m in re.finditer(r"^AIPHYS_RESULT\s+(\S+)\s+(\S+)\s*$", log_text, re.M):
        key, val = m.group(1), m.group(2)
        try:
            out[key] = float(val)
        except ValueError:
            out[key] = val
    return out
