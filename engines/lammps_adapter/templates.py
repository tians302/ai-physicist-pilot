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

TEMPLATES = {"relax_v1": RELAX_V1, "static_v1": STATIC_V1}

# -------- strict renderer -------------------------------------------------

_PLACEHOLDER = re.compile(r"(?<!\{)\{([a-z_][a-z0-9_]*)\}(?!\})")
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
