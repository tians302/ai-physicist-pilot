"""Canonical-units module (WP1). Dependency-free dimensional algebra.

Every quantity crossing a contract boundary carries a unit string. This
module parses unit strings, checks dimensional consistency, and converts
to canonical SI. Rejecting a bad unit at the schema boundary is a gate
against silent unit errors (a classic simulation failure mode).

Grammar (flat, no parentheses): tokens joined by '*' and '/'; each token
is NAME or NAME^INT (e.g. 'W/m/K', 'eV/Ang^3', 'm^2/s', 'GPa', '1').
Leading '1' allowed for reciprocals: '1/K'. Offset units (Celsius) are
NOT supported by design — use K.

Dimensions are 7-tuples of SI base exponents: (m, kg, s, K, A, mol, cd).
"""
from __future__ import annotations

DIM_NAMES = ("m", "kg", "s", "K", "A", "mol", "cd")
DIMLESS = (0, 0, 0, 0, 0, 0, 0)


class DimensionError(ValueError):
    pass


def _d(**kw):
    return tuple(kw.get(n, 0) for n in DIM_NAMES)


# name -> (dimension, factor to canonical SI)
_UNITS: dict[str, tuple[tuple, float]] = {
    "1": (DIMLESS, 1.0),
    # SI base
    "m": (_d(m=1), 1.0),
    "kg": (_d(kg=1), 1.0),
    "s": (_d(s=1), 1.0),
    "K": (_d(K=1), 1.0),
    "A": (_d(A=1), 1.0),
    "mol": (_d(mol=1), 1.0),
    "cd": (_d(cd=1), 1.0),
    # derived SI
    "N": (_d(kg=1, m=1, s=-2), 1.0),
    "J": (_d(kg=1, m=2, s=-2), 1.0),
    "W": (_d(kg=1, m=2, s=-3), 1.0),
    "Pa": (_d(kg=1, m=-1, s=-2), 1.0),
    "Hz": (_d(s=-1), 1.0),
    # scaled / non-SI common in materials physics
    "g": (_d(kg=1), 1e-3),
    "cm": (_d(m=1), 1e-2),
    "mm": (_d(m=1), 1e-3),
    "um": (_d(m=1), 1e-6),
    "nm": (_d(m=1), 1e-9),
    "Ang": (_d(m=1), 1e-10),
    "ps": (_d(s=1), 1e-12),
    "fs": (_d(s=1), 1e-15),
    "eV": (_d(kg=1, m=2, s=-2), 1.602176634e-19),
    "GPa": (_d(kg=1, m=-1, s=-2), 1e9),
    "bar": (_d(kg=1, m=-1, s=-2), 1e5),
    "THz": (_d(s=-1), 1e12),
    "amu": (_d(kg=1), 1.66053906892e-27),
}


def _parse_token(tok: str) -> tuple[tuple, float]:
    tok = tok.strip()
    if "^" in tok:
        name, _, p = tok.partition("^")
        try:
            power = int(p)
        except ValueError as e:
            raise DimensionError(f"non-integer power in {tok!r}") from e
    else:
        name, power = tok, 1
    name = name.strip()
    if name not in _UNITS:
        raise DimensionError(f"unknown unit {name!r}")
    dim, fac = _UNITS[name]
    return tuple(x * power for x in dim), fac ** power


def parse_unit(unit: str) -> tuple[tuple, float]:
    """Return (dimension 7-tuple, factor to canonical SI) for a unit string."""
    if not isinstance(unit, str) or not unit.strip():
        raise DimensionError(f"empty or non-string unit: {unit!r}")
    dim = list(DIMLESS)
    factor = 1.0
    sign = +1
    # split keeping '/' boundaries: a/b*c means a / b * c evaluated left-to-right,
    # with '/' flipping the sign of the NEXT token only (flat grammar, documented).
    token = ""
    ops = []
    for ch in unit:
        if ch in "*/":
            ops.append((token, sign))
            sign = -1 if ch == "/" else +1
            token = ""
        else:
            token += ch
    ops.append((token, sign))
    for tok, sg in ops:
        d, f = _parse_token(tok)
        dim = [a + sg * b for a, b in zip(dim, d)]
        factor *= f ** sg
    return tuple(dim), factor


def dim_of(unit: str) -> tuple:
    return parse_unit(unit)[0]


def same_dimension(u1: str, u2: str) -> bool:
    return dim_of(u1) == dim_of(u2)


def to_canonical(value: float, unit: str) -> tuple[float, str]:
    """Convert value to canonical SI; return (si_value, canonical_unit_str)."""
    dim, fac = parse_unit(unit)
    return value * fac, canonical_name(dim)


def canonical_name(dim: tuple) -> str:
    if dim == DIMLESS:
        return "1"
    num = [f"{n}^{p}" if p != 1 else n for n, p in zip(DIM_NAMES, dim) if p > 0]
    den = [f"{n}^{-p}" if p != -1 else n for n, p in zip(DIM_NAMES, dim) if p < 0]
    out = "*".join(num) if num else "1"
    for d in den:
        out += f"/{d}"
    return out


def convert(value: float, from_unit: str, to_unit: str) -> float:
    d1, f1 = parse_unit(from_unit)
    d2, f2 = parse_unit(to_unit)
    if d1 != d2:
        raise DimensionError(
            f"dimension mismatch: {from_unit!r} {d1} vs {to_unit!r} {d2}")
    return value * f1 / f2
