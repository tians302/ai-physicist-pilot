"""Regression fixtures for the pinned SW-Si model (WP3 smoke test).

RULE (PLAN.md): published same-model values are REGRESSION FIXTURES ONLY —
they verify our build/templates reproduce the potential as published; they
are never a physics answer key. Physics keys come from independent
experimental data (curated at WP4 with DOI + uncertainty).

Values: the SW potential (Stillinger & Weber 1985) was parameterized to
give diamond-Si a0 = 5.431 A and cohesive energy 4.3364 eV/atom (see the
LAMMPS/OpenKIM SW model documentation). Tolerances are tight because this
is a determinism check, not a physics claim.
"""

SW_SI_REGRESSION = {
    "a0_A": {"value": 5.431, "tol": 0.01},
    "ecoh_eV_per_atom": {"value": -4.3364, "tol": 0.01},
    "residual_pressure_bar": {"tol": 50.0},   # after box/relax at 0 K
}
