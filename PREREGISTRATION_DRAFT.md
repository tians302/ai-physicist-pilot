# Pre-registration (DRAFT — NOT FROZEN)

**Status: DRAFT.** The freeze (WP7) happens only when: (1) T signs off,
(2) all quantitative answer keys used for scoring are `verified: true`,
(3) this file is committed with the word DRAFT removed and the commit
hash recorded below. Any change after the freeze requires a logged
justification in PLAN.md's decision log. No headline (WP10) run may
start before the freeze.

## 1. Benchmarks and answer keys

| Benchmark | Capability | Answer key (scored) | Key status |
|---|---|---|---|
| silicon_boundary | si_kappa_callaway | qualitative: suppression at all T; relative suppression larger at low T (Spearman ≤ −0.9) | registered (structural) |
| si_eos | si_eos_sw_lammps | qualitative: interior minimum; B0 > 0 | registered (structural) |
| si_elastic | si_elastic_sw_lammps | Born stability + cubic symmetry (verified qualitative entry) | verified |
| si_expansion (WP8) | si_expansion_sw_lammps | qualitative: a(T) increasing over 300–900 K (α > 0) | registered (structural) |
| lj_diffusion (WP8) | lj_diffusion_lammps | qualitative: MSD linear regime exists; D > 0 | registered (structural) |
| lj_kappa_gk (WP8) | lj_kappa_gk_lammps | qualitative: HCACF integral converges to positive plateau | registered (structural) |

Quantitative experimental comparisons (Si a0/B0/Cij from McSkimin &
Andreatch 1964 / Hall 1967; LJ literature values) are reported INFO-ONLY
until the corresponding key entries are `verified: true`; if verified
before the freeze, the quantitative check is |value − key| within the
key's stated uncertainty × 3, reported alongside the qualitative check.

## 2. Gate thresholds (frozen as of this commit)

Exact values live in code and are version-tagged: `gates/__init__.py`
v1.0 (Callaway: χ²/dof ≤ 4, |lag-1 ac| ≤ 0.5, CV ≤ 0.5, n_ok ≥ 50,
holdout χ²/dof ≤ 6), `gates/eos.py` v0.1 (coverage ≤0.98/≥1.02, interior
margin ≥ 0.10, resid/span ≤ 5e-3, form ΔB0 ≤ 0.15, B0 ∈ [1, 1000] GPa),
`gates/elastic.py` v0.1 (linearity ≤ 0.02, window ≤ 0.10, ref-stress ≤
0.05 GPa, symmetry ≤ 0.05/off-block ≤ 0.08, Born + eigenvalue stability,
C ∈ [1, 2000] GPa). WP8 gate configs join this list before freeze.
The freeze pins the git tree hash of `gates/`.

## 3. Corruption set

The named corruptions in `corruptions/__init__.py` (17 as of draft),
with the intended-gate map used for the WP9 selectivity matrix. New
corruptions may be added before freeze only.

## 4. Primary metrics (computed only by `scripts/run_battery.py`)

1. **Recovery rate**: fraction of clean gates-on runs with
   `status=completed`, all gates passed, and benchmark success, per
   benchmark per condition.
2. **Unflagged-false-claim rate (HEADLINE)**: fraction of corrupted runs
   whose conclusion is `valid=True` with benchmark success — i.e. a wrong
   result presented as a valid claim. Compared gates-on vs gates-off
   (identical seeds/corruptions; the only difference is enforcement).
3. **Unsupported-hypothesis rejection precision/recall** on the WP2
   planted fixture battery (13 fixtures as of draft; may grow to ≥20
   before freeze).
4. **Gate sensitivity/selectivity matrix**: per (corruption, gate):
   intended-gate detection rate; collateral-failure rate.

## 5. Design

- **N = 30 seeds** per benchmark per arm (clean gates-on; corrupted
  gates-on; corrupted gates-off; clean gates-off as control). Seeds
  0–29, fixed in advance.
- Corrupted arms cycle deterministically through the corruption list
  (seed i gets corruption i mod K).
- **CI method: Wilson score interval, 95%**, computed by the harness
  (`scripts/stats.py`); differences reported with Newcombe hybrid CIs.
- **Exclusion rules**: no exclusions. Engine failures/timeouts count
  against recovery (they are measured outcomes). If infrastructure
  failure (e.g. Slurm kill) is documented, the seed is re-run with the
  same seed and the event logged; the original artifact is retained.
- **Compute**: laptop/CARC; per-run resource budgets in the plans;
  battery totals recorded in the cost ledger.

## 6. What is NOT claimed

Gates measure evidentiary validity, not truth: they cannot catch silent
protocol inadequacy (documented example: affine C44 = 109.8 GPa passes
all gates; the physical relaxed-ion value is 56.45) or physical-model
bias (SW-Si Cij differ from experiment by construction). All claims are
model-conditional. The 10 µm boundary-length suppression magnitudes in
silicon_boundary remain qualitative-only.

## 7. Freeze record

- Frozen by: ________ (T)
- Date: ________
- Commit hash: ________
- Answer keys verified: [ ] a0 [ ] C11 [ ] C12 [ ] C44 [ ] B0 [ ] WP8 refs
