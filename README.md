# ai-physicist

**A capability-constrained, gate-checked closed-loop AI scientist for simulation-driven materials physics.**

LLM-driven simulation agents usually *self-judge* their results. This framework inverts that: an untrusted proposer (the LLM) may only select from a **registry of typed, versioned capabilities**; plans are **schema-validated before execution**; engines run only **fixed, reviewed input templates** (never generated code); fixed analyzers turn raw runs into unit-checked observations; and **hard validity gates run before any narration**. Reliability is not asserted — it is *measured*: rejection rates for unsupported hypotheses, detection rates for planted corruptions, and false-claim rates with and without gates.

Builds on ideas from [SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist) (full-loop autonomy) and [A3HT](https://github.com/tians302/A3HT) (high-throughput MD execution), adding the discipline layer: registry + typed plans + gates + measured reliability.

## Measured results (sandbox scale, 2026-07)

| Metric | Result |
|---|---|
| Benchmark battery | 6 capabilities, 4 property families, 2 engines — all complete, all gates green |
| SW-Si physics reproduced | a0 = 5.4309 Å, Ecoh = −4.3366 eV/atom, B0 = 101.5 GPa, C11/C12/C44 = 151.4/76.4/56.5 GPa (relaxed-ion) |
| Cross-protocol consistency | B0 from EOS vs (C11+2C12)/3: 0.06 % |
| Transport (LJ, reduced units) | D* = 0.028, κ*(Green–Kubo) = 7.0 at ρ*=0.8442, T*=0.722 |
| Unsupported-hypothesis rejection | 8/8 planted rejected before planning; precision = recall = 1.0 |
| Gate matrix | 26/26 sensitivity (intended gate fires), 26/26 selectivity (zero collateral) |
| **Gate ablation (pilot, N=6)** | unflagged false claims **0/6 gates-ON vs 5/6 gates-OFF** (Wilson 95 % CIs) |
| Cross-model sensitivity | SW-1985 vs Tersoff-1988: a0 0 %, B0 3.8 %, **C44 138 %** — both pass all gates (gates bound evidence, not model adequacy) |

Pre-registered N=30 statistics on HPC are pending the freeze of [`PREREGISTRATION_DRAFT.md`](PREREGISTRATION_DRAFT.md). Paper skeleton: [`PAPER1_DRAFT.md`](PAPER1_DRAFT.md).

## Architecture

**Trust boundary.**
- *Untrusted proposer*: LLM hypothesis text, routing suggestions, plan content within schemas, post-gate narration.
- *Trusted control plane*: capability registry, schemas, validators, state machine, engine allowlist, analyzers, gates, answer keys, note templates.
- *Fallible instruments*: engines and physical models — successful execution is **not** evidence of validity.

**Fixed 12-stage loop** (`core/loop2.py`; the agent fills content, never order):

```
load-capabilities → hypothesize → route → plan → validate-plan → execute
→ analyze → gate-check → score → interpret → iterate-or-conclude → write-note
```

Key rules: unsupported hypotheses end the run as `unsupported_hypothesis` (a measured behavior, not an error); engines produce checksummed `RawRun` evidence; gates run on analyzer diagnostics *before* the LLM sees anything; scores come from the harness only; a gate failure voids the conclusion regardless of any narrative.

**Concept separation** — hypothesis ≠ property ≠ protocol ≠ engine ≠ physical model ≠ analyzer ≠ benchmark. Each is a separate, versioned, named thing; capabilities bind one combination together with its gates and validity domain.

## Repository layout

```
contracts/          WP1 core contracts (pydantic v2, extra="forbid")
  units.py            canonical-units module: dimensional algebra, unit-slip rejection
  schemas.py          CapabilityManifest, BaseExperimentPlan, RawRun, ObservationBundle,
                      Scalar/Curve/Tensor values, dimensional Bounds
  lammps.py           typed LAMMPS plans (EOS, elastic, NPT expansion, LJ MSD, LJ GK)
registry/           WP2 closed capability catalog + deterministic fail-closed router
  builtin.py          the 6 registered capabilities (manifests with model cards)
  router.py           route(request, registry): pure function, unit-converted
                      condition checks, per-capability rejection reasons
engines/
  callaway.py         Callaway-RTA κ(T) engine (Phase-1A physics, unchanged)
  callaway_adapter.py legacy engine behind the generic execute→RawRun interface
  lammps_adapter/     subprocess-isolated LAMMPS adapter: fixed templates,
                      pinned potentials (checksummed), structure lineage,
                      budget timeouts, fail-closed everything
analyzers/          fixed, versioned analysis: BM3 EOS fit, stress-strain Cij,
                    NPT expansion, MSD diffusivity, Green-Kubo κ, Callaway repackager
gates/              validity gates per capability kind (+ Phase-1A common gates);
                    thresholds documented a priori, frozen at pre-registration
corruptions/        26 named corruptions + intended-gate map + clean-diag factories
benchmarks/         registered hypotheses, answer-key policy (verified-only scoring),
                    harness scorers (v2.py); answer_keys.py refuses unverified keys
core/               loop2.py (12-stage loop), note2.py (generic renderer),
                    llm.py (provider-agnostic: OpenAI → Anthropic → scripted),
                    loop.py/note.py (Phase-1A loop, kept for parity regression)
scripts/            run_battery.py (multi-seed ablation, Wilson CIs, --arm chunking)
                    gate_matrix.py · cross_model.py · selection_baseline.py
                    make_figures.py · validate_lammps.py · stats.py
data/               si_kappa_reference.csv; reference/ answer keys (provenance-first)
runs/m1/            committed M1 demo runs incl. the negative suite
reports/            gate matrix, battery metrics, cross-model, figures
docs/INSTALL_LAMMPS.md   pin-disciplined install guide (laptop + cluster)
PREREGISTRATION_DRAFT.md · PAPER1_DRAFT.md
tests/              64 tests; CI-safe (fake engine binary) + real-engine (auto-skip)
```

## Installation

```bash
git clone https://github.com/tians302/ai-physicist-pilot.git
cd ai-physicist
python -m venv .venv && source .venv/bin/activate   # or conda env create -f environment.yml
pip install -r requirements.txt                     # pinned versions
python -m pytest tests/ -q                          # everything except real-engine tests
```

**LAMMPS (optional but recommended).** Needed for the EOS/elastic/expansion/LJ capabilities; the Callaway benchmark and all CI-safe tests run without it. Follow [`docs/INSTALL_LAMMPS.md`](docs/INSTALL_LAMMPS.md) (conda-forge on a laptop, module/cmake on a cluster — one pinned release everywhere), then validate:

```bash
python scripts/validate_lammps.py --binary lmp --tag "your build note"
# writes environment/lammps_manifest_<host>.json — commit it
```

The validator checks the binary/packages, runs the SW-Si regression smoke (a0, Ecoh), EOS convexity, and the fail-closed negative tests. Potentials ship in-repo, sha256-pinned (SW 1985; Tersoff 1988 from the official LAMMPS 2025-07-22 release), so no external potential downloads are needed. Point the framework at your binary with `AIPHYS_LMP_BIN=/path/to/lmp` if it is not `lmp` on PATH.

## Quickstart

Run a benchmark through the full loop (each run directory gets `request.json`, `route_decision.json`, `plan.json`, `rawrun.json` with checksummed artifacts, `bundle.json`, `gate_report.json`, `score_report.json`, `note.md`, `loop_trace.json`):

```bash
# no LAMMPS needed:
python run_loop2.py --benchmark silicon_boundary --run-dir runs/demo_kappa

# with LAMMPS:
python run_loop2.py --benchmark si_eos       --run-dir runs/demo_eos
python run_loop2.py --benchmark si_elastic   --run-dir runs/demo_cij
python run_loop2.py --benchmark si_expansion --run-dir runs/demo_alpha
python run_loop2.py --benchmark lj_diffusion --run-dir runs/demo_msd
python run_loop2.py --benchmark lj_kappa_gk  --run-dir runs/demo_gk
```

Harness experiments (the mechanisms behind the headline metric):

```bash
# inject a named corruption -> gates fail -> conclusion voided
python run_loop2.py --benchmark silicon_boundary --run-dir runs/x1 --corruption bad_fit

# same corruption with gates NOT enforced -> an unflagged false claim
python run_loop2.py --benchmark silicon_boundary --run-dir runs/x2 --corruption bad_fit --no-gates
```

Read `runs/*/note.md` — gate failures and ablation mode are banner-flagged; every number links back to raw, checksummed artifacts.

## Statistics and reports

```bash
# multi-seed ablation battery (4 arms: clean/corrupted x gates-on/off)
python scripts/run_battery.py --benchmarks silicon_boundary --seeds 30 --out reports/battery/full
#   chunked execution (e.g., Slurm array): --arm clean_on | clean_off | corr_on | corr_off,
#   then rerun without --arm to aggregate cached arms

python scripts/gate_matrix.py           # 26-pair sensitivity/selectivity matrix
python scripts/cross_model.py           # SW vs Tersoff EOS+elastic spread + cost ledger
python scripts/selection_baseline.py    # experiment-selection arms (scripted/grid/random; llm with API key)
python scripts/make_figures.py          # regenerates all paper figures from artifacts
```

Primary metrics (recovery rate, unflagged-false-claim rate, rejection precision/recall, gate matrix) and their exact definitions, N, and CI method are fixed in [`PREREGISTRATION_DRAFT.md`](PREREGISTRATION_DRAFT.md). Headline runs may only happen after that document is frozen (human sign-off + verified answer keys).

## Gates (summary)

| Kind | Gates |
|---|---|
| Callaway (v1.0) | metrics exist, finite values, χ²/dof ≤ 4, \|lag-1 autocorr\| ≤ 0.5, mandatory σ, bootstrap CV ≤ 0.5 (n_ok ≥ 50), block-holdout χ²/dof ≤ 6 |
| EOS (v0.1) | points, finite, volume coverage, interior minimum, residuals/span ≤ 5e-3, BM3-vs-Murnaghan ΔB0 ≤ 15 %, magnitude band |
| Elastic (v0.1) | data, finite, linearity ≤ 2 %, half-window stability ≤ 10 %, reference-state stress ≤ 0.05 GPa, tensor symmetry, Born + eigenvalue stability, magnitude band |
| Expansion (v0.1) | temps, finite, equilibration drift ≤ 0.5 %, fit residuals, magnitude band |
| Diffusion (v0.1) | checkpoints, finite, diffusive regime (MSD ≥ σ²), fit residuals ≤ 5 %, magnitude band |
| Green–Kubo (v0.1) | finite, positive components, component consistency ≤ 50 % (convergence proxy), magnitude band |

Magnitude bands are order-of-magnitude **unit-slip catchers** (eV/Å³↔GPa is ×160, bar↔GPa is ×10⁴), never answer keys. Two measured limits of gating, kept deliberately in the record: the affine-shear C44 (109.8 GPa — passes every gate, wrong protocol for diamond structures; the default protocol relaxes ions) and the Tersoff-1988 C44 (10.3 GPa — passes every gate, 138 % from SW; only cross-model comparison flags it). Gates bound *evidentiary validity*; protocol review and cross-model spread bound *physical adequacy*. All claims are model-conditional.

## Provenance rules

- Reference values enter `data/reference/` only with DOI, stated uncertainty, temperature/conditions, and transcription method; `verified:false` entries are **refused by the scorer**. No numbers from memory or LLM output.
- Published same-model values (e.g., SW-Si a0/Ecoh/Cij) are **regression fixtures only**, never physics answer keys.
- Every run records: plan sha256, engine version, pinned potential checksums, structure lineage (parent hashes for every transformation), seeds, environment lock, and per-artifact sha256.
- Pins change only deliberately: potentials are checksummed in `engines/lammps_adapter/models.py`; unverified KIM entries are refused at execute time; dependency pins in `requirements.txt`.

## Testing

```bash
python -m pytest tests/ -q                    # 64 tests
AIPHYS_LMP_BIN=/path/to/lmp python -m pytest tests/ -q   # + real-engine tests (auto-skip otherwise)
```

CI (GitHub Actions, py3.10/3.12) runs the CI-safe set: contracts round-trips and dimensional checks, router fixture battery (100 % rejection), template-injection guards, fake-binary fail-closed suite (nonzero exit / timeout / missing sentinels / unverified model), corruption→intended-gate acceptance for all six capability kinds, and v1↔v2 Callaway parity.

## LLM modes

Scripted mode (default, no key) keeps every run deterministic. With `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` (order: OpenAI → Anthropic; override with `AIPHYS_LLM_PROVIDER`), the LLM contributes hypothesis text and post-gate narration only — it never sees ungated results, never picks stage order, never writes engine input. In the selection baseline, LLM choices are menu-constrained and validator-checked; off-menu replies fall back to the grid arm.

## Status and roadmap

Done: Phases 1A–1B and the Phase-2 scaffolding (WP0–WP13 of `PHASE1B2_EXECUTION_PLAN.md`, sandbox scale). Pending, in order: freeze pre-registration (sign-off + verified answer keys) → N=30 HPC battery → LLM-mode arms with a real key → final Paper 1. Phase 3 applies the framework to glassy-carbon thermal transport via A3HT (see the project plan). Lab record: `LAB_NOTES.md` in the parent project folder.

## Citation

Paper in preparation (see `PAPER1_DRAFT.md`). Until then, cite this repository:

```
T. Sang, "ai-physicist: a capability-constrained, gate-checked closed-loop
AI scientist for simulation-driven materials physics," 2026,
github.com/tians302/ai-physicist-pilot.
```
