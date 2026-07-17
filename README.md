# AI Physicist

Closed-loop AI-scientist framework for thermal transport.

The agent runs a fixed hypothesize → plan → execute → gate-check → interpret → conclude loop: hypotheses become **typed, schema-validated experiment plans**, executed only by **allowlisted physics engines** (no code generation), and every result must pass **hard validity gates** (fit quality, residual structure, mandatory uncertainties, bootstrap stability, holdout validation) *before* any LLM judgment. Validation is by rediscovery on silicon κ(T) with an answer key scored by the harness; application targets glassy-carbon thermal conductivity via the [A3HT](https://github.com/tians302/A3HT) pipeline on ALCF.

Builds on ideas from [SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist) (full-loop autonomy) and A3HT (high-throughput MD execution), adding the discipline layer: gates + typed plans + uncertainty quantification.

## Status

Phase 1 (hackathon): loop state machine + Callaway-RTA silicon engine + gate set + one rediscovery task end-to-end. Code landing here incrementally; the reference pilot implementation lives outside this repo.

## Roadmap

1. **Validate** — rediscovery benchmark (silicon κ(T), Glassbrenner–Slack ground truth), gate-on/gate-off ablation
2. **Extend** — small LAMMPS NEMD engine, benchmark battery
3. **Apply** — glassy-carbon campaigns via A3HT on ALCF

