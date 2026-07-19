"""Research-note generator: markdown + figure, with full provenance."""
import json
import platform
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_figure(results, path):
    T = np.array(results["curves"]["T_K"])
    cal = np.array(results["calibration_points"])
    hold = np.array(results["holdout_points"])
    c = results["constants"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(cal[:, 0], cal[:, 1], yerr=cal[:, 2], fmt="o", ms=4, capsize=2,
                label="reference (calibration)", color="tab:blue")
    ax.errorbar(hold[:, 0], hold[:, 1], yerr=hold[:, 2], fmt="s", ms=4, capsize=2,
                label="reference (holdout)", color="tab:cyan")
    ax.plot(T, results["curves"]["kappa_baseline"], "-", color="tab:blue",
            label=f"model, L = {c['L_baseline_m']*1e3:g} mm")
    ax.plot(T, results["curves"]["kappa_intervention"], "--", color="tab:red",
            label=f"model, L = {c['L_intervention_m']*1e6:g} um")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("T (K)"); ax.set_ylabel(r"$\kappa$ (W m$^{-1}$ K$^{-1}$)")
    ax.legend(fontsize=8); ax.set_title("Si thermal conductivity: boundary-length intervention")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def write_note(run_dir, plan, results, gate_report, score_report, narrative,
               llm_mode, seed, gates_enforced):
    fig_path = run_dir / "figure_kappa.png"
    make_figure(results, fig_path)

    fit = results["fit"]
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    flag = ""
    if not gate_report["passed"]:
        flag = ("\n> **GATE FAILURE — conclusions below are NOT valid claims.**"
                " This note is retained for the record only.\n")
    if not gates_enforced:
        flag += "\n> **Gates were NOT enforced in this run (ablation mode).**\n"

    gate_rows = "\n".join(
        f"| {g['name']} | {'PASS' if g['passed'] else 'FAIL'} | {g['detail']} |"
        for g in gate_report["gates"])
    score_rows = "\n".join(
        f"| {c['name']} | {'PASS' if c['passed'] else 'FAIL'} | {c['detail']} |"
        for c in score_report["checks"])

    note = f"""# Research note: {plan.plan_id}
{flag}
*Generated {ts} · engine `{results['engine']}` · LLM mode `{llm_mode}` · seed {seed}*

## Hypothesis

{plan.hypothesis}

## Methods

Callaway-style RTA model of kappa(T) with boundary (v/L), point-defect (A w^4) and
Umklapp (B w^2 T exp[-Theta_D/3T]) scattering; Theta_D = {results['constants']['theta_D_K']} K,
v = {results['constants']['v_m_s']} m/s (fixed). Only A, B fitted, on the calibration
split of the Glassbrenner-Slack reference data ({results.get('split_mode', plan.split)} split;
holdout never seen during fitting). Baseline L = {results['constants']['L_baseline_m']:g} m;
intervention L = {results['constants']['L_intervention_m']:g} m.
T range {plan.T_min_K:g}-{plan.T_max_K:g} K.

**Scope caveat.** L is an *effective* boundary-scattering parameter of a simplified
Callaway/Holland model with fixed Theta_D and v. The intervention curves support only
the qualitative claim (suppression at every T, relatively stronger at low T); the
numerical suppression magnitudes are NOT quantitative predictions for a physical
{results['constants']['L_intervention_m']*1e6:g}-um sample.

## Fitted parameters (bootstrap uncertainties, n = {results['n_bootstrap_ok']})

| Param | Value | Sigma | Units |
|---|---|---|---|
| A | {fit['A']['value']:.3e} | {fit['A']['sigma']:.2e} | {fit['A']['units']} |
| B | {fit['B']['value']:.3e} | {fit['B']['sigma']:.2e} | {fit['B']['units']} |

chi2/dof: calibration = {results['chi2_dof_calibration']:.2f}, holdout = {results['chi2_dof_holdout']:.2f}

## Validity gates (v{gate_report['version']}, run before any LLM judgment)

| Gate | Verdict | Detail |
|---|---|---|
{gate_rows}

## Answer-key score (harness-computed)

| Check | Verdict | Detail |
|---|---|---|
{score_rows}

**Benchmark success: {score_report['success']}**

## Results

![kappa vs T](figure_kappa.png)

{narrative}

## Provenance

- Plan: [`plan.json`](plan.json) (schema-validated before execution)
- Gate report: [`gate_report.json`](gate_report.json) · Score: [`score_report.json`](score_report.json)
- Reference data: `data/si_kappa_reference.csv` (Glassbrenner & Slack 1964 via Touloukian 1970, ~5% stated uncertainty)
- Python {platform.python_version()}, seed {seed}
"""
    (run_dir / "note.md").write_text(note)
    return note
