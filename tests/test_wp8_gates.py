"""WP8 gate acceptance: clean synthetic diagnostics pass; every named
corruption is caught by its intended gate (mirrors WP4 pattern)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corruptions import CORRUPTIONS, INTENDED_GATE
from gates.expansion import run_expansion_gates
from gates.transport import run_diffusion_gates, run_gk_gates

CLEAN = {
    "expansion": {
        "n_temps": 4, "T_K": [300, 500, 700, 900],
        "a_A": [5.435, 5.440, 5.446, 5.452], "alpha_per_K": 1.0e-5,
        "a_ref_A": 5.435, "equilibration_drift_rel": 8e-4,
        "fit_resid_rel": 0.05, "spearman_like_monotonic": 0.999,
    },
    "diffusion": {
        "n_checkpoints": 20, "t_lj": [], "msd_lj": [], "D_lj": 0.031,
        "msd_final_lj": 5.6, "fit_resid_rel": 0.012,
        "fit_window_start_index": 8, "state_point": {},
    },
    "gk": {
        "k_components_lj": [8.41, 6.86, 5.78], "kappa_lj": 7.02,
        "component_spread_rel": 0.154, "all_positive": True,
        "state_point": {},
    },
}
RUNNERS = {"expansion": run_expansion_gates, "diffusion": run_diffusion_gates,
           "gk": run_gk_gates}


def test_clean_diagnostics_pass_all_gates():
    for kind, diag in CLEAN.items():
        report = RUNNERS[kind](diag)
        assert report["passed"], (kind, report)


def test_every_wp8_corruption_caught_by_intended_gate():
    for kind in ("expansion", "diffusion", "gk"):
        for cname, fn in CORRUPTIONS[kind].items():
            report = RUNNERS[kind](fn(CLEAN[kind]))
            verdicts = {g["name"]: g["passed"] for g in report["gates"]}
            intended = INTENDED_GATE[(kind, cname)]
            assert not report["passed"], (kind, cname)
            assert not verdicts[intended], (kind, cname, verdicts)


def test_intended_gate_map_is_complete():
    for kind, table in CORRUPTIONS.items():
        for cname in table:
            assert (kind, cname) in INTENDED_GATE


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "pytest", __file__, "-q"]))
