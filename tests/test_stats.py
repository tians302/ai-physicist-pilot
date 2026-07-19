"""WP0 regression tests: true bootstrap (multiplicity kept) + block holdout.

Run with: pytest tests/ -q  (or python tests/test_stats.py from repo root).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from engines.callaway import (CallawayEngine, bootstrap_AB, split_data,
                              N_HOLDOUT_BLOCKS)
from plans import ExperimentPlan, validate_plan


def test_bootstrap_preserves_multiplicity():
    """Resamples must be passed to the fit WITH duplicates (true bootstrap)."""
    cal = np.column_stack([np.linspace(50, 500, 10),
                           np.linspace(1000, 60, 10),
                           np.full(10, 5.0)])
    seen = []

    def spy_fit(sub):
        seen.append(sub.copy())
        return 1.0, 1.0

    rng = np.random.default_rng(0)
    boot = bootstrap_AB(cal, 1e-3, (0.0, 0.0), rng, n_boot=50, fit_fn=spy_fit)

    assert len(boot) == len(seen) > 0
    # every resample has the full sample size (nothing collapsed to unique)
    assert all(len(s) == len(cal) for s in seen)
    # with n=10 draws from 10 rows, duplicates are near-certain in most resamples
    n_with_dupes = sum(len(np.unique(s[:, 0])) < len(s) for s in seen)
    assert n_with_dupes > len(seen) // 2, "resamples show no duplicated rows"


def test_bootstrap_skips_degenerate_resamples():
    cal = np.column_stack([np.linspace(50, 500, 6),
                           np.linspace(1000, 60, 6),
                           np.full(6, 5.0)])

    class ConstRng:
        def integers(self, lo, hi, size):
            return np.zeros(size, dtype=int)  # 1 unique point: degenerate

    boot = bootstrap_AB(cal, 1e-3, (0.0, 0.0), ConstRng(), n_boot=10,
                        fit_fn=lambda sub: (1.0, 1.0))
    assert len(boot) == 0


def test_block_split_properties():
    d = np.column_stack([np.arange(16.0), np.arange(16.0), np.ones(16)])
    cal, hold = split_data(d, "block")
    # partition: disjoint, complete
    all_T = sorted(cal[:, 0].tolist() + hold[:, 0].tolist())
    assert all_T == list(np.arange(16.0))
    # holdout is contiguous blocks, not interleaved points
    blocks = np.array_split(np.arange(16), N_HOLDOUT_BLOCKS)
    expected_hold = np.concatenate(blocks[1::2])
    assert hold[:, 0].tolist() == [float(i) for i in expected_hold]
    # each holdout block spans >1 consecutive points (non-interpolative gap)
    diffs = np.diff(hold[:, 0])
    assert (diffs == 1).any(), "holdout should contain consecutive runs"


def test_alternating_split_retained_for_comparison():
    d = np.column_stack([np.arange(8.0), np.arange(8.0), np.ones(8)])
    cal, hold = split_data(d, "alternating")
    assert cal[:, 0].tolist() == [0, 2, 4, 6] and hold[:, 0].tolist() == [1, 3, 5, 7]
    with pytest.raises(ValueError):
        split_data(d, "bogus")


def test_plan_defaults_to_block_split():
    plan = ExperimentPlan(plan_id="t", hypothesis="Boundary scattering suppresses kappa at low T.")
    assert plan.split == "block"
    assert validate_plan(plan) == []
    # legacy mode still validates (comparison only)
    plan_alt = ExperimentPlan(plan_id="t", hypothesis="Boundary scattering suppresses kappa at low T.",
                              split="alternating")
    assert validate_plan(plan_alt) == []


def test_engine_end_to_end_block_split():
    """WP0 acceptance: Callaway rerun green with the new statistics."""
    eng = CallawayEngine()
    plan = ExperimentPlan(plan_id="t", hypothesis="Boundary scattering suppresses kappa at low T.")
    res = eng.run(plan, np.random.default_rng(0))
    assert res["split_mode"] == "block"
    assert res["bootstrap_method"].startswith("nonparametric_row_resample")
    assert res["n_bootstrap_ok"] >= 50
    for p in res["fit"].values():
        assert np.isfinite(p["sigma"]) and p["sigma"] > 0


if __name__ == "__main__":
    for t in [test_bootstrap_preserves_multiplicity, test_bootstrap_skips_degenerate_resamples,
              test_block_split_properties, test_alternating_split_retained_for_comparison,
              test_plan_defaults_to_block_split, test_engine_end_to_end_block_split]:
        print(t.__name__)
        t()
    print("STATS TESTS PASSED")
