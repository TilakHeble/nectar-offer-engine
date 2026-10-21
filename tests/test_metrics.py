"""Unit tests for the evaluation metrics on small synthetic experiments."""

import numpy as np

from nectar.evaluate import metrics as M


def _experiment(n=200_000, seed=0):
    """Randomised experiment where treatment adds +1 for x > 0 and -1 otherwise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = rng.integers(0, 2, n)
    tau = np.where(x > 0, 1.0, -1.0)
    y = rng.normal(size=n) + t * tau
    return x, t, y, tau


def test_ipw_policy_value_matches_truth():
    x, t, y, tau = _experiment()
    treat = x > 0
    # true incremental value of "treat if x > 0" = number of rows with x > 0 (each gains +1)
    est = M.incremental_policy_value(treat, t, y)
    assert abs(est - treat.sum()) / treat.sum() < 0.03


def test_treat_nobody_has_zero_incremental_value():
    x, t, y, _ = _experiment(10_000)
    assert M.incremental_policy_value(np.zeros(len(t), bool), t, y) == 0.0


def test_observed_curve_peaks_near_correct_share():
    x, t, y, _ = _experiment()
    grid, curve = M.observed_uplift_curve(x, t, y)
    assert abs(grid[curve.argmax()] - 0.5) <= 0.05
    assert curve[0] == 0.0


def test_true_curve_and_oracle():
    tu = np.array([3.0, -1.0, 2.0, -5.0])
    assert M.oracle_profit(tu) == 5.0
    grid, curve = M.true_profit_curve(tu, tu, grid=np.array([0, 0.5, 1.0]))
    assert list(curve) == [0.0, 5.0, -1.0]
    assert M.true_policy_profit(tu > 0, tu) == 5.0


def test_good_ranking_beats_bad_ranking_on_auuc():
    x, t, y, _ = _experiment(50_000)
    assert M.auuc(x, t, y) > M.auuc(-x, t, y)


def test_bootstrap_interval_contains_estimate():
    x, t, y, _ = _experiment(20_000)
    groups = np.arange(len(t)) // 5
    est, lo, hi = M.bootstrap_incremental_policy_value(x > 0, t, y, groups, n_boot=100)
    assert lo < est < hi
