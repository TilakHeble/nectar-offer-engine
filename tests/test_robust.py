"""Risk-aware allocation: shrinkage, lower bounds and the optimiser's curse on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from nectar.optimise.allocate import Constraints, solve_ilp
from nectar.optimise.robust import BaggedUplift, adjusted_values, lower_bound, shrink


def test_shrinkage_pulls_noisy_estimates_harder():
    mu = np.array([1.0, 1.0])
    out = shrink(mu, np.array([0.1, 1.0]), prior_mean=0.0, prior_var=0.25)
    assert out[0] == pytest.approx(1.0 * 0.25 / (0.25 + 0.01))
    assert out[1] == pytest.approx(1.0 * 0.25 / (0.25 + 1.0))
    assert out[1] < out[0] < 1.0


def test_lower_bound_and_dispatch():
    mu, sd = np.array([1.0, 2.0]), np.array([0.5, 2.0])
    np.testing.assert_allclose(lower_bound(mu, sd, 1.0), [0.5, 0.0])
    np.testing.assert_allclose(adjusted_values(mu, sd, "mean"), mu)
    with pytest.raises(ValueError):
        adjusted_values(mu, sd, "nope")


def test_risk_aware_plan_beats_naive_plan_when_noise_varies():
    """Pairs with noisy estimates look best by chance; the lower bound avoids them."""
    rng = np.random.default_rng(0)
    n = 4000
    true = rng.normal(0, 1, n)
    sd = np.where(rng.random(n) < 0.5, 0.2, 3.0)  # half the estimates are very noisy
    mu = true + rng.normal(0, sd)
    cand = pd.DataFrame({"member_id": np.arange(n), "offer_id": "A", "cost": 1.0})
    c = Constraints(budget=200, max_per_member=1)
    naive = solve_ilp(cand.assign(value=mu), c).selected
    robust = solve_ilp(cand.assign(value=lower_bound(mu, sd, 1.0)), c).selected
    assert true[robust].sum() > true[naive].sum()
    # the naive plan overstates its own value far more
    assert mu[naive].sum() / true[naive].sum() > mu[robust].sum() / true[robust].sum()


def test_bagged_uplift_gives_mean_and_positive_spread():
    rng = np.random.default_rng(1)
    n = 3000
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    t = rng.integers(0, 2, n)
    y = X.x2 + t * 0.5 * X.x1 + rng.normal(0, 0.5, n)
    bag = BaggedUplift(n_models=4).fit(X, t, y, groups=np.arange(n) // 3)
    mu, sd = bag.predict(X)
    assert mu.shape == sd.shape == (n,)
    assert (sd > 0).all()
    assert np.corrcoef(mu, 0.5 * X.x1)[0, 1] > 0.6
