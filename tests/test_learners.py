"""Uplift learners recover a known treatment effect on synthetic data (fast, no warehouse)."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from nectar.models.learners import (
    CausalForestModel,
    DRLearnerModel,
    EnsembleModel,
    RandomTargeting,
    ResponseModel,
    SLearnerModel,
    TLearnerModel,
    XLearnerModel,
)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 8_000
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "x3": rng.normal(size=n)})
    t = rng.integers(0, 2, n)
    tau = 0.5 * X.x1  # uplift varies with x1 only
    y = X.x2 + t * tau + rng.normal(scale=0.5, size=n)
    return X, t, y.to_numpy(), tau.to_numpy()


@pytest.mark.parametrize(
    "model",
    [
        TLearnerModel(),
        SLearnerModel(),
        XLearnerModel(),
        DRLearnerModel(),
        CausalForestModel(n_estimators=100, min_samples_leaf=20),
    ],
    ids=lambda m: m.name,
)
def test_uplift_learners_rank_true_uplift(model, data):
    X, t, y, tau = data
    model.fit(X, t, y)
    rho = spearmanr(model.score(X), tau).statistic
    assert rho > 0.6, f"{model.name}: rho={rho:.2f}"
    assert model.has_profit_threshold


def test_ensemble_averages_members(data):
    X, t, y, _ = data
    members = [SLearnerModel(), XLearnerModel()]
    ens = EnsembleModel(members).fit(X, t, y)
    expected = np.mean([m.score(X) for m in members], axis=0)
    np.testing.assert_allclose(ens.score(X), expected)


def test_baselines_have_no_profit_threshold(data):
    X, t, y, _ = data
    resp = ResponseModel().fit(X, t, (y > 0).astype(int))
    assert not resp.has_profit_threshold
    assert resp.score(X).shape == (len(X),)
    rnd = RandomTargeting().fit(X, t, y)
    np.testing.assert_array_equal(rnd.score(X), RandomTargeting().score(X))
