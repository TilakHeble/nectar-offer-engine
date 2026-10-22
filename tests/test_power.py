"""Sample-size maths and CUPED variance reduction."""

import math

import numpy as np
import pytest
from scipy.stats import norm

from nectar.evaluate.power import cuped, minimum_detectable_effect, sample_size_per_arm


def test_sample_size_matches_textbook_formula():
    z = norm.ppf(0.975) + norm.ppf(0.8)  # ~2.80
    assert sample_size_per_arm(sigma=10, delta=1) == math.ceil(2 * z**2 * 100)


def test_cuped_scales_sample_size_by_one_minus_rho_squared():
    base = sample_size_per_arm(50, 0.5)
    adj = sample_size_per_arm(50, 0.5, rho=0.8)
    assert adj == pytest.approx(base * (1 - 0.8**2), rel=1e-4)


def test_mde_inverts_sample_size():
    n = sample_size_per_arm(30, 0.7, rho=0.5)
    assert minimum_detectable_effect(n, 30, rho=0.5) == pytest.approx(0.7, rel=1e-3)


def test_cuped_keeps_the_effect_and_cuts_variance():
    rng = np.random.default_rng(0)
    n = 200_000
    x = rng.gamma(2, 20, n)  # pre-period spend
    t = rng.integers(0, 2, n)
    y = 0.8 * x + rng.normal(0, 10, n) + 0.5 * t  # true effect 0.5
    ya, _ = cuped(y, x)
    raw = y[t == 1].mean() - y[t == 0].mean()
    adj = ya[t == 1].mean() - ya[t == 0].mean()
    rho = np.corrcoef(x, y)[0, 1]
    assert ya.var() == pytest.approx(y.var() * (1 - rho**2), rel=0.01)
    se_adj = np.sqrt(ya[t == 1].var() / (t == 1).sum() + ya[t == 0].var() / (t == 0).sum())
    se_raw = np.sqrt(y[t == 1].var() / (t == 1).sum() + y[t == 0].var() / (t == 0).sum())
    assert abs(adj - 0.5) < 3 * se_adj  # still unbiased
    assert abs(raw - 0.5) < 3 * se_raw
    assert se_adj < 0.65 * se_raw  # and far more precise


def test_invalid_inputs():
    with pytest.raises(ValueError):
        sample_size_per_arm(10, 0)
