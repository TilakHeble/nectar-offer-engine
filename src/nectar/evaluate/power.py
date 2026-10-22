"""Sample size for a live member-level offer test, with and without CUPED.

Outcome: each member's gross margin over one 4-week offer cycle. Members' spending varies hugely,
so this outcome is very noisy compared with the effect we hope to detect (tens of pence per
member).

CUPED (Controlled-experiment Using Pre-Experiment Data; Deng et al., 2013) removes the part of the
outcome predicted by each member's own pre-test spending:

    y_adj = y - theta * (x - mean(x)),   theta = cov(x, y) / var(x)

Randomisation makes x independent of treatment, so the treatment effect is unchanged while the
variance falls by a factor (1 - rho^2), where rho = corr(x, y). The required sample shrinks by
the same factor.

Two-sided test, two equal arms:

    n_per_arm = 2 * (z_{1-alpha/2} + z_{power})^2 * sigma^2 * (1 - rho^2) / delta^2
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def sample_size_per_arm(
    sigma: float, delta: float, alpha: float = 0.05, power: float = 0.8, rho: float = 0.0
) -> int:
    """Members per arm needed to detect a mean difference `delta`."""
    if delta <= 0 or sigma <= 0:
        raise ValueError("sigma and delta must be positive")
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return math.ceil(2 * z**2 * sigma**2 * (1 - rho**2) / delta**2)


def minimum_detectable_effect(
    n_per_arm: int, sigma: float, alpha: float = 0.05, power: float = 0.8, rho: float = 0.0
) -> float:
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return z * sigma * math.sqrt(2 * (1 - rho**2) / n_per_arm)


def cuped(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """CUPED-adjusted outcome and theta."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    theta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    return y - theta * (x - x.mean()), theta


def member_outcomes(
    con, outcome_start: int = 92, cycle_weeks: int = 4, pre_weeks: int = 12, margin: float = 0.30
):
    """Real per-member gross margin in one cycle (y) and the pre-period margin (x)."""
    lo, hi = outcome_start, outcome_start + cycle_weeks - 1
    pre_lo = outcome_start - pre_weeks
    df = (
        con.execute(f"""
        SELECT member_id,
               SUM(basket_value) FILTER (WHERE week_no BETWEEN {lo} AND {hi}) AS y_spend,
               SUM(basket_value) FILTER (WHERE week_no BETWEEN {pre_lo} AND {lo - 1}) AS x_spend
        FROM marts.fct_basket
        GROUP BY 1
        HAVING MAX(week_no) FILTER (WHERE week_no < {lo}) > {lo - 1} - 26
    """)
        .df()
        .fillna(0.0)
    )
    return margin * df.y_spend.to_numpy(), margin * df.x_spend.to_numpy()
