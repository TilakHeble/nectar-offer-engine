"""Evaluation for offer-targeting policies.

Two families of metrics:

* Observed-outcome metrics use only (treatment, outcome) and the randomisation. They are what a
  real team can compute from a live randomised test, and they drive model selection here.
* Ground-truth metrics use the simulator's `gt_` columns. Only possible in simulation, they show
  how close each model gets to the best achievable targeting.
"""

from __future__ import annotations

import numpy as np


def ipw_policy_value(
    treat: np.ndarray, t: np.ndarray, y: np.ndarray, p_treat: float = 0.5
) -> float:
    """Estimated mean outcome per pair if `treat` (bool per row) were the policy.

    Inverse-propensity weighting: keep the rows where the random assignment happens to agree
    with the policy, and reweight by the probability of that assignment.
    """
    treat = np.asarray(treat, bool)
    t = np.asarray(t, int)
    agree = (t == treat).astype(float)
    prob = np.where(treat, p_treat, 1 - p_treat)
    return float(np.mean(y * agree / prob))


def incremental_policy_value(treat, t, y, p_treat: float = 0.5) -> float:
    """Estimated total incremental outcome vs sending no offers, over all rows."""
    none = np.zeros(len(t), bool)
    return len(t) * (ipw_policy_value(treat, t, y, p_treat) - ipw_policy_value(none, t, y, p_treat))


def observed_uplift_curve(scores, t, y, grid: np.ndarray | None = None):
    """Estimated cumulative incremental outcome when treating the top-k fraction by score.

    At each k: (mean outcome of treated in top k - mean outcome of control in top k) x k rows.
    Returns (fractions, incremental_totals).
    """
    grid = np.linspace(0, 1, 101) if grid is None else grid
    order = np.argsort(-np.asarray(scores), kind="stable")
    t_s, y_s = np.asarray(t)[order], np.asarray(y, float)[order]
    n_t, n_c = np.cumsum(t_s), np.cumsum(1 - t_s)
    s_t, s_c = np.cumsum(y_s * t_s), np.cumsum(y_s * (1 - t_s))
    out = []
    for f in grid:
        k = int(round(f * len(t_s)))
        if k == 0:
            out.append(0.0)
            continue
        i = k - 1
        mt = s_t[i] / n_t[i] if n_t[i] else 0.0
        mc = s_c[i] / n_c[i] if n_c[i] else 0.0
        out.append((mt - mc) * k)
    return grid, np.array(out)


def auuc(scores, t, y) -> float:
    """Area under the observed incremental-outcome curve (x in fractions)."""
    x, c = observed_uplift_curve(scores, t, y)
    return float(np.trapezoid(c, x))


def true_profit_curve(scores, true_uplift, grid: np.ndarray | None = None):
    """True cumulative incremental outcome when treating the top-k fraction by score."""
    grid = np.linspace(0, 1, 101) if grid is None else grid
    order = np.argsort(-np.asarray(scores), kind="stable")
    cum = np.concatenate([[0.0], np.cumsum(np.asarray(true_uplift)[order])])
    idx = np.round(grid * len(order)).astype(int)
    return grid, cum[idx]


def oracle_profit(true_uplift) -> float:
    """Best achievable incremental outcome: treat exactly the pairs with positive true uplift."""
    tu = np.asarray(true_uplift)
    return float(tu[tu > 0].sum())


def true_policy_profit(treat, true_uplift) -> float:
    return float(np.asarray(true_uplift)[np.asarray(treat, bool)].sum())


def bootstrap_incremental_policy_value(
    treat, t, y, groups, n_boot: int = 500, seed: int = 0, p_treat: float = 0.5
) -> tuple[float, float, float]:
    """Estimate + 95% interval, resampling whole members (rows of a member move together)."""
    treat, t, y, groups = map(np.asarray, (treat, t, y, groups))
    uniq, inv = np.unique(groups, return_inverse=True)
    rows_by_group = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    rng = np.random.default_rng(seed)
    est = incremental_policy_value(treat, t, y, p_treat)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_by_group[g] for g in pick])
        stats.append(incremental_policy_value(treat[idx], t[idx], y[idx], p_treat))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return est, float(lo), float(hi)
