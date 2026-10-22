"""Budget-constrained offer allocation.

Given, for each candidate (member, offer) pair,
    value_ij = predicted incremental gross profit if offered   (from the uplift champion)
    cost_ij  = expected retailer-funded discount cost if offered (from the cost model)
choose which offers to send:

    maximise    sum_ij value_ij * x_ij
    subject to  sum_ij cost_ij * x_ij        <= budget          (retailer discount budget)
                sum_j  x_ij                  <= max_per_member  (app / email real estate)
                sum_i  x_ij                  <= offer_cap_j     (supplier funding / stock)
                x_ij in {0, 1};  pairs with value_ij <= 0 are never offered

Two solvers share one interface:
* `solve_ilp`    — exact integer programme (PuLP + CBC). Also returns the budget's shadow
                   price from the LP relaxation: the extra profit one more $1 of budget buys.
* `solve_greedy` — value-per-cost ranking with the same constraints; a fast, explainable
                   fallback for very large candidate sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pulp


@dataclass(frozen=True)
class Constraints:
    budget: float
    max_per_member: int = 3
    offer_caps: dict[str, int] = field(default_factory=dict)  # offer_id -> max offers sent


@dataclass
class Allocation:
    selected: np.ndarray  # bool per candidate row
    objective: float  # predicted incremental profit of the selection
    spend: float  # expected discount cost of the selection
    method: str
    shadow_price: float | None = None  # profit per extra $1 of budget (LP relaxation)

    def summary(self) -> dict:
        return {
            "method": self.method,
            "offers_sent": int(self.selected.sum()),
            "predicted_profit": self.objective,
            "expected_cost": self.spend,
            "shadow_price": self.shadow_price,
        }


_REQUIRED = ["member_id", "offer_id", "value", "cost"]


def _check(candidates: pd.DataFrame) -> None:
    missing = set(_REQUIRED) - set(candidates.columns)
    if missing:
        raise ValueError(f"candidates missing columns: {sorted(missing)}")
    if (candidates.cost < 0).any():
        raise ValueError("cost must be non-negative")


def _result(candidates, selected, method, shadow=None) -> Allocation:
    return Allocation(
        selected=selected,
        objective=float(candidates.value.to_numpy()[selected].sum()),
        spend=float(candidates.cost.to_numpy()[selected].sum()),
        method=method,
        shadow_price=shadow,
    )


def solve_greedy(candidates: pd.DataFrame, c: Constraints) -> Allocation:
    """Take positive-value pairs in order of value per $ of cost while constraints allow."""
    _check(candidates)
    v = candidates.value.to_numpy()
    cost = candidates.cost.to_numpy()
    ratio = np.where(cost > 0, v / np.maximum(cost, 1e-12), np.inf)
    order = np.lexsort((-v, -ratio))  # best ratio first, ties by value
    selected = np.zeros(len(candidates), bool)
    per_member: dict = {}
    per_offer: dict = {}
    spent = 0.0
    members = candidates.member_id.to_numpy()
    offers = candidates.offer_id.to_numpy()
    for i in order:
        if v[i] <= 0:
            continue
        if spent + cost[i] > c.budget + 1e-9:
            continue
        m, o = members[i], offers[i]
        if per_member.get(m, 0) >= c.max_per_member:
            continue
        if o in c.offer_caps and per_offer.get(o, 0) >= c.offer_caps[o]:
            continue
        selected[i] = True
        spent += cost[i]
        per_member[m] = per_member.get(m, 0) + 1
        per_offer[o] = per_offer.get(o, 0) + 1
    return _result(candidates, selected, "greedy")


def _build(candidates: pd.DataFrame, c: Constraints, relax: bool):
    pos = np.flatnonzero(candidates.value.to_numpy() > 0)  # never offer a loss-making pair
    sub = candidates.iloc[pos]
    prob = pulp.LpProblem("offer_allocation", pulp.LpMaximize)
    cat = pulp.LpContinuous if relax else pulp.LpBinary
    x = [pulp.LpVariable(f"x{k}", 0, 1, cat) for k in range(len(sub))]
    prob += pulp.lpSum(float(v) * xi for v, xi in zip(sub.value, x, strict=True))
    budget = pulp.lpSum(float(cst) * xi for cst, xi in zip(sub.cost, x, strict=True)) <= c.budget
    prob += budget, "budget"
    for _, idx in sub.groupby("member_id").indices.items():
        if len(idx) > c.max_per_member:
            prob += pulp.lpSum(x[k] for k in idx) <= c.max_per_member
    for oid, idx in sub.groupby("offer_id").indices.items():
        if oid in c.offer_caps and len(idx) > c.offer_caps[oid]:
            prob += pulp.lpSum(x[k] for k in idx) <= c.offer_caps[oid]
    return prob, x, pos


def solve_ilp(candidates: pd.DataFrame, c: Constraints, time_limit: int = 60) -> Allocation:
    _check(candidates)
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)

    prob, x, pos = _build(candidates, c, relax=False)
    prob.solve(solver)
    if pulp.LpStatus[prob.status] not in ("Optimal", "Not Solved"):
        raise RuntimeError(f"ILP status: {pulp.LpStatus[prob.status]}")
    selected = np.zeros(len(candidates), bool)
    selected[pos] = np.array([(xi.value() or 0) > 0.5 for xi in x], bool)

    # Shadow price of the budget from the LP relaxation (dual of the budget constraint).
    lp, _, _ = _build(candidates, c, relax=True)
    lp.solve(solver)
    shadow = lp.constraints["budget"].pi
    return _result(candidates, selected, "ilp", float(shadow) if shadow is not None else None)
