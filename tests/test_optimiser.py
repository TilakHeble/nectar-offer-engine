"""The allocation solvers respect every constraint and the ILP is optimal."""

import numpy as np
import pandas as pd
import pytest

from nectar.optimise.allocate import Constraints, solve_greedy, solve_ilp


def _cands(values, costs, members=None, offers=None):
    n = len(values)
    return pd.DataFrame(
        {
            "member_id": members if members is not None else np.arange(n),
            "offer_id": offers if offers is not None else ["OFR01"] * n,
            "value": np.asarray(values, float),
            "cost": np.asarray(costs, float),
        }
    )


def test_ilp_beats_ratio_greedy_on_classic_knapsack():
    # Greedy takes A (best ratio) and then nothing else fits; the optimum is B + C.
    c = _cands([10, 7, 7], [6, 5, 5])
    ilp = solve_ilp(c, Constraints(budget=10, max_per_member=1))
    greedy = solve_greedy(c, Constraints(budget=10, max_per_member=1))
    assert ilp.objective == 14 and list(ilp.selected) == [False, True, True]
    assert greedy.objective == 10


@pytest.mark.parametrize("solver", [solve_ilp, solve_greedy])
def test_never_offers_loss_making_pairs(solver):
    c = _cands([5, -1, 0, 2], [1, 0, 0, 1])
    alloc = solver(c, Constraints(budget=100))
    assert list(alloc.selected) == [True, False, False, True]


@pytest.mark.parametrize("solver", [solve_ilp, solve_greedy])
def test_respects_budget_member_and_offer_caps(solver):
    rng = np.random.default_rng(0)
    n = 300
    c = _cands(
        rng.normal(0.5, 1, n),
        rng.uniform(0.1, 2, n),
        members=rng.integers(0, 40, n),
        offers=rng.choice(["OFR01", "OFR02", "OFR03"], n),
    )
    cons = Constraints(budget=25, max_per_member=2, offer_caps={"OFR01": 5})
    alloc = solver(c, cons)
    sel = c[alloc.selected]
    assert sel.cost.sum() <= cons.budget + 1e-6
    assert sel.groupby("member_id").size().max() <= 2
    assert (sel.offer_id == "OFR01").sum() <= 5
    assert (sel.value > 0).all()


def test_ilp_is_at_least_as_good_as_greedy():
    rng = np.random.default_rng(1)
    for _ in range(5):
        n = 200
        c = _cands(rng.normal(0.3, 1, n), rng.uniform(0.1, 3, n), members=rng.integers(0, 30, n))
        cons = Constraints(budget=20, max_per_member=2)
        assert solve_ilp(c, cons).objective >= solve_greedy(c, cons).objective - 1e-6


def test_shadow_price_positive_when_budget_binds_and_zero_when_slack():
    c = _cands([3, 2, 1], [1, 1, 1])
    tight = solve_ilp(c, Constraints(budget=1.5))
    slack = solve_ilp(c, Constraints(budget=100))
    assert tight.shadow_price > 0
    assert slack.shadow_price == pytest.approx(0, abs=1e-9)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        solve_greedy(pd.DataFrame({"member_id": [1], "value": [1.0]}), Constraints(budget=1))
    with pytest.raises(ValueError):
        solve_ilp(_cands([1], [-1]), Constraints(budget=1))
