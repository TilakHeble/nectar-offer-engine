"""Fairness report on small hand-made plans."""

import pandas as pd
import pytest

from nectar.evaluate.fairness import group_report, member_plan, summarise


def _plan():
    # members 1-4 eligible; 1 gets 2 offers, 2 gets 1, 3 and 4 get none
    rows = [
        (1, "A", True),
        (1, "B", True),
        (2, "A", True),
        (2, "B", False),
        (3, "A", False),
        (4, "B", False),
    ]
    df = pd.DataFrame(rows, columns=["member_id", "offer_id", "selected"])
    df["cost"], df["value"] = 1.0, 0.5
    return df


def test_member_plan_counts_offers_and_includes_unreached():
    m = member_plan(_plan()).set_index("member_id")
    assert m.offers.to_dict() == {1: 2, 2: 1, 3: 0, 4: 0}
    assert m.reached.tolist() == [True, True, False, False]
    assert m.loc[1, "discount"] == 2.0


def test_group_report_flags_low_reach_and_skips_small_groups():
    members = pd.DataFrame(
        {
            "member_id": range(200),
            "reached": [True] * 90 + [False] * 10 + [True] * 50 + [False] * 50,
            "offers": 1,
            "discount": 0.2,
            "group": ["high"] * 100 + ["low"] * 100,
        }
    )
    members.loc[len(members)] = [999, False, 0, 0.0, "tiny"]
    rep = group_report(members, "group", min_group_size=30)
    assert rep.loc["high", "reach_ratio"] == pytest.approx(1.0)
    assert rep.loc["low", "reach_ratio"] == pytest.approx(0.5 / 0.9)
    assert bool(rep.loc["low", "flag"]) and not bool(rep.loc["high", "flag"])
    assert not bool(rep.loc["tiny", "judged"]) and not bool(rep.loc["tiny", "flag"])
    s = summarise(rep)
    assert s["flagged"] == ["low"] and s["groups_judged"] == 2


def test_missing_group_values_are_excluded():
    members = pd.DataFrame(
        {
            "member_id": [1, 2],
            "reached": [True, False],
            "offers": [1, 0],
            "discount": [0.1, 0.0],
            "group": ["x", None],
        }
    )
    rep = group_report(members, "group", min_group_size=1)
    assert rep.members.sum() == 1
