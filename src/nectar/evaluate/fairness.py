"""Fairness check for an offer plan: who gets offers, and how much discount value.

The model never sees demographics (and the data has them for only a third of members), but a
behaviour-based targeting rule can still skew offers towards or away from groups. This module
compares groups on:

    reach              share of eligible members who receive at least one offer
    offers_per_member  average number of offers
    discount_per_member expected retailer-funded discount per member

and flags groups whose reach falls below `ratio_threshold` x the best-reached group (the
"four-fifths rule" used in adverse-impact testing). Groups smaller than `min_group_size`
are reported but not judged: their rates are too noisy.

Note this checks *allocation* parity. Offers are a benefit, so a group receiving fewer offers
pays a higher effective price; whether a gap is justified (e.g. by lower price sensitivity)
is a business decision this report informs, not one it makes.
"""

from __future__ import annotations

import pandas as pd

FOUR_FIFTHS = 0.8


def member_plan(candidates: pd.DataFrame) -> pd.DataFrame:
    """One row per eligible member: offers received, expected discount cost, predicted value."""
    sel = candidates[candidates.selected]
    per = sel.groupby("member_id").agg(
        offers=("offer_id", "size"), discount=("cost", "sum"), value=("value", "sum")
    )
    members = pd.DataFrame({"member_id": candidates.member_id.unique()})
    out = members.merge(per, on="member_id", how="left").fillna(
        {"offers": 0, "discount": 0.0, "value": 0.0}
    )
    out["offers"] = out.offers.astype(int)
    out["reached"] = out.offers > 0
    return out


def group_report(
    members: pd.DataFrame,
    group_col: str,
    min_group_size: int = 30,
    ratio_threshold: float = FOUR_FIFTHS,
) -> pd.DataFrame:
    """Per-group reach and value, with a reach ratio against the best judged group."""
    g = (
        members.dropna(subset=[group_col])
        .groupby(group_col)
        .agg(
            members=("member_id", "size"),
            reach=("reached", "mean"),
            offers_per_member=("offers", "mean"),
            discount_per_member=("discount", "mean"),
        )
    )
    g["judged"] = g.members >= min_group_size
    best = g.loc[g.judged, "reach"].max() if g.judged.any() else float("nan")
    g["reach_ratio"] = g.reach / best
    g["flag"] = g.judged & (g.reach_ratio < ratio_threshold)
    return g.sort_values("reach_ratio")


def summarise(report: pd.DataFrame) -> dict:
    judged = report[report.judged]
    return {
        "groups": int(len(report)),
        "groups_judged": int(len(judged)),
        "min_reach_ratio": float(judged.reach_ratio.min()) if len(judged) else None,
        "flagged": judged.index[judged.flag].astype(str).tolist(),
    }
