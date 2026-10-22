"""A small synthetic serving bundle for tests, CI smoke tests and demos.

It needs no warehouse and no solver, so it can be built anywhere the serving image runs:

    python -m nectar.serving.demo_bundle /tmp/artifacts

The models are real (an S-learner and the discount cost model) but trained on synthetic data:
use it to check that the service works, never for offer decisions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nectar.features.schema import OFFER_IDS, model_matrix
from nectar.models.learners import SLearnerModel
from nectar.optimise.cost_model import DiscountCostModel
from nectar.serving.bundle import Bundle


def synthetic_features(n_members: int = 150, seed: int = 0) -> pd.DataFrame:
    """Schema-valid (member, offer) feature rows with plausible ranges."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(1, n_members + 1):
        member = {
            "m_trips_per_week": rng.uniform(0.2, 3),
            "m_spend_per_week": rng.uniform(5, 80),
            "m_weeks_since_last_trip": int(rng.integers(0, 10)),
            "m_member_discount_rate": rng.uniform(0.05, 0.3),
            "m_private_label_share": rng.uniform(0.05, 0.4),
            "m_median_basket": rng.uniform(5, 40),
            "m_trips_per_week_13w": rng.uniform(0, 3),
        }
        for i, oid in enumerate(OFFER_IDS):
            share = rng.uniform(0, 0.5)
            rows.append(
                {
                    "member_id": m,
                    "offer_id": oid,
                    "commodity": f"CATEGORY {i + 1}",
                    "discount_pct": [0.1, 0.2, 0.3][i % 3],
                    **member,
                    "c_trip_share": share,
                    "c_trip_share_13w": rng.uniform(0, 0.5),
                    "c_spend_per_week": share * 3,
                    "c_spend_per_trip": rng.uniform(1, 6),
                    "c_weeks_since_last": int(rng.integers(0, 60)),
                    "c_discount_line_share": rng.uniform(0, 1),
                    "c_ever_bought": bool(share > 0.05),
                    "o_spend_per_trip": 3.0,
                }
            )
    return pd.DataFrame(rows)


def build_demo_bundle(n_members: int = 150, max_per_member: int = 3, seed: int = 0) -> Bundle:
    df = synthetic_features(n_members, seed)
    rng = np.random.default_rng(seed + 1)
    df["treatment"] = rng.integers(0, 2, len(df))
    # Offers help occasional category buyers and lose money on habitual ones.
    effect = 0.3 - 1.2 * df.c_trip_share
    df["gross_profit"] = df.c_trip_share * 2 + df.treatment * effect + rng.normal(0, 0.2, len(df))
    df["discount_cost"] = df.treatment * df.discount_pct * rng.uniform(0, 2, len(df))

    X = model_matrix(df)
    model = SLearnerModel(seed).fit(X, df.treatment.to_numpy(), df.gross_profit.to_numpy())
    cost_model = DiscountCostModel().fit(df)
    cand = df[["member_id", "offer_id", "commodity", "discount_pct"]].copy()
    cand["value"] = model.score(X)
    cand["cost"] = cost_model.predict(df)
    # The last member is eligible but every offer loses money, so it gets none.
    cand.loc[cand.member_id == n_members, "value"] = -1.0
    # Plan without a solver: each member's best positive-value offers, up to the cap.
    rank = cand.groupby("member_id").value.rank(ascending=False, method="first")
    cand["selected"] = (cand.value > 0) & (rank <= max_per_member)
    manifest = {
        "bundle_version": "demo",
        "champion": "S-learner (synthetic demo)",
        "decision_rule": "offer when predicted incremental profit > 0",
        "offer_window_weeks": [102, 105],
        "constraints": {"max_per_member": max_per_member},
        "plan": {
            "offers_sent": int(cand.selected.sum()),
            "eligible_members": n_members,
            "members_reached": int(cand.loc[cand.selected, "member_id"].nunique()),
        },
        "warning": "synthetic demo bundle: for service checks only",
    }
    return Bundle(model, cost_model, cand, manifest)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path, nargs="?", default=Path("artifacts-demo"))
    args = p.parse_args()
    path = build_demo_bundle().save(args.out)
    print(f"demo bundle written to {path}")


if __name__ == "__main__":
    main()
