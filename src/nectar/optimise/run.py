"""Allocate offers to the held-out test members under a discount budget.

Uses the champion's predicted incremental profit as `value` and the cost model's expected
discount cost as `cost`, then solves the allocation and reports predicted and (simulation-only)
true results.

Run:
    python -m nectar.optimise.run --budget 150 --max-per-member 99
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from nectar.config import PROCESSED_DIR, ROOT
from nectar.models.train import load_experiment, split_by_member
from nectar.optimise.allocate import Constraints, solve_greedy, solve_ilp
from nectar.optimise.cost_model import DiscountCostModel
from nectar.simulate.simulator import SimConfig

log = logging.getLogger(__name__)


def true_expected_cost(df: pd.DataFrame, cfg: SimConfig | None = None) -> np.ndarray:
    """Simulation ground truth for the retailer discount cost of offering each pair."""
    cfg = cfg or SimConfig()
    d = df.discount_pct.to_numpy()
    lift = 1 + cfg.stock_up_per_discount * d
    return (
        (1 - cfg.supplier_funding)
        * d
        * df.gt_p1.to_numpy()
        * df.gt_expected_spend.to_numpy()
        * lift
    )


def build_candidates(seed: int = 0) -> pd.DataFrame:
    """Test-split candidates with champion value and predicted cost (plus truth for evaluation)."""
    meta = json.loads((ROOT / "models" / "champion.json").read_text())
    champion = meta["champion"]
    df = load_experiment()
    df["split"] = split_by_member(df, seed)
    cost_model = DiscountCostModel().fit(df[df.split == "train"])

    scores = pd.read_parquet(PROCESSED_DIR / "holdout_scores.parquet")
    test = scores[scores.split == "test"].copy()
    test["value"] = test[f"score__{champion}"]
    test["cost"] = cost_model.predict(test)
    test["true_cost"] = true_expected_cost(test)
    test.attrs["champion"] = champion
    return test


def evaluate(candidates: pd.DataFrame, selected: np.ndarray) -> dict:
    return {
        "true_incremental_profit": float(candidates.gt_profit_uplift.to_numpy()[selected].sum()),
        "true_discount_cost": float(candidates.true_cost.to_numpy()[selected].sum()),
        "members_reached": int(pd.unique(candidates.member_id.to_numpy()[selected]).size),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--budget", type=float, default=150.0)
    p.add_argument("--max-per-member", type=int, default=99)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cand = build_candidates()
    c = Constraints(budget=args.budget, max_per_member=args.max_per_member)
    rows = []
    for solver in (solve_ilp, solve_greedy):
        alloc = solver(cand, c)
        rows.append({**alloc.summary(), **evaluate(cand, alloc.selected)})
        if alloc.method == "ilp":
            out = cand.loc[
                alloc.selected,
                ["member_id", "offer_id", "commodity", "discount_pct", "value", "cost"],
            ]
            out.to_parquet(PROCESSED_DIR / "allocation.parquet", index=False)
    with pd.option_context("display.float_format", "{:,.2f}".format, "display.width", 160):
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
