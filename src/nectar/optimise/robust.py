"""Risk-aware offer allocation: correcting the optimiser's curse.

The problem
-----------
The optimiser sends the offers with the highest *predicted* incremental profit. Predictions are
noisy, so the top of the ranking is enriched with pairs whose noise happened to be positive:
the plan systematically overstates its own profit (week 7: by 34% at a $100 budget) and wastes
budget on pairs that only *look* good.

This is the same problem as estimation error in mean-variance portfolio optimisation, where
maximising on sample means "maximises estimation error". The remedies carry over:

1. Measure uncertainty. Refit the champion on B member-level bootstrap resamples of the
   training data: each pair gets a mean prediction mu and a spread sigma.
2. Shrink noisy estimates (empirical Bayes / James-Stein). Treat each pair's true effect as
   drawn from a prior N(m, tau^2) with tau^2 = var(mu) - mean(sigma^2), the spread of mu that
   noise does not explain. The posterior mean is

       mu_shrunk = m + (mu - m) * tau^2 / (tau^2 + sigma^2)

   so confident estimates move little, and noisy ones are pulled towards the average.
3. Or be explicitly risk-averse: rank by a lower confidence bound, mu - z * sigma.

Nothing here uses the ground truth. The risk parameter z is chosen on the validation split
with the same observed-outcome estimate used to select the champion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nectar.models.learners import XLearnerModel


@dataclass
class BaggedUplift:
    """B models refit on member-level bootstrap resamples; predicts mean and spread."""

    n_models: int = 30
    seed: int = 0

    def fit(self, X: pd.DataFrame, t: np.ndarray, y: np.ndarray, groups: np.ndarray):
        rng = np.random.default_rng(self.seed)
        members = np.unique(groups)
        rows = pd.Series(np.arange(len(groups))).groupby(groups).apply(np.array)
        self.models_ = []
        for b in range(self.n_models):
            pick = rng.choice(members, size=len(members), replace=True)
            idx = np.concatenate(rows.loc[pick].to_numpy())
            m = XLearnerModel(seed=self.seed + b)
            m.fit(X.iloc[idx].reset_index(drop=True), t[idx], y[idx])
            self.models_.append(m)
        return self

    def predict_all(self, X: pd.DataFrame) -> np.ndarray:
        """Array of shape (n_models, n_rows)."""
        return np.vstack([m.score(X) for m in self.models_])

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        p = self.predict_all(X)
        return p.mean(axis=0), p.std(axis=0, ddof=1)


def shrink(
    mu: np.ndarray,
    sigma: np.ndarray,
    prior_mean: float | None = None,
    prior_var: float | None = None,
) -> np.ndarray:
    """Empirical-Bayes posterior mean of each pair's effect under a normal prior.

    The prior is estimated from the predictions themselves unless given: its mean is the
    average prediction and its variance is the part of their spread noise does not explain.
    Pass the prior estimated on training data to apply it unchanged to new pairs.
    """
    mu, sigma = np.asarray(mu, float), np.asarray(sigma, float)
    m = float(np.mean(mu)) if prior_mean is None else prior_mean
    if prior_var is None:
        prior_var = max(float(np.var(mu) - np.mean(sigma**2)), 1e-12)
    weight = prior_var / (prior_var + sigma**2)
    return m + (mu - m) * weight


def lower_bound(mu: np.ndarray, sigma: np.ndarray, z: float) -> np.ndarray:
    """Risk-averse value: mu - z * sigma (z = 0 is risk-neutral)."""
    return np.asarray(mu, float) - z * np.asarray(sigma, float)


def adjusted_values(mu, sigma, method: str, z: float = 1.0, prior=None) -> np.ndarray:
    if method == "mean":
        return np.asarray(mu, float)
    if method == "shrink":
        pm, pv = prior if prior else (None, None)
        return shrink(mu, sigma, pm, pv)
    if method == "lcb":
        return lower_bound(mu, sigma, z)
    raise ValueError(f"unknown method {method!r}")


# ---------- evaluation and selection (CLI) ----------

Z_GRID = (0.25, 0.5, 1.0, 1.5, 2.0)
EVAL_BUDGETS = (25, 50, 100, 150, 200, 300)
SELECT_BUDGETS = (50, 100, 150)
CAPS = (3, 99)


def _plan(part: pd.DataFrame, values: np.ndarray, budget: float, cap: int) -> np.ndarray:
    from nectar.optimise.allocate import Constraints, solve_ilp

    c = part[["member_id", "offer_id"]].copy()
    c["value"], c["cost"] = values, part.cost.to_numpy()
    return solve_ilp(c, Constraints(budget=budget, max_per_member=cap)).selected


def evaluate(parts: dict[str, pd.DataFrame], prior: tuple[float, float]) -> pd.DataFrame:
    """Plans for every method x budget x cap on each split.

    `forecast` is the plan's own profit forecast: the raw prediction for the point and mean
    methods, and the empirical-Bayes posterior mean for the risk-adjusted ones.
    """
    from nectar.evaluate import metrics as M

    rows = []
    for split, p in parts.items():
        mu, sd = p.mu.to_numpy(), p.sd.to_numpy()
        post = shrink(mu, sd, *prior)
        methods = {
            "point": (p.point.to_numpy(), p.point.to_numpy()),
            "bagged mean": (mu, mu),
            "shrink": (post, post),
        }
        for z in Z_GRID:
            methods[f"lcb z={z}"] = (lower_bound(mu, sd, z), post)
        for cap in CAPS:
            for b in EVAL_BUDGETS:
                for name, (values, forecast) in methods.items():
                    sel = _plan(p, values, b, cap)
                    row = {
                        "split": split,
                        "cap": cap,
                        "budget": b,
                        "method": name,
                        "offers": int(sel.sum()),
                        "forecast": float(forecast[sel].sum()),
                        "observed_profit": M.incremental_policy_value(
                            sel, p.treatment.to_numpy(), p.gross_profit.to_numpy()
                        ),
                    }
                    if "gt_profit_uplift" in p:
                        row["true_profit"] = float(p.gt_profit_uplift.to_numpy()[sel].sum())
                    rows.append(row)
    return pd.DataFrame(rows)


def choose_z(results: pd.DataFrame) -> float:
    """Risk level with the best average observed (IPW) profit on validation, no ground truth."""
    v = results[
        (results.split == "valid")
        & results.method.str.startswith("lcb")
        & results.budget.isin(SELECT_BUDGETS)
    ]
    best = v.groupby("method")["observed_profit"].mean().idxmax()
    return float(best.split("=")[1])


def main() -> None:
    import json
    import logging
    import os
    import time

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import joblib

    from nectar.config import PROCESSED_DIR, ROOT
    from nectar.features.schema import model_matrix
    from nectar.models.train import load_experiment, split_by_member
    from nectar.optimise.cost_model import DiscountCostModel

    # Import the class by its package path: run as `python -m`, this file is `__main__`, and a
    # model pickled from `__main__.BaggedUplift` cannot be loaded by any other program.
    from nectar.optimise.robust import BaggedUplift as Bagged

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger(__name__)
    t0 = time.time()
    df = load_experiment()
    df["split"] = split_by_member(df, 0)
    X = model_matrix(df)
    tr = (df.split == "train").to_numpy()
    bag = Bagged().fit(
        X[tr],
        df.treatment.to_numpy()[tr],
        df.gross_profit.to_numpy()[tr],
        df.member_id.to_numpy()[tr],
    )
    mu_tr, sd_tr = bag.predict(X[tr])
    prior = (float(mu_tr.mean()), max(float(mu_tr.var() - (sd_tr**2).mean()), 1e-12))
    cost = DiscountCostModel().fit(df[tr])
    champion = json.loads((ROOT / "models" / "champion.json").read_text())["champion"]
    scores = pd.read_parquet(PROCESSED_DIR / "holdout_scores.parquet")
    parts = {}
    for split in ("valid", "test"):
        p = df[df.split == split].copy()
        s = scores[scores.split == split].set_index(["member_id", "offer_id"])[f"score__{champion}"]
        p["point"] = s.reindex(pd.MultiIndex.from_frame(p[["member_id", "offer_id"]])).to_numpy()
        p["mu"], p["sd"] = bag.predict(model_matrix(p))
        p["cost"] = cost.predict(p)
        parts[split] = p
    results = evaluate(parts, prior)
    results.to_csv(ROOT / "models" / "robust_eval.csv", index=False)
    joblib.dump(bag, ROOT / "models" / "bagged.joblib")
    z = choose_z(results)
    meta = {
        "method": "lcb",
        "z": z,
        "n_models": bag.n_models,
        "prior_mean": prior[0],
        "prior_var": prior[1],
        "selected_on": "validation observed incremental profit, budgets "
        f"{list(SELECT_BUDGETS)}, caps {list(CAPS)}",
    }
    (ROOT / "models" / "robust.json").write_text(json.dumps(meta, indent=2))
    log.info(
        "Chosen risk level z=%s (%.0fs). Saved models/bagged.joblib, robust.json",
        z,
        time.time() - t0,
    )


if __name__ == "__main__":
    main()
