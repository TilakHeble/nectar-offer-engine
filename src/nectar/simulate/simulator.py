"""Randomised Nectar-offer experiment simulator with a known ground truth.

Why simulate: the real campaigns were sent to heavier shoppers (notebook 01), so their effect
cannot be separated from who was chosen. Here every eligible (member, offer) pair is randomly
assigned to treatment or control, and outcomes come from a model whose true uplift we know.

How the simulated world is built
--------------------------------
1. Baseline (no offer): p0(x) = probability the member buys the offer category in the 4-week
   offer window. It is a logistic regression fitted to the REAL purchases in weeks 92-95, so the
   control group behaves like the real members did.
2. Persuadability: c(x) = probability that an offer converts a member who would NOT otherwise
   have bought. Driven by price sensitivity, category familiarity, lapsed category buying,
   overall spend level and discount depth, plus an unobserved member-level component.
3. Potential outcomes: Y(0) ~ Bernoulli(p0); Y(1) = Y(0) OR Bernoulli(c). True conversion uplift
   is therefore tau(x) = (1 - p0) * c. Habitual buyers ("sure things") have high p0 and little
   room to move; members who never buy the category have low c ("lost causes").
4. Money: buyers spend their usual category amount (with noise); treated buyers stock up a
   little. Half of each discount is funded by the brand (as with supplier-funded Nectar Prices),
   the rest by the retailer. Profit = margin x spend - retailer discount cost, so a discount
   given to a sure thing is pure margin loss.

Ground-truth columns are prefixed `gt_` and must never be used as model inputs.

Run:
    python -m nectar.simulate.simulator
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nectar.config import PROCESSED_DIR, WAREHOUSE_PATH
from nectar.features.offer_features import build_offer_features, real_category_purchases

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimConfig:
    cutoff_week: int = 92  # offer window starts here; features use earlier weeks only
    window_weeks: int = 4  # offer validity / outcome window
    treat_share: float = 0.5  # randomisation ratio
    target_mean_uplift: float = 0.04  # average conversion uplift (4 percentage points)
    margin: float = 0.30  # assumed grocery gross margin on shelf value
    supplier_funding: float = 0.5  # share of the discount funded by the brand (Nectar360-style)
    stock_up_per_discount: float = 0.5  # treated buyers spend +5% per 10% of discount
    spend_noise_sigma: float = 0.4
    member_effect_sd: float = 0.5  # unobserved member-level persuadability (not in features)
    seed: int = 42


# Coefficients of the persuadability index (logit scale), on standardised inputs.
PERSUADABILITY_WEIGHTS = {
    "z_member_discount_rate": 1.2,  # members who rely on member prices respond more
    "z_category_discount_share": 0.6,  # ...especially if they buy this category on deal
    "ever_bought": 0.8,  # some familiarity with the category is needed
    "lapsed_category": 0.8,  # bought before but not in the last 4-26 weeks: win-back
    "z_log_spend": -0.5,  # heavy spenders are less price-driven
    "discount_depth": 8.0,  # per unit of discount (0.1 -> 0.3 adds 1.6 on logit scale)
}

BASELINE_NUMERIC = [
    "log_c_trip_share",
    "log_c_trip_share_13w",
    "log_m_trips_13w",
    "log_c_weeks_since_last",
    "log_m_weeks_since_last_trip",
    "c_ever_bought",
]


def _baseline_design(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "log_c_trip_share": np.log(df.c_trip_share + 0.01),
            "log_c_trip_share_13w": np.log(df.c_trip_share_13w + 0.01),
            "log_m_trips_13w": np.log1p(df.m_trips_per_week_13w),
            "log_c_weeks_since_last": np.log1p(df.c_weeks_since_last),
            "log_m_weeks_since_last_trip": np.log1p(df.m_weeks_since_last_trip),
            "c_ever_bought": df.c_ever_bought.astype(float),
            "offer_id": df.offer_id,
        }
    )


def _baseline_model() -> object:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), BASELINE_NUMERIC),
            ("offer", OneHotEncoder(drop="first"), ["offer_id"]),
        ]
    )
    return make_pipeline(pre, LogisticRegression(C=1.0, max_iter=2000))


def fit_baseline(features: pd.DataFrame, y_real: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fit p0 on real purchases. Returns in-world p0 and calibration diagnostics."""
    X = _baseline_design(features)
    model = _baseline_model()
    cv_pred = cross_val_predict(
        model, X, y_real, cv=GroupKFold(5), groups=features.member_id, method="predict_proba"
    )[:, 1]
    model.fit(X, y_real)
    p0 = model.predict_proba(X)[:, 1]
    diag = {
        "real_purchase_rate": float(y_real.mean()),
        "baseline_mean_p0": float(p0.mean()),
        "baseline_cv_auc": float(roc_auc_score(y_real, cv_pred)),
    }
    return p0, diag


def _z(x: pd.Series) -> pd.Series:
    return (x - x.mean()) / x.std()


def persuadability_index(features: pd.DataFrame, member_effect: np.ndarray) -> np.ndarray:
    w = PERSUADABILITY_WEIGHTS
    lapsed = features.c_ever_bought & features.c_weeks_since_last.between(4, 26)
    return (
        w["z_member_discount_rate"] * _z(features.m_member_discount_rate)
        + w["z_category_discount_share"] * _z(features.c_discount_line_share)
        + w["ever_bought"] * features.c_ever_bought.astype(float)
        + w["lapsed_category"] * lapsed.astype(float)
        + w["z_log_spend"] * _z(np.log1p(features.m_spend_per_week))
        + w["discount_depth"] * (features.discount_pct - 0.2)
        + member_effect
    ).to_numpy()


def calibrate_intercept(index: np.ndarray, p0: np.ndarray, target: float) -> float:
    """Intercept that makes the mean true uplift equal `target`."""
    return brentq(lambda a: np.mean((1 - p0) * expit(a + index)) - target, -15, 5)


def simulate(
    cfg: SimConfig | None = None, con: duckdb.DuckDBPyConnection | None = None
) -> tuple[pd.DataFrame, dict]:
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    features = build_offer_features(cfg.cutoff_week, con=con)
    real = real_category_purchases(cfg.cutoff_week, cfg.window_weeks, con=con)
    y_real = (
        features[["member_id", "offer_id"]]
        .merge(real.assign(y=1), how="left", on=["member_id", "offer_id"])
        .y.fillna(0)
        .to_numpy()
    )

    # 1. Baseline calibrated to real behaviour.
    p0, diag = fit_baseline(features, y_real)

    # 2. Persuadability with an unobserved member-level component.
    members = features.member_id.unique()
    effect = pd.Series(rng.normal(0, cfg.member_effect_sd, len(members)), index=members)
    index = persuadability_index(features, features.member_id.map(effect).to_numpy())
    intercept = calibrate_intercept(index, p0, cfg.target_mean_uplift)
    c = expit(intercept + index)
    p1 = p0 + (1 - p0) * c
    tau = p1 - p0

    # 3. Random assignment and potential outcomes (common random numbers).
    n = len(features)
    treatment = rng.random(n) < cfg.treat_share
    y0 = rng.random(n) < p0
    y1 = y0 | (rng.random(n) < c)
    purchased = np.where(treatment, y1, y0)

    # 4. Money. Expected spend given purchase = usual spend per category trip x expected trips.
    exp_trips = np.maximum(
        1.0, cfg.window_weeks * features.c_trip_share * features.m_trips_per_week
    )
    exp_spend = (features.c_spend_per_trip * exp_trips).to_numpy()
    noise = rng.lognormal(-(cfg.spend_noise_sigma**2) / 2, cfg.spend_noise_sigma, n)
    lift = 1 + cfg.stock_up_per_discount * features.discount_pct.to_numpy()
    spend = purchased * exp_spend * noise * np.where(treatment, lift, 1.0)
    retailer_rate = (1 - cfg.supplier_funding) * features.discount_pct.to_numpy()
    discount_cost = treatment * retailer_rate * spend
    gross_profit = cfg.margin * spend - discount_cost

    exp_profit0 = cfg.margin * p0 * exp_spend
    exp_profit1 = (cfg.margin - retailer_rate) * p1 * exp_spend * lift
    out = features.assign(
        treatment=treatment.astype(int),
        purchased=purchased.astype(int),
        spend=spend,
        discount_cost=discount_cost,
        gross_profit=gross_profit,
        real_purchased=y_real.astype(int),
        gt_p0=p0,
        gt_p1=p1,
        gt_persuadability=c,
        gt_uplift=tau,
        gt_expected_spend=exp_spend,
        gt_profit_uplift=exp_profit1 - exp_profit0,
    )
    diag.update(
        {
            "rows": n,
            "members": int(len(members)),
            "offers": int(features.offer_id.nunique()),
            "persuadability_intercept": float(intercept),
            "mean_true_uplift": float(tau.mean()),
            "share_negative_profit_uplift": float((out.gt_profit_uplift < 0).mean()),
            "config": asdict(cfg),
        }
    )
    return out, diag


def write(df: pd.DataFrame, warehouse_path=WAREHOUSE_PATH) -> None:
    path = PROCESSED_DIR / "offer_experiment.parquet"
    df.to_parquet(path, index=False)
    con = duckdb.connect(str(warehouse_path))
    con.execute("CREATE SCHEMA IF NOT EXISTS sim")
    con.execute(
        f"CREATE OR REPLACE TABLE sim.offer_experiment AS SELECT * FROM read_parquet('{path}')"
    )
    con.close()


def observable_columns(df: pd.DataFrame) -> list[str]:
    """Columns a model may use: everything except ground truth and outcomes."""
    hidden = {"treatment", "purchased", "spend", "discount_cost", "gross_profit", "real_purchased"}
    return [c for c in df.columns if not c.startswith("gt_") and c not in hidden]


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a randomised offer experiment.")
    parser.add_argument("--seed", type=int, default=SimConfig.seed)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df, diag = simulate(SimConfig(seed=args.seed))
    write(df)
    for k, v in diag.items():
        log.info("%-30s %s", k, v)


if __name__ == "__main__":
    main()
