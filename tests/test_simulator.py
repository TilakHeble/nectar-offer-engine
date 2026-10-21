"""Tests for the offer feature builder and the experiment simulator."""

import duckdb
import numpy as np
import pandas as pd
import pytest

from nectar.config import WAREHOUSE_PATH
from nectar.simulate.simulator import (
    SimConfig,
    calibrate_intercept,
    observable_columns,
    persuadability_index,
    simulate,
)

# ---------- unit tests (no data needed) ----------


def test_calibrate_intercept_hits_target():
    rng = np.random.default_rng(0)
    index = rng.normal(0, 1, 10_000)
    p0 = rng.uniform(0, 0.9, 10_000)
    a = calibrate_intercept(index, p0, target=0.05)
    tau = (1 - p0) / (1 + np.exp(-(a + index)))
    assert abs(tau.mean() - 0.05) < 1e-9


def _toy_features(n=200, discount=0.2):
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "m_member_discount_rate": rng.uniform(0.05, 0.3, n),
            "c_discount_line_share": rng.uniform(0, 1, n),
            "c_ever_bought": rng.random(n) < 0.7,
            "c_weeks_since_last": rng.integers(0, 60, n),
            "m_spend_per_week": rng.uniform(5, 100, n),
            "discount_pct": np.full(n, discount),
        }
    )


def test_deeper_discount_raises_persuadability():
    low = persuadability_index(_toy_features(discount=0.1), np.zeros(200))
    high = persuadability_index(_toy_features(discount=0.3), np.zeros(200))
    assert np.all(high > low)


def test_observable_columns_hide_truth_and_outcomes():
    df = pd.DataFrame(
        columns=[
            "member_id",
            "c_trip_share",
            "treatment",
            "purchased",
            "spend",
            "discount_cost",
            "gross_profit",
            "real_purchased",
            "gt_uplift",
        ]
    )
    assert observable_columns(df) == ["member_id", "c_trip_share"]


# ---------- integration tests on the real warehouse ----------


def _marts_built() -> bool:
    if not WAREHOUSE_PATH.exists():
        return False
    con = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    try:
        return (
            con.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = 'marts' AND table_name = 'fct_line'"
            ).fetchone()[0]
            == 1
        )
    finally:
        con.close()


needs_data = pytest.mark.skipif(not _marts_built(), reason="dbt marts not built")


@pytest.fixture(scope="module")
def sim():
    return simulate(SimConfig())


@needs_data
def test_reproducible_with_seed(sim):
    again, _ = simulate(SimConfig())
    pd.testing.assert_frame_equal(sim[0], again)


@needs_data
def test_baseline_is_calibrated_to_real_behaviour(sim):
    df, diag = sim
    assert abs(df.gt_p0.mean() - df.real_purchased.mean()) < 0.005
    assert diag["baseline_cv_auc"] > 0.75
    # control group behaves like the real members did, offer by offer
    ctrl = df[df.treatment == 0].groupby("offer_id")
    gap = (ctrl.purchased.mean() - ctrl.real_purchased.mean()).abs()
    assert gap.max() < 0.03


@needs_data
def test_randomisation_is_balanced(sim):
    df, _ = sim
    assert abs(df.treatment.mean() - 0.5) < 0.01
    for col in ["m_spend_per_week", "m_member_discount_rate", "c_trip_share", "c_weeks_since_last"]:
        t, c = df.loc[df.treatment == 1, col], df.loc[df.treatment == 0, col]
        smd = abs(t.mean() - c.mean()) / np.sqrt((t.var() + c.var()) / 2)
        assert smd < 0.05, col


@needs_data
def test_ground_truth_is_consistent(sim):
    df, _ = sim
    assert abs(df.gt_uplift.mean() - SimConfig().target_mean_uplift) < 1e-6
    assert (df.gt_p1 >= df.gt_p0).all()
    assert np.allclose(df.gt_uplift, df.gt_p1 - df.gt_p0)


@needs_data
def test_difference_in_means_recovers_true_ate(sim):
    df, _ = sim
    t, c = df[df.treatment == 1].purchased, df[df.treatment == 0].purchased
    est = t.mean() - c.mean()
    se = np.sqrt(t.var() / len(t) + c.var() / len(c))
    assert abs(est - df.gt_uplift.mean()) < 3 * se


@needs_data
def test_features_do_not_use_data_from_the_offer_window():
    """Features built on data truncated at the cutoff equal features built on all data."""
    from nectar.features.offer_features import build_offer_features

    full = build_offer_features(92)
    con = duckdb.connect()
    con.execute(f"ATTACH '{WAREHOUSE_PATH}' AS wh (READ_ONLY)")
    con.execute("CREATE SCHEMA marts")
    con.execute(
        "CREATE VIEW marts.fct_basket AS SELECT * FROM wh.marts.fct_basket WHERE week_no < 92"
    )
    con.execute("CREATE VIEW marts.fct_line AS SELECT * FROM wh.marts.fct_line WHERE week_no < 92")
    truncated = build_offer_features(92, con=con)
    pd.testing.assert_frame_equal(full, truncated)
