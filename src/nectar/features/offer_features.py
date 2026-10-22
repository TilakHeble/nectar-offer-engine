"""Point-in-time member x category features for offer targeting.

Every feature is computed from weeks strictly before `cutoff_week`, so anything measured in or
after the offer window can never leak into the features. The same function builds training
features (historical cutoff) and scoring features (latest cutoff).
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import pandas as pd

from nectar.config import WAREHOUSE_PATH


@dataclass(frozen=True)
class Offer:
    offer_id: str
    commodity: str
    discount_pct: float


# 18 widely bought categories across grocery, fresh and household, with three discount depths.
# Depths rotate so each depth covers a mix of category types.
_CATEGORIES = [
    "SOFT DRINKS",
    "CHEESE",
    "BAG SNACKS",
    "BAKED BREAD/BUNS/ROLLS",
    "BEEF",
    "FLUID MILK PRODUCTS",
    "SOUP",
    "CRACKERS/MISC BKD FD",
    "EGGS",
    "TROPICAL FRUIT",
    "CHICKEN",
    "BAKED SWEET GOODS",
    "FD WRAPS/BAGS/TRSH BG",
    "TOMATOES",
    "PNT BTR/JELLY/JAMS",
    "FRUIT - SHELF STABLE",
    "MEAT - SHELF STABLE",
    "WATER - CARBONATED/FLVRD DRINK",
]
_DEPTHS = [0.10, 0.20, 0.30]
OFFERS: list[Offer] = [
    Offer(f"OFR{i + 1:02d}", c, _DEPTHS[i % 3]) for i, c in enumerate(_CATEGORIES)
]

FIRST_STABLE_WEEK = 17
ACTIVE_LOOKBACK_WEEKS = 26  # eligible members shopped in the last 26 weeks before the cutoff
# "Weeks since last category purchase" is capped at a fixed horizon, and members who never
# bought the category get the cap. The cap used to be the feature-window length, which grows
# every week, so the never-bought value crept upward and showed as drift (found by the drift
# monitor). 75 = the training window (weeks 17-91), so training features are unchanged.
WEEKS_SINCE_CAP = 75


def offers_frame(offers: list[Offer] = OFFERS) -> pd.DataFrame:
    return pd.DataFrame([o.__dict__ for o in offers])


def build_offer_features(
    cutoff_week: int,
    offers: list[Offer] = OFFERS,
    con: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """One row per (eligible member, offer) with member- and category-level features.

    Args:
        cutoff_week: first week of the offer window; features use weeks [17, cutoff_week).
    """
    own = con is None
    con = con or duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    con.register("offers", offers_frame(offers))
    lo, hi = FIRST_STABLE_WEEK, cutoff_week - 1
    n_weeks = hi - lo + 1
    D = "DECIMAL(18,6)"  # exact money sums (see the note at the end of this function)
    cap = WEEKS_SINCE_CAP
    sql = f"""
    WITH base AS (
        SELECT * FROM marts.fct_basket WHERE week_no BETWEEN {lo} AND {hi}
    ),
    member AS (
        SELECT member_id,
               COUNT(*) / {n_weeks}.0                        AS m_trips_per_week,
               SUM(basket_value::{D}) / {n_weeks}.0               AS m_spend_per_week,
               {hi} - MAX(week_no)                           AS m_weeks_since_last_trip,
               SUM(member_discount::{D}) / SUM(shelf_value::{D})
                                                             AS m_member_discount_rate,
               SUM((basket_value * private_label_share)::{D}) / SUM(basket_value::{D})
                                                             AS m_private_label_share,
               MEDIAN(basket_value)                          AS m_median_basket,
               COUNT(*) FILTER (WHERE week_no > {hi} - 13) / 13.0 AS m_trips_per_week_13w
        FROM base GROUP BY 1
        HAVING MAX(week_no) > {hi} - {ACTIVE_LOOKBACK_WEEKS}
    ),
    lines AS (
        SELECT l.member_id, l.basket_id, l.week_no, l.sales_value, l.member_discount, o.offer_id
        FROM marts.fct_line l JOIN offers o ON l.commodity = o.commodity
        WHERE l.week_no BETWEEN {lo} AND {hi}
    ),
    cat AS (
        SELECT member_id, offer_id,
               COUNT(DISTINCT basket_id)                     AS c_trips,
               COUNT(DISTINCT basket_id) FILTER (WHERE week_no > {hi} - 13) AS c_trips_13w,
               SUM(sales_value::{D}) / {n_weeks}.0                AS c_spend_per_week,
               SUM(sales_value::{D}) / COUNT(DISTINCT basket_id)  AS c_spend_per_trip,
               {hi} - MAX(week_no)                           AS c_weeks_since_last,
               AVG((member_discount > 0)::INT)               AS c_discount_line_share
        FROM lines GROUP BY 1, 2
    ),
    member_trips AS (
        SELECT member_id, COUNT(*) AS trips,
               COUNT(*) FILTER (WHERE week_no > {hi} - 13) AS trips_13w
        FROM base GROUP BY 1
    ),
    offer_level AS (
        SELECT offer_id,
               SUM(sales_value::{D}) / COUNT(DISTINCT basket_id) AS o_spend_per_trip
        FROM lines GROUP BY 1
    )
    SELECT
        m.member_id, o.offer_id, o.commodity, o.discount_pct,
        m.m_trips_per_week, m.m_spend_per_week, m.m_weeks_since_last_trip,
        m.m_member_discount_rate, m.m_private_label_share, m.m_median_basket,
        m.m_trips_per_week_13w,
        COALESCE(c.c_trips, 0) / mt.trips                             AS c_trip_share,
        COALESCE(c.c_trips_13w, 0) / GREATEST(mt.trips_13w, 1)        AS c_trip_share_13w,
        COALESCE(c.c_spend_per_week, 0)                               AS c_spend_per_week,
        COALESCE(c.c_spend_per_trip, ol.o_spend_per_trip)             AS c_spend_per_trip,
        LEAST(COALESCE(c.c_weeks_since_last, {cap}), {cap})          AS c_weeks_since_last,
        COALESCE(c.c_discount_line_share, m.m_member_discount_rate)   AS c_discount_line_share,
        (c.c_trips IS NOT NULL)                                       AS c_ever_bought,
        ol.o_spend_per_trip
    FROM member m
    JOIN member_trips mt USING (member_id)
    CROSS JOIN offers o
    LEFT JOIN cat c ON c.member_id = m.member_id AND c.offer_id = o.offer_id
    JOIN offer_level ol ON ol.offer_id = o.offer_id
    ORDER BY m.member_id, o.offer_id
    """
    df = con.execute(sql).df()
    con.unregister("offers")
    if own:
        con.close()
    df["c_ever_bought"] = df.c_ever_bought.astype(bool)
    # Money is summed as DECIMAL above: DuckDB sums floats in parallel, so float totals can
    # differ in the last bits between runs and flip values across bin edges. Exact decimal
    # sums make features bit-for-bit reproducible (identical inputs -> identical plans).
    floats = df.select_dtypes(include=["float", "object"]).columns.intersection(
        [c for c in df.columns if c.startswith(("m_", "c_", "o_"))]
    )
    df[floats] = df[floats].astype(float)
    return df


def real_category_purchases(
    start_week: int,
    n_weeks: int,
    offers: list[Offer] = OFFERS,
    con: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """(member_id, offer_id) pairs that really bought the offer category in the window."""
    own = con is None
    con = con or duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    con.register("offers", offers_frame(offers))
    df = con.execute(
        f"""SELECT DISTINCT l.member_id, o.offer_id
            FROM marts.fct_line l JOIN offers o ON l.commodity = o.commodity
            WHERE l.week_no BETWEEN {start_week} AND {start_week + n_weeks - 1}"""
    ).df()
    con.unregister("offers")
    if own:
        con.close()
    return df
