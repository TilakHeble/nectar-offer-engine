"""Data-quality contract for the ingested warehouse.

These run against data/warehouse/nectar.duckdb and are skipped if it has not been built
(e.g. in CI, where raw data is not committed).
"""

import duckdb
import pytest

from nectar.config import WAREHOUSE_PATH

pytestmark = pytest.mark.skipif(not WAREHOUSE_PATH.exists(), reason="warehouse not built")


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    yield c
    c.close()


def scalar(con, sql):
    return con.execute(sql).fetchone()[0]


@pytest.mark.parametrize(
    "table, expected_rows",
    [
        ("transactions", 2_595_732),
        ("products", 92_353),
        ("households", 801),
        ("campaigns", 30),
        ("campaign_members", 7_208),
        ("coupons", 124_548),
        ("coupon_redemptions", 2_318),
        ("promotions", 36_786_524),
    ],
)
def test_row_counts(con, table, expected_rows):
    assert scalar(con, f"SELECT COUNT(*) FROM raw.{table}") == expected_rows


@pytest.mark.parametrize(
    "table, key",
    [
        ("products", "product_id"),
        ("households", "household_id"),
        ("campaigns", "campaign_id"),
    ],
)
def test_primary_keys_unique_and_not_null(con, table, key):
    total, distinct, nulls = con.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT {key}), COUNT(*) - COUNT({key}) FROM raw.{table}"
    ).fetchone()
    assert nulls == 0
    assert total == distinct


def test_no_duplicate_transaction_lines(con):
    dupes = scalar(
        con,
        """
        SELECT COUNT(*) FROM (
            SELECT basket_id, product_id, COUNT(*) n
            FROM raw.transactions GROUP BY ALL HAVING n > 1
        )""",
    )
    assert dupes == 0


def test_each_basket_belongs_to_one_household(con):
    bad = scalar(
        con,
        """
        SELECT COUNT(*) FROM (
            SELECT basket_id FROM raw.transactions
            GROUP BY 1 HAVING COUNT(DISTINCT household_id) > 1
        )""",
    )
    assert bad == 0


def test_transactions_reference_known_products(con):
    orphans = scalar(
        con, "SELECT COUNT(*) FROM raw.transactions ANTI JOIN raw.products USING (product_id)"
    )
    assert orphans == 0


def test_discounts_are_non_positive(con):
    positives = scalar(
        con,
        """SELECT COUNT(*) FROM raw.transactions
           WHERE retail_disc > 0 OR coupon_disc > 0 OR coupon_match_disc > 0""",
    )
    # A handful of positive retail_disc values exist in the source; keep them visible, cap them.
    assert positives < 0.001 * scalar(con, "SELECT COUNT(*) FROM raw.transactions")


def test_day_and_week_ranges(con):
    lo, hi, wlo, whi = con.execute(
        "SELECT MIN(day), MAX(day), MIN(week_no), MAX(week_no) FROM raw.transactions"
    ).fetchone()
    assert (lo, hi) == (1, 711)
    assert (wlo, whi) == (1, 102)


def test_campaign_windows_valid(con):
    assert scalar(con, "SELECT COUNT(*) FROM raw.campaigns WHERE end_day < start_day") == 0


def test_redemptions_link_to_campaigns(con):
    orphans = scalar(
        con,
        "SELECT COUNT(*) FROM raw.coupon_redemptions ANTI JOIN raw.campaigns USING (campaign_id)",
    )
    assert orphans == 0
