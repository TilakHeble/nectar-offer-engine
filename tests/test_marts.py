"""Checks on the dbt marts. Skipped until `make dbt` has built them."""

import duckdb
import pytest

from nectar.config import WAREHOUSE_PATH


def _marts_built() -> bool:
    if not WAREHOUSE_PATH.exists():
        return False
    con = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    try:
        n = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'marts'"
        ).fetchone()[0]
    finally:
        con.close()
    return n > 0


pytestmark = pytest.mark.skipif(not _marts_built(), reason="dbt marts not built")


@pytest.fixture(scope="module")
def con():
    c = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    yield c
    c.close()


def scalar(con, sql):
    return con.execute(sql).fetchone()[0]


def test_every_member_has_one_segment(con):
    members = scalar(con, "SELECT COUNT(*) FROM marts.dim_member")
    segmented = scalar(con, "SELECT COUNT(DISTINCT member_id) FROM marts.mart_member_rfm")
    assert members == segmented == 2500


def test_basket_spend_reconciles_with_lines(con):
    lines = scalar(con, "SELECT SUM(sales_value) FROM marts.fct_line")
    baskets = scalar(con, "SELECT SUM(basket_value) FROM marts.fct_basket")
    assert abs(lines - baskets) < 0.01


def test_champions_outspend_lost(con):
    med = dict(
        con.execute(
            "SELECT segment, MEDIAN(monetary) FROM marts.mart_member_rfm GROUP BY 1"
        ).fetchall()
    )
    assert med["Champions"] > med["Loyal"] * 0.8 > med["Lost"]


def test_every_targeted_pair_is_in_funnel(con):
    targeted = scalar(con, "SELECT COUNT(*) FROM raw.campaign_members")
    in_funnel = scalar(con, "SELECT COUNT(*) FROM marts.fct_member_campaign WHERE is_targeted")
    assert targeted == in_funnel
