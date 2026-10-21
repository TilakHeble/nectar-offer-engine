"""Load Dunnhumby "The Complete Journey" CSVs into typed Parquet files and a DuckDB warehouse.

Business framing used across the project:
    household          -> Nectar member
    coupon / campaign  -> Nectar Prices offer / offer campaign
    basket             -> a shopping trip

Run:
    python -m nectar.ingest.load_raw --raw-dir data/raw
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb

from nectar.config import PROCESSED_DIR, RAW_DIR, WAREHOUSE_PATH

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TableSpec:
    name: str
    file_prefix: str  # raw files may carry a random suffix, e.g. transaction_data-YJ8RJS.csv
    select_sql: str  # typed, renamed projection over the raw CSV (aliased as `src`)


# Explicit types and snake_case names. The Kaggle copy anonymises some demographic labels
# (classification_1..5); we keep them as neutral names and document what is known.
TABLES: list[TableSpec] = [
    TableSpec(
        "transactions",
        "transaction_data",
        """
        SELECT
            CAST(household_key AS INTEGER)      AS household_id,
            CAST(BASKET_ID AS BIGINT)           AS basket_id,
            CAST(DAY AS SMALLINT)               AS day,
            CAST(WEEK_NO AS SMALLINT)           AS week_no,
            CAST(TRANS_TIME AS SMALLINT)        AS trans_time_hhmm,
            CAST(STORE_ID AS INTEGER)           AS store_id,
            CAST(PRODUCT_ID AS INTEGER)         AS product_id,
            CAST(QUANTITY AS INTEGER)           AS quantity,
            CAST(SALES_VALUE AS DOUBLE)         AS sales_value,
            CAST(RETAIL_DISC AS DOUBLE)         AS retail_disc,
            CAST(COUPON_DISC AS DOUBLE)         AS coupon_disc,
            CAST(COUPON_MATCH_DISC AS DOUBLE)   AS coupon_match_disc
        FROM src
        """,
    ),
    TableSpec(
        "products",
        "product",
        """
        SELECT
            CAST(PRODUCT_ID AS INTEGER)         AS product_id,
            CAST(MANUFACTURER AS INTEGER)       AS manufacturer_id,
            TRIM(DEPARTMENT)                    AS department,
            TRIM(BRAND)                         AS brand_type,
            TRIM(COMMODITY_DESC)                AS commodity,
            TRIM(SUB_COMMODITY_DESC)            AS sub_commodity,
            NULLIF(TRIM(CURR_SIZE_OF_PRODUCT), '') AS pack_size
        FROM src
        """,
    ),
    TableSpec(
        "households",
        "hh_demographic",
        """
        SELECT
            CAST(household_key AS INTEGER)      AS household_id,
            classification_1                    AS age_group,
            classification_2                    AS marital_group,
            classification_3                    AS income_level,
            classification_4                    AS household_size,
            classification_5                    AS hh_composition_group,
            HOMEOWNER_DESC                      AS homeowner,
            KID_CATEGORY_DESC                   AS kids
        FROM src
        """,
    ),
    TableSpec(
        "campaigns",
        "campaign_desc",
        """
        SELECT
            CAST(CAMPAIGN AS SMALLINT)          AS campaign_id,
            DESCRIPTION                         AS campaign_type,
            CAST(START_DAY AS SMALLINT)         AS start_day,
            CAST(END_DAY AS SMALLINT)           AS end_day
        FROM src
        """,
    ),
    TableSpec(
        "campaign_members",
        "campaign_table",
        """
        SELECT
            CAST(household_key AS INTEGER)      AS household_id,
            CAST(CAMPAIGN AS SMALLINT)          AS campaign_id,
            DESCRIPTION                         AS campaign_type
        FROM src
        """,
    ),
    TableSpec(
        "coupons",
        "coupon",
        """
        SELECT
            CAST(COUPON_UPC AS BIGINT)          AS coupon_upc,
            CAST(PRODUCT_ID AS INTEGER)         AS product_id,
            CAST(CAMPAIGN AS SMALLINT)          AS campaign_id
        FROM src
        """,
    ),
    TableSpec(
        "coupon_redemptions",
        "coupon_redempt",
        """
        SELECT
            CAST(household_key AS INTEGER)      AS household_id,
            CAST(DAY AS SMALLINT)               AS day,
            CAST(COUPON_UPC AS BIGINT)          AS coupon_upc,
            CAST(CAMPAIGN AS SMALLINT)          AS campaign_id
        FROM src
        """,
    ),
    TableSpec(
        "promotions",
        "causal_data",
        """
        SELECT
            CAST(PRODUCT_ID AS INTEGER)         AS product_id,
            CAST(STORE_ID AS INTEGER)           AS store_id,
            CAST(WEEK_NO AS SMALLINT)           AS week_no,
            CAST(display AS VARCHAR)            AS display_code,
            CAST(mailer AS VARCHAR)             AS mailer_code
        FROM src
        """,
    ),
]


def find_raw_file(raw_dir: Path, prefix: str) -> Path:
    """Return the single CSV in raw_dir whose name starts with `prefix` + '-' or '.'."""
    matches = sorted(
        p for p in raw_dir.glob("*.csv") if p.stem == prefix or p.stem.startswith(prefix + "-")
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one CSV for '{prefix}' in {raw_dir}, found {matches}")
    return matches[0]


def ingest(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    warehouse_path: Path = WAREHOUSE_PATH,
) -> dict[str, int]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    warehouse_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(warehouse_path))
    con.execute("SET enable_progress_bar = false")
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    row_counts: dict[str, int] = {}

    for spec in TABLES:
        t0 = time.time()
        csv_path = find_raw_file(raw_dir, spec.file_prefix)
        parquet_path = processed_dir / f"{spec.name}.parquet"
        src = f"read_csv('{csv_path.as_posix()}', header=true, all_varchar=true)"
        query = spec.select_sql.replace("FROM src", f"FROM {src} AS src")
        con.execute(
            f"COPY ({query}) TO '{parquet_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        con.execute(
            f"CREATE OR REPLACE TABLE raw.{spec.name} AS "
            f"SELECT * FROM read_parquet('{parquet_path.as_posix()}')"
        )
        n = con.execute(f"SELECT COUNT(*) FROM raw.{spec.name}").fetchone()[0]
        row_counts[spec.name] = n
        log.info("%-20s %12s rows  %5.1fs", spec.name, f"{n:,}", time.time() - t0)

    con.close()
    return row_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ingest(raw_dir=args.raw_dir)


if __name__ == "__main__":
    main()
