"""Unit tests for ingestion helpers (no real data needed)."""

from pathlib import Path

import duckdb
import pytest

import nectar.ingest.load_raw as load_raw
from nectar.ingest.load_raw import TableSpec, find_raw_file, ingest


def test_find_raw_file_matches_suffix(tmp_path: Path):
    (tmp_path / "transaction_data-ABC123.csv").write_text("x\n1\n")
    (tmp_path / "product-ZZZ.csv").write_text("x\n1\n")
    assert find_raw_file(tmp_path, "transaction_data").name == "transaction_data-ABC123.csv"


def test_find_raw_file_does_not_confuse_prefixes(tmp_path: Path):
    (tmp_path / "coupon-A.csv").write_text("x\n1\n")
    (tmp_path / "coupon_redempt-B.csv").write_text("x\n1\n")
    assert find_raw_file(tmp_path, "coupon").name == "coupon-A.csv"
    assert find_raw_file(tmp_path, "coupon_redempt").name == "coupon_redempt-B.csv"


def test_find_raw_file_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        find_raw_file(tmp_path, "transaction_data")


def test_ingest_end_to_end_on_tiny_fixture(tmp_path: Path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "campaign_desc-X.csv").write_text(
        "DESCRIPTION,CAMPAIGN,START_DAY,END_DAY\nTypeA,1,10,20\nTypeB,2,15,30\n"
    )
    spec = next(s for s in load_raw.TABLES if s.name == "campaigns")
    monkeypatch.setattr(
        load_raw, "TABLES", [TableSpec(spec.name, spec.file_prefix, spec.select_sql)]
    )

    wh = tmp_path / "wh.duckdb"
    counts = ingest(raw_dir=raw, processed_dir=tmp_path / "proc", warehouse_path=wh)

    assert counts == {"campaigns": 2}
    con = duckdb.connect(str(wh), read_only=True)
    cols = [r[0] for r in con.execute("DESCRIBE raw.campaigns").fetchall()]
    assert cols == ["campaign_id", "campaign_type", "start_day", "end_day"]
    assert (tmp_path / "proc" / "campaigns.parquet").exists()
