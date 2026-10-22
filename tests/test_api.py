"""API tests against a small synthetic bundle (no warehouse needed, so they run in CI)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nectar.features.schema import OFFER_IDS
from nectar.serving.bundle import Bundle
from nectar.serving.demo_bundle import build_demo_bundle, synthetic_features


@pytest.fixture(scope="module")
def bundle(tmp_path_factory) -> Path:
    return build_demo_bundle(n_members=150).save(tmp_path_factory.mktemp("b") / "artifacts")


@pytest.fixture(scope="module")
def client(bundle, monkeypatch_module):
    monkeypatch_module.setenv("NECTAR_ARTIFACTS_DIR", str(bundle))
    from nectar.serving.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def _row(member_id=1, offer_id=OFFER_IDS[0]) -> dict:
    df = synthetic_features(1)
    r = df[df.offer_id == offer_id].drop(columns="commodity").iloc[0].to_dict()
    r["member_id"] = member_id
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}


def test_health_and_ready(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["bundle_version"] == "demo"
    assert client.get("/ready").status_code == 200
    assert "x-process-time-ms" in client.get("/health").headers


def test_model_returns_manifest(client):
    assert client.get("/model").json()["bundle_version"] == "demo"


def test_member_offers_respect_plan(client, bundle):
    plan = pd.read_parquet(bundle / "candidates.parquet")
    member = int(plan[plan.selected].member_id.value_counts().index[0])
    body = client.get(f"/members/{member}/offers").json()
    expected = set(plan[(plan.member_id == member) & plan.selected].offer_id)
    assert {o["offer_id"] for o in body["offers"]} == expected
    assert len(body["offers"]) <= 3
    values = [o["predicted_incremental_profit"] for o in body["offers"]]
    assert values == sorted(values, reverse=True)


def test_eligible_member_without_offers_gets_empty_list(client):
    r = client.get("/members/150/offers")
    assert r.status_code == 200 and r.json()["offers"] == []


def test_unknown_member_is_404(client):
    assert client.get("/members/99999/offers").status_code == 404


def test_score_matches_bundle(client, bundle):
    rows = [_row(offer_id=o) for o in OFFER_IDS[:5]]
    r = client.post("/score", json={"rows": rows})
    assert r.status_code == 200
    got = pd.DataFrame(r.json()["results"])
    expected = Bundle.load(bundle).score(pd.DataFrame(rows))
    np.testing.assert_allclose(
        got.predicted_incremental_profit, expected.predicted_incremental_profit
    )
    assert (got.recommend == (got.predicted_incremental_profit > 0)).all()


@pytest.mark.parametrize(
    "field, value",
    [
        ("c_trip_share", 1.5),  # share outside [0, 1]
        ("m_spend_per_week", -3.0),  # negative spend
        ("offer_id", "OFR99"),  # unknown offer
        ("discount_pct", 0.5),  # discount depth not on offer
        ("unexpected", 1),  # extra field
    ],
)
def test_invalid_rows_are_rejected(client, field, value):
    row = _row()
    row[field] = value
    assert client.post("/score", json={"rows": [row]}).status_code == 422


def test_duplicate_pairs_are_rejected(client):
    assert client.post("/score", json={"rows": [_row(), _row()]}).status_code == 422


def test_batch_limits(client):
    assert client.post("/score", json={"rows": []}).status_code == 422
    too_many = [_row(member_id=i) for i in range(1, 1002)]
    assert client.post("/score", json={"rows": too_many}).status_code == 422


def test_starts_degraded_without_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("NECTAR_ARTIFACTS_DIR", str(tmp_path / "missing"))
    from nectar.serving.app import app

    with TestClient(app) as c:
        h = c.get("/health")
        assert h.status_code == 200 and h.json()["status"] == "degraded"
        assert c.get("/ready").status_code == 503
        assert c.get("/members/1/offers").status_code == 503


def test_bundle_save_is_atomic_and_reloadable(bundle):
    b = Bundle.load(bundle)
    assert not bundle.with_name(bundle.name + ".tmp").exists()
    assert b.manifest["bundle_version"] == "demo"
    assert b.candidates.selected.dtype == bool
