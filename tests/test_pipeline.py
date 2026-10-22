"""Weekly pipeline gates and publishing, with the heavy steps replaced by synthetic stand-ins."""

import csv
import json
import os

import pytest

import nectar.pipeline.weekly as W
from nectar.monitoring.drift import ReferenceProfile, compare
from nectar.serving.demo_bundle import build_demo_bundle, synthetic_features

REF = synthetic_features(2000, seed=0)
PROFILE = ReferenceProfile.fit(REF, source="synthetic reference")


@pytest.fixture
def cfg(tmp_path):
    return W.CycleConfig(
        artifacts_dir=tmp_path / "artifacts",
        runs_dir=tmp_path / "runs",
        history_dir=tmp_path / "history",
        reference_path=tmp_path / "ref.json",
        budget=1e9,
    )


@pytest.fixture
def fake_steps(monkeypatch):
    """Swap data-dependent steps for synthetic ones; `state` controls the scenario."""
    state = {"spend_shift": 1.0, "members": 150}

    def features(cutoff_week):
        f = synthetic_features(2000, seed=1)
        f["m_spend_per_week"] *= state["spend_shift"]
        return f

    def drift(profile, features, cutoff_week, out_dir):
        rep = compare(profile, features, label=f"week {cutoff_week}")
        rep.save(out_dir)
        return rep

    def plan(cfg, features):
        b = build_demo_bundle(n_members=state["members"])
        b.manifest["bundle_version"] = f"v-{state['members']}"
        b.manifest["plan"]["expected_cost"] = 1.0
        return b

    monkeypatch.setattr(W, "load_or_build_reference", lambda path: PROFILE)
    monkeypatch.setattr(W, "build_features", features)
    monkeypatch.setattr(W, "check_drift", drift)
    monkeypatch.setattr(W, "build_plan", plan)
    return state


def test_happy_path_publishes_and_archives(cfg, fake_steps):
    rec = W.run_cycle(cfg)
    assert rec.status == "published", rec.message
    assert (cfg.artifacts_dir / "manifest.json").exists()
    assert (cfg.history_dir / rec.published_version / "manifest.json").exists()
    run_dir = cfg.runs_dir / rec.run_id
    assert {p.name for p in run_dir.iterdir()} == {
        "run.json",
        "drift_report.json",
        "drift_report.html",
    }
    manifest = json.loads((cfg.artifacts_dir / "manifest.json").read_text())
    assert manifest["pipeline_run_id"] == rec.run_id
    steps = [s["step"] for s in rec.steps]
    assert steps == ["reference", "features", "drift", "plan", "plan_checks", "publish"]


def test_drift_alert_blocks_and_keeps_live_plan(cfg, fake_steps):
    first = W.run_cycle(cfg)
    live_before = (cfg.artifacts_dir / "manifest.json").read_text()
    fake_steps["spend_shift"] = 1.8  # members suddenly spend 80% more
    rec = W.run_cycle(cfg)
    assert rec.status == "blocked"
    assert "m_spend_per_week" in rec.message
    assert (cfg.artifacts_dir / "manifest.json").read_text() == live_before
    assert first.published_version and rec.published_version is None
    assert "plan" not in [s["step"] for s in rec.steps]


def test_force_publishes_despite_alert(cfg, fake_steps):
    fake_steps["spend_shift"] = 1.8
    cfg.force = True
    rec = W.run_cycle(cfg)
    assert rec.status.startswith("published")


def test_large_plan_change_is_blocked(cfg, fake_steps):
    assert W.run_cycle(cfg).status == "published"
    fake_steps["members"] = 40  # far fewer offers than the live plan
    rec = W.run_cycle(cfg)
    assert rec.status == "blocked" and "offers sent changed" in rec.message


def test_plan_checks_catch_budget_and_cap_violations(cfg):
    b = build_demo_bundle(n_members=50, max_per_member=3)
    b.manifest["plan"]["expected_cost"] = 10.0
    cfg.budget, cfg.max_per_member = 5.0, 2
    problems = W.check_plan(b, cfg)
    assert any("exceeds budget" in p for p in problems)
    assert any("per-member offer cap" in p for p in problems)


def test_history_csv_records_every_run(cfg, fake_steps):
    W.run_cycle(cfg)
    fake_steps["spend_shift"] = 1.8
    W.run_cycle(cfg)
    rows = list(csv.DictReader((cfg.runs_dir / "history.csv").open()))
    assert [r["status"] for r in rows] == ["published", "blocked"]
    assert rows[1]["drift_status"] == "alert"


def test_step_failure_is_recorded_and_raised(cfg, fake_steps, monkeypatch):
    def broken(cutoff_week):
        raise ValueError("warehouse unavailable")

    monkeypatch.setattr(W, "build_features", broken)
    with pytest.raises(ValueError):
        W.run_cycle(cfg)
    run = json.loads(next(cfg.runs_dir.glob("*/run.json")).read_text())
    assert run["status"] == "failed" and "warehouse unavailable" in run["message"]
    assert run["steps"][-1] == {
        "step": "features",
        "status": "failed",
        "error": "warehouse unavailable",
        "seconds": run["steps"][-1]["seconds"],
    }


def test_api_picks_up_a_newly_published_bundle(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import nectar.serving.app as A

    path = tmp_path / "artifacts"
    first = build_demo_bundle(n_members=60)
    first.manifest["bundle_version"] = "week-1"
    first.save(path)
    monkeypatch.setenv("NECTAR_ARTIFACTS_DIR", str(path))
    monkeypatch.setattr(A, "RELOAD_CHECK_SECONDS", 0.0)
    with TestClient(A.app) as c:
        assert c.get("/health").json()["bundle_version"] == "week-1"
        second = build_demo_bundle(n_members=60)
        second.manifest["bundle_version"] = "week-2"
        second.save(path)
        # make sure the modification time differs even on coarse filesystem clocks
        m = path / "manifest.json"
        st = m.stat()
        os.utime(m, (st.st_atime, st.st_mtime + 5))
        assert c.get("/health").json()["bundle_version"] == "week-2"


def test_api_keeps_serving_when_new_bundle_is_broken(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import nectar.serving.app as A

    path = tmp_path / "artifacts"
    b = build_demo_bundle(n_members=60)
    b.manifest["bundle_version"] = "good"
    b.save(path)
    monkeypatch.setenv("NECTAR_ARTIFACTS_DIR", str(path))
    monkeypatch.setattr(A, "RELOAD_CHECK_SECONDS", 0.0)
    with TestClient(A.app) as c:
        (path / "champion.joblib").write_bytes(b"corrupt")
        m = path / "manifest.json"
        os.utime(m, (m.stat().st_atime, m.stat().st_mtime + 5))
        r = c.get("/members/1/offers")
        assert r.status_code == 200
        assert c.get("/health").json()["bundle_version"] == "good"
