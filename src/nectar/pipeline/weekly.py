"""The weekly offer cycle: refresh -> features -> checks -> score + allocate -> publish.

Each step is a Prefect task, so a run gets retries, logs, timings and a UI for free. Safety
gates stop a bad run before it reaches customers; the currently published plan stays live:

    1. refresh      (optional) reload raw data and rebuild the dbt warehouse
    2. features     build features at the cutoff and validate them against the contract
    3. drift gate   compare with the training population: ALERT blocks, WARN publishes + flags
    4. plan         score with the champion and allocate offers under the budget
    5. plan gate    sanity-check the plan (budget, caps, size vs the live plan)
    6. publish      write the bundle atomically and archive it; the API picks it up by itself

Every run writes runs/<run_id>/ with run.json (status, steps, timings, summaries) and the drift
report (HTML + JSON), and appends a line to runs/history.csv.

Run once:              python -m nectar.pipeline.weekly
Schedule (Mondays 6am): python -m nectar.pipeline.weekly --serve
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from nectar.config import ROOT  # noqa: E402
from nectar.features.offer_features import build_offer_features  # noqa: E402
from nectar.features.schema import model_matrix, validate_features  # noqa: E402
from nectar.monitoring.drift import DriftReport, ReferenceProfile, compare  # noqa: E402
from nectar.serving.bundle import DEFAULT_DIR, LATEST_CUTOFF_WEEK, Bundle  # noqa: E402

log = logging.getLogger("nectar.pipeline")

RUNS_DIR = ROOT / "runs"
HISTORY_DIR = ROOT / "artifacts_history"
REFERENCE_PATH = ROOT / "models" / "reference_profile.json"
TRAINING_CUTOFF_WEEK = 92  # the champion was trained on features up to week 91


@dataclass
class CycleConfig:
    cutoff_week: int = LATEST_CUTOFF_WEEK
    budget: float = 600.0
    max_per_member: int = 3
    refresh: bool = False  # rebuild warehouse from raw CSVs first
    risk_aware: bool = True  # rank offers by a lower confidence bound (see optimise/robust.py)
    force: bool = False  # publish even if a gate fails (logged)
    max_plan_change: float = 0.5  # block if offers sent moves more than +/-50% vs live plan
    artifacts_dir: Path = DEFAULT_DIR
    runs_dir: Path = RUNS_DIR
    history_dir: Path = HISTORY_DIR
    reference_path: Path = REFERENCE_PATH


class GateFailed(RuntimeError):
    """A safety check failed; nothing was published."""


@dataclass
class RunRecord:
    run_id: str
    started_at: str
    config: dict
    status: str = "running"
    steps: list[dict] = field(default_factory=list)
    drift: dict | None = None
    plan: dict | None = None
    published_version: str | None = None
    message: str = ""

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "run.json").write_text(json.dumps(asdict(self), indent=2, default=str))


# ---------- steps (plain functions; the flow wraps them as Prefect tasks) ----------


def refresh_data() -> dict:
    import subprocess

    from nectar.ingest.load_raw import ingest

    counts = ingest()
    subprocess.run(
        ["dbt", "build", "--profiles-dir", "."],
        cwd=ROOT / "dbt",
        check=True,
        capture_output=True,
        text=True,
    )
    return {"rows": counts}


def load_or_build_reference(path: Path) -> ReferenceProfile:
    """Training-population profile; built once from the training features and the champion."""
    if path.exists():
        return ReferenceProfile.load(path)
    model = joblib.load(ROOT / "models" / "champion.joblib")
    ref = validate_features(build_offer_features(TRAINING_CUTOFF_WEEK))
    profile = ReferenceProfile.fit(
        ref,
        model.score(model_matrix(ref)),
        source=f"training features, cutoff week {TRAINING_CUTOFF_WEEK}",
    )
    profile.save(path)
    return profile


def build_features(cutoff_week: int) -> pd.DataFrame:
    return validate_features(build_offer_features(cutoff_week))


def check_drift(
    profile: ReferenceProfile, features: pd.DataFrame, cutoff_week: int, out_dir: Path
) -> DriftReport:
    model = joblib.load(ROOT / "models" / "champion.joblib")
    report = compare(
        profile,
        features,
        model.score(model_matrix(features)),
        label=f"scoring features, cutoff week {cutoff_week}",
    )
    report.save(out_dir)
    return report


def build_plan(cfg: CycleConfig, features: pd.DataFrame) -> Bundle:
    from nectar.serving.bundle import build_bundle

    return build_bundle(
        cfg.budget,
        cfg.max_per_member,
        cfg.cutoff_week,
        features=features,
        risk_aware=cfg.risk_aware,
    )


def check_plan(bundle: Bundle, cfg: CycleConfig) -> list[str]:
    """Return a list of problems (empty = plan is safe to publish)."""
    problems = []
    plan, sel = bundle.manifest["plan"], bundle.candidates[bundle.candidates.selected]
    if plan["offers_sent"] == 0:
        problems.append("plan sends no offers")
    if plan["expected_cost"] > cfg.budget * 1.0001:
        problems.append(f"expected cost {plan['expected_cost']:.2f} exceeds budget {cfg.budget}")
    if len(sel) and sel.groupby("member_id").size().max() > cfg.max_per_member:
        problems.append("a member exceeds the per-member offer cap")
    if (sel.value <= 0).any():
        problems.append("plan includes offers with non-positive predicted value")
    live = cfg.artifacts_dir / "manifest.json"
    if live.exists():
        prev = json.loads(live.read_text())["plan"]["offers_sent"]
        change = plan["offers_sent"] / max(prev, 1) - 1
        if abs(change) > cfg.max_plan_change:
            problems.append(
                f"offers sent changed {change:+.0%} vs live plan ({prev:,} -> "
                f"{plan['offers_sent']:,}); limit is +/-{cfg.max_plan_change:.0%}"
            )
    return problems


def publish(bundle: Bundle, cfg: CycleConfig, run_id: str) -> str:
    bundle.manifest["pipeline_run_id"] = run_id
    version = bundle.manifest["bundle_version"]
    bundle.save(cfg.artifacts_dir)  # atomic swap; the API reloads on its own
    archive = cfg.history_dir / version
    if archive.exists():
        shutil.rmtree(archive)
    shutil.copytree(cfg.artifacts_dir, archive)
    return version


# ---------- orchestration ----------


def _identity(fn=None, **_kw):
    return fn if fn is not None else (lambda f: f)


def run_cycle(cfg: CycleConfig, task=_identity) -> RunRecord:
    """Run one cycle. `task` wraps each step (prefect.task inside the flow; a no-op in tests)."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.runs_dir / run_id
    rec = RunRecord(
        run_id=run_id,
        started_at=datetime.now(timezone.utc).isoformat(),
        config={k: str(v) for k, v in asdict(cfg).items()},
    )

    def step(name, fn, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            out = task(fn, name=name)(*args, **kwargs)
            rec.steps.append(
                {"step": name, "status": "ok", "seconds": round(time.perf_counter() - t0, 2)}
            )
            return out
        except Exception as e:
            rec.steps.append(
                {
                    "step": name,
                    "status": "failed",
                    "error": str(e),
                    "seconds": round(time.perf_counter() - t0, 2),
                }
            )
            raise

    try:
        if cfg.refresh:
            step("refresh", refresh_data)
        profile = step("reference", load_or_build_reference, cfg.reference_path)
        features = step("features", build_features, cfg.cutoff_week)
        report = step("drift", check_drift, profile, features, cfg.cutoff_week, run_dir)
        rec.drift = report.summary()
        if report.status == "alert" and not cfg.force:
            raise GateFailed(
                f"drift ALERT on {', '.join(rec.drift['features_drifted'])}: "
                "kept the live plan; investigate or retrain, or rerun with --force"
            )
        bundle = step("plan", build_plan, cfg, features)
        rec.plan = bundle.manifest["plan"]
        problems = step("plan_checks", check_plan, bundle, cfg)
        if problems and not cfg.force:
            raise GateFailed("plan checks failed: " + "; ".join(problems))
        rec.published_version = step("publish", publish, bundle, cfg, run_id)
        rec.status = (
            "published_with_warning" if report.status == "warn" or problems else "published"
        )
        rec.message = "published" + (
            " (drift WARN: schedule retraining)" if report.status == "warn" else ""
        )
    except GateFailed as e:
        rec.status, rec.message = "blocked", str(e)
    except Exception as e:  # noqa: BLE001 - recorded, then re-raised for the scheduler
        rec.status, rec.message = "failed", f"{type(e).__name__}: {e}"
        rec.save(run_dir)
        _append_history(cfg.runs_dir, rec)
        raise
    rec.save(run_dir)
    _append_history(cfg.runs_dir, rec)
    return rec


def _append_history(runs_dir: Path, rec: RunRecord) -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / "history.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(
                [
                    "run_id",
                    "status",
                    "drift_status",
                    "max_feature_psi",
                    "offers_sent",
                    "members_reached",
                    "published_version",
                    "message",
                ]
            )
        w.writerow(
            [
                rec.run_id,
                rec.status,
                (rec.drift or {}).get("status"),
                round((rec.drift or {}).get("max_feature_psi", 0), 4) if rec.drift else None,
                (rec.plan or {}).get("offers_sent"),
                (rec.plan or {}).get("members_reached"),
                rec.published_version,
                rec.message,
            ]
        )


def _prefect_flow():
    from prefect import flow, task

    @flow(name="weekly-offer-cycle", log_prints=True)
    def weekly_offer_cycle(
        cutoff_week: int = LATEST_CUTOFF_WEEK,
        budget: float = 600.0,
        max_per_member: int = 3,
        refresh: bool = False,
        force: bool = False,
        risk_aware: bool = True,
    ) -> dict:
        cfg = CycleConfig(
            cutoff_week=cutoff_week,
            budget=budget,
            max_per_member=max_per_member,
            refresh=refresh,
            force=force,
            risk_aware=risk_aware,
        )
        rec = run_cycle(cfg, task=lambda fn, name: task(fn, name=name))
        print(f"run {rec.run_id}: {rec.status} - {rec.message}")
        return asdict(rec)

    return weekly_offer_cycle


def main() -> None:
    p = argparse.ArgumentParser(description="Run the weekly offer cycle.")
    p.add_argument("--cutoff-week", type=int, default=LATEST_CUTOFF_WEEK)
    p.add_argument("--budget", type=float, default=600.0)
    p.add_argument("--max-per-member", type=int, default=3)
    p.add_argument("--refresh", action="store_true", help="rebuild the warehouse first")
    p.add_argument("--force", action="store_true", help="publish even if a gate fails")
    p.add_argument("--no-risk", action="store_true", help="rank by point predictions instead")
    p.add_argument("--no-prefect", action="store_true", help="run without Prefect")
    p.add_argument("--serve", action="store_true", help="schedule weekly (Mondays 06:00)")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # Prefect's API calls are noisy

    if a.no_prefect:
        cfg = CycleConfig(
            cutoff_week=a.cutoff_week,
            budget=a.budget,
            max_per_member=a.max_per_member,
            refresh=a.refresh,
            force=a.force,
            risk_aware=not a.no_risk,
        )
        rec = run_cycle(cfg)
        print(f"run {rec.run_id}: {rec.status} - {rec.message}")
        raise SystemExit(0 if rec.status.startswith("published") else 2)

    flow = _prefect_flow()
    if a.serve:
        flow.serve(
            name="weekly",
            cron="0 6 * * 1",
            parameters={
                "budget": a.budget,
                "max_per_member": a.max_per_member,
                "refresh": True,
                "risk_aware": not a.no_risk,
            },
        )
        return
    result = flow(
        cutoff_week=a.cutoff_week,
        budget=a.budget,
        max_per_member=a.max_per_member,
        refresh=a.refresh,
        force=a.force,
        risk_aware=not a.no_risk,
    )
    raise SystemExit(0 if result["status"].startswith("published") else 2)


if __name__ == "__main__":
    main()
