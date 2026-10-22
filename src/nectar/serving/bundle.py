"""The serving bundle: everything the API needs, built by a weekly batch job.

Batch-then-serve is how most loyalty offer engines run: offers are decided once per cycle, then
looked up with low latency when the member opens the app. The bundle is a directory:

    artifacts/
        champion.joblib       uplift model: predicted incremental profit per (member, offer)
        cost_model.joblib     expected retailer-funded discount cost per (member, offer)
        candidates.parquet    every eligible (member, offer) with value, cost and `selected`
        manifest.json         version, model, cutoff, constraints and plan summary

Build it:
    python -m nectar.serving.bundle --budget 600 --max-per-member 3
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nectar.config import ROOT
from nectar.features.offer_features import build_offer_features
from nectar.features.schema import model_matrix, validate_features

log = logging.getLogger(__name__)

DEFAULT_DIR = ROOT / "artifacts"
LATEST_CUTOFF_WEEK = 102  # features from weeks 17-101 (the last full week); offers for week 102+
# value    = what the optimiser maximised (risk-adjusted when the risk-aware plan is on)
# forecast = the plan's best estimate of each offer's incremental profit (shrunk posterior mean)
PLAN_COLUMNS = [
    "member_id",
    "offer_id",
    "commodity",
    "discount_pct",
    "value",
    "forecast",
    "value_sd",
    "cost",
    "selected",
]


@dataclass
class Bundle:
    model: object
    cost_model: object
    candidates: pd.DataFrame
    manifest: dict

    # ---------- scoring ----------
    def score(self, features: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
        """Return value, cost and recommendation per row.

        `validate=False` skips the Pandera contract for callers that have already validated the
        rows (the API does, field by field with Pydantic, which saves ~17 ms per request).
        """
        if validate:
            validate_features(features)
        X = model_matrix(features)
        value = np.asarray(self.model.score(X), float)
        cost = np.asarray(self.cost_model.predict(features, X=X), float)
        return pd.DataFrame(
            {
                "member_id": features.member_id.to_numpy(),
                "offer_id": features.offer_id.to_numpy(),
                "predicted_incremental_profit": value,
                "predicted_discount_cost": cost,
                "recommend": value > 0,
            }
        )

    # ---------- plan lookups ----------
    def has_member(self, member_id: int) -> bool:
        return member_id in self._members

    def offers_for(self, member_id: int) -> pd.DataFrame:
        rows = self._by_member.get(member_id)
        if rows is None:
            return self.candidates.iloc[0:0]
        return rows

    def __post_init__(self) -> None:
        sel = self.candidates[self.candidates.selected]
        self._members = set(self.candidates.member_id.unique().tolist())
        self._by_member = {
            int(m): g.sort_values("value", ascending=False) for m, g in sel.groupby("member_id")
        }

    # ---------- persistence ----------
    def save(self, out_dir: Path) -> Path:
        tmp = out_dir.with_name(out_dir.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        joblib.dump(self.model, tmp / "champion.joblib")
        joblib.dump(self.cost_model, tmp / "cost_model.joblib")
        cols = [c for c in PLAN_COLUMNS if c in self.candidates]
        self.candidates[cols].to_parquet(tmp / "candidates.parquet", index=False)
        (tmp / "manifest.json").write_text(json.dumps(self.manifest, indent=2))
        # Swap in atomically so a running API never sees a half-written bundle.
        if out_dir.exists():
            shutil.rmtree(out_dir)
        tmp.rename(out_dir)
        return out_dir

    @classmethod
    def load(cls, path: Path) -> Bundle:
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text())
        return cls(
            model=joblib.load(path / "champion.joblib"),
            cost_model=joblib.load(path / "cost_model.joblib"),
            candidates=pd.read_parquet(path / "candidates.parquet"),
            manifest=manifest,
        )


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001 - version info is best effort
        return "unknown"


def build_bundle(
    budget: float,
    max_per_member: int,
    cutoff_week: int = LATEST_CUTOFF_WEEK,
    seed: int = 0,
    features: pd.DataFrame | None = None,
    risk_aware: bool = True,
) -> Bundle:
    """Score every eligible member at `cutoff_week` and allocate offers under the constraints.

    Pass `features` (already built and validated, e.g. by the weekly pipeline) to skip
    rebuilding them.

    With `risk_aware` (default) and the bootstrap ensemble available (`python -m
    nectar.optimise.robust`), the optimiser ranks offers by a lower confidence bound
    mu - z * sigma, with z chosen on validation data, which corrects the optimiser's curse. The
    plan's profit forecast uses the shrunk (posterior-mean) estimate. Without it, the
    champion's point prediction is used for both.
    """
    from nectar.models.train import load_experiment, split_by_member
    from nectar.optimise.allocate import Constraints, solve_ilp
    from nectar.optimise.cost_model import DiscountCostModel

    meta = json.loads((ROOT / "models" / "champion.json").read_text())
    model = joblib.load(ROOT / "models" / "champion.joblib")

    # The cost model learns from the randomised experiment's training split, like the champion.
    exp = load_experiment()
    exp["split"] = split_by_member(exp, seed)
    cost_model = DiscountCostModel().fit(exp[exp.split == "train"])

    if features is None:
        features = validate_features(build_offer_features(cutoff_week))
    X = model_matrix(features)
    candidates = features[["member_id", "offer_id", "commodity", "discount_pct"]].copy()
    candidates["cost"] = cost_model.predict(features, X=X)
    risk = _load_risk() if risk_aware else None
    if risk:
        from nectar.optimise.robust import lower_bound, shrink

        mu, sd = risk["bag"].predict(X)
        candidates["value"] = lower_bound(mu, sd, risk["z"])
        candidates["forecast"] = shrink(mu, sd, risk["prior_mean"], risk["prior_var"])
        candidates["value_sd"] = sd
        risk_info = {"method": "lcb", "z": risk["z"], "n_models": risk["n_models"]}
    else:
        candidates["value"] = model.score(X)
        candidates["forecast"] = candidates["value"]
        risk_info = {"method": "point"}

    c = Constraints(budget=budget, max_per_member=max_per_member)
    alloc = solve_ilp(candidates, c)
    candidates["selected"] = alloc.selected

    now = datetime.now(timezone.utc)
    manifest = {
        "bundle_version": f"{now:%Y%m%d-%H%M%S}-{_git_sha()}",
        "created_at": now.isoformat(timespec="seconds"),
        "champion": meta["champion"],
        "decision_rule": meta["decision_rule"],
        "features_cutoff_week": cutoff_week,
        "offer_window_weeks": [cutoff_week, cutoff_week + 3],
        "constraints": {"budget": budget, "max_per_member": max_per_member},
        "risk": risk_info,
        "plan": {
            **alloc.summary(),
            # honest forecast of the plan's profit (the objective is risk-adjusted)
            "forecast_profit": float(candidates.forecast[alloc.selected].sum()),
            "eligible_members": int(candidates.member_id.nunique()),
            "candidate_pairs": int(len(candidates)),
            "members_reached": int(candidates.loc[alloc.selected, "member_id"].nunique()),
        },
        "model_test_metrics": meta.get("test_metrics", {}),
        # Loading the pickled models needs these exact versions (see docker/requirements-serve.txt).
        "library_versions": {
            pkg: version(pkg)
            for pkg in ["scikit-learn", "lightgbm", "econml", "numpy", "pandas", "scipy"]
        },
    }
    return Bundle(model, cost_model, candidates, manifest)


def _load_risk() -> dict | None:
    """Bootstrap ensemble + chosen risk level, if `python -m nectar.optimise.robust` was run."""
    meta_path, bag_path = ROOT / "models" / "robust.json", ROOT / "models" / "bagged.joblib"
    if not (meta_path.exists() and bag_path.exists()):
        log.warning(
            "Risk-aware plan unavailable (run python -m nectar.optimise.robust); "
            "using point predictions"
        )
        return None
    meta = json.loads(meta_path.read_text())
    return {**meta, "bag": joblib.load(bag_path)}


def main() -> None:
    p = argparse.ArgumentParser(description="Build the weekly serving bundle.")
    p.add_argument("--budget", type=float, default=600.0)
    p.add_argument("--max-per-member", type=int, default=3)
    p.add_argument("--out", type=Path, default=DEFAULT_DIR)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    bundle = build_bundle(args.budget, args.max_per_member)
    bundle.save(args.out)
    log.info(json.dumps(bundle.manifest["plan"], indent=2))
    log.info("Bundle %s written to %s", bundle.manifest["bundle_version"], args.out)


if __name__ == "__main__":
    main()
