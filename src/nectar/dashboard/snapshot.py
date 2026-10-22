"""Export the aggregated results the dashboard needs into dashboards/snapshot/.

Only aggregates leave the project (no member-level rows), so the snapshot can be committed and
the dashboard hosted publicly (e.g. Streamlit Community Cloud) without the raw data or the
ML stack.

    python -m nectar.dashboard.snapshot
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nectar.config import PROCESSED_DIR, ROOT, WAREHOUSE_PATH  # noqa: E402
from nectar.evaluate import metrics as M  # noqa: E402
from nectar.evaluate.fairness import group_report, member_plan  # noqa: E402
from nectar.features.offer_features import build_offer_features  # noqa: E402
from nectar.features.schema import model_matrix  # noqa: E402
from nectar.monitoring.drift import PREDICTION, ReferenceProfile, compare  # noqa: E402
from nectar.serving.bundle import DEFAULT_DIR, Bundle  # noqa: E402

log = logging.getLogger(__name__)
OUT = ROOT / "dashboards" / "snapshot"


def _write(df: pd.DataFrame, name: str, out: Path) -> None:
    df.to_csv(out / f"{name}.csv", index=False, float_format="%.6g")


def plan_tables(bundle: Bundle, con) -> dict[str, pd.DataFrame]:
    cand = bundle.candidates
    sel = cand[cand.selected]
    by_cat = (
        cand.groupby(["offer_id", "commodity", "discount_pct"])
        .apply(
            lambda g: pd.Series(
                {
                    "eligible_members": g.member_id.nunique(),
                    "offers_sent": int(g.selected.sum()),
                    "expected_discount_cost": g.cost[g.selected].sum(),
                    "predicted_incremental_profit": g.value[g.selected].sum(),
                    "share_positive_value": (g.value > 0).mean(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .sort_values("offers_sent", ascending=False)
    )
    members = member_plan(cand)
    rfm = con.execute("SELECT member_id, segment FROM marts.mart_member_rfm").df()
    seg = (
        members.merge(rfm, on="member_id", how="left")
        .groupby("segment")
        .agg(
            members=("member_id", "size"),
            reach=("reached", "mean"),
            offers_per_member=("offers", "mean"),
            discount_per_member=("discount", "mean"),
            predicted_profit=("value", "sum"),
        )
        .reset_index()
        .sort_values("members", ascending=False)
    )
    dist = (
        members.offers.value_counts().sort_index().rename_axis("offers").reset_index(name="members")
    )
    assert len(sel) == members.offers.sum()
    return {"plan_by_category": by_cat, "plan_by_segment": seg, "offers_per_member": dist}


def fairness_tables(bundle: Bundle, con) -> dict[str, pd.DataFrame]:
    members = member_plan(bundle.candidates)
    demo = con.execute(
        "SELECT member_id, age_group, income_level, household_size, kids "
        "FROM marts.dim_member WHERE has_demographics"
    ).df()
    m = members.merge(demo, on="member_id", how="inner")
    out = {}
    for col in ["age_group", "income_level", "household_size", "kids"]:
        rep = group_report(m, col).reset_index().rename(columns={col: "group"})
        rep.insert(0, "attribute", col)
        out[f"fairness_{col}"] = rep
    return {"fairness": pd.concat(out.values(), ignore_index=True)}


def budget_curve(
    bundle: Bundle, budgets=(100, 200, 300, 400, 600, 800, 1000, 1500)
) -> pd.DataFrame:
    from nectar.optimise.allocate import Constraints, solve_ilp

    cand = bundle.candidates[["member_id", "offer_id", "value", "cost"]]
    cap = bundle.manifest["constraints"]["max_per_member"]
    rows = []
    for b in budgets:
        a = solve_ilp(cand, Constraints(budget=float(b), max_per_member=cap))
        rows.append(
            {
                "budget": b,
                "predicted_profit": a.objective,
                "expected_cost": a.spend,
                "offers_sent": int(a.selected.sum()),
                "members_reached": int(cand.member_id[a.selected].nunique()),
                "shadow_price": a.shadow_price,
            }
        )
    return pd.DataFrame(rows)


def model_tables() -> dict[str, pd.DataFrame]:
    comp = pd.read_csv(ROOT / "models" / "model_comparison.csv")
    test = comp[comp.split == "test"].drop(columns="split")
    scores = pd.read_parquet(PROCESSED_DIR / "holdout_scores.parquet")
    te = scores[scores.split == "test"]
    t, y = te.treatment.to_numpy(), te.gross_profit.to_numpy()
    curves = []
    for name in test.model:
        g, obs = M.observed_uplift_curve(te[f"score__{name}"], t, y)
        _, true = M.true_profit_curve(te[f"score__{name}"], te.gt_profit_uplift)
        curves.append(
            pd.DataFrame(
                {"model": name, "share_offered": g, "observed_profit": obs, "true_profit": true}
            )
        )
    g, oracle = M.true_profit_curve(te.gt_profit_uplift, te.gt_profit_uplift)
    curves.append(
        pd.DataFrame(
            {
                "model": "Perfect ranking (truth)",
                "share_offered": g,
                "observed_profit": np.nan,
                "true_profit": oracle,
            }
        )
    )
    return {"model_comparison": test, "uplift_curves": pd.concat(curves, ignore_index=True)}


def drift_tables(bundle: Bundle) -> dict[str, pd.DataFrame]:
    profile = ReferenceProfile.load(ROOT / "models" / "reference_profile.json")
    model = bundle.model
    hist, latest = [], None
    for cut in range(57, 103, 5):
        f = build_offer_features(cut)
        rep = compare(profile, f, model.score(model_matrix(f)), label=f"week {cut}")
        s = rep.summary()
        hist.append(
            {
                "cutoff_week": cut,
                "status": s["status"],
                "max_feature_psi": s["max_feature_psi"],
                "prediction_psi": s["prediction_psi"],
            }
        )
        latest = rep
    feats = pd.DataFrame(
        [
            {k: r[k] for k in ("feature", "psi", "ks", "ref_mean", "cur_mean", "status")}
            for r in latest.results
        ]
    )
    feats["feature"] = feats.feature.replace({PREDICTION: "model prediction"})
    return {"drift_history": pd.DataFrame(hist), "drift_latest": feats}


def build(out: Path = OUT, artifacts: Path = DEFAULT_DIR) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    bundle = Bundle.load(artifacts)
    con = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    tables = {}
    tables.update(plan_tables(bundle, con))
    tables.update(fairness_tables(bundle, con))
    tables["budget_curve"] = budget_curve(bundle)
    tables.update(model_tables())
    tables.update(drift_tables(bundle))
    runs = ROOT / "runs" / "history.csv"
    if runs.exists():
        shutil.copy(runs, out / "run_history.csv")
    for name, df in tables.items():
        _write(df, name, out)
    meta = {
        "bundle_version": bundle.manifest["bundle_version"],
        "champion": bundle.manifest["champion"],
        "features_cutoff_week": bundle.manifest["features_cutoff_week"],
        "constraints": bundle.manifest["constraints"],
        "plan": bundle.manifest["plan"],
        "model_test_metrics": bundle.manifest.get("model_test_metrics", {}),
        "currency": "USD (Dunnhumby data is US grocery spend)",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
    con.close()
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = build()
    log.info("Snapshot written to %s: %s", path, sorted(p.name for p in path.iterdir()))


if __name__ == "__main__":
    main()
