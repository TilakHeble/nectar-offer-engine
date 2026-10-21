"""Train, compare and select offer-targeting models; log everything to MLflow.

Protocol
--------
* Split by member (no member appears in two splits): 50% train, 25% validation, 25% test.
* Every model is trained on train. Each produces a targeting policy:
    - uplift models: offer when predicted incremental profit > 0 (their own decision rule);
    - baselines: offer to the top 20% by score (a typical business rule of thumb).
* The champion is chosen on VALIDATION using the observed-outcome IPW estimate of incremental
  profit. Ground truth is never used for training or selection.
* The TEST split reports both the observed estimate and the simulator's ground truth.

Run:
    python -m nectar.models.train
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import duckdb  # noqa: E402
import joblib  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

from nectar.config import PROCESSED_DIR, ROOT, WAREHOUSE_PATH  # noqa: E402
from nectar.evaluate import metrics as M  # noqa: E402
from nectar.features.schema import model_matrix, validate_features  # noqa: E402
from nectar.models.learners import EnsembleModel, TargetingModel, all_models  # noqa: E402

log = logging.getLogger(__name__)

MODELS_DIR = ROOT / "models"
BASELINE_TOP_SHARE = 0.20
ENSEMBLE_MEMBERS = ["S-learner", "X-learner", "DR-learner", "Causal forest"]


def load_experiment(path: Path = WAREHOUSE_PATH) -> pd.DataFrame:
    con = duckdb.connect(str(path), read_only=True)
    df = con.execute("SELECT * FROM sim.offer_experiment").df()
    con.close()
    return df


def split_by_member(df: pd.DataFrame, seed: int = 0) -> pd.Series:
    """Label each row train / valid / test (50 / 25 / 25 by member)."""
    idx = np.arange(len(df))
    train, rest = next(
        GroupShuffleSplit(1, test_size=0.5, random_state=seed).split(idx, groups=df.member_id)
    )
    valid_rel, test_rel = next(
        GroupShuffleSplit(1, test_size=0.5, random_state=seed + 1).split(
            rest, groups=df.member_id.iloc[rest]
        )
    )
    split = pd.Series("train", index=df.index)
    split.iloc[rest[valid_rel]] = "valid"
    split.iloc[rest[test_rel]] = "test"
    return split


def policy(model: TargetingModel, scores: np.ndarray) -> np.ndarray:
    if model.has_profit_threshold:
        return scores > 0
    cut = np.quantile(scores, 1 - BASELINE_TOP_SHARE)
    return scores >= cut


def evaluate(model, scores, part: pd.DataFrame) -> dict:
    t, y = part.treatment.to_numpy(), part.gross_profit.to_numpy()
    treat = policy(model, scores)
    out = {
        "share_offered": float(treat.mean()),
        "observed_incremental_profit": M.incremental_policy_value(treat, t, y),
        "observed_auuc": M.auuc(scores, t, y),
    }
    if "gt_profit_uplift" in part:
        gt = part.gt_profit_uplift.to_numpy()
        oracle = M.oracle_profit(gt)
        realised = M.true_policy_profit(treat, gt)
        out.update(
            {
                "true_incremental_profit": realised,
                "oracle_profit": oracle,
                "share_of_oracle": realised / oracle,
                "spearman_true_profit_uplift": float(spearmanr(scores, gt).statistic),
                "spearman_true_conversion_uplift": float(
                    spearmanr(scores, part.gt_uplift).statistic
                ),
            }
        )
    return out


def run(seed: int = 0, experiment: str = "nectar-offer-engine") -> pd.DataFrame:
    df = validate_features(load_experiment())
    df["split"] = split_by_member(df, seed)
    X = model_matrix(df)
    parts = {s: df.index[df.split == s] for s in ["train", "valid", "test"]}
    tr = parts["train"]

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(experiment)
    rows, preds, fitted = [], {}, {}

    def record(model: TargetingModel, fit_seconds: float) -> None:
        with mlflow.start_run(run_name=model.name):
            mlflow.log_params(
                {
                    "model": model.name,
                    "kind": model.kind,
                    "seed": seed,
                    "outcome": model.outcome,
                    **model.params,
                }
            )
            for split in ["valid", "test"]:
                idx = parts[split]
                scores = model.score(X.loc[idx])
                res = evaluate(model, scores, df.loc[idx])
                mlflow.log_metrics({f"{split}_{k}": v for k, v in res.items()})
                rows.append({"model": model.name, "kind": model.kind, "split": split, **res})
                preds[(model.name, split)] = scores
            mlflow.log_metric("fit_seconds", fit_seconds)
        fitted[model.name] = model
        log.info("%-18s %5.1fs", model.name, fit_seconds)

    for model in all_models(seed):
        t0 = time.time()
        y_train = df.loc[tr, model.outcome].to_numpy()
        model.fit(X.loc[tr], df.loc[tr, "treatment"].to_numpy(), y_train)
        record(model, time.time() - t0)

    # The ensemble reuses the already-fitted members (no refit).
    record(EnsembleModel([fitted[n] for n in ENSEMBLE_MEMBERS], seed), 0.0)

    results = pd.DataFrame(rows)
    valid = results[results.split == "valid"].set_index("model")
    champion = valid.observed_incremental_profit.idxmax()
    log.info("Champion (by validation observed incremental profit): %s", champion)

    # Persist champion, metadata and test-set scores for every model.
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(fitted[champion], MODELS_DIR / "champion.joblib")
    scored = []
    for split in ["valid", "test"]:
        part = df.loc[parts[split]].copy()
        for name in fitted:
            part[f"score__{name}"] = preds[(name, split)]
        scored.append(part)
    pd.concat(scored).to_parquet(PROCESSED_DIR / "holdout_scores.parquet", index=False)
    results.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
    meta = {
        "champion": champion,
        "seed": seed,
        "selection_metric": "valid_observed_incremental_profit",
        "decision_rule": "offer when predicted incremental profit > 0",
        "features": list(X.columns),
        "train_rows": int(len(tr)),
        "test_metrics": results[(results.split == "test") & (results.model == champion)]
        .drop(columns=["model", "kind", "split"])
        .iloc[0]
        .to_dict(),
    }
    (MODELS_DIR / "champion.json").write_text(json.dumps(meta, indent=2))
    with mlflow.start_run(run_name=f"champion: {champion}"):
        mlflow.log_dict(meta, "champion.json")
        mlflow.log_artifact(str(MODELS_DIR / "model_comparison.csv"))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = run(seed=args.seed)
    cols = [
        "model",
        "share_offered",
        "observed_incremental_profit",
        "true_incremental_profit",
        "share_of_oracle",
        "spearman_true_profit_uplift",
    ]
    with pd.option_context("display.float_format", "{:,.3f}".format, "display.width", 140):
        print(res[res.split == "test"][cols].to_string(index=False))


if __name__ == "__main__":
    main()
