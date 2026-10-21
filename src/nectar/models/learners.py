"""Targeting models behind one interface: fit(X, treatment, outcome) -> score(X).

Uplift learners estimate the *incremental gross profit* of sending an offer (CATE on profit),
so a positive score means "this offer is expected to pay for itself". Baselines score something
else and are kept to show why it fails:

- RandomTargeting: no model, the blanket-offer benchmark.
- ResponseModel:   P(buy | offered), the classic "likely responder" model. It ranks members
                   but cannot say whether an offer pays, so it has no natural threshold.

Hyperparameters are deliberately smooth (shallow trees, large leaves): profit outcomes are noisy
and mostly zero, and the uplift signal is small next to baseline variation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from econml.dr import DRLearner
from econml.metalearners import SLearner, TLearner, XLearner
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.dummy import DummyClassifier

warnings.filterwarnings("ignore", category=UserWarning)

OUTCOME_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    verbose=-1,
)
CATE_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.03,
    num_leaves=7,
    min_child_samples=400,
    subsample=0.8,
    subsample_freq=1,
    verbose=-1,
)


def _outcome(seed: int) -> LGBMRegressor:
    return LGBMRegressor(**OUTCOME_PARAMS, random_state=seed)


def _cate(seed: int) -> LGBMRegressor:
    return LGBMRegressor(**CATE_PARAMS, random_state=seed)


def _known_propensity() -> DummyClassifier:
    # Treatment was randomised 50/50, so the propensity is known and constant. Using the
    # empirical treatment share avoids fitting noise into the propensity model.
    return DummyClassifier(strategy="prior")


@dataclass
class TargetingModel:
    name: str
    kind: str  # "baseline" or "uplift"
    seed: int = 0
    params: dict = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> TargetingModel:
        raise NotImplementedError

    def score(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    # Which observed column the model is trained on.
    outcome: ClassVar[str] = "gross_profit"

    @property
    def has_profit_threshold(self) -> bool:
        """True if score > 0 means 'expected to increase profit'."""
        return self.kind == "uplift"


class RandomTargeting(TargetingModel):
    def __init__(self, seed: int = 0):
        super().__init__("Random (blanket)", "baseline", seed)

    def fit(self, X, t, y):
        return self

    def score(self, X):
        return np.random.default_rng(self.seed).random(len(X))


class ResponseModel(TargetingModel):
    """P(buy | offered): trained on the offer group's purchase outcome."""

    outcome = "purchased"

    def __init__(self, seed: int = 0):
        super().__init__("Response model", "baseline", seed, {"model": "LGBMClassifier"})

    def fit(self, X, t, y):
        self.model_ = LGBMClassifier(**OUTCOME_PARAMS, random_state=self.seed)
        self.model_.fit(X[t == 1], y[t == 1])
        return self

    def score(self, X):
        return self.model_.predict_proba(X)[:, 1]


class _EconMLUplift(TargetingModel):
    def _build(self):
        raise NotImplementedError

    def fit(self, X, t, y):
        self.model_ = self._build()
        self.model_.fit(y, t, X=X.to_numpy())
        return self

    def score(self, X):
        return np.asarray(self.model_.effect(X.to_numpy())).ravel()


class TLearnerModel(_EconMLUplift):
    def __init__(self, seed: int = 0):
        super().__init__("T-learner", "uplift", seed, {"base": "LGBMRegressor"})

    def _build(self):
        return TLearner(models=_outcome(self.seed))


class SLearnerModel(_EconMLUplift):
    def __init__(self, seed: int = 0):
        super().__init__("S-learner", "uplift", seed, {"base": "LGBMRegressor"})

    def _build(self):
        return SLearner(overall_model=_outcome(self.seed))


class XLearnerModel(_EconMLUplift):
    def __init__(self, seed: int = 0):
        super().__init__("X-learner", "uplift", seed, {"base": "LGBMRegressor"})

    def _build(self):
        return XLearner(
            models=_outcome(self.seed),
            propensity_model=_known_propensity(),
            cate_models=_cate(self.seed),
        )


class DRLearnerModel(_EconMLUplift):
    def __init__(self, seed: int = 0):
        super().__init__("DR-learner", "uplift", seed, {"base": "LGBMRegressor", "cv": 3})

    def _build(self):
        return DRLearner(
            model_propensity=_known_propensity(),
            model_regression=_outcome(self.seed),
            model_final=_cate(self.seed),
            cv=3,
            random_state=self.seed,
        )


class CausalForestModel(_EconMLUplift):
    def __init__(self, seed: int = 0, n_estimators: int = 500, min_samples_leaf: int = 50):
        super().__init__(
            "Causal forest",
            "uplift",
            seed,
            {"n_estimators": n_estimators, "min_samples_leaf": min_samples_leaf, "cv": 3},
        )

    def _build(self):
        return CausalForestDML(
            model_y=_outcome(self.seed),
            model_t=_known_propensity(),
            discrete_treatment=True,
            n_estimators=self.params["n_estimators"],
            min_samples_leaf=self.params["min_samples_leaf"],
            cv=3,
            random_state=self.seed,
        )

    def score_interval(self, X: pd.DataFrame, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.model_.effect_interval(X.to_numpy(), alpha=alpha)
        return np.asarray(lo).ravel(), np.asarray(hi).ravel()


class EnsembleModel(TargetingModel):
    """Average predicted incremental profit of several uplift learners.

    All members predict on the same scale (incremental profit per pair), so the plain mean keeps
    the "offer when > 0" decision rule. Averaging reduces the variance of any single learner.
    """

    def __init__(self, members: list[TargetingModel], seed: int = 0):
        super().__init__(
            "Ensemble", "uplift", seed, {"members": ", ".join(m.name for m in members)}
        )
        self.members = members

    def fit(self, X, t, y):
        for m in self.members:
            m.fit(X, t, y)
        return self

    def score(self, X):
        return np.mean([m.score(X) for m in self.members], axis=0)


def all_models(seed: int = 0) -> list[TargetingModel]:
    return [
        RandomTargeting(seed),
        ResponseModel(seed),
        TLearnerModel(seed),
        SLearnerModel(seed),
        XLearnerModel(seed),
        DRLearnerModel(seed),
        CausalForestModel(seed),
    ]
