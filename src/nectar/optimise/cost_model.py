"""Expected retailer discount cost of sending an offer.

The optimiser needs the cost side of each decision: if we send this offer, how much discount
will the retailer fund? That is P(buy | offered) x spend x retailer-funded rate, which a
regressor learns directly from the offer group's observed `discount_cost` (zero for
non-buyers). Trained on the training split only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from nectar.features.schema import model_matrix
from nectar.models.learners import OUTCOME_PARAMS


class DiscountCostModel:
    def fit(self, df: pd.DataFrame) -> DiscountCostModel:
        offered = df[df.treatment == 1]
        self.model_ = LGBMRegressor(**OUTCOME_PARAMS, random_state=0)
        self.model_.fit(model_matrix(offered), offered.discount_cost)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model_.predict(model_matrix(df)), 0, None)
