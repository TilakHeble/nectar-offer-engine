"""Feature contract for the offer models: validation and model-matrix construction.

The schema fails fast if upstream data drifts out of the ranges the models were built for
(negative spend, shares outside [0, 1], missing values, unknown offers). The same contract is
used at training and at scoring time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa

from nectar.features.offer_features import OFFERS

OFFER_IDS = [o.offer_id for o in OFFERS]

NUMERIC_FEATURES = [
    "discount_pct",
    "m_trips_per_week",
    "m_spend_per_week",
    "m_weeks_since_last_trip",
    "m_member_discount_rate",
    "m_private_label_share",
    "m_median_basket",
    "m_trips_per_week_13w",
    "c_trip_share",
    "c_trip_share_13w",
    "c_spend_per_week",
    "c_spend_per_trip",
    "c_weeks_since_last",
    "c_discount_line_share",
    "o_spend_per_trip",
]
BOOL_FEATURES = ["c_ever_bought"]

_share = pa.Check.in_range(0, 1)
_non_negative = pa.Check.ge(0)

FEATURE_SCHEMA = pa.DataFrameSchema(
    {
        "member_id": pa.Column(int, nullable=False, coerce=True),
        "offer_id": pa.Column(str, pa.Check.isin(OFFER_IDS)),
        "discount_pct": pa.Column(float, pa.Check.isin([0.10, 0.20, 0.30])),
        "m_trips_per_week": pa.Column(float, pa.Check.gt(0)),
        "m_spend_per_week": pa.Column(float, pa.Check.gt(0)),
        "m_weeks_since_last_trip": pa.Column(int, _non_negative, coerce=True),
        "m_member_discount_rate": pa.Column(float, _share),
        "m_private_label_share": pa.Column(float, _share),
        "m_median_basket": pa.Column(float, pa.Check.gt(0)),
        "m_trips_per_week_13w": pa.Column(float, _non_negative),
        "c_trip_share": pa.Column(float, _share),
        "c_trip_share_13w": pa.Column(float, _share),
        "c_spend_per_week": pa.Column(float, _non_negative),
        "c_spend_per_trip": pa.Column(float, pa.Check.gt(0)),
        "c_weeks_since_last": pa.Column(int, _non_negative, coerce=True),
        "c_discount_line_share": pa.Column(float, _share),
        "c_ever_bought": pa.Column(bool),
        "o_spend_per_trip": pa.Column(float, pa.Check.gt(0)),
    },
    unique=["member_id", "offer_id"],
    strict=False,  # outcome and ground-truth columns may be present; they are never model inputs
    coerce=False,
)


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raise a SchemaErrors report listing every failing check, or return the validated frame."""
    return FEATURE_SCHEMA.validate(df, lazy=True)


MODEL_COLUMNS = NUMERIC_FEATURES + BOOL_FEATURES + [f"offer_{oid}" for oid in OFFER_IDS]


def model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric features + boolean flags + one-hot offer (fixed column order, all offers present).

    Built as one numpy array: column-by-column pandas assignment costs ~18 ms per call, which
    dominates single-row API latency.
    """
    offers = df.offer_id.to_numpy()
    onehot = (offers[:, None] == np.array(OFFER_IDS)[None, :]).astype(float)
    values = np.hstack(
        [
            df[NUMERIC_FEATURES].to_numpy(dtype=float),
            df[BOOL_FEATURES].to_numpy(dtype=float),
            onehot,
        ]
    )
    return pd.DataFrame(values, columns=MODEL_COLUMNS, index=df.index)
