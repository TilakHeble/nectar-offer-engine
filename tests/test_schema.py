"""The feature contract rejects bad data and builds a stable model matrix."""

import pandas as pd
import pandera.errors as pe
import pytest

from nectar.features.schema import NUMERIC_FEATURES, OFFER_IDS, model_matrix, validate_features


def _row(**overrides):
    row = {
        "member_id": 1,
        "offer_id": OFFER_IDS[0],
        "discount_pct": 0.1,
        "m_trips_per_week": 1.0,
        "m_spend_per_week": 20.0,
        "m_weeks_since_last_trip": 0,
        "m_member_discount_rate": 0.15,
        "m_private_label_share": 0.2,
        "m_median_basket": 15.0,
        "m_trips_per_week_13w": 1.0,
        "c_trip_share": 0.1,
        "c_trip_share_13w": 0.1,
        "c_spend_per_week": 0.5,
        "c_spend_per_trip": 3.0,
        "c_weeks_since_last": 2,
        "c_discount_line_share": 0.4,
        "c_ever_bought": True,
        "o_spend_per_trip": 3.2,
    }
    row.update(overrides)
    return row


def test_valid_rows_pass():
    validate_features(pd.DataFrame([_row(), _row(offer_id=OFFER_IDS[1])]))


@pytest.mark.parametrize(
    "bad",
    [
        {"c_trip_share": 1.5},
        {"m_spend_per_week": -1.0},
        {"offer_id": "UNKNOWN"},
        {"discount_pct": 0.5},
    ],
)
def test_invalid_values_fail(bad):
    with pytest.raises(pe.SchemaErrors):
        validate_features(pd.DataFrame([_row(**bad)]))


def test_duplicate_member_offer_fails():
    with pytest.raises(pe.SchemaErrors):
        validate_features(pd.DataFrame([_row(), _row()]))


def test_model_matrix_has_fixed_columns():
    X = model_matrix(pd.DataFrame([_row()]))
    assert list(X.columns[: len(NUMERIC_FEATURES)]) == NUMERIC_FEATURES
    assert X.shape[1] == len(NUMERIC_FEATURES) + 1 + len(OFFER_IDS)
    assert X[f"offer_{OFFER_IDS[0]}"].iloc[0] == 1.0
