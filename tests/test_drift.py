"""Drift maths and reporting on synthetic data (no warehouse needed)."""

import numpy as np
import pandas as pd
import pytest

from nectar.monitoring.drift import (
    ALERT_PSI,
    MONITORED,
    PREDICTION,
    FeatureProfile,
    ReferenceProfile,
    compare,
    psi_from_props,
    render_html,
    status_for,
)
from nectar.serving.demo_bundle import synthetic_features


def test_psi_is_zero_for_identical_distributions():
    p = np.array([0.2, 0.3, 0.5])
    assert psi_from_props(p, p) == 0.0


def test_psi_matches_hand_calculation():
    ref, cur = np.array([0.5, 0.5]), np.array([0.25, 0.75])
    expected = (0.25 - 0.5) * np.log(0.25 / 0.5) + (0.75 - 0.5) * np.log(0.75 / 0.5)
    assert psi_from_props(ref, cur) == pytest.approx(expected)


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(0)
    prof = FeatureProfile.fit("x", rng.normal(0, 1, 20_000))
    psis = [prof.compare(rng.normal(shift, 1, 20_000))["psi"] for shift in (0, 0.25, 0.5, 1.0)]
    assert psis[0] < 0.01
    assert psis == sorted(psis)
    assert psis[-1] > ALERT_PSI


def test_binary_and_point_mass_features():
    flags = np.r_[np.zeros(700), np.ones(300)]
    prof = FeatureProfile.fit("flag", flags)
    assert prof.kind == "binary" and prof.proportions == pytest.approx([0.7, 0.3])
    # a feature with a big point mass at zero keeps that mass in a single bin
    zeros = np.r_[np.zeros(600), np.linspace(1, 10, 400)]
    zp = FeatureProfile.fit("z", zeros)
    assert zp.proportions[0] == pytest.approx(0.6)
    assert zp.compare(zeros)["psi"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "psi, status",
    [(0.0, "ok"), (0.099, "ok"), (0.1, "warn"), (0.249, "warn"), (0.25, "alert"), (2.0, "alert")],
)
def test_status_thresholds(psi, status):
    assert status_for(psi) == status


# PSI has sampling noise of roughly (bins - 1) / n_members for member-level features: at 300
# members pure noise reaches the 0.10 warn line, at 2,000 it is ~0.01. Use realistic sizes.
N_MEMBERS = 2000


@pytest.fixture(scope="module")
def ref():
    return synthetic_features(N_MEMBERS, seed=0)


def test_only_the_shifted_feature_is_flagged(ref):
    prof = ReferenceProfile.fit(ref, predictions=ref.c_trip_share.to_numpy(), source="test")
    cur = synthetic_features(N_MEMBERS, seed=1)
    cur["m_spend_per_week"] *= 1.6  # members spend 60% more
    rep = compare(prof, cur, predictions=cur.c_trip_share.to_numpy())
    flagged = rep.summary()["features_drifted"]
    assert flagged == ["m_spend_per_week"]
    assert rep.status == "alert"
    assert {r["feature"] for r in rep.results} == set(MONITORED) | {PREDICTION}


def test_unshifted_sample_is_ok(ref):
    rep = compare(ReferenceProfile.fit(ref), synthetic_features(N_MEMBERS, seed=2))
    assert rep.status == "ok"


def test_profile_round_trip_and_report_files(ref, tmp_path):
    prof = ReferenceProfile.fit(ref, source="test")
    prof.save(tmp_path / "p.json")
    loaded = ReferenceProfile.load(tmp_path / "p.json")
    assert loaded.features["c_trip_share"] == prof.features["c_trip_share"]
    rep = compare(loaded, ref, label="same data")
    j, h = rep.save(tmp_path / "out")
    assert j.exists() and h.exists()
    page = render_html(rep)
    assert "<!doctype html>" in page and "OK" in page and "Category share of trips" in page


def test_report_handles_missing_prediction_reference(ref):
    rep = compare(ReferenceProfile.fit(ref), ref, predictions=np.zeros(len(ref)))
    assert rep.summary()["prediction_psi"] is None
    assert isinstance(pd.DataFrame(rep.results), pd.DataFrame)
