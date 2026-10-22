"""The dashboard renders every tab from the committed snapshot without errors."""

from pathlib import Path

import pytest

SNAP = Path(__file__).parents[1] / "dashboards" / "snapshot"
pytestmark = pytest.mark.skipif(not (SNAP / "meta.json").exists(), reason="no snapshot")


def test_dashboard_renders_without_exceptions():
    st_testing = pytest.importorskip("streamlit.testing.v1")
    at = st_testing.AppTest.from_file(str(SNAP.parent / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert at.title[0].value == "Nectar Offer Engine"
    assert len(at.tabs) == 5
    assert any("Uplift targeting" in s.value for s in at.subheader)


def test_fairness_selector_switches_attribute():
    st_testing = pytest.importorskip("streamlit.testing.v1")
    at = st_testing.AppTest.from_file(str(SNAP.parent / "app.py"), default_timeout=60)
    at.run()
    at.selectbox[0].select("income_level").run()
    assert not at.exception
