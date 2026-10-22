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
    assert len(at.tabs) == 6
    assert any("Uplift targeting" in s.value for s in at.subheader)
    assert any("optimiser's curse" in s.value for s in at.subheader)


def test_risk_tab_cap_toggle():
    st_testing = pytest.importorskip("streamlit.testing.v1")
    at = st_testing.AppTest.from_file(str(SNAP.parent / "app.py"), default_timeout=60)
    at.run()
    at.radio(key="risk_cap").set_value(99).run()
    assert not at.exception
    assert any("100 budget" in i.value for i in at.info)


def test_fairness_selector_switches_attribute():
    st_testing = pytest.importorskip("streamlit.testing.v1")
    at = st_testing.AppTest.from_file(str(SNAP.parent / "app.py"), default_timeout=60)
    at.run()
    at.selectbox[0].select("income_level").run()
    assert not at.exception
