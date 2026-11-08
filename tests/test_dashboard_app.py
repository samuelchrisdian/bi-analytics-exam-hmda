import pytest
from streamlit.testing.v1 import AppTest

from src import config

APP = str(config.ROOT / "app" / "dashboard.py")

pytestmark = pytest.mark.skipif(not (config.MODELS / "final_model.joblib").exists(), reason="model belum dilatih")


def test_dashboard_renders_without_exception():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert len(at.tabs) == 3


def test_dashboard_scores_default_form():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.button(key="btn_score").click().run()
    assert not at.exception
    assert any("Probabilitas" in m.label for m in at.metric)


def test_dashboard_random_example_prefills_and_scores():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.button(key="btn_random").click().run()
    assert not at.exception
    assert any("Probabilitas" in m.label for m in at.metric)


def test_dashboard_shows_probability_as_percent():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.button(key="btn_score").click().run()
    value = next(m.value for m in at.metric if "Probabilitas" in m.label)
    assert value.endswith("%") and "," in value
