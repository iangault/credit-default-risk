"""The Streamlit app runs end to end on the committed model and data (AppTest).

AppTest runs app.py as Streamlit would, without a browser. The first run loads the model
and the split (cached for the rest of the session), so these tests take a few seconds.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "app.py")


@pytest.fixture
def app():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def sidebar_text(at) -> str:
    return " ".join(m.value for m in at.sidebar.markdown)


def probability(at) -> str:
    return at.metric[0].value


def test_app_starts_with_three_tabs(app):
    assert [tab.label for tab in app.tabs] == ["About", "Try a client", "Cost ratio"]
    assert "16.7%" in sidebar_text(app)  # default ratio 5: cutoff 1/6


def test_moving_the_slider_changes_the_cutoff(app):
    app.slider(key="ratio").set_value(2).run()
    assert not app.exception
    assert "33.3%" in sidebar_text(app)


def test_each_example_client_shows_a_probability(app):
    for key in ("caught", "missed", "false_alarm", "cleared"):
        app.button(key=f"example_{key}").click().run()
        assert not app.exception
        assert probability(app).endswith("%")


def test_changing_payment_behavior_changes_the_probability(app):
    app.button(key="example_caught").click().run()
    before = probability(app)
    app.selectbox(key="scenario").set_value("Pays in full every month").run()
    assert not app.exception
    assert probability(app) != before
    assert any("Your changes moved" in m.value for m in app.markdown)


def test_picking_another_client_resets_the_controls(app):
    app.selectbox(key="scenario").set_value("Pays in full every month").run()
    app.button(key="example_missed").click().run()
    assert app.selectbox(key="scenario").value == "As recorded"
    assert not any("Your changes moved" in m.value for m in app.markdown)


def test_fairness_controls_run(app):
    app.selectbox(key="SEX").set_value(1).run()
    app.selectbox(key="SEX").set_value(2).run()
    app.slider(key="AGE").set_value(40).run()
    assert not app.exception


def test_reset_button_restores_the_client(app):
    app.selectbox(key="scenario").set_value("Fell 2 months behind").run()
    app.button(key="reset").click().run()
    assert app.selectbox(key="scenario").value == "As recorded"


def test_random_client_runs(app):
    app.button(key="random").click().run()
    assert not app.exception


def test_client_above_the_limit_range_renders(app):
    # 47 test clients have limits above the slider's usual 500,000 maximum.
    from credit_risk.config import load_config
    from credit_risk.showcase import load_assets

    assets = load_assets(load_config(ROOT / "configs" / "base.yaml"))
    position = int((assets.test_X["LIMIT_BAL"] > 500_000).to_numpy().nonzero()[0][0])
    limit = int(assets.test_X.loc[position, "LIMIT_BAL"])
    # Select the client, then let the app's own callback set the controls (as picking a
    # client does). Writing a widget's value straight into session_state is overridden
    # by the widget's stored value in AppTest, so it would not test the app.
    app.session_state["client"] = position
    app.button(key="reset").click().run()
    assert not app.exception
    assert app.slider(key="limit").value == limit
    assert app.slider(key="limit").max == limit


def test_examples_say_which_cutoff_grouped_them(app):
    # The examples are grouped at the saved 5x cutoff; at other ratios the decision shown
    # can differ from the group name, so the app says so next to the buttons.
    app.slider(key="ratio").set_value(20).run()
    app.button(key="example_cleared").click().run()
    assert any("grouped at the saved model's cutoff" in c.value for c in app.caption)


def test_unexpected_startup_error_shows_a_message(monkeypatch):
    # e.g. the host cannot unpickle the model: the visitor sees a short message, not a
    # traceback. The cached assets from other tests are cleared so the failure is reached.
    import streamlit as st

    from credit_risk import showcase

    def broken(cfg):
        raise RuntimeError("cannot load model")

    monkeypatch.setattr(showcase, "load_assets", broken)
    st.cache_resource.clear()
    try:
        at = AppTest.from_file(APP, default_timeout=120)
        at.run()
        assert not at.exception
        assert "could not start" in at.error[0].value
    finally:
        st.cache_resource.clear()  # later tests load the real assets again
