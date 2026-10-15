"""The Streamlit app runs end to end on the committed model and data (AppTest).

AppTest runs app.py as Streamlit would, without a browser. The first run loads the model
and the split (cached for the rest of the session), so these tests take a few seconds.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from credit_risk import showcase as sc

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "app.py")


@pytest.fixture
def app():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def tab(at, label):
    return next(t for t in at.tabs if t.label == label)


def cost_text(at) -> str:
    return " ".join(m.value for m in tab(at, "Cost ratio").markdown)


def prediction(at) -> str:
    """The colored prediction box: red (st.error) if flagged, green (st.success) if not."""
    boxes = [b.value for b in tab(at, "Try a client").error]
    boxes += [b.value for b in tab(at, "Try a client").success]
    assert len(boxes) == 1, boxes
    return boxes[0]


def test_app_starts_with_three_tabs_and_no_sidebar(app):
    assert [t.label for t in app.tabs] == ["About", "Try a client", "Cost ratio"]
    assert len(app.sidebar.children) == 0
    assert "16.7%" in cost_text(app)  # default ratio 5: cutoff 1/6


def test_clicking_the_ratio_up_and_down_changes_the_cutoff(app):
    app.number_input(key="ratio").increment().run()
    assert "14.3%" in cost_text(app)  # ratio 6: 1/7
    app.number_input(key="ratio").set_value(2).run()
    assert not app.exception
    assert "33.3%" in cost_text(app)


def test_ratio_of_one_draws_one_marker(app):
    # At 1x the cutoff is 0.5, the same as the gray reference, so only one dot is drawn.
    app.number_input(key="ratio").set_value(1).run()
    assert not app.exception


def test_each_example_client_shows_a_colored_prediction(app):
    expected = {
        "caught": "Predicted to default",
        "missed": "Predicted to pay",
        "false_alarm": "Predicted to default",
        "cleared": "Predicted to pay",
    }
    for key, headline in expected.items():
        app.button(key=f"example_{key}").click().run()
        assert not app.exception
        assert headline in prediction(app)
        assert "%" in prediction(app)


def test_client_tab_uses_the_saved_cutoff_whatever_the_ratio(app):
    # The cost ratio control only affects the Cost ratio tab.
    app.number_input(key="ratio").set_value(20).run()
    app.button(key="example_cleared").click().run()
    assert "Predicted to pay" in prediction(app)
    assert "17.0%" in prediction(app)  # the saved model's cutoff, 0.1697


def test_changing_payment_behavior_changes_the_probability(app):
    app.button(key="example_caught").click().run()
    before = prediction(app)
    app.radio(key="scenario").set_value("Pays in full every month").run()
    assert not app.exception
    assert prediction(app) != before
    assert any("Your changes" in m.value for m in app.markdown)


def test_picking_another_client_resets_the_controls(app):
    app.radio(key="scenario").set_value("Pays in full every month").run()
    app.button(key="example_missed").click().run()
    assert app.radio(key="scenario").value == "As recorded"
    assert not any("Your changes" in m.value for m in app.markdown)


def test_fairness_controls_run(app):
    app.selectbox(key="SEX").set_value(1).run()
    app.selectbox(key="SEX").set_value(2).run()
    app.slider(key="AGE").set_value(40).run()
    assert not app.exception


def test_reset_button_restores_the_client(app):
    app.radio(key="scenario").set_value("Fell 2 months behind").run()
    app.button(key="reset").click().run()
    assert app.radio(key="scenario").value == "As recorded"


def test_random_client_runs(app):
    app.button(key="random").click().run()
    assert not app.exception
    assert prediction(app)


def test_examples_sit_in_a_confusion_matrix(app):
    # Rows: what actually happened. Columns: what the model decided at the saved cutoff.
    text = " ".join(m.value for m in tab(app, "Try a client").markdown)
    for heading in ("Flagged", "Not flagged", "Defaulted", "Paid"):
        assert heading in text
    labels = {key: app.button(key=f"example_{key}").label for key in sc.EXAMPLE_LABELS}
    assert labels == {
        "caught": "Caught (true positive)",
        "missed": "Missed (false negative)",
        "false_alarm": "False alarm (false positive)",
        "cleared": "Cleared (true negative)",
    }


def test_your_changes_line_is_colored_by_direction(app):
    # Paying in full lowers the caught defaulter's risk: a green down arrow.
    app.button(key="example_caught").click().run()
    app.radio(key="scenario").set_value("Pays in full every month").run()
    change = [m.value for m in app.markdown if "Your changes" in m.value]
    assert len(change) == 1
    assert change[0].startswith("### ")
    assert ":green[▼" in change[0]


def test_client_data_is_a_full_width_panel_comparing_recorded_and_changed(app):
    client = tab(app, "Try a client")
    assert any("The data the model reads for this client" in m.value for m in client.markdown)
    assert not [e for e in client.expander if "payment history" in e.label]
    assert len(client.dataframe) == 2


def test_result_column_follows_the_scroll(app):
    # As on the Cost ratio tab: the CSS rule is checked, not the scrolling (no browser).
    assert any("st-key-result_panel" in m.value for m in app.markdown)


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


def test_cost_panel_shows_the_key_numbers_and_follows_the_scroll(app):
    # The panel on the left of the Cost ratio tab holds the ratio and three counts. A CSS
    # rule makes its column sticky; AppTest has no browser, so the test checks the rule
    # is on the page and the counts update, not the scrolling itself.
    cost = tab(app, "Cost ratio")
    labels = [m.label for m in cost.metric]
    assert labels == ["Defaulters caught", "False alarms", "Cost per client"]
    before = [m.value for m in cost.metric]
    app.number_input(key="ratio").increment().run()
    assert [m.value for m in tab(app, "Cost ratio").metric] != before
    assert any("position: sticky" in m.value for m in app.markdown)


def test_about_tab_leads_with_the_headline_numbers(app):
    about = tab(app, "About")
    metrics = {m.label: m.value for m in about.metric}
    assert metrics == {
        "Test clients the model never saw": "6,000",
        "Defaulters caught": "77%",
        "Lower cost per client than cutoff 0.5": "29%",
        "Cutoff for a 5x cost ratio": "0.17",
    }
    text = " ".join(m.value for m in about.markdown)
    for heading in ("The data", "The model", "The key idea", "The result", "Try a client"):
        assert heading in text
    assert "github.com/iangault/credit-default-risk" in text


def test_title_sits_above_the_tabs(app):
    assert app.title[0].value == "Credit Card Default Predictor"


@pytest.mark.parametrize(
    ("button", "tab_label"), [("go_client", "Try a client"), ("go_cost", "Cost ratio")]
)
def test_about_buttons_open_their_tab(app, button, tab_label):
    app.button(key=button).click().run()
    assert not app.exception
    assert app.session_state["view"] == tab_label


def test_theme_sets_one_accent_color_for_buttons_and_controls():
    # Streamlit reads .streamlit/config.toml from the folder it is started in (the repo
    # root, for `make app` and for Community Cloud). primaryColor colors primary buttons,
    # radios, sliders and the number box.
    import tomllib

    theme = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())["theme"]
    assert theme["primaryColor"] == "#6d28d9"
