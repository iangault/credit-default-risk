"""Streamlit app: try the credit default model on clients it never saw.

Layout only. The logic (example clients, payment scenarios, scoring, SHAP
contributions, test-set counts) is in src/credit_risk/showcase.py, which has tests.
Run locally with `make app`. Deployed on Streamlit Community Cloud from main
(app/requirements.txt): https://credit-default-risk-2005-taiwan.streamlit.app/

Streamlit runs this whole file again after every click. Values that must survive a
rerun (which client is shown, the control values) live in st.session_state; the model
and data are loaded once per app process with st.cache_resource.

The two views are kept apart: Try a client uses the saved model's own cutoff, and only
the Cost ratio tab lets the visitor change what a miss costs.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Community Cloud installs app/requirements.txt but not this repo's package, so put
# src/ on the import path. Locally the package is installed too, and this changes nothing.
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from credit_risk import showcase as sc
from credit_risk.config import ConfigError, load_config
from credit_risk.data import DataFileError
from credit_risk.evaluate import plot_tradeoff_curves

REPO = "https://github.com/iangault/credit-default-risk"
RESULTS = f"{REPO}/blob/main/docs/results.md"

TITLE = "Credit Card Default Predictor"

st.set_page_config(page_title=TITLE, layout="wide")


@st.cache_resource(show_spinner="Loading the model and the test clients...")
def get_assets() -> sc.Assets:
    """Loaded once per app process and shared by every visitor (it never changes)."""
    return sc.load_assets(load_config(ROOT / "configs" / "base.yaml"))


try:
    assets = get_assets()
except (ConfigError, DataFileError) as error:
    # The expected failures (model missing or stale, raw CSV changed) have clear messages.
    st.error(f"The app could not start: {error}")
    st.stop()
except Exception as error:
    # Anything else, e.g. the host cannot load the model with its library versions. A
    # visitor sees a short message instead of a traceback; the host's log keeps the details.
    print(f"Startup failed: {error!r}", file=sys.stderr)
    st.error("The app could not start because of an unexpected error. Please try again later.")
    st.stop()


# ---- Which client is shown, and the controls -------------------------------------


def current_row() -> pd.DataFrame:
    """The selected test client, as a one-row frame (double brackets keep it a frame)."""
    return assets.test_X.iloc[[st.session_state.client]]


def reset_controls() -> None:
    """Set every control to the current client's recorded values."""
    client = current_row().iloc[0]
    st.session_state.scenario = sc.SCENARIOS[0]
    st.session_state.limit = int(client["LIMIT_BAL"])
    for column in sc.DEMOGRAPHICS:
        st.session_state[column] = int(client[column])


def show_example(key: str) -> None:
    st.session_state.client = assets.examples[key]
    st.session_state.pick = key  # its button in the grid is highlighted
    st.session_state.shown = sc.EXAMPLE_LABELS[key]
    reset_controls()


def show_random() -> None:
    st.session_state.client = int(np.random.default_rng().integers(len(assets.test_X)))
    st.session_state.pick = None  # no example button highlighted
    st.session_state.shown = "A random test client"
    reset_controls()


# First run in a browser session: start on the first example (the caught defaulter).
if "client" not in st.session_state:
    show_example(next(iter(assets.examples)))

# The cutoff that sorted the examples into their groups, so each example's decision
# matches its group name.
CLIENT_CUTOFF = assets.saved_cutoff
# Button text for each example in the confusion-matrix grid.
MATRIX_LABELS = {
    "caught": "Caught (true positive)",
    "missed": "Missed (false negative)",
    "false_alarm": "False alarm (false positive)",
    "cleared": "Cleared (true negative)",
}
FLAG_COLOR, REFERENCE_COLOR = "#e4572e", "#7f7f7f"  # orange-red stands out from the blue curve

TABS = ["About", "Try a client", "Cost ratio"]


def open_tab(label: str) -> None:
    """Switch to another tab (for the About tab's buttons)."""
    st.session_state.view = label


# Keeps one column of a tab in view while the other scrolls past: the result on Try a
# client, and the ratio with its counts on Cost ratio. Streamlit gives a container with
# key="result_panel" the CSS class st-key-result_panel; :has() finds the column holding
# it. Only on screens wide enough for side-by-side columns: on a phone they stack and a
# sticky panel would cover what is below it.
STICKY_PANELS = """
<style>
@media (min-width: 640px) {
  [data-testid="stColumn"]:has(.st-key-result_panel),
  [data-testid="stColumn"]:has(.st-key-ratio_panel) {
    position: sticky;
    top: 4rem;
    align-self: flex-start;
  }
}
</style>
"""
st.markdown(STICKY_PANELS, unsafe_allow_html=True)

# Above the tabs, so it shows on every tab.
st.title(TITLE)
# key + on_change="rerun" make the selected tab part of session_state, so a button can
# change it. Every tab's content still runs on each rerun (none of it checks .open).
about, client_tab, cost_tab = st.tabs(TABS, key="view", on_change="rerun")

# ---- Tab 1: About ------------------------------------------------------------------

# The About tab's text, one card per entry. The numbers are the test-set results in
# docs/results.md (section 6), read once at the end; the app does not recompute them.
ABOUT_CARDS = [
    (
        ":material/database: The data",
        "30,000 real, anonymized credit card clients of a Taiwanese bank in 2005, "
        "published by the UCI Machine Learning Repository. About 22% missed the next "
        "payment.",
    ),
    (
        ":material/model_training: The model",
        "LightGBM, which ranked clients best of four model types compared with "
        "cross-validation. Its probabilities are calibrated, so 30% means about 30% of "
        "such clients default.",
    ),
    (
        ":material/balance: The key idea",
        "Missing a defaulter costs more than a false alarm on a client who would have "
        "paid. If a miss costs 5 times as much, the best cutoff is about 0.17, not the "
        "usual 0.5.",
    ),
    (
        ":material/task_alt: The result",
        "On 6,000 clients the model never saw, it caught 77% of defaulters. The cost per "
        "client was 29% lower than with a 0.5 cutoff, using the same model.",
    ),
]
# The About tab's headline numbers: label, value, an optional line under it, and the
# tooltip (the ⓘ next to the label). From the test-set table in docs/results.md.
HEADLINE_NUMBERS = [
    (
        "Test clients the model never saw",
        "6,000",
        None,
        "20% of the clients were set aside before any modelling and scored once at the "
        "end. No choice about the model was made from them.",
    ),
    (
        "Defaulters caught",
        "77%",
        None,
        "1,016 of the 1,327 test clients who defaulted had a probability at or above "
        "the 0.17 cutoff, so they would have been flagged for review.",
    ),
    (
        "Cost of mistakes vs the usual 0.5 cutoff",
        "29% lower",
        "0.53 vs 0.75 per client",
        "A missed defaulter costs 5, a false alarm costs 1. Averaged over 6,000 test "
        "clients, the 0.17 cutoff's mistakes cost 0.53 per client and 0.5's cost 0.75. "
        "Same model; only the cutoff differs.",
    ),
    (
        "Cutoff for a 5x cost ratio",
        "0.17",
        None,
        "Flag a client when their probability of default is 0.17 or more. With a miss "
        "costing 5 times a false alarm, this cutoff had the lowest cost in a search on "
        "the training clients (close to 1 / (1 + 5)).",
    ),
]
TRY_CARDS = [
    (
        "go_client",
        "Try a client",
        ":material/person_search:",
        "Pick a real test client, change their payment behavior, credit limit or "
        "demographics, and watch the prediction and the reasons behind it change.",
    ),
    (
        "go_cost",
        "Cost ratio",
        ":material/tune:",
        "Click the cost of a miss up or down and see how the cutoff moves, how many "
        "defaulters are caught, and how many clients who would have paid are flagged.",
    ),
]

with about:
    st.subheader("Who will miss next month's credit card payment?")
    st.markdown(
        "A model that ranks credit card clients by their risk of default, and a cutoff "
        "chosen from what each kind of mistake costs."
    )

    with st.container(border=True):
        for column, (label, value, note, help_text) in zip(
            st.columns(4), HEADLINE_NUMBERS, strict=True
        ):
            # note: a small gray line under the number (a delta with no arrow or color).
            column.metric(
                label, value, delta=note, delta_color="off", delta_arrow="off", help=help_text
            )

    for column, (title, body) in zip(st.columns(4), ABOUT_CARDS, strict=True):
        with column.container(border=True, height="stretch"):
            st.markdown(f"**{title}**")
            st.markdown(body)

    st.subheader("What to try")
    for column, (key, label, icon, body) in zip(st.columns(2), TRY_CARDS, strict=True):
        with column.container(border=True, height="stretch"):
            st.markdown(f"**{icon} {label}**")
            st.markdown(body)
            st.button(
                f"Open {label}",
                key=key,
                type="primary",
                icon=":material/arrow_forward:",
                icon_position="right",
                on_click=open_tab,
                args=(label,),
            )

    st.markdown(f":material/code: [Code on GitHub]({REPO}) · [Full write-up]({RESULTS})")

# ---- Tab 2: Try a client -----------------------------------------------------------

with client_tab:
    left, right = st.columns(2, gap="large")
    row = current_row()
    position = st.session_state.client

    with left:
        st.subheader("Pick a test client")
        # The four examples laid out as a confusion matrix: rows are what actually
        # happened, columns what the model decided at the saved cutoff.
        with st.container(border=True):
            corner, flagged, cleared = st.columns([1, 2, 2])
            flagged.markdown("**Flagged**")
            cleared.markdown("**Not flagged**")
            for actual, keys in (
                ("Defaulted", ("caught", "missed")),
                ("Paid", ("false_alarm", "cleared")),
            ):
                label_col, *cells = st.columns([1, 2, 2], vertical_alignment="center")
                label_col.markdown(f"**{actual}**")
                for cell, key in zip(cells, keys, strict=True):
                    cell.button(
                        MATRIX_LABELS[key],
                        key=f"example_{key}",
                        on_click=show_example,
                        args=(key,),
                        disabled=key not in assets.examples,
                        # The selected example stands out.
                        type="primary" if st.session_state.pick == key else "secondary",
                        width="stretch",
                    )
        st.button("A random test client", key="random", on_click=show_random)
        st.caption(f"Showing: {st.session_state.shown} (test client ID {int(row['ID'].iloc[0])})")

        st.subheader("Change a few things")
        st.radio("Payment behavior", sc.SCENARIOS, key="scenario")
        low, high = sc.widen(
            assets.ranges.limit_min, assets.ranges.limit_max, int(row["LIMIT_BAL"].iloc[0])
        )
        st.slider("Credit limit (NT$)", low, high, step=1000, key="limit")
        with st.expander("Fairness check: change only demographics"):
            st.selectbox("Sex", list(sc.SEX_LABELS), format_func=sc.SEX_LABELS.get, key="SEX")
            st.selectbox(
                "Education",
                list(sc.EDUCATION_LABELS),
                format_func=sc.EDUCATION_LABELS.get,
                key="EDUCATION",
            )
            st.selectbox(
                "Marital status",
                list(sc.MARRIAGE_LABELS),
                format_func=sc.MARRIAGE_LABELS.get,
                key="MARRIAGE",
            )
            age_low, age_high = sc.widen(
                assets.ranges.age_min, assets.ranges.age_max, int(row["AGE"].iloc[0])
            )
            st.slider("Age", age_low, age_high, key="AGE")
            st.markdown(
                "- The model was trained with these columns, so it can use them. "
                "Sex: female ranks 10th of 28 columns by average effect and lowers scores "
                "slightly.\n"
                "- In many places a lender could not legally base a decision on some of "
                "them.\n"
                "- Changing one client is a spot check, not a fairness audit. The proper "
                "checks (error rates by group, and the model with and without these "
                f"columns) are listed in the [write-up]({RESULTS}) as future work."
            )
        st.button("Reset to original", key="reset", on_click=reset_controls)
        st.caption(
            "Scenarios rewrite all six months using patterns common in the data; bills "
            "stay as recorded."
        )

    demographics = {column: st.session_state[column] for column in sc.DEMOGRAPHICS}
    edited = sc.edit_client(row, st.session_state.scenario, st.session_state.limit, demographics)

    with right, st.container(key="result_panel"):
        before = sc.score_client(assets.model, row)
        after = sc.score_client(assets.model, edited)
        # Red box if the model flags the client, green if not.
        box = st.error if sc.is_flagged(after, CLIENT_CUTOFF) else st.success
        box(
            f"### {sc.prediction_headline(after, CLIENT_CUTOFF)}\n"
            f"**{after:.1%}** chance of default. The model flags clients at "
            f"{CLIENT_CUTOFF:.1%} or more."
        )
        if sc.is_edited(row, edited):
            # A heading-sized line, red when the edits raise the risk, green when they
            # lower it (rounded as the text is, so "No change" is gray).
            points = round((after - before) * 100, 1)
            color = "red" if points > 0 else "green" if points < 0 else "gray"
            st.markdown(f"### Your changes: :{color}[{sc.change_from_original(before, after)}]")
        actual = "Yes" if assets.test_y[position] == 1 else "No"
        st.markdown(f"**Did this client actually default?** {actual}")
        st.altair_chart(
            sc.plot_contributions(sc.top_contributions(assets.model, edited)), width="stretch"
        )
        st.caption(
            "Each bar is how much one column pushed this client's score up (toward "
            "default) or down, compared with an average client (SHAP values). The "
            f"cutoff is the saved model's ({CLIENT_CUTOFF:.3f}, from a 5x cost ratio); "
            "the Cost ratio tab shows other choices."
        )

    # Below both columns, full width: the data itself, recorded and edited side by side,
    # kept apart from the controls and the result above.
    with st.container(border=True):
        st.markdown("#### :material/table_view: The data the model reads for this client")
        st.caption(
            "**Recorded** is the client's real (anonymized) data from the 2005 dataset. "
            "**With your changes** is the same data after the controls above; the "
            "prediction is scored on it. Highlighted cells are the ones your changes "
            "altered. These are all 23 columns the model reads."
        )
        details = sc.compare_details(row, edited)
        history = sc.compare_history(row, edited)
        facts, months = st.columns([2, 5], gap="large")
        with facts:
            st.markdown("**Details**")
            st.dataframe(details.style.apply(sc.change_styles, axis=None), hide_index=True)
        with months:
            st.markdown("**Last six months, newest first**")
            st.dataframe(history.style.apply(sc.change_styles, axis=None), hide_index=True)

# ---- Tab 3: Cost ratio -------------------------------------------------------------

with cost_tab:
    panel, main = st.columns([1, 3], gap="large")

    with panel, st.container(key="ratio_panel"):
        # A number box with - / + buttons: each click moves the marker one step.
        ratio = st.number_input(
            "A missed defaulter costs N times a false alarm", 1, 20, 5, step=1, key="ratio"
        )
        cutoff = sc.cutoff_for(ratio)
        st.markdown(f"Flag clients with a default risk of **{cutoff:.1%}** or more.")
        outcomes = sc.outcomes_at_ratio(assets.test_y, assets.proba, ratio)
        defaulters = int(assets.test_y.sum())
        payers = len(assets.test_y) - defaulters
        now, at_half = outcomes.iloc[0], outcomes.iloc[1]  # this cutoff, cutoff 0.5
        st.metric(
            "Defaulters caught",
            f"{int(now['caught']):,} ({now['caught'] / defaulters:.0%})",
        )
        st.metric(
            "False alarms",
            f"{int(now['false_alarms']):,} ({now['false_alarms'] / payers:.0%})",
        )
        # delta_color="inverse": a lower cost than cutoff 0.5 shows green.
        st.metric(
            "Cost per client",
            f"{now['cost_per_client']:.3f}",
            delta=f"{now['cost_per_client'] - at_half['cost_per_client']:+.3f} vs cutoff 0.5",
            delta_color="inverse",
        )
        st.caption(
            "Cutoff = 1 / (1 + N), with a false alarm costing 1. The saved model's cutoff "
            f"at 5x came from a search on training data: {assets.saved_cutoff:.3f}. So "
            "counts at 5x differ slightly from the write-up."
        )

    with main:
        st.subheader(f"All {len(assets.test_y):,} test clients at a {ratio}x cost ratio")
        table = pd.DataFrame(
            {
                "Cutoff": [
                    "none flagged" if pd.isna(c) else f"{c:.3f}" for c in outcomes["cutoff"]
                ],
                f"Defaulters caught (of {defaulters:,})": outcomes["caught"],
                "Defaulters missed": outcomes["missed"],
                f"False alarms (of {payers:,})": outcomes["false_alarms"],
                "Cost per client": outcomes["cost_per_client"].round(3),
            },
        )
        table.index = outcomes["setup"]
        st.dataframe(table)
        st.caption(f"Cost per client: a miss costs {ratio}, a false alarm costs 1, averaged.")
        # The current cutoff in orange-red, and 0.5 in gray as a fixed reference (one
        # dot at 1x, where the two are the same).
        marks = pd.DataFrame({"label": [f"{ratio}x: cutoff {cutoff:.3f}"], "threshold": [cutoff]})
        colors = [FLAG_COLOR]
        if ratio != 1:
            marks.loc[1] = ["cutoff 0.5", 0.5]
            colors.append(REFERENCE_COLOR)
        st.altair_chart(
            plot_tradeoff_curves(
                assets.test_y,
                assets.proba,
                marks,
                points_note=f"the cutoff for {ratio}x, and 0.5",
                colors=colors,
            )
        )
        st.markdown(
            "The model's probabilities stay the same when you change the ratio. Only the "
            "line drawn through them moves: a higher cost of a miss lowers the cutoff, so "
            "more defaulters are caught and more clients who would have paid are flagged."
        )
