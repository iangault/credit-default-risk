"""Streamlit app: try the credit default model on clients it never saw.

Layout only. The logic (example clients, payment scenarios, scoring, SHAP
contributions, test-set counts) is in src/credit_risk/showcase.py, which has tests.
Run locally with `make app`; deployed on Streamlit Community Cloud (app/requirements.txt).

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

st.set_page_config(page_title="Credit default risk", layout="wide")


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
    st.session_state.pick = key  # the example list shows which one is selected
    st.session_state.shown = sc.EXAMPLE_LABELS[key]
    reset_controls()


def show_picked() -> None:
    """Called when the example list changes (on_change)."""
    show_example(st.session_state.pick)


def show_random() -> None:
    st.session_state.client = int(np.random.default_rng().integers(len(assets.test_X)))
    st.session_state.pick = None  # no example selected in the list
    st.session_state.shown = "A random test client"
    reset_controls()


# First run in a browser session: start on the first example (the caught defaulter).
if "client" not in st.session_state:
    show_example(next(iter(assets.examples)))

# The cutoff that sorted the examples into their groups, so each example's decision
# matches its group name.
CLIENT_CUTOFF = assets.saved_cutoff
FLAG_COLOR, REFERENCE_COLOR = "#e4572e", "#7f7f7f"  # orange-red stands out from the blue curve

about, client_tab, cost_tab = st.tabs(["About", "Try a client", "Cost ratio"])

# ---- Tab 1: About ------------------------------------------------------------------

with about:
    st.header("Who will miss next month's credit card payment?")
    st.markdown(
        "This app shows a model that predicts whether a credit card client will miss "
        "next month's payment.\n\n"
        "- **The data:** 30,000 real, anonymized clients of a Taiwanese bank in 2005, "
        "published by the UCI Machine Learning Repository.\n"
        "- **The model:** LightGBM, with probabilities calibrated so that 30% means "
        "about 30% of such clients default.\n"
        "- **The key idea:** missing a defaulter costs more than a false alarm on a "
        "client who would have paid. If a miss costs 5 times as much, the best cutoff is "
        "about 0.17, not the usual 0.5.\n"
        "- **The result:** on 6,000 clients the model never saw, it caught 77% of "
        "defaulters, and the cost per client was 29% lower than with a 0.5 cutoff.\n\n"
        "**Try a client** lets you pick a real client, change a few things, and see the "
        "model's prediction at its saved cutoff. **Cost ratio** lets you change how much "
        "a miss costs and shows what the cutoff then does across all 6,000 clients.\n\n"
        f"[Code on GitHub]({REPO}) · [Full write-up]({RESULTS})"
    )

# ---- Tab 2: Try a client -----------------------------------------------------------

with client_tab:
    left, right = st.columns(2, gap="large")
    row = current_row()
    position = st.session_state.client

    with left:
        st.subheader("Pick a test client")
        # One list instead of a button per example keeps the controls short enough to
        # sit beside the result. Its value is None after "random" (nothing selected).
        st.radio(
            "Example clients",
            [key for key in sc.EXAMPLE_LABELS if key in assets.examples],
            format_func=sc.EXAMPLE_LABELS.get,
            key="pick",
            on_change=show_picked,
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

    with right:
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
            st.markdown(f"**Your changes:** {sc.change_from_original(before, after)}")
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

    # Below both columns, collapsed, so the controls and the result fit on one screen.
    with st.expander("Client details and payment history"):
        st.dataframe(sc.describe_client(edited), hide_index=True)
        st.dataframe(sc.payment_history(edited), hide_index=True)

# ---- Tab 3: Cost ratio -------------------------------------------------------------

with cost_tab:
    # A number box with - / + buttons: each click moves the marker one step.
    ratio = st.number_input(
        "A missed defaulter costs N times a false alarm", 1, 20, 5, step=1, key="ratio"
    )
    cutoff = sc.cutoff_for(ratio)
    st.markdown(f"Flag clients with a default risk of **{cutoff:.1%}** or more.")
    st.caption(
        "Cutoff = 1 / (1 + N), with a false alarm costing 1. The saved model's cutoff at "
        f"5x came from a search on training data: {assets.saved_cutoff:.3f}. So counts "
        "at 5x differ slightly from the write-up."
    )
    outcomes = sc.outcomes_at_ratio(assets.test_y, assets.proba, ratio)
    defaulters = int(assets.test_y.sum())
    payers = len(assets.test_y) - defaulters
    st.subheader(f"All {len(assets.test_y):,} test clients at a {ratio}x cost ratio")
    table = pd.DataFrame(
        {
            "Cutoff": ["none flagged" if pd.isna(c) else f"{c:.3f}" for c in outcomes["cutoff"]],
            f"Defaulters caught (of {defaulters:,})": outcomes["caught"],
            "Defaulters missed": outcomes["missed"],
            f"False alarms (of {payers:,})": outcomes["false_alarms"],
            "Cost per client": outcomes["cost_per_client"].round(3),
        },
    )
    table.index = outcomes["setup"]
    st.dataframe(table)
    st.caption(f"Cost per client: a miss costs {ratio}, a false alarm costs 1, averaged.")
    # The current cutoff in orange-red, and 0.5 in gray as a fixed reference (one dot
    # at 1x, where the two are the same).
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
        "The model's probabilities stay the same when you move the slider. Only the line "
        "drawn through them moves: a higher cost of a miss lowers the cutoff, so more "
        "defaulters are caught and more clients who would have paid are flagged."
    )
