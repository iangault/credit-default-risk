"""Logic behind the Streamlit app (app/app.py), kept here so it can be tested.

The app shows the final model on the test clients: pick a client, try a few changes,
and see the probability of default, the decision at a cost-based cutoff, and which
columns moved the score. A second view shows how the cost ratio changes the counts on
all test clients. This module has no Streamlit code; app.py only lays things out.

Nothing here changes the model or chooses a setting. Like make evaluate, the app scores
the test clients once and then counts them at different cutoffs.

Payment behavior scenarios rewrite all six months together, so an edited client stays
plausible instead of, say, "paid in full" this month with every earlier month late. The
rules use patterns measured on the training clients (2026-10-14):
- 80% of clients who paid in full paid at least the previous month's bill;
- clients paying the minimum paid a median 6% of the previous bill;
- clients 2 months late in both of the last two months paid a median of about 4%, and
  "2 then 2" is the most common history for clients 2 months late now.
Bills stay as recorded. Credit limit and demographics change alone, since no other
column depends on them.

A "row" is a one-row DataFrame with the cleaned columns minus the target (ID included),
the shape model.predict_proba takes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import altair as alt
import numpy as np
import pandas as pd
from sklearn.model_selection import TunedThresholdClassifierCV

from credit_risk.config import Config, Costs
from credit_risk.data import (
    BILL_AMTS,
    PAY_AMTS,
    PAY_STATUS,
    TARGET,
    clean,
    load_raw,
    split,
    verify_raw,
)
from credit_risk.decision import load_model
from credit_risk.evaluate import decision_metrics
from credit_risk.explain import nearest_to_median, shap_explanation

# The data's six months, newest first: column 1 (PAY_1, BILL_AMT1, PAY_AMT1) is
# September 2005 and column 6 is April 2005.
MONTHS = ["Sep 2005", "Aug 2005", "Jul 2005", "Jun 2005", "May 2005", "Apr 2005"]

# Category codes after cleaning (data.py merges undocumented codes into "other").
SEX_LABELS = {1: "Male", 2: "Female"}
EDUCATION_LABELS = {1: "Graduate school", 2: "University", 3: "High school", 4: "Other or unknown"}
MARRIAGE_LABELS = {1: "Married", 2: "Single", 3: "Other"}

# Columns the fairness check lets a visitor change. Nothing else in a row depends on them.
DEMOGRAPHICS = ("SEX", "EDUCATION", "MARRIAGE", "AGE")

# The model's 28 input columns (after scaling and one-hot encoding) in plain words, for
# the "what moved this score" chart. One-hot columns name the category they stand for;
# SEX_1 is not there because the encoder drops one category of a two-value column.
PLAIN_LABELS = {
    "LIMIT_BAL": "Credit limit",
    "AGE": "Age",
    **{f"BILL_AMT{i}": f"Bill, {month}" for i, month in enumerate(MONTHS, start=1)},
    **{f"PAY_AMT{i}": f"Payment, {month}" for i, month in enumerate(MONTHS, start=1)},
    **{f"PAY_{i}": f"Payment status, {month}" for i, month in enumerate(MONTHS, start=1)},
    "SEX_2": "Sex: female",
    **{f"EDUCATION_{k}": f"Education: {v.lower()}" for k, v in EDUCATION_LABELS.items()},
    **{f"MARRIAGE_{k}": f"Marital status: {v.lower()}" for k, v in MARRIAGE_LABELS.items()},
}

# The four example clients the app offers, by the group each comes from.
EXAMPLE_LABELS = {
    "caught": "A defaulter the model caught",
    "missed": "A defaulter the model missed",
    "false_alarm": "A false alarm (paid, but flagged)",
    "cleared": "A low-risk client, correctly cleared",
}

# Payment behavior options, in the order the app lists them.
SCENARIOS = (
    "As recorded",
    "Pays in full every month",
    "Pays the minimum every month",
    "Fell 2 months behind",
)
MIN_PAYMENT_SHARE = 0.06  # median minimum payment, as a share of the previous bill
LATE_PAYMENT_SHARE = 0.04  # median payment of clients 2 months late, same basis


@dataclass(frozen=True)
class Ranges:
    """Slider ranges, from the training clients: limit min to 99th percentile (so a few
    very high limits do not stretch the slider), age min to max."""

    limit_min: int
    limit_max: int
    age_min: int
    age_max: int


@dataclass(frozen=True)
class Assets:
    """Everything the app loads once at startup. None of it changes while it runs."""

    model: TunedThresholdClassifierCV
    test_X: pd.DataFrame  # test clients without the target; index 0 to n-1
    test_y: np.ndarray  # 1 = defaulted
    proba: np.ndarray  # calibrated probability of default per test client
    ranges: Ranges
    examples: dict[str, int]  # EXAMPLE_LABELS key -> row position in test_X
    saved_cutoff: float  # the saved model's cutoff (searched on train at the config's costs)


def status_label(code: int) -> str:
    """Plain words for a repayment status code (PAY_1 to PAY_6)."""
    named = {-2: "No card use", -1: "Paid in full", 0: "Paid the minimum"}
    if code in named:
        return named[code]
    return f"{code} month{'s' if code > 1 else ''} late"


def cutoff_for(ratio: float) -> float:
    """The cutoff when a missed defaulter costs `ratio` times a false alarm.

    For a client with probability p, flagging costs (1 - p) on average (a false alarm if
    they pay) and not flagging costs ratio * p (a miss if they default). Flagging is the
    cheaper choice when p is above 1 / (1 + ratio). Section 5 of docs/results.md shows the
    search on training data lands within about 0.01 of this.
    """
    return 1.0 / (1.0 + ratio)


def is_flagged(proba: float, cutoff: float) -> bool:
    """`>=`, as in the saved model's predict and evaluate.decision_metrics."""
    return proba >= cutoff


def widen(low: int, high: int, value: int) -> tuple[int, int]:
    """A slider range that also includes `value`, e.g. a client above the usual range."""
    return min(low, value), max(high, value)


def _share_of_previous_bill(out: pd.DataFrame, month: int, share: float) -> pd.Series:
    """`share` of the bill a payment in column `month` (0-based) pays: the next column's.

    A payment in September pays the August statement, which is the next column (data
    columns run newest first). A negative bill (a credit) gives 0, never a negative
    payment.
    """
    return (share * out[BILL_AMTS[month + 1]].clip(lower=0)).round()


def apply_scenario(row: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """A copy of `row` with one of SCENARIOS applied (see the module docstring)."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; expected one of {SCENARIOS}.")
    out = row.copy()  # never change the caller's row
    if scenario == "Pays in full every month":
        out[PAY_STATUS] = -1
        for month in range(5):  # the oldest payment's bill is not in the data; keep it
            out[PAY_AMTS[month]] = _share_of_previous_bill(out, month, 1.0)
    elif scenario == "Pays the minimum every month":
        out[PAY_STATUS] = 0
        for month in range(5):
            out[PAY_AMTS[month]] = _share_of_previous_bill(out, month, MIN_PAYMENT_SHARE)
    elif scenario == "Fell 2 months behind":
        out[PAY_STATUS[:2]] = 2
        for month in range(2):
            out[PAY_AMTS[month]] = _share_of_previous_bill(out, month, LATE_PAYMENT_SHARE)
    return out


def edit_client(
    row: pd.DataFrame, scenario: str, limit: float, demographics: dict[str, int]
) -> pd.DataFrame:
    """A copy of `row` with a scenario, a credit limit and demographic values applied.

    `demographics` maps any of DEMOGRAPHICS to a new value; columns left out keep theirs.
    """
    unknown = set(demographics) - set(DEMOGRAPHICS)
    if unknown:
        raise ValueError(f"Only {DEMOGRAPHICS} can be changed, not {sorted(unknown)}.")
    out = apply_scenario(row, scenario)
    out["LIMIT_BAL"] = float(limit)  # LIMIT_BAL is stored as a float in the cleaned data
    for column, value in demographics.items():
        out[column] = value
    return out


def is_edited(original: pd.DataFrame, edited: pd.DataFrame) -> bool:
    """Whether any value differs, comparing numbers (so 50000 and 50000.0 match)."""
    return not np.allclose(original.to_numpy(dtype=float), edited.to_numpy(dtype=float))


def example_clients(y: Sequence[int], proba: Sequence[float], threshold: float) -> dict[str, int]:
    """Row positions of one typical client per group (keys as in EXAMPLE_LABELS).

    Groups at `threshold`: defaulters flagged (caught) or not (missed), payers flagged
    (false alarm) or not (cleared). In each, the client nearest the group's median
    probability, the rule make explain uses. A group with no clients is left out.
    """
    y, proba = np.asarray(y), np.asarray(proba)
    flagged = proba >= threshold
    groups = {
        "caught": (y == 1) & flagged,
        "missed": (y == 1) & ~flagged,
        "false_alarm": (y == 0) & flagged,
        "cleared": (y == 0) & ~flagged,
    }
    picks = {}
    for name, in_group in groups.items():
        position = nearest_to_median(proba, in_group)
        if position is not None:
            picks[name] = position
    return picks


def describe_client(row: pd.DataFrame) -> pd.DataFrame:
    """The client's details in plain words, as a two-column table."""
    client = row.iloc[0]
    details = {
        "Credit limit": f"{client['LIMIT_BAL']:,.0f} NT$",
        "Age": f"{int(client['AGE'])}",
        "Sex": SEX_LABELS[int(client["SEX"])],
        "Education": EDUCATION_LABELS[int(client["EDUCATION"])],
        "Marital status": MARRIAGE_LABELS[int(client["MARRIAGE"])],
        "Most recent payment": status_label(int(client["PAY_1"])),
    }
    return pd.DataFrame({"Detail": list(details), "Value": list(details.values())})


def payment_history(row: pd.DataFrame) -> pd.DataFrame:
    """Six months of status, bill and payment, newest first, as display text."""
    client = row.iloc[0]
    return pd.DataFrame(
        {
            "Month": MONTHS,
            "Status": [status_label(int(client[c])) for c in PAY_STATUS],
            "Bill (NT$)": [f"{client[c]:,.0f}" for c in BILL_AMTS],
            "Payment (NT$)": [f"{client[c]:,.0f}" for c in PAY_AMTS],
        }
    )


def score_client(model: TunedThresholdClassifierCV, row: pd.DataFrame) -> float:
    """Calibrated probability of default for one client (the full saved pipeline)."""
    return float(model.predict_proba(row)[0, 1])


def top_contributions(
    model: TunedThresholdClassifierCV, row: pd.DataFrame, n: int = 8
) -> pd.DataFrame:
    """The `n` model columns that moved this client's score most, in plain words.

    Values are SHAP values (see explain.py): log-odds for LightGBM, before calibration.
    Positive pushes toward default. Sorted by size, largest first.
    """
    explanation = shap_explanation(model, row)
    table = pd.DataFrame({"feature": explanation.feature_names, "shap": explanation.values[0]})
    table["label"] = table["feature"].map(PLAIN_LABELS)
    table["direction"] = np.where(table["shap"] > 0, "Raises risk", "Lowers risk")
    order = table["shap"].abs().sort_values(ascending=False).index
    return table.loc[order].head(n).reset_index(drop=True)


def plot_contributions(table: pd.DataFrame) -> alt.Chart:
    """Horizontal bars from top_contributions: right raises risk, left lowers it."""
    return (
        alt.Chart(table)
        .mark_bar()
        .encode(
            x=alt.X("shap:Q", title="Effect on the model's score (right: toward default)"),
            # sort=None keeps the table's order (largest effect at the top).
            y=alt.Y("label:N", sort=None, title=None),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Raises risk", "Lowers risk"], range=["#c0392b", "#2e86c1"]
                ),
                title=None,
            ),
            tooltip=["label:N", alt.Tooltip("shap:Q", format=".2f")],
        )
        .properties(title="What moved this client's score most")
    )


def outcomes_at_ratio(y: Sequence[int], proba: Sequence[float], ratio: float) -> pd.DataFrame:
    """Counts and cost per client on the test clients at three settings, for one ratio.

    This cutoff (1 / (1 + ratio)), the default 0.5, and flag nobody. Cost per client
    uses the same ratio for all three (a miss costs `ratio`, a false alarm 1), so the
    rows can be compared. Flag nobody uses an infinite cutoff, which flags no one.
    Not named test_*: pytest would collect it as a test when a test file imports it.
    """
    costs = Costs(false_negative=float(ratio), false_positive=1.0)
    settings = {"This cutoff": cutoff_for(ratio), "Cutoff 0.5": 0.5, "Flag nobody": np.inf}
    rows = []
    for setup, cutoff in settings.items():
        metrics = decision_metrics(y, proba, cutoff, costs)
        rows.append(
            {
                "setup": setup,
                "cutoff": cutoff if np.isfinite(cutoff) else np.nan,
                "caught": metrics["caught"],
                "missed": metrics["missed"],
                "false_alarms": metrics["false_alarms"],
                "cost_per_client": metrics["expected_cost"],
            }
        )
    return pd.DataFrame(rows)


def load_assets(cfg: Config) -> Assets:
    """Load the saved model and rebuild the train/test split in memory.

    The split comes from the committed raw CSV with the config's seed, so it is the same
    6,000 test clients as make evaluate, and nothing is written to disk (data/processed
    is not in git). Raises DataFileError if the model is missing or stale, or the raw
    CSV does not match its checksum.
    """
    model = load_model(cfg)  # checks artifacts/model.json against the config
    verify_raw(cfg)
    train, test = split(clean(load_raw(cfg)), cfg)
    test_X = test.drop(columns=TARGET).reset_index(drop=True)
    test_y = test[TARGET].to_numpy()
    proba = model.predict_proba(test_X)[:, 1]  # one scoring, as in make evaluate
    ranges = Ranges(
        limit_min=int(train["LIMIT_BAL"].min()),
        limit_max=int(train["LIMIT_BAL"].quantile(0.99)),
        age_min=int(train["AGE"].min()),
        age_max=int(train["AGE"].max()),
    )
    saved_cutoff = float(model.best_threshold_)
    return Assets(
        model=model,
        test_X=test_X,
        test_y=test_y,
        proba=proba,
        ranges=ranges,
        # Chosen once at the saved cutoff, so each example stays the same person when the
        # app's cost slider moves.
        examples=example_clients(test_y, proba, saved_cutoff),
        saved_cutoff=saved_cutoff,
    )
