"""Logic behind the Streamlit app (app/app.py), kept here so it can be tested.

The app shows the final model on the test clients: pick a client, try a few changes,
and see the probability of default, the decision at a cost-based cutoff, and which
columns moved the score. A second view shows how the cost ratio changes the counts on
all test clients. This module has no Streamlit code; app.py only lays things out.

Nothing here changes the model or chooses a setting. Like make evaluate, the app scores
the test clients once and then counts them at different cutoffs.

A "row" is a one-row DataFrame with the cleaned columns minus the target (ID included),
the shape model.predict_proba takes.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from credit_risk.data import BILL_AMTS, PAY_AMTS, PAY_STATUS
from credit_risk.explain import nearest_to_median

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
