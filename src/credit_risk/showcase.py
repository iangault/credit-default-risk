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

# Payment behavior options, in the order the app lists them.
SCENARIOS = (
    "As recorded",
    "Pays in full every month",
    "Pays the minimum every month",
    "Fell 2 months behind",
)
MIN_PAYMENT_SHARE = 0.06  # median minimum payment, as a share of the previous bill
LATE_PAYMENT_SHARE = 0.04  # median payment of clients 2 months late, same basis


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
