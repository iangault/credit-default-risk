"""Engineered features (M4): new columns built from each client's own row.

Each group adds a few columns that summarize information the raw columns hold
only indirectly. The groups come from the EDA (notebooks/01-eda.ipynb, section 6)
and the M4 line in the design spec:

- delinquency: months_late (count of PAY_1..PAY_6 >= 1) and max_delay (the worst
  PAY code). The EDA found these the strongest single signals (AUC 0.72 and 0.71).
- payment: pay_ratio, the share of the previous statement (BILL_AMT2) paid in
  PAY_AMT1, through log1p because a few clients pay many times their bill;
  no_bill, a flag for BILL_AMT2 <= 0, where the ratio has no meaning (set to 0);
  months_zero_pay, the number of months with no payment.
- utilization: BILL_AMT1 / LIMIT_BAL, how much of the credit limit is in use.
- trend: util_change, (BILL_AMT1 - BILL_AMT6) / LIMIT_BAL, whether debt grew over
  the six months; delay_trend, the mean PAY code of the latest three months minus
  that of the three before, whether the client is falling further behind.

Why utilization is also binned (in preprocess.build_encoder): the EDA showed a
U-shape. Default risk is higher for clients using almost none of their limit,
lowest around 1% to 13%, and highest at or over the limit. Logistic regression
gives each column one weight, so a column's effect on the log-odds of default is a
straight line: as utilization rises, predicted risk can only rise steadily or fall
steadily. A straight line can fit one side of the U, not both. Splitting
utilization into 10 bins, each a 0/1 column with its own weight, lets the model
give every range its own risk level and so follow the U step by step. Tree models
do not need this, since they can split utilization at several points; the raw
column is kept for them.

Every feature uses only the client's own row and learns nothing from other rows,
so adding them before the CV split cannot leak information between folds. (The bin
edges do learn from data, which is why they are fitted inside the pipeline.)
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from credit_risk.data import PAY_AMTS, PAY_STATUS

# group name -> the columns it adds, in order.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "delinquency": ("months_late", "max_delay"),
    "payment": ("pay_ratio", "no_bill", "months_zero_pay"),
    "utilization": ("utilization",),
    "trend": ("util_change", "delay_trend"),
}
GROUP_NAMES = tuple(FEATURE_GROUPS)

# New columns that the encode step also splits into quantile bins (see docstring).
BINNED = ("utilization",)


def _delinquency(X: pd.DataFrame) -> dict[str, pd.Series]:
    pay = X[PAY_STATUS]
    # Code 1 and above means a payment was late. (Code 1 is rare after PAY_1, a quirk
    # noted in the EDA, so the count mostly reflects codes 2+ in earlier months.)
    return {"months_late": (pay >= 1).sum(axis=1), "max_delay": pay.max(axis=1)}


def _payment(X: pd.DataFrame) -> dict[str, pd.Series]:
    bill = X["BILL_AMT2"]
    has_bill = bill > 0
    # bill.where(has_bill) turns non-positive bills into NaN, so the division gives NaN
    # there instead of a divide-by-zero; fillna(0) then sets those ratios to 0.
    ratio = (X["PAY_AMT1"] / bill.where(has_bill)).fillna(0)
    return {
        "pay_ratio": np.log1p(ratio),  # log1p(x) = log(1 + x): 0 stays 0, big values shrink
        "no_bill": (~has_bill).astype(int),
        "months_zero_pay": (X[PAY_AMTS] == 0).sum(axis=1),
    }


def _utilization(X: pd.DataFrame) -> dict[str, pd.Series]:
    # LIMIT_BAL is never 0 (checked in the EDA). Negative bills give negative values.
    return {"utilization": X["BILL_AMT1"] / X["LIMIT_BAL"]}


def _trend(X: pd.DataFrame) -> dict[str, pd.Series]:
    recent, earlier = PAY_STATUS[:3], PAY_STATUS[3:]  # PAY_1-3 and PAY_4-6
    return {
        "util_change": (X["BILL_AMT1"] - X["BILL_AMT6"]) / X["LIMIT_BAL"],
        "delay_trend": X[recent].mean(axis=1) - X[earlier].mean(axis=1),
    }


_BUILDERS = {
    "delinquency": _delinquency,
    "payment": _payment,
    "utilization": _utilization,
    "trend": _trend,
}


def feature_columns(groups: Iterable[str]) -> list[str]:
    """Names of the columns the given groups add, in the order given.

    Raise ValueError naming the valid options if a group is not in GROUP_NAMES.
    """
    groups = list(groups)
    unknown = [g for g in groups if g not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(
            f"Unknown feature group(s) {unknown}. Valid names: {', '.join(GROUP_NAMES)}"
        )
    return [column for g in groups for column in FEATURE_GROUPS[g]]


def add_features(X: pd.DataFrame, groups: Iterable[str]) -> pd.DataFrame:
    """Return a copy of X with the columns of each requested group appended."""
    groups = list(groups)
    feature_columns(groups)  # fail early on an unknown group name
    X = X.copy()  # leave the caller's DataFrame unchanged
    for group in groups:
        for name, values in _BUILDERS[group](X).items():
            X[name] = values
    return X


def build_feature_step(groups: Iterable[str]) -> FunctionTransformer:
    """A pipeline step that adds the given groups. It learns nothing, so fit does nothing."""
    # FunctionTransformer wraps a plain function as a pipeline step; kw_args are passed
    # to it on every call. A tuple, so the step's settings cannot be changed afterwards.
    return FunctionTransformer(add_features, kw_args={"groups": tuple(groups)})
