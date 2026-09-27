"""Column transformations applied before any model.

- NUMERIC (credit limit, age, bill and payment amounts): standardized, so linear
  models see them on one scale. Tree models are unaffected by scaling.
- CATEGORICAL (sex, education, marriage): one-hot encoded. The codes are labels,
  not quantities, so "EDUCATION = 3" should not mean "three times EDUCATION = 1".
  A level not seen during fitting is encoded as all zeros instead of raising.
- ORDINAL (PAY_1 to PAY_6): kept as numbers, since -2 to 8 runs roughly from least
  to most risky.
- ID and any other column are dropped.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.data import BILL_AMTS, PAY_AMTS, PAY_STATUS

NUMERIC = ["LIMIT_BAL", "AGE", *BILL_AMTS, *PAY_AMTS]
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
ORDINAL = PAY_STATUS


def build_preprocessor() -> ColumnTransformer:
    """A fresh, unfitted ColumnTransformer that outputs a pandas DataFrame."""
    preprocessor = make_column_transformer(
        (StandardScaler(), NUMERIC),
        (
            OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
            CATEGORICAL,
        ),
        ("passthrough", ORDINAL),
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor.set_output(transform="pandas")
