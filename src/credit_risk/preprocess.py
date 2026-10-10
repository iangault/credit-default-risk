"""Column transformations applied before any model.

- NUMERIC (credit limit, age, bill and payment amounts): standardized, so linear
  models see them on one scale. Tree models are unaffected by scaling.
- CATEGORICAL (sex, education, marriage): one-hot encoded. The codes are labels,
  not quantities, so "EDUCATION = 3" should not mean "three times EDUCATION = 1".
  A level not seen during fitting is encoded as all zeros instead of raising.
- ORDINAL (PAY_1 to PAY_6): kept as numbers, since -2 to 8 runs roughly from least
  to most risky.
- ID and any other column are dropped.

Since M3 the work is done in two steps, so a resampler can sit between them:

    build_scaler():  scale NUMERIC; keep CATEGORICAL and ORDINAL as raw code columns
    build_encoder(): one-hot CATEGORICAL; pass NUMERIC and ORDINAL through

SMOTENC (samplers.py) needs that middle state. The amounts must already be scaled,
because SMOTENC finds neighbours by distance and an unscaled LIMIT_BAL (up to
1,000,000) would outweigh AGE (20 to 79). The categories must still be one column
each, so SMOTENC can give a synthetic row a real code instead of a fractional
one-hot value such as EDUCATION_2 = 0.4.

build_preprocessor() chains the two steps with nothing between them. Its output
(columns, order and values) matches the single-step M2 version.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.data import BILL_AMTS, PAY_AMTS, PAY_STATUS

NUMERIC = ["LIMIT_BAL", "AGE", *BILL_AMTS, *PAY_AMTS]
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
ORDINAL = PAY_STATUS


def build_scaler() -> ColumnTransformer:
    """Step 1, a fresh unfitted ColumnTransformer: scale NUMERIC, pass the codes through."""
    scaler = make_column_transformer(
        (StandardScaler(), NUMERIC),
        # Left as raw codes here; build_encoder one-hot encodes CATEGORICAL later,
        # after any resampling.
        ("passthrough", CATEGORICAL + ORDINAL),
        remainder="drop",  # drops ID
        verbose_feature_names_out=False,
    )
    # pandas output keeps the column names, which SMOTENC uses to find the
    # categorical columns.
    return scaler.set_output(transform="pandas")


def build_encoder() -> ColumnTransformer:
    """Step 2, a fresh unfitted ColumnTransformer: one-hot CATEGORICAL, pass the rest through.

    Expects the output of build_scaler (or of a resampler placed after it).
    """
    encoder = make_column_transformer(
        # All three groups are listed (instead of remainder="passthrough") to keep
        # the M2 column order: amounts, one-hot columns, then PAY codes.
        ("passthrough", NUMERIC),
        (
            OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
            CATEGORICAL,
        ),
        ("passthrough", ORDINAL),
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return encoder.set_output(transform="pandas")


def build_preprocessor() -> Pipeline:
    """Both steps chained with no resampling: raw columns in, model-ready DataFrame out.

    pipeline.py uses the two steps separately; this is for notebooks and tests that
    want the whole transformation as one object.
    """
    return make_pipeline(build_scaler(), build_encoder())
