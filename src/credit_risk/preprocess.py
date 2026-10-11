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

Since M4 both steps accept the engineered columns from features.py: extra numeric
columns are scaled with the amounts, and columns listed as binned (utilization)
are also split into quantile bins by the encoder. See build_encoder for why.
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

from credit_risk.data import BILL_AMTS, PAY_AMTS, PAY_STATUS

NUMERIC = ["LIMIT_BAL", "AGE", *BILL_AMTS, *PAY_AMTS]
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
ORDINAL = PAY_STATUS


def build_scaler(extra_numeric: Sequence[str] = ()) -> ColumnTransformer:
    """Step 1, a fresh unfitted ColumnTransformer: scale NUMERIC, pass the codes through.

    `extra_numeric`: engineered columns (features.py) to scale along with the amounts.
    """
    scaler = make_column_transformer(
        (StandardScaler(), [*NUMERIC, *extra_numeric]),
        # Left as raw codes here; build_encoder one-hot encodes CATEGORICAL later,
        # after any resampling.
        ("passthrough", CATEGORICAL + ORDINAL),
        remainder="drop",  # drops ID
        verbose_feature_names_out=False,
    )
    # pandas output keeps the column names, which SMOTENC uses to find the
    # categorical columns.
    return scaler.set_output(transform="pandas")


def build_encoder(
    extra_numeric: Sequence[str] = (), binned: Sequence[str] = ()
) -> ColumnTransformer:
    """Step 2, a fresh unfitted ColumnTransformer: one-hot CATEGORICAL, pass the rest through.

    Expects the output of build_scaler (or of a resampler placed after it).
    `extra_numeric`: engineered columns to pass through. `binned`: columns (already
    in extra_numeric) to also split into quantile bins.
    """
    parts = [
        # Listing each group (instead of remainder="passthrough") keeps the column
        # order fixed: amounts, engineered columns, one-hot columns, bins, PAY codes.
        ("passthrough", [*NUMERIC, *extra_numeric]),
        (
            OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
            CATEGORICAL,
        ),
    ]
    if binned:
        # Why bins (M4): logistic regression gives each column one weight, so the
        # column's effect on the log-odds of default is a straight line. Utilization
        # has a U-shaped link to default (higher at near-zero use, lowest around 1% to
        # 13%, highest at or over the limit), and a straight line can follow only one
        # side of a U. Ten bins, each a 0/1 column with its own weight, let the model
        # give each range its own risk level. "quantile" puts about the same number of
        # clients in each bin. The edges are learned when the pipeline is fitted, so in
        # cross-validation they come from the training folds only. The bins are cut
        # from the scaled column, which gives the same bins as the raw one.
        parts.append(
            (
                KBinsDiscretizer(
                    n_bins=10,
                    encode="onehot-dense",
                    strategy="quantile",
                    quantile_method="averaged_inverted_cdf",
                ),
                list(binned),
            )
        )
    parts.append(("passthrough", ORDINAL))
    encoder = make_column_transformer(*parts, remainder="drop", verbose_feature_names_out=False)
    return encoder.set_output(transform="pandas")


def build_preprocessor() -> Pipeline:
    """Both steps chained with no resampling: raw columns in, model-ready DataFrame out.

    pipeline.py uses the two steps separately; this is for notebooks and tests that
    want the whole transformation as one object.
    """
    return make_pipeline(build_scaler(), build_encoder())
