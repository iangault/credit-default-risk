from pathlib import Path

import pytest

from credit_risk.config import load_config
from credit_risk.data import TARGET, clean, load_raw
from credit_risk.features import add_features
from credit_risk.preprocess import (
    CATEGORICAL,
    NUMERIC,
    ORDINAL,
    build_encoder,
    build_preprocessor,
    build_scaler,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def features():
    cleaned = clean(load_raw(load_config(ROOT / "configs" / "base.yaml")))
    return cleaned.drop(columns=TARGET).head(2000)


def test_output_has_expected_columns(features):
    out = build_preprocessor().fit_transform(features)
    assert out.shape == (2000, 28)
    assert "ID" not in out.columns
    assert "SEX_2" in out.columns  # binary category becomes one column
    assert {"EDUCATION_1", "EDUCATION_4", "MARRIAGE_3"} <= set(out.columns)
    assert "PAY_1" in out.columns


def test_pay_status_passes_through_unscaled(features):
    out = build_preprocessor().fit_transform(features)
    assert out["PAY_1"].tolist() == features["PAY_1"].tolist()


def test_numeric_columns_are_scaled(features):
    out = build_preprocessor().fit_transform(features)
    assert abs(out["LIMIT_BAL"].mean()) < 1e-9
    assert out["LIMIT_BAL"].std(ddof=0) == pytest.approx(1.0)


def test_unseen_category_is_ignored(features):
    fit_rows = features[features["EDUCATION"] != 4]
    pre = build_preprocessor().fit(fit_rows)
    with pytest.warns(UserWarning, match="unknown categories"):
        out = pre.transform(features[features["EDUCATION"] == 4])
    education = [c for c in out.columns if c.startswith("EDUCATION_")]
    assert (out[education] == 0).all().all()


def test_scaler_keeps_codes_as_single_columns(features):
    # The middle state SMOTENC needs: amounts scaled, codes still one column each.
    out = build_scaler().fit_transform(features)
    assert list(out.columns) == [*NUMERIC, *CATEGORICAL, *ORDINAL]
    codes = CATEGORICAL + ORDINAL
    assert out[codes].values.tolist() == features[codes].values.tolist()
    assert abs(out["LIMIT_BAL"].mean()) < 1e-9


def test_preprocessor_column_order_matches_m2(features):
    # Pins the M2 layout so the split into two steps cannot reorder columns
    # (tree models can shift slightly when column order changes).
    out = build_preprocessor().fit_transform(features)
    one_hot = ["SEX_2", "EDUCATION_1", "EDUCATION_2", "EDUCATION_3", "EDUCATION_4"]
    one_hot += ["MARRIAGE_1", "MARRIAGE_2", "MARRIAGE_3"]
    assert list(out.columns) == [*NUMERIC, *one_hot, *ORDINAL]


def test_scaler_scales_extra_numeric_columns(features):
    X = add_features(features, ["utilization"])
    out = build_scaler(["utilization"]).fit_transform(X)
    assert list(out.columns) == [*NUMERIC, "utilization", *CATEGORICAL, *ORDINAL]
    assert abs(out["utilization"].mean()) < 1e-9


def test_bins_are_one_hot_with_one_bin_per_row(features):
    X = add_features(features, ["utilization"])
    scaled = build_scaler(["utilization"]).fit_transform(X)
    out = build_encoder(["utilization"], ["utilization"]).fit_transform(scaled)
    bins = [f"utilization_{i}.0" for i in range(10)]
    assert bins == [c for c in out.columns if c.startswith("utilization_")]
    assert "utilization" in out.columns  # the raw (scaled) column is kept too
    assert (out[bins].sum(axis=1) == 1).all()
