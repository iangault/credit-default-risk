from pathlib import Path

import pytest

from credit_risk.config import load_config
from credit_risk.data import TARGET, clean, load_raw
from credit_risk.preprocess import build_preprocessor

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
