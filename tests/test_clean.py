from pathlib import Path

import pandas as pd
import pytest

from credit_risk.config import load_config
from credit_risk.data import (
    CLEAN_COLUMNS,
    RAW_COLUMNS,
    TARGET,
    DataFileError,
    clean,
    load_raw,
)

ROOT = Path(__file__).resolve().parents[1]


def raw_frame(n=4, **columns):
    """A tiny raw-format DataFrame with valid defaults; keyword args override columns."""
    data = {col: [0] * n for col in RAW_COLUMNS}
    data.update(
        ID=list(range(1, n + 1)),
        LIMIT_BAL=[50000.0] * n,
        SEX=[1] * n,
        EDUCATION=[2] * n,
        MARRIAGE=[1] * n,
        AGE=[30] * n,
    )
    data.update(columns)
    return pd.DataFrame(data)


def test_clean_renames_pay_0_and_target():
    out = clean(raw_frame())
    assert list(out.columns) == CLEAN_COLUMNS
    assert "PAY_1" in out.columns and "PAY_0" not in out.columns
    assert TARGET in out.columns


def test_clean_merges_undocumented_education_codes():
    out = clean(raw_frame(EDUCATION=[0, 5, 6, 1]))
    assert out["EDUCATION"].tolist() == [4, 4, 4, 1]


def test_clean_merges_undocumented_marriage_code():
    out = clean(raw_frame(MARRIAGE=[0, 1, 2, 3]))
    assert out["MARRIAGE"].tolist() == [3, 1, 2, 3]


def test_clean_keeps_pay_status_and_negative_bills():
    out = clean(raw_frame(PAY_0=[-2, -1, 0, 8], BILL_AMT1=[-500, 0, 10, 20]))
    assert out["PAY_1"].tolist() == [-2, -1, 0, 8]
    assert out["BILL_AMT1"].tolist() == [-500, 0, 10, 20]


def test_clean_does_not_modify_input():
    raw = raw_frame(EDUCATION=[0, 5, 6, 1], MARRIAGE=[0, 1, 2, 3])
    before = raw.copy()
    clean(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_clean_rejects_unknown_category_code():
    with pytest.raises(DataFileError, match=r"EDUCATION.*7"):
        clean(raw_frame(EDUCATION=[7, 1, 1, 1]))


def test_clean_rejects_out_of_range_pay_status():
    with pytest.raises(DataFileError, match="PAY_1"):
        clean(raw_frame(PAY_0=[9, 0, 0, 0]))


def test_clean_rejects_missing_column():
    with pytest.raises(DataFileError, match="AGE"):
        clean(raw_frame().drop(columns="AGE"))


def test_clean_on_real_data():
    out = clean(load_raw(load_config(ROOT / "configs" / "base.yaml")))
    assert out.shape == (30000, 25)
    assert set(out["EDUCATION"]) == {1, 2, 3, 4}
    assert set(out["MARRIAGE"]) == {1, 2, 3}
    assert out[TARGET].mean() == pytest.approx(0.2212, abs=0.0001)
