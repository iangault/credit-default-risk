from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import load_config
from credit_risk.data import BILL_AMTS, PAY_AMTS, PAY_STATUS, TARGET, clean, load_raw
from credit_risk.features import (
    BINNED,
    FEATURE_GROUPS,
    GROUP_NAMES,
    add_features,
    build_feature_step,
    feature_columns,
)

ROOT = Path(__file__).resolve().parents[1]


def client(**values):
    """One made-up client: every amount and PAY code 0 unless given."""
    row = {"LIMIT_BAL": 100_000, **dict.fromkeys(PAY_STATUS + BILL_AMTS + PAY_AMTS, 0)}
    row.update(values)
    return pd.DataFrame([row])


def test_group_names_and_binned_columns():
    assert GROUP_NAMES == ("delinquency", "payment", "utilization", "trend")
    assert BINNED == ("utilization",)
    assert feature_columns(["trend", "delinquency"]) == [
        "util_change",
        "delay_trend",
        "months_late",
        "max_delay",
    ]


def test_delinquency_counts_late_months_and_max_delay():
    X = client(PAY_1=2, PAY_2=1, PAY_3=0, PAY_4=-1, PAY_5=-2, PAY_6=3)
    out = add_features(X, ["delinquency"])
    assert out.loc[0, "months_late"] == 3  # codes 2, 1 and 3 are >= 1
    assert out.loc[0, "max_delay"] == 3


def test_pay_ratio_uses_previous_bill_with_log1p():
    out = add_features(client(PAY_AMT1=500, BILL_AMT2=1000), ["payment"])
    assert out.loc[0, "pay_ratio"] == pytest.approx(np.log1p(0.5))
    assert out.loc[0, "no_bill"] == 0


@pytest.mark.parametrize("bill", [0, -200])
def test_no_bill_sets_flag_and_zero_ratio(bill):
    out = add_features(client(PAY_AMT1=500, BILL_AMT2=bill), ["payment"])
    assert out.loc[0, "pay_ratio"] == 0
    assert out.loc[0, "no_bill"] == 1


def test_months_zero_pay_counts_zero_payments():
    X = client(PAY_AMT1=10, PAY_AMT2=0, PAY_AMT3=5, PAY_AMT4=0, PAY_AMT5=0, PAY_AMT6=1)
    assert add_features(X, ["payment"]).loc[0, "months_zero_pay"] == 3


def test_utilization_and_trend():
    X = client(BILL_AMT1=50_000, BILL_AMT6=20_000, PAY_1=2, PAY_2=2, PAY_3=2)
    out = add_features(X, ["utilization", "trend"])
    assert out.loc[0, "utilization"] == pytest.approx(0.5)
    assert out.loc[0, "util_change"] == pytest.approx(0.3)  # (50k - 20k) / 100k
    assert out.loc[0, "delay_trend"] == pytest.approx(2.0)  # mean(2, 2, 2) - mean(0, 0, 0)


def test_only_requested_groups_are_added_and_input_is_unchanged():
    X = client()
    out = add_features(X, ["utilization"])
    assert list(out.columns) == [*X.columns, "utilization"]
    assert "utilization" not in X.columns


def test_unknown_group_lists_valid_names():
    with pytest.raises(ValueError, match="delinquency, payment, utilization, trend"):
        feature_columns(["delinquncy"])


def test_repeated_group_is_refused():
    # Listing a group twice would add its columns twice, which only fails once fitting starts.
    with pytest.raises(ValueError, match="more than once"):
        feature_columns(["delinquency", "trend", "delinquency"])


def test_feature_step_adds_columns_in_a_pipeline_step():
    out = build_feature_step(["delinquency"]).fit_transform(client(PAY_1=1))
    assert out.loc[0, "months_late"] == 1


def test_no_missing_or_infinite_values_on_real_data():
    cleaned = clean(load_raw(load_config(ROOT / "configs" / "base.yaml")))
    out = add_features(cleaned.drop(columns=TARGET), GROUP_NAMES)
    new = out[[c for cols in FEATURE_GROUPS.values() for c in cols]]
    assert np.isfinite(new.to_numpy(dtype=float)).all()
