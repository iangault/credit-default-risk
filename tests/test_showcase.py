from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from credit_risk.config import Final, load_config
from credit_risk.data import TARGET, clean, load_raw
from credit_risk.decision import train_final
from credit_risk.explain import inner_pipeline, nearest_to_median
from credit_risk.showcase import (
    EXAMPLE_LABELS,
    MONTHS,
    PLAIN_LABELS,
    cutoff_for,
    describe_client,
    example_clients,
    is_flagged,
    payment_history,
    status_label,
    widen,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = load_config(ROOT / "configs" / "base.yaml")


def small_cfg(tmp_path):
    paths = replace(
        BASE.paths, reports_dir=tmp_path / "reports", artifacts_dir=tmp_path / "artifacts"
    )
    return replace(BASE, paths=paths, cv_folds=2, final=Final("lgbm", "sigmoid"))


@pytest.fixture(scope="module")
def clients():
    # Samples of the cleaned raw data stand in for train and test (the split is not read).
    rows = clean(load_raw(BASE)).sample(1200, random_state=0)
    return rows.iloc[:1000], rows.iloc[1000:].drop(columns=TARGET)


@pytest.fixture(scope="module")
def model(clients, tmp_path_factory):
    train, _ = clients
    return train_final(small_cfg(tmp_path_factory.mktemp("m")), train)


@pytest.fixture
def row(clients):
    """One client with a hand-set history: a negative bill in month 3, all 1 month late."""
    _, test = clients
    one = test.iloc[[0]].copy()
    bills = [1000.0, 2000.0, -500.0, 4000.0, 5000.0, 6000.0]
    one[[f"BILL_AMT{i}" for i in range(1, 7)]] = bills
    one[[f"PAY_AMT{i}" for i in range(1, 7)]] = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    one[[f"PAY_{i}" for i in range(1, 7)]] = 1
    one[["LIMIT_BAL", "AGE", "SEX", "EDUCATION", "MARRIAGE"]] = [50000.0, 30, 2, 2, 1]
    return one


def test_cutoff_for_five_to_one_is_one_sixth():
    assert cutoff_for(5) == pytest.approx(1 / 6)
    assert cutoff_for(1) == pytest.approx(0.5)


def test_is_flagged_at_the_cutoff():
    # >=, as in the saved model's predict and decision_metrics.
    assert is_flagged(0.2, 0.2)
    assert not is_flagged(0.19, 0.2)


def test_widen_includes_the_clients_own_value():
    assert widen(10, 20, 15) == (10, 20)
    assert widen(10, 20, 25) == (10, 25)
    assert widen(10, 20, 5) == (5, 20)


def test_nearest_to_median_picks_the_middle_client_and_skips_empty_groups():
    proba = np.array([0.9, 0.8, 0.6, 0.1])
    assert nearest_to_median(proba, np.array([True, True, True, False])) == 1
    assert nearest_to_median(proba, np.zeros(4, dtype=bool)) is None


def test_example_clients_are_each_in_their_group():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])
    proba = np.array([0.9, 0.8, 0.6, 0.2, 0.7, 0.55, 0.3, 0.1, 0.2])
    picks = example_clients(y, proba, 0.5)
    assert set(picks) == set(EXAMPLE_LABELS) == {"caught", "missed", "false_alarm", "cleared"}
    assert picks["caught"] == 1  # 0.8 is the median of 0.9, 0.8, 0.6
    assert y[picks["missed"]] == 1 and proba[picks["missed"]] < 0.5
    assert y[picks["false_alarm"]] == 0 and proba[picks["false_alarm"]] >= 0.5
    assert y[picks["cleared"]] == 0 and proba[picks["cleared"]] < 0.5


def test_example_clients_skip_an_empty_group():
    picks = example_clients(np.zeros(4, dtype=int), np.array([0.1, 0.2, 0.7, 0.9]), 0.5)
    assert set(picks) == {"false_alarm", "cleared"}


def test_status_labels():
    assert [status_label(c) for c in (-2, -1, 0, 1, 2)] == [
        "No card use",
        "Paid in full",
        "Paid the minimum",
        "1 month late",
        "2 months late",
    ]


def test_describe_client_in_plain_words(row):
    table = describe_client(row).set_index("Detail")["Value"]
    assert table["Credit limit"] == "50,000 NT$"
    assert table["Age"] == "30"
    assert table["Sex"] == "Female"
    assert table["Education"] == "University"
    assert table["Marital status"] == "Married"
    assert table["Most recent payment"] == "1 month late"


def test_payment_history_has_six_months_newest_first(row):
    history = payment_history(row)
    assert history["Month"].tolist() == MONTHS
    assert MONTHS[0] == "Sep 2005"
    assert history["Status"].iloc[0] == "1 month late"
    assert history["Bill (NT$)"].iloc[2] == "-500"
    assert history["Payment (NT$)"].iloc[0] == "11"


def test_plain_labels_cover_every_model_column(model, clients):
    _, test = clients
    columns = inner_pipeline(model)[:-1].transform(test.head(2)).columns
    assert set(columns) == set(PLAIN_LABELS)
