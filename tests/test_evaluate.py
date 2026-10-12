import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import Costs, Final, load_config
from credit_risk.data import TARGET, clean, load_raw
from credit_risk.decision import decision_table_path, train_final
from credit_risk.evaluate import (
    COLUMNS,
    calibration_plot_path,
    decision_metrics,
    evaluate,
    metrics_table_path,
    plot_calibration,
    save_evaluation,
    train_estimate,
)
from credit_risk.experiments import experiments_table_path

ROOT = Path(__file__).resolve().parents[1]
BASE = load_config(ROOT / "configs" / "base.yaml")
COSTS = Costs(false_negative=5.0, false_positive=1.0)


def small_cfg(tmp_path, **changes):
    paths = replace(
        BASE.paths, reports_dir=tmp_path / "reports", artifacts_dir=tmp_path / "artifacts"
    )
    changes.setdefault("final", Final("logreg", "sigmoid"))
    return replace(BASE, paths=paths, cv_folds=2, **changes)


@pytest.fixture(scope="module")
def data():
    # Samples of the cleaned raw data stand in for train and test; the test split is
    # not read by the test suite.
    rows = clean(load_raw(BASE)).sample(1500, random_state=0)
    return rows.iloc[:1000], rows.iloc[1000:]


def test_decision_metrics_hand_worked_example():
    # Flagged at 0.5: clients 0, 2, 4. Defaulters 0 and 4 caught, defaulter 1 missed,
    # client 2 a false alarm. Cost (5 * 1 + 1 * 1) / 5 = 1.2.
    m = decision_metrics([1, 1, 0, 0, 1], [0.9, 0.1, 0.6, 0.2, 0.5], 0.5, COSTS)
    assert (m["flagged"], m["caught"], m["missed"], m["false_alarms"]) == (3, 2, 1, 1)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["expected_cost"] == pytest.approx(1.2)
    assert m["threshold"] == 0.5


def test_decision_metrics_with_nobody_flagged_has_zero_precision():
    m = decision_metrics([1, 0], [0.1, 0.2], 0.5, COSTS)
    assert m["flagged"] == 0
    assert m["precision"] == 0  # zero_division=0, not a warning
    assert m["expected_cost"] == pytest.approx(2.5)


def write_tables(cfg):
    experiments_table_path(cfg).parent.mkdir(parents=True)
    pd.DataFrame({"experiment": ["logreg"], "roc_auc_mean": [0.72], "pr_auc_mean": [0.50]}).to_csv(
        experiments_table_path(cfg), index=False
    )
    pd.DataFrame(
        {
            "experiment": ["logreg", "logreg"],
            "calibration": ["sigmoid", "sigmoid"],
            "false_negative_cost": [5.0, 3.0],
            "false_positive_cost": [1.0, 1.0],
            "threshold": [0.17, 0.25],
            "expected_cost": [0.57, 0.41],
        }
    ).to_csv(decision_table_path(cfg), index=False)


def test_train_estimate_reads_the_chosen_setup(tmp_path):
    cfg = small_cfg(tmp_path)
    write_tables(cfg)
    assert train_estimate(cfg) == {
        "threshold": 0.17,
        "roc_auc": 0.72,
        "pr_auc": 0.50,
        "expected_cost": 0.57,
    }


def test_train_estimate_is_blank_when_the_false_alarm_cost_changed(tmp_path):
    cfg = small_cfg(tmp_path, costs=Costs(false_negative=5.0, false_positive=2.0))
    write_tables(cfg)
    estimate = train_estimate(cfg)
    assert math.isnan(estimate["threshold"]) and math.isnan(estimate["expected_cost"])


def test_train_estimate_is_blank_when_tables_lack_the_setup(tmp_path):
    cfg = small_cfg(tmp_path, costs=Costs(false_negative=7.0, false_positive=1.0))
    write_tables(cfg)
    estimate = train_estimate(cfg)
    assert estimate["roc_auc"] == 0.72  # experiments.csv still has the model
    assert math.isnan(estimate["threshold"]) and math.isnan(estimate["expected_cost"])
    # No tables at all: every cell blank, no error.
    assert all(math.isnan(v) for v in train_estimate(small_cfg(tmp_path / "empty")).values())


def test_evaluate_table_rows_and_baselines(data, tmp_path):
    train, test = data
    cfg = small_cfg(tmp_path)
    model = train_final(cfg, train)
    table, proba = evaluate(model, test, cfg)
    assert list(table.columns) == COLUMNS
    assert table["setup"].tolist() == [
        "model at tuned threshold",
        "model at 0.5",
        "flag nobody",
        "flag everyone",
        "train CV estimate",
    ]
    rows = table.set_index("setup")
    y = test[TARGET]
    assert rows.loc["model at tuned threshold", "threshold"] == pytest.approx(
        model.best_threshold_
    )
    assert rows.loc["flag nobody", "expected_cost"] == pytest.approx(5.0 * y.mean())
    assert rows.loc["flag everyone", "expected_cost"] == pytest.approx(1.0 * (1 - y.mean()))
    assert rows.loc["flag everyone", "recall"] == 1
    assert len(proba) == len(test)
    assert np.array_equal(proba >= model.best_threshold_, model.predict(test.drop(columns=TARGET)))


def test_calibration_plot_has_points_and_diagonal():
    rng = np.random.default_rng(0)
    proba = rng.uniform(0, 1, 500)
    y = (rng.uniform(0, 1, 500) < proba).astype(int)
    spec = plot_calibration(y, proba).to_dict()
    assert len(spec["layer"]) == 2  # observed points/line, and the diagonal
    assert "Calibration" in spec["title"]["text"]


def test_save_evaluation_writes_table_and_plot(data, tmp_path):
    train, test = data
    cfg = small_cfg(tmp_path)
    table, proba = evaluate(train_final(cfg, train), test, cfg)
    table_path, plot_path = save_evaluation(table, test[TARGET], proba, cfg)
    assert table_path == metrics_table_path(cfg) == tmp_path / "reports/tables/test_metrics.csv"
    assert plot_path == calibration_plot_path(cfg)
    assert pd.read_csv(table_path)["setup"].tolist() == table["setup"].tolist()
    assert plot_path.read_bytes()[:4] == b"\x89PNG"
