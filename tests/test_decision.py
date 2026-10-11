from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV

from credit_risk.config import ConfigError, Costs, Decision, Final, load_config
from credit_risk.data import TARGET, DataFileError, clean, load_raw
from credit_risk.decision import (
    build_decision_model,
    cost_plot_path,
    cost_scorer,
    decision_table_path,
    expected_cost,
    load_model,
    model_path,
    run_decision,
    save_decision,
    save_model,
    train_final,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = load_config(ROOT / "configs" / "base.yaml")
COSTS = Costs(false_negative=5.0, false_positive=1.0)


@pytest.fixture(scope="module")
def sample():
    train = clean(load_raw(BASE)).sample(1000, random_state=0)
    return train.drop(columns=TARGET), train[TARGET]


def small_cfg(tmp_path, **changes):
    paths = replace(
        BASE.paths, reports_dir=tmp_path / "reports", artifacts_dir=tmp_path / "artifacts"
    )
    return replace(BASE, paths=paths, cv_folds=2, **changes)


def experiment(name):
    return {e.name: e for e in BASE.experiments}[name]


def test_expected_cost_hand_worked_example():
    # 4 clients: one defaulter caught, one missed (cost 5), one false alarm (cost 1),
    # one good client passed. (5 + 1) / 4 = 1.5 per client.
    assert expected_cost([1, 1, 0, 0], [1, 0, 1, 0], COSTS) == pytest.approx(1.5)


def test_expected_cost_is_zero_when_every_prediction_is_right():
    assert expected_cost([1, 0, 0], [1, 0, 0], COSTS) == 0


def test_expected_cost_of_flagging_nobody_is_false_negative_cost_times_default_rate():
    y = np.array([1, 0, 0, 0, 0])  # 20% default rate
    assert expected_cost(y, np.zeros(5), COSTS) == pytest.approx(5.0 * 0.2)


class FixedPredictions(ClassifierMixin, BaseEstimator):
    """A stand-in model whose predict returns the list it was given."""

    classes_ = np.array([0, 1])

    def __init__(self, predictions=None):
        self.predictions = predictions

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.asarray(self.predictions)


def test_scorer_ranks_cheaper_predictions_higher():
    scorer = cost_scorer(COSTS)
    X, y = np.zeros((4, 1)), np.array([1, 1, 0, 0])
    costly = scorer(FixedPredictions([1, 0, 1, 0]), X, y)
    perfect = scorer(FixedPredictions([1, 1, 0, 0]), X, y)
    # scikit-learn picks the highest score, so the scorer reports the cost negated.
    assert costly == pytest.approx(-1.5)
    assert perfect > costly


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_decision_model_fits_and_predicts_at_its_threshold(sample, tmp_path, method):
    X, y = sample
    model = build_decision_model(experiment("lgbm"), method, COSTS, small_cfg(tmp_path))
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    assert 0 < model.best_threshold_ < 1
    # predict is predict_proba compared with the tuned cutoff, not with 0.5.
    assert (model.predict(X) == (proba >= model.best_threshold_)).all()
    assert len(model.cv_results_["thresholds"]) == 100


def test_calibration_keeps_one_fitted_pipeline(sample, tmp_path):
    X, y = sample
    model = build_decision_model(experiment("logreg"), "sigmoid", COSTS, small_cfg(tmp_path))
    model.fit(X, y)
    calibrated = model.estimator_  # the refit CalibratedClassifierCV
    assert isinstance(calibrated, CalibratedClassifierCV)
    assert calibrated.method == "sigmoid"
    # ensemble=False: one pipeline fitted on all rows, not one per fold (keeps SHAP simple).
    assert len(calibrated.calibrated_classifiers_) == 1


def test_threshold_is_below_half_when_misses_cost_more(sample, tmp_path):
    # With misses costing 5x a false alarm the rule of thumb is about 0.17. A flipped
    # scorer sign would push the cutoff to an extreme instead.
    X, y = sample
    model = build_decision_model(experiment("lgbm"), "sigmoid", COSTS, small_cfg(tmp_path))
    assert model.fit(X, y).best_threshold_ < 0.4


def small_decision_cfg(tmp_path):
    decision = Decision(("logreg", "lgbm"), ("sigmoid",), (3.0, 5.0))
    return small_cfg(tmp_path, decision=decision)


def test_run_decision_has_one_row_per_combination(sample, tmp_path):
    X, y = sample
    table, curves = run_decision(small_decision_cfg(tmp_path), X.assign(**{TARGET: y}))
    assert list(table.columns) == [
        "experiment",
        "calibration",
        "false_negative_cost",
        "threshold",
        "expected_cost",
    ]
    assert table[["experiment", "false_negative_cost"]].values.tolist() == [
        ["logreg", 3.0],
        ["logreg", 5.0],
        ["lgbm", 3.0],
        ["lgbm", 5.0],
    ]
    assert table["threshold"].between(0, 1).all()
    assert (table["expected_cost"] > 0).all()
    assert len(curves) == 4 * 100
    assert list(curves.columns) == list(table.columns)


def test_run_decision_without_section_is_a_config_error(sample, tmp_path):
    X, y = sample
    with pytest.raises(ConfigError, match="decision"):
        run_decision(small_cfg(tmp_path, decision=None), X.assign(**{TARGET: y}))


def test_save_decision_writes_table_and_plot(sample, tmp_path):
    X, y = sample
    cfg = small_decision_cfg(tmp_path)
    table, curves = run_decision(cfg, X.assign(**{TARGET: y}))
    table_path, plot_path = save_decision(table, curves, cfg)
    assert (
        table_path == decision_table_path(cfg) == tmp_path / "reports" / "tables" / "decision.csv"
    )
    assert plot_path == cost_plot_path(cfg)
    assert pd.read_csv(table_path)["experiment"].tolist() == table["experiment"].tolist()
    assert plot_path.read_bytes()[:4] == b"\x89PNG"


def test_train_final_saves_a_model_that_loads_and_predicts(sample, tmp_path):
    X, y = sample
    cfg = small_cfg(tmp_path, final=Final("logreg", "sigmoid"))
    model = train_final(cfg, X.assign(**{TARGET: y}))
    path = save_model(model, cfg)
    assert path == model_path(cfg) == tmp_path / "artifacts" / "model.joblib"
    loaded = load_model(cfg)
    assert loaded.best_threshold_ == model.best_threshold_
    assert (loaded.predict(X) == model.predict(X)).all()


def test_train_without_final_is_a_config_error(sample, tmp_path):
    X, y = sample
    with pytest.raises(ConfigError, match="final"):
        train_final(small_cfg(tmp_path, final=None), X.assign(**{TARGET: y}))


def test_load_model_before_train_is_a_data_file_error(tmp_path):
    with pytest.raises(DataFileError, match="make train"):
        load_model(small_cfg(tmp_path))
