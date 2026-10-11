from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV

from credit_risk.config import Costs, load_config
from credit_risk.data import TARGET, clean, load_raw
from credit_risk.decision import build_decision_model, cost_scorer, expected_cost

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
