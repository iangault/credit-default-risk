import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from credit_risk.config import Costs
from credit_risk.decision import cost_scorer, expected_cost

COSTS = Costs(false_negative=5.0, false_positive=1.0)


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
