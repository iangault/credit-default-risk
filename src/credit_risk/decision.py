"""Decision layer (M5): turn a model's scores into honest probabilities and a yes/no call.

Three separate ideas, each handled by one piece:

- Costs (config `costs`): what each kind of mistake costs. Missing a defaulter (a false
  negative) costs more than a false alarm on a client who would have paid (a false
  positive). This is a business judgment, not something learned from data.
  expected_cost() turns 0/1 predictions into an average cost per client.
- Calibration (CalibratedClassifierCV, below): a translation table from the model's raw
  score to how often clients with that score actually default, learned from
  out-of-fold predictions. After it, 0.2 means "about 20 in 100 such clients default".
  Class weights (M3) push scores up; calibration undoes that. It keeps the order of
  clients, so ROC-AUC does not change.
- Threshold (TunedThresholdClassifierCV, below): the cutoff above which a client is
  flagged. It tries about 100 cutoffs and keeps the one with the lowest expected cost.
  With honest probabilities there is a rule of thumb to check it against: flagging
  costs false_positive * (1 - p), not flagging costs false_negative * p, so flag when
  p > false_positive / (false_positive + false_negative). With costs 5 and 1 that is
  1 / 6, about 0.17.

The threshold is not a hyperparameter in the usual sense: it does not change how the
model learns, only how its finished scores are turned into decisions. It is chosen the
same honest way, from predictions on rows the model did not train on.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import make_scorer

from credit_risk.config import Costs


def expected_cost(y_true: Sequence[int], y_pred: Sequence[int], costs: Costs) -> float:
    """Average cost per client of 0/1 predictions.

    (false_negative * missed defaulters + false_positive * false alarms) / clients.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    missed = np.sum((y_true == 1) & (y_pred == 0))  # defaulters not flagged
    false_alarms = np.sum((y_true == 0) & (y_pred == 1))  # good clients flagged
    total = costs.false_negative * missed + costs.false_positive * false_alarms
    return float(total / len(y_true))


def cost_scorer(costs: Costs):
    """A scikit-learn scorer for expected cost, for TunedThresholdClassifierCV.

    A scorer is a function scorer(model, X, y): it calls model.predict(X) and passes the
    0/1 predictions to expected_cost along with `costs`. scikit-learn always picks the
    highest score, so greater_is_better=False makes the scorer return the cost negated
    (-0.55 beats -0.60). Without it, the search would pick the most expensive cutoff.
    """
    return make_scorer(expected_cost, greater_is_better=False, costs=costs)
