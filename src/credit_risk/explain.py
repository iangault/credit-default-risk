"""Explain the saved model's predictions with SHAP (M6).

SHAP values split one prediction into a contribution from each column. For a client,
    model score = base value + sum of the client's SHAP values,
where the base value is the model's average score. A positive SHAP value means that
column pushed this client's score toward default; negative, away from it. The values come
from game theory (Shapley values): each column gets its average contribution over every
order in which columns could be added. For tree models shap.TreeExplainer computes them
exactly and fast.

Units: SHAP explains the model step inside the saved pipeline, so values are in that
model's score before calibration: log-odds for LightGBM and XGBoost (a value of +0.7
multiplies the odds of default by about 2), probability for random forest. Sigmoid
calibration keeps the order of clients, so "pushes risk up" means the same either way.

Plots (reports/figures/):
- shap_summary.png (beeswarm): one row per column, most important at the top; one dot
  per test client, placed left or right by its SHAP value, colored by the column's value
  (red high, blue low). Example reading: red dots far right on PAY_1 mean recent late
  payment pushes risk up a lot.
- shap_client_flagged.png, shap_client_missed.png (waterfalls): one client each, from the
  base value at the bottom to their score at the top, one bar per column.

The model sees transformed columns (scaled amounts, one-hot categories). The plots show
each column's original value where the column keeps its name (LIMIT_BAL = 50000, not
-0.9); one-hot columns such as SEX_2 show 0 or 1.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import shap
from imblearn.pipeline import Pipeline
from sklearn.model_selection import TunedThresholdClassifierCV

from credit_risk.config import Config, ConfigError

# Models shap.TreeExplainer can explain. The final model must be one of these.
TREE_MODELS = ("lgbm", "xgboost", "forest")


def check_tree_model(cfg: Config) -> None:
    """Stop early if the final setup's model is not a tree model."""
    if cfg.final is None:
        raise ConfigError("No 'final' section in the config; set it and run `make train`.")
    experiment = {e.name: e for e in cfg.experiments}[cfg.final.experiment]
    if experiment.model not in TREE_MODELS:
        raise ConfigError(
            f"make explain supports tree models ({', '.join(TREE_MODELS)}); the final "
            f"setup '{experiment.name}' uses '{experiment.model}'."
        )


def inner_pipeline(model: TunedThresholdClassifierCV) -> Pipeline:
    """The fitted pipeline inside the saved model (see decision.py's module docstring).

    model.estimator_ is the CalibratedClassifierCV refit on all of train. With
    ensemble=False it holds one (pipeline, calibrator) pair in calibrated_classifiers_,
    and .estimator is that pipeline, fitted on all of train.
    """
    return model.estimator_.calibrated_classifiers_[0].estimator


def shap_explanation(model: TunedThresholdClassifierCV, X: pd.DataFrame) -> shap.Explanation:
    """SHAP values for each row of X, with original (unscaled) values for display."""
    pipe = inner_pipeline(model)
    # pipe[:-1] is every step except the model: features, scaling, encoding. Its output
    # is what the model was trained on, so SHAP must explain the model on that.
    transformed = pipe[:-1].transform(X)
    explanation = shap.TreeExplainer(pipe.named_steps["model"])(transformed)
    values, base = explanation.values, explanation.base_values
    if values.ndim == 3:
        # Random forest returns one set per class (no default, default); keep default.
        values, base = values[:, :, 1], base[:, 1]
    display = transformed.copy()
    kept = [column for column in display.columns if column in X.columns]
    display[kept] = X[kept].to_numpy()  # same rows in the same order
    return shap.Explanation(
        values=values,
        base_values=base,
        data=display.to_numpy(),
        feature_names=list(display.columns),
    )


def pick_clients(y: Sequence[int], proba: Sequence[float], threshold: float) -> dict[str, int]:
    """Row numbers of a typical correctly flagged defaulter and a typical missed one.

    A fixed rule, chosen before seeing the test set, so the examples are not hand-picked:
    in each group, the client whose probability is nearest the group's median. A group
    with no clients (possible on a small test set) is left out.
    """
    y, proba = np.asarray(y), np.asarray(proba)
    flagged = proba >= threshold
    groups = {"flagged": (y == 1) & flagged, "missed": (y == 1) & ~flagged}
    picks = {}
    for name, in_group in groups.items():
        rows = np.flatnonzero(in_group)
        if len(rows) == 0:
            continue
        distance = np.abs(proba[rows] - np.median(proba[rows]))
        picks[name] = int(rows[np.argmin(distance)])  # ties: the first such row
    return picks
