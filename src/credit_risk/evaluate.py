"""Score the saved model on the test set (M6).

The test set (20% of clients, split off in M1) has not been used for any choice: model,
features, class weights, calibration, threshold and costs were all chosen on train. So
the numbers here estimate how the model does on clients it has never seen. Per the spec,
no setting is changed after seeing them; a disappointing result is reported, not tuned.

The table has five rows, so each number can be judged against something:
- model at tuned threshold: the saved model as it would be used (cutoff from M5).
- model at 0.5: the same scores with the default cutoff, to show what M5's threshold did.
- flag nobody / flag everyone: the cost of not using a model at all.
- train CV estimate: what cross-validation on train predicted, read from the committed
  experiments.csv and decision.csv. Close agreement means the estimates were trustworthy.

ROC-AUC and PR-AUC measure ranking, so they do not depend on the cutoff; the two model
rows share them. Counts: flagged (predicted defaulters), caught and missed (defaulters
flagged or not), false_alarms (flagged clients who paid).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TunedThresholdClassifierCV

from credit_risk.config import Config, Costs
from credit_risk.data import TARGET
from credit_risk.decision import decision_table_path, expected_cost
from credit_risk.experiments import experiments_table_path

COLUMNS = [
    "setup",
    "threshold",
    "roc_auc",
    "pr_auc",
    "recall",
    "precision",
    "expected_cost",
    "flagged",
    "caught",
    "missed",
    "false_alarms",
]


def _outcome_metrics(y: np.ndarray, flagged: np.ndarray, costs: Costs) -> dict:
    """Recall, precision, cost and counts for a given set of flagged clients."""
    return {
        "recall": recall_score(y, flagged, zero_division=0),
        # zero_division=0: report 0 (not a warning) when nobody is flagged.
        "precision": precision_score(y, flagged, zero_division=0),
        "expected_cost": expected_cost(y, flagged.astype(int), costs),
        "flagged": int(flagged.sum()),
        "caught": int((flagged & (y == 1)).sum()),
        "missed": int((~flagged & (y == 1)).sum()),
        "false_alarms": int((flagged & (y == 0)).sum()),
    }


def decision_metrics(
    y: Sequence[int], proba: Sequence[float], threshold: float, costs: Costs
) -> dict:
    """Metrics when clients with probability >= threshold are flagged.

    `>=` matches the saved model's predict (TunedThresholdClassifierCV).
    """
    y, proba = np.asarray(y), np.asarray(proba)
    return {"threshold": threshold, **_outcome_metrics(y, proba >= threshold, costs)}


def train_estimate(cfg: Config) -> dict:
    """The train cross-validation numbers for the chosen setup, from the committed tables.

    ROC-AUC and PR-AUC come from experiments.csv (the uncalibrated model; sigmoid
    calibration keeps the order of clients, so ranking metrics match). Threshold and cost
    come from decision.csv. A value not found (no table yet, or costs changed without
    rerunning make decide) is NaN, so the row shows blanks instead of failing.
    """
    estimate = dict.fromkeys(["threshold", "roc_auc", "pr_auc", "expected_cost"], np.nan)
    final, fn_cost = cfg.final, cfg.costs.false_negative
    if experiments_table_path(cfg).is_file():
        experiments = pd.read_csv(experiments_table_path(cfg))
        match = experiments[experiments["experiment"] == final.experiment]
        if len(match) == 1:
            estimate["roc_auc"] = float(match["roc_auc_mean"].iloc[0])
            estimate["pr_auc"] = float(match["pr_auc_mean"].iloc[0])
    if decision_table_path(cfg).is_file():
        decision = pd.read_csv(decision_table_path(cfg))
        match = decision[
            (decision["experiment"] == final.experiment)
            & (decision["calibration"] == final.calibration)
            & np.isclose(decision["false_negative_cost"], fn_cost)
        ]
        if len(match) == 1:
            estimate["threshold"] = float(match["threshold"].iloc[0])
            estimate["expected_cost"] = float(match["expected_cost"].iloc[0])
    return estimate


def evaluate(
    model: TunedThresholdClassifierCV, test: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, np.ndarray]:
    """The five-row comparison table on `test`, and the calibrated test probabilities."""
    X, y = test.drop(columns=TARGET), test[TARGET].to_numpy()
    proba = model.predict_proba(X)[:, 1]  # calibrated probability of default
    ranking = {"roc_auc": roc_auc_score(y, proba), "pr_auc": average_precision_score(y, proba)}
    nobody, everyone = np.zeros(len(y), dtype=bool), np.ones(len(y), dtype=bool)
    rows = [
        {
            "setup": "model at tuned threshold",
            **ranking,
            **decision_metrics(y, proba, model.best_threshold_, cfg.costs),
        },
        {"setup": "model at 0.5", **ranking, **decision_metrics(y, proba, 0.5, cfg.costs)},
        {"setup": "flag nobody", **_outcome_metrics(y, nobody, cfg.costs)},
        {"setup": "flag everyone", **_outcome_metrics(y, everyone, cfg.costs)},
        {"setup": "train CV estimate", **train_estimate(cfg)},
    ]
    return pd.DataFrame(rows, columns=COLUMNS), proba


def plot_calibration(y: Sequence[int], proba: Sequence[float]) -> alt.LayerChart:
    """Predicted probability against the actual default rate, in 10 equal-size groups.

    Clients are sorted by predicted probability and cut into 10 groups of about the same
    size ("quantile"). Each point is one group: its average prediction (x) and the share
    of its clients who defaulted (y). Points on the diagonal mean the probabilities are
    honest: clients given about 30% default about 30% of the time.
    """
    observed, predicted = calibration_curve(y, proba, n_bins=10, strategy="quantile")
    groups = pd.DataFrame({"predicted": predicted, "observed": observed})
    top = float(max(predicted.max(), observed.max()))
    diagonal = pd.DataFrame({"predicted": [0.0, top], "observed": [0.0, top]})
    points = (
        alt.Chart(groups)
        .mark_line(point=True)
        .encode(
            x=alt.X("predicted:Q", title="Average predicted probability of default"),
            y=alt.Y("observed:Q", title="Actual default rate"),
        )
    )
    ideal = (
        alt.Chart(diagonal)
        .mark_line(color="gray", strokeDash=[4, 4])
        .encode(x="predicted:Q", y="observed:Q")
    )
    return alt.layer(points, ideal).properties(
        title=alt.TitleParams(
            "Calibration on the test set",
            subtitle="10 equal-size groups of clients; dashed line: perfect calibration",
        ),
        width=400,
        height=400,
    )


def metrics_table_path(cfg: Config) -> Path:
    """Where the test metrics are written: reports/tables/test_metrics.csv."""
    return cfg.paths.reports_dir / "tables" / "test_metrics.csv"


def calibration_plot_path(cfg: Config) -> Path:
    """Where the calibration plot is written: reports/figures/calibration_test.png."""
    return cfg.paths.reports_dir / "figures" / "calibration_test.png"


def save_evaluation(
    table: pd.DataFrame, y: Sequence[int], proba: Sequence[float], cfg: Config
) -> tuple[Path, Path]:
    """Write the table (rounded to 4 decimals) and the calibration plot; return both paths."""
    table_path, plot_path = metrics_table_path(cfg), calibration_plot_path(cfg)
    for path in (table_path, plot_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(table_path, index=False)
    plot_calibration(y, proba).save(plot_path, scale_factor=2)
    return table_path, plot_path
