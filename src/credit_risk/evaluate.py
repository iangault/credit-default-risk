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

Two figures go with the table: a calibration plot, and the ROC and precision-recall
curves with the tuned cutoffs marked on them.
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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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
    calibration keeps the order of clients, so ranking metrics match; isotonic can merge
    nearby scores into ties, so they may differ slightly). Threshold and cost
    come from decision.csv. A value not found (no table yet, or costs changed without
    rerunning make decide) is NaN, so the row shows blanks instead of failing.
    """
    estimate = dict.fromkeys(["threshold", "roc_auc", "pr_auc", "expected_cost"], np.nan)
    final, costs = cfg.final, cfg.costs
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
            & np.isclose(decision["false_negative_cost"], costs.false_negative)
            & np.isclose(decision["false_positive_cost"], costs.false_positive)
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


def marked_thresholds(model: TunedThresholdClassifierCV, cfg: Config) -> pd.DataFrame:
    """The cutoffs to mark on the ROC and PR curves: 0.5, and the tuned one per cost ratio.

    The current cost ratio uses the saved model's own threshold. The other ratios in
    decision.csv (same setup and false alarm cost) show where the cutoff would sit if a
    miss were judged cheaper or dearer; those thresholds were tuned on train for the
    same setup, so they can be placed on the test curves without refitting anything.
    If decision.csv is missing, only 0.5 and the saved model's cutoff are marked.
    """
    final, costs = cfg.final, cfg.costs
    rows = [
        {"label": "default 0.5", "threshold": 0.5},
        {
            "label": f"misses cost {costs.false_negative:g}x (saved model)",
            "threshold": float(model.best_threshold_),
        },
    ]
    if decision_table_path(cfg).is_file():
        decision = pd.read_csv(decision_table_path(cfg))
        others = decision[
            (decision["experiment"] == final.experiment)
            & (decision["calibration"] == final.calibration)
            & np.isclose(decision["false_positive_cost"], costs.false_positive)
            & ~np.isclose(decision["false_negative_cost"], costs.false_negative)
        ]
        rows += [
            {"label": f"misses cost {fn:g}x", "threshold": float(t)}
            for fn, t in zip(others["false_negative_cost"], others["threshold"], strict=True)
        ]
    # Sorted by cutoff, so the legend reads from most flagged to least flagged.
    return pd.DataFrame(rows).sort_values("threshold", ignore_index=True)


def plot_tradeoff_curves(
    y: Sequence[int], proba: Sequence[float], marks: pd.DataFrame
) -> alt.HConcatChart:
    """ROC curve and precision-recall curve side by side, with the cutoffs in `marks`.

    Both curves trace every possible cutoff. ROC: share of defaulters caught (true
    positive rate) against share of paying clients flagged (false positive rate); the
    dashed diagonal is a model that guesses at random. PR: share of flagged clients who
    default (precision) against share of defaulters caught (recall); the dashed line is
    the default rate, which is the precision of flagging everyone. With about 22% of
    clients defaulting, PR shows the cost of false alarms more plainly than ROC, since
    false positive rate divides them by the large number of paying clients.

    The cost search in M5 picks one point on these curves for a given cost ratio; the
    markers show where those points fall and how far apart the ratios put them.
    """
    y, proba = np.asarray(y), np.asarray(proba)
    fpr, tpr, _ = roc_curve(y, proba)
    precision, recall, _ = precision_recall_curve(y, proba)
    roc = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    pr = pd.DataFrame({"recall": recall, "precision": precision})

    # Each marker is the test-set outcome of flagging clients at or above its cutoff
    # (`>=`, as in decision_metrics). Ties in proba can put a marker just off the line.
    points = []
    for label, threshold in zip(marks["label"], marks["threshold"], strict=True):
        flagged = proba >= threshold
        caught = recall_score(y, flagged, zero_division=0)  # true positive rate = recall
        points.append(
            {
                "label": label,
                "threshold": threshold,
                "tpr": caught,
                "fpr": flagged[y == 0].mean(),  # share of paying clients flagged
                "recall": caught,
                "precision": precision_score(y, flagged, zero_division=0),
            }
        )
    points = pd.DataFrame(points)
    color = alt.Color("label:N", title="Cutoff", sort=list(marks["label"]))
    tooltip = ["label:N", alt.Tooltip("threshold:Q", format=".3f")]

    def panel(curve, x, x_title, y_field, y_title, baseline, title):
        line = (
            alt.Chart(curve)
            .mark_line(color="steelblue")
            .encode(
                x=alt.X(f"{x}:Q", title=x_title, scale=alt.Scale(domain=[0, 1])),
                y=alt.Y(f"{y_field}:Q", title=y_title, scale=alt.Scale(domain=[0, 1])),
            )
        )
        ref = (
            alt.Chart(baseline)
            .mark_line(color="gray", strokeDash=[4, 4])
            .encode(x=f"{x}:Q", y=f"{y_field}:Q")
        )
        dots = (
            alt.Chart(points)
            .mark_point(size=90, filled=True)
            .encode(x=f"{x}:Q", y=f"{y_field}:Q", color=color, tooltip=tooltip)
        )
        return alt.layer(line, ref, dots).properties(title=title, width=330, height=330)

    default_rate = float(y.mean())
    roc_panel = panel(
        roc,
        "fpr",
        "False positive rate (paying clients flagged)",
        "tpr",
        "True positive rate (defaulters caught)",
        pd.DataFrame({"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]}),
        "ROC curve",
    )
    pr_panel = panel(
        pr,
        "recall",
        "Recall (defaulters caught)",
        "precision",
        "Precision (flagged clients who default)",
        pd.DataFrame({"recall": [0.0, 1.0], "precision": [default_rate, default_rate]}),
        "Precision-recall curve",
    )
    return alt.hconcat(roc_panel, pr_panel).properties(
        title=alt.TitleParams(
            "ROC and precision-recall curves on the test set",
            subtitle=(
                "Dashed lines: random guessing (ROC) and the default rate, "
                f"{default_rate:.0%} (PR). Points: cutoffs tuned for each cost ratio, and 0.5"
            ),
        )
    )


def metrics_table_path(cfg: Config) -> Path:
    """Where the test metrics are written: reports/tables/test_metrics.csv."""
    return cfg.paths.reports_dir / "tables" / "test_metrics.csv"


def calibration_plot_path(cfg: Config) -> Path:
    """Where the calibration plot is written: reports/figures/calibration_test.png."""
    return cfg.paths.reports_dir / "figures" / "calibration_test.png"


def tradeoff_plot_path(cfg: Config) -> Path:
    """Where the ROC and PR curves are written: reports/figures/roc_pr_test.png."""
    return cfg.paths.reports_dir / "figures" / "roc_pr_test.png"


def save_evaluation(
    table: pd.DataFrame,
    y: Sequence[int],
    proba: Sequence[float],
    marks: pd.DataFrame,
    cfg: Config,
) -> tuple[Path, Path, Path]:
    """Write the table (rounded to 4 decimals), the calibration plot and the ROC/PR plot.

    `marks` is the cutoffs to show on the ROC/PR plot (see marked_thresholds). Returns
    the three paths.
    """
    table_path = metrics_table_path(cfg)
    calibration_path, tradeoff_path = calibration_plot_path(cfg), tradeoff_plot_path(cfg)
    for path in (table_path, calibration_path, tradeoff_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(table_path, index=False)
    plot_calibration(y, proba).save(calibration_path, scale_factor=2)
    plot_tradeoff_curves(y, proba, marks).save(tradeoff_path, scale_factor=2)
    return table_path, calibration_path, tradeoff_path
