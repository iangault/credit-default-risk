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

How the model is built (build_decision_model): three layers, each wrapping the one
inside it.

    TunedThresholdClassifierCV         outer: picks the cutoff
      CalibratedClassifierCV           middle: maps scores to real default rates
        build_pipeline(experiment)     inner: features, scaling, encoding, model

What .fit(X, y) on the outer layer does, with k folds (5 in base.yaml):

1. Outer layer, for each of its k folds: fit the middle layer on the other k - 1
   folds, get calibrated probabilities for the held-out fold, and compute the expected
   cost at each of about 100 cutoffs. Average each cutoff's cost over the k folds and
   keep the cheapest cutoff. Then refit the middle layer on all rows.
2. Middle layer, each time it is fitted (ensemble=False): get out-of-fold scores from
   the inner pipeline with its own k folds, fit the translation from score to default
   rate on them (sigmoid: a smooth S-curve; isotonic: a staircase that only goes up),
   then refit the inner pipeline on all the rows it was given. Its predict_proba is
   the translation applied to the inner pipeline's score.
3. Inner pipeline: the same pipeline as `make experiments` (pipeline.py).

So the threshold is picked on rows that neither the calibration nor the model saw, and
the inner pipeline is fitted (k + 1) * (k + 1) times: 36 with 5 folds.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import TunedThresholdClassifierCV

from credit_risk.config import Config, ConfigError, Costs, Experiment
from credit_risk.data import TARGET, DataFileError, load_split
from credit_risk.experiments import make_folds
from credit_risk.pipeline import build_pipeline


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


def build_decision_model(
    experiment: Experiment, calibration: str, costs: Costs, cfg: Config
) -> TunedThresholdClassifierCV:
    """An unfitted calibrated model with a cost-tuned threshold (see the module docstring).

    `calibration` is "sigmoid" or "isotonic" (config.CALIBRATION_METHODS).
    """
    folds = make_folds(cfg)  # the same seeded folds as `make experiments`
    calibrated = CalibratedClassifierCV(
        build_pipeline(experiment, cfg.seed),
        method=calibration,
        cv=folds,
        # False: learn the calibration from out-of-fold scores, then keep one pipeline
        # refit on all rows. True (the default) would keep k pipelines and average them,
        # which would make M6's SHAP explanations harder.
        ensemble=False,
    )
    return TunedThresholdClassifierCV(
        calibrated,
        scoring=cost_scorer(costs),
        cv=folds,
        # Keep the cost at every cutoff tried (cv_results_), for the cost vs threshold plot.
        store_cv_results=True,
    )


def run_decision(
    cfg: Config, train: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every candidate x calibration x cost combination in cfg.decision.

    Returns (table, curves). table: one row per combination with the chosen threshold
    and its cross-validated expected cost per client. curves: the expected cost at every
    cutoff tried, for the plot. `train` is optional so tests can pass a small sample.
    """
    if cfg.decision is None:
        raise ConfigError("No 'decision' section in the config; add one to run `make decide`.")
    if train is None:
        train = load_split("train", cfg)
    X, y = train.drop(columns=TARGET), train[TARGET]
    experiments = {e.name: e for e in cfg.experiments}

    rows, curves = [], []
    for name in cfg.decision.candidates:
        for method in cfg.decision.calibration:
            for false_negative_cost in cfg.decision.false_negative_costs:
                costs = Costs(false_negative_cost, cfg.costs.false_positive)
                model = build_decision_model(experiments[name], method, costs, cfg).fit(X, y)
                setup = {
                    "experiment": name,
                    "calibration": method,
                    "false_negative_cost": false_negative_cost,
                }
                # The scorer reports costs negated (see cost_scorer), so flip them back.
                # best_score_ is the cheapest cutoff's cost averaged over the folds. It is
                # slightly optimistic, since that cutoff was picked on the same folds.
                rows.append(
                    {
                        **setup,
                        "threshold": model.best_threshold_,
                        "expected_cost": -model.best_score_,
                    }
                )
                curves.append(
                    pd.DataFrame(
                        {
                            **setup,
                            "threshold": model.cv_results_["thresholds"],
                            "expected_cost": -model.cv_results_["scores"],
                        }
                    )
                )
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


def plot_cost_curves(curves: pd.DataFrame, false_positive_cost: float) -> alt.FacetChart:
    """Expected cost per client against threshold: one panel per cost of a missed
    defaulter, one line per candidate and calibration method.

    The dashed gray line marks the rule of thumb false_positive / (false_positive +
    false_negative), where the best cutoff lands if the probabilities are honest.
    """
    data = curves.assign(
        rule_of_thumb=false_positive_cost / (false_positive_cost + curves["false_negative_cost"])
    )
    lines = (
        alt.Chart()
        .mark_line()
        .encode(
            x=alt.X("threshold:Q", title="Threshold"),
            # zero=False: start the axis near the curve's minimum so the dip is visible.
            y=alt.Y(
                "expected_cost:Q", title="Expected cost per client", scale=alt.Scale(zero=False)
            ),
            color=alt.Color("experiment:N", title="Candidate"),
            strokeDash=alt.StrokeDash("calibration:N", title="Calibration"),
        )
    )
    rule = alt.Chart().mark_rule(color="gray", strokeDash=[2, 2]).encode(x="mean(rule_of_thumb):Q")
    return (
        alt.layer(lines, rule, data=data)
        .facet(column=alt.Column("false_negative_cost:O", title="Cost of a missed defaulter"))
        # Each panel's costs sit on a different scale (higher miss cost, higher cost).
        .resolve_scale(y="independent")
    )


def decision_table_path(cfg: Config) -> Path:
    """Where the comparison table is written: reports/tables/decision.csv."""
    return cfg.paths.reports_dir / "tables" / "decision.csv"


def cost_plot_path(cfg: Config) -> Path:
    """Where the plot is written: reports/figures/cost_vs_threshold.png."""
    return cfg.paths.reports_dir / "figures" / "cost_vs_threshold.png"


def save_decision(table: pd.DataFrame, curves: pd.DataFrame, cfg: Config) -> tuple[Path, Path]:
    """Write the table (rounded to 4 decimals) and the plot; return both paths."""
    table_path, plot_path = decision_table_path(cfg), cost_plot_path(cfg)
    for path in (table_path, plot_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(table_path, index=False)
    # Altair saves PNGs through vl-convert (in environment.yml); scale_factor=2 for sharpness.
    plot_cost_curves(curves, cfg.costs.false_positive).save(plot_path, scale_factor=2)
    return table_path, plot_path


def train_final(cfg: Config, train: pd.DataFrame | None = None) -> TunedThresholdClassifierCV:
    """Fit the `final` setup from the config on all of train, with the config's costs.

    Calibration and threshold come from cross-validation inside train (see the module
    docstring); the test split is not read.
    """
    if cfg.final is None:
        raise ConfigError(
            "No 'final' section in the config. Run `make decide`, pick a setup from "
            "reports/tables/decision.csv, and add it under 'final'."
        )
    if train is None:
        train = load_split("train", cfg)
    X, y = train.drop(columns=TARGET), train[TARGET]
    experiment = {e.name: e for e in cfg.experiments}[cfg.final.experiment]
    return build_decision_model(experiment, cfg.final.calibration, cfg.costs, cfg).fit(X, y)


def model_path(cfg: Config) -> Path:
    """Where the trained model is saved: artifacts/model.joblib (gitignored)."""
    return cfg.paths.artifacts_dir / "model.joblib"


def save_model(model: TunedThresholdClassifierCV, cfg: Config) -> Path:
    """Save the fitted model (pipeline + calibration + threshold) with joblib."""
    path = model_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    # joblib writes the whole fitted object to one file; loading it gives back the same
    # object, with predict_proba (calibrated) and predict (0/1 at the threshold).
    joblib.dump(model, path)
    return path


def load_model(cfg: Config) -> TunedThresholdClassifierCV:
    """Load the model saved by `make train`."""
    path = model_path(cfg)
    if not path.is_file():
        raise DataFileError(f"{path} not found. Run `make train` first.")
    # joblib files can run code when loaded, so only load files this project wrote.
    return joblib.load(path)
