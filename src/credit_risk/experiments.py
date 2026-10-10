"""Compare model setups with cross-validation on the training split.

This is the src/ version of the comparison in notebooks/scratch.ipynb (cells 9 to 13:
shared folds, the `run_cv` loop over models, and the comparison table). Moving it
here means it is tested, reruns the same way every time, and can be run from one
command (`make experiments`, added in M2 Task 7) instead of by running cells.

What it does, for each experiment listed in configs/base.yaml:
1. Build a preprocessor + model pipeline (pipeline.py).
2. Run 5-fold cross-validation on the train split: fit on 4 folds, score on the
   5th, repeat 5 times so every row is used for validation once.
3. Average the 5 scores per metric into one row of the results table.

Every experiment uses the same folds (one StratifiedKFold with a fixed seed), so
differences in the table come from the setups, not from which rows landed in
which fold. The test split is not touched here; it is kept for the final check.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from credit_risk.config import Config, ConfigError
from credit_risk.data import TARGET, load_split
from credit_risk.pipeline import build_pipeline

# Metric name in the table -> scikit-learn scorer. A string is a built-in scorer name;
# make_scorer wraps a metric function with extra settings. zero_division=0 reports
# precision and F1 as 0 (instead of warning) when a model predicts no defaulters at
# all, which the dummy model does. The scratch notebook hid that warning instead.
SCORERS = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "recall": "recall",
    "precision": make_scorer(precision_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
}
METRICS = tuple(SCORERS)


def run_experiments(cfg: Config, train: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cross-validate every experiment in cfg and return one summary row per experiment.

    `train` is optional so tests can pass a small sample; normally it is loaded from
    data/processed (written by `make split`).
    """
    if not cfg.experiments:
        raise ConfigError("No experiments defined; add an 'experiments' list to the config.")
    if train is None:
        train = load_split("train", cfg)
    # X = the features, y = the 0/1 default column the model learns to predict.
    X, y = train.drop(columns=TARGET), train[TARGET]

    # Build every pipeline first, so a typo in a model name fails before any CV runs.
    # Otherwise a typo in the last experiment would only show up after the others
    # had spent minutes training.
    pipelines = {e.name: (e, build_pipeline(e, cfg.seed)) for e in cfg.experiments}
    # Stratified: each fold keeps the ~22% default rate. shuffle + random_state: the
    # rows are mixed before splitting, the same way on every run.
    folds = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)

    rows = []
    for name, (experiment, pipe) in pipelines.items():
        # cross_validate does the fit-on-4-folds, score-on-1 loop. It returns a dict
        # of arrays with one value per fold, e.g. scores["test_roc_auc"] has 5 values.
        # return_train_score=True also scores each fold's training rows, for the gap.
        scores = cross_validate(pipe, X, y, cv=folds, scoring=SCORERS, return_train_score=True)
        row = {"experiment": name, "model": experiment.model}
        for metric in METRICS:
            # Mean = typical score; std = how much it varies from fold to fold.
            row[f"{metric}_mean"] = scores[f"test_{metric}"].mean()
            row[f"{metric}_std"] = scores[f"test_{metric}"].std()
        # Compare with roc_auc_mean: a large gap means the model memorizes train rows.
        row["train_roc_auc_mean"] = scores["train_roc_auc"].mean()
        rows.append(row)
    # One row per experiment, in the order listed in the config.
    return pd.DataFrame(rows)


def experiments_table_path(cfg: Config) -> Path:
    """Where the results table is written: reports/tables/experiments.csv."""
    return cfg.paths.reports_dir / "tables" / "experiments.csv"


def save_experiments(table: pd.DataFrame, cfg: Config) -> Path:
    """Write the results table as CSV (rounded to 4 decimals) and return its path."""
    path = experiments_table_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)  # create reports/tables/ if missing
    table.round(4).to_csv(path, index=False)
    return path
