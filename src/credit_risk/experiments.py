"""Compare model setups with cross-validation on the training split.

Every experiment uses the same folds (one StratifiedKFold with a fixed seed), so
differences in the table come from the setups, not from which rows landed in
which fold.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from credit_risk.config import Config, ConfigError
from credit_risk.data import TARGET, load_split
from credit_risk.pipeline import build_pipeline

# Metric name in the table -> scikit-learn scorer. zero_division=0 reports precision
# and F1 as 0 (instead of warning) when a model predicts no defaulters at all.
SCORERS = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "recall": "recall",
    "precision": make_scorer(precision_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
}
METRICS = tuple(SCORERS)


def run_experiments(cfg: Config, train: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cross-validate every experiment in cfg and return one summary row per experiment."""
    if not cfg.experiments:
        raise ConfigError("No experiments defined; add an 'experiments' list to the config.")
    if train is None:
        train = load_split("train", cfg)
    X, y = train.drop(columns=TARGET), train[TARGET]

    # Build every pipeline first, so a typo in a model name fails before any CV runs.
    pipelines = {e.name: (e, build_pipeline(e, cfg.seed)) for e in cfg.experiments}
    folds = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)

    rows = []
    for name, (experiment, pipe) in pipelines.items():
        scores = cross_validate(pipe, X, y, cv=folds, scoring=SCORERS, return_train_score=True)
        row = {"experiment": name, "model": experiment.model}
        for metric in METRICS:
            row[f"{metric}_mean"] = scores[f"test_{metric}"].mean()
            row[f"{metric}_std"] = scores[f"test_{metric}"].std()
        row["train_roc_auc_mean"] = scores["train_roc_auc"].mean()
        row["fit_time_mean"] = scores["fit_time"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def experiments_table_path(cfg: Config) -> Path:
    return cfg.paths.reports_dir / "tables" / "experiments.csv"


def save_experiments(table: pd.DataFrame, cfg: Config) -> Path:
    path = experiments_table_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(path, index=False)
    return path
