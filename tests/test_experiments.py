import warnings
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from credit_risk.config import ConfigError, Experiment, load_config
from credit_risk.data import clean, load_raw
from credit_risk.experiments import (
    METRICS,
    experiments_table_path,
    run_experiments,
    save_experiments,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = load_config(ROOT / "configs" / "base.yaml")


def small_cfg(tmp_path, **changes):
    paths = replace(BASE.paths, reports_dir=tmp_path / "reports")
    return replace(BASE, paths=paths, cv_folds=2, **changes)


@pytest.fixture(scope="module")
def train():
    return clean(load_raw(BASE)).sample(1000, random_state=0)


def test_one_row_per_experiment_with_all_metrics(train, tmp_path):
    table = run_experiments(small_cfg(tmp_path), train)
    assert table["experiment"].tolist() == [e.name for e in BASE.experiments]
    assert "sampler" in table.columns
    assert "features" in table.columns
    for metric in METRICS:
        assert f"{metric}_mean" in table.columns
        assert f"{metric}_std" in table.columns
    assert "train_roc_auc_mean" in table.columns
    # Timings change on every run, so they are left out to keep the CSV diff clean.
    assert "fit_time_mean" not in table.columns


def test_sampler_column_names_the_resampler(train, tmp_path):
    experiments = (
        Experiment("lr", "logreg"),
        Experiment("lr_under", "logreg", sampler="undersample"),
    )
    table = run_experiments(small_cfg(tmp_path, experiments=experiments), train)
    assert table["sampler"].tolist() == ["none", "undersample"]


def test_features_column_names_the_groups(train, tmp_path):
    experiments = (
        Experiment("lr", "logreg"),
        Experiment("lr_feat", "logreg", features=("delinquency", "trend")),
    )
    table = run_experiments(small_cfg(tmp_path, experiments=experiments), train)
    assert table["features"].tolist() == ["none", "delinquency+trend"]


def test_dummy_is_chance_level_and_models_beat_it(train, tmp_path):
    # The dummy is the negative control: no real signal, so AUC 0.5 is known in
    # advance. 0.6 is a loose floor that only a broken model should miss.
    table = run_experiments(small_cfg(tmp_path), train).set_index("experiment")
    assert table.loc["dummy", "roc_auc_mean"] == pytest.approx(0.5)
    assert table.loc["dummy", "precision_mean"] == 0
    for model in ["logreg", "forest", "lgbm", "xgboost"]:
        assert table.loc[model, "roc_auc_mean"] > 0.6


def test_no_undefined_metric_warnings(train, tmp_path):
    cfg = small_cfg(tmp_path, experiments=(Experiment("dummy", "dummy"),))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run_experiments(cfg, train)


def test_results_are_repeatable(train, tmp_path):
    cfg = small_cfg(tmp_path, experiments=(Experiment("lr", "logreg"),))
    first = run_experiments(cfg, train)
    second = run_experiments(cfg, train)
    pd.testing.assert_frame_equal(first, second)


def test_unknown_model_fails_before_any_cv(train, tmp_path):
    cfg = small_cfg(tmp_path, experiments=(Experiment("ok", "logreg"), Experiment("typo", "lgmb")))
    with pytest.raises(ValueError, match="lgbm"):
        run_experiments(cfg, train)


def test_no_experiments_is_a_config_error(train, tmp_path):
    with pytest.raises(ConfigError, match="experiments"):
        run_experiments(small_cfg(tmp_path, experiments=()), train)


def test_save_writes_csv_under_reports_tables(train, tmp_path):
    cfg = small_cfg(tmp_path, experiments=(Experiment("dummy", "dummy"),))
    path = save_experiments(run_experiments(cfg, train), cfg)
    assert (
        path == experiments_table_path(cfg) == tmp_path / "reports" / "tables" / "experiments.csv"
    )
    assert pd.read_csv(path)["experiment"].tolist() == ["dummy"]
