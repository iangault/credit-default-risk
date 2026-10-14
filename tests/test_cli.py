from pathlib import Path

import pandas as pd
import pytest

from credit_risk.cli import main
from credit_risk.config import load_config
from credit_risk.data import sha256_of

ROOT = Path(__file__).resolve().parents[1]
BASE = str(ROOT / "configs" / "base.yaml")


def test_data_stage_passes_on_committed_csv(capsys):
    assert main(["--config", BASE, "data"]) == 0
    assert "OK" in capsys.readouterr().out


def test_missing_config_prints_error_and_returns_1(tmp_path, capsys):
    code = main(["--config", str(tmp_path / "nope.yaml"), "data"])
    err = capsys.readouterr().err
    assert code == 1
    assert err.startswith("error:")
    assert "not found" in err


def test_changed_raw_file_prints_error_and_returns_1(tmp_path, capsys):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (tmp_path / "raw.csv").write_bytes(b"not the real data")
    config = tmp_path / "cfg.yaml"
    base_text = (ROOT / "configs" / "base.yaml").read_text()
    config.write_text(base_text.replace("data/raw/UCI_Credit_Card.csv", "raw.csv"))
    code = main(["--config", str(config), "data"])
    err = capsys.readouterr().err
    assert code == 1
    assert "changed" in err
    assert "Traceback" not in err


def test_no_stage_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_malformed_yaml_prints_error_not_traceback(tmp_path, capsys):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    config = tmp_path / "bad.yaml"
    config.write_text("seed: [1\n")
    code = main(["--config", str(config), "data"])
    err = capsys.readouterr().err
    assert code == 1
    assert err.startswith("error:")
    assert "Traceback" not in err


def test_cli_finds_default_config_from_subfolder(monkeypatch, capsys):
    monkeypatch.chdir(ROOT / "src")
    assert main(["data"]) == 0
    assert "OK" in capsys.readouterr().out


def test_makefile_runs_stages_in_order():
    lines = (ROOT / "Makefile").read_text().splitlines()
    assert ".NOTPARALLEL:" in lines


def make_small_project(tmp_path, n_rows=200):
    """A throwaway project: the first n_rows of the real CSV and a matching config."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    raw = tmp_path / "raw.csv"
    pd.read_csv(ROOT / "data" / "raw" / "UCI_Credit_Card.csv", nrows=n_rows).to_csv(
        raw, index=False
    )
    real_sha = load_config(BASE).raw_sha256
    text = (ROOT / "configs" / "base.yaml").read_text()
    text = text.replace("data/raw/UCI_Credit_Card.csv", "raw.csv")
    text = text.replace(real_sha, sha256_of(raw))
    config = tmp_path / "cfg.yaml"
    config.write_text(text)
    return config


def test_split_stage_writes_train_and_test(tmp_path, capsys):
    config = make_small_project(tmp_path)
    assert main(["--config", str(config), "split"]) == 0
    processed = tmp_path / "data" / "processed"
    train = pd.read_parquet(processed / "train.parquet")
    test = pd.read_parquet(processed / "test.parquet")
    assert len(train) + len(test) == 200
    assert "train" in capsys.readouterr().out


def test_experiments_stage_writes_table(tmp_path, capsys):
    config = make_small_project(tmp_path, n_rows=300)
    assert main(["--config", str(config), "split"]) == 0
    assert main(["--config", str(config), "experiments"]) == 0
    table = pd.read_csv(tmp_path / "reports" / "tables" / "experiments.csv")
    assert table["experiment"].tolist() == [e.name for e in load_config(config).experiments]
    assert "roc_auc" in capsys.readouterr().out


def test_makefile_all_includes_every_stage():
    lines = (ROOT / "Makefile").read_text().splitlines()
    assert "all: data split experiments decide train evaluate explain" in lines


def shrink_decision(config):
    """Cut the decision stage down to one small setup so CLI tests run in seconds."""
    text = config.read_text()
    for old, new in [
        ("cv_folds: 5", "cv_folds: 2"),
        ("candidates: [lgbm, lgbm_weighted, logreg_weighted_allfeatures]", "candidates: [logreg]"),
        ("calibration: [sigmoid, isotonic]", "calibration: [sigmoid]"),
        ("false_negative_costs: [3, 5, 10]", "false_negative_costs: [5]"),
    ]:
        assert old in text
        text = text.replace(old, new)
    config.write_text(text)


def test_decide_stage_writes_table_and_plot(tmp_path, capsys):
    config = make_small_project(tmp_path, n_rows=300)
    shrink_decision(config)
    assert main(["--config", str(config), "split"]) == 0
    assert main(["--config", str(config), "decide"]) == 0
    table = pd.read_csv(tmp_path / "reports" / "tables" / "decision.csv")
    assert table["experiment"].tolist() == ["logreg"]
    assert (tmp_path / "reports" / "figures" / "cost_vs_threshold.png").is_file()
    assert "threshold" in capsys.readouterr().out


def test_train_stage_writes_model(tmp_path, capsys):
    config = make_small_project(tmp_path, n_rows=300)
    shrink_decision(config)
    assert main(["--config", str(config), "split"]) == 0
    assert main(["--config", str(config), "train"]) == 0
    assert (tmp_path / "artifacts" / "model.joblib").is_file()
    assert "Threshold" in capsys.readouterr().out


def test_train_before_split_prints_error(tmp_path, capsys):
    config = make_small_project(tmp_path)
    code = main(["--config", str(config), "train"])
    err = capsys.readouterr().err
    assert code == 1
    assert "make split" in err
    assert "Traceback" not in err


# The 300-row project's test split can hold an EDUCATION code its 240 training rows never
# saw; the encoder sets it to all zeros by design (preprocess.py) and warns. Expected here.
@pytest.mark.filterwarnings("ignore:Found unknown categories:UserWarning")
def test_evaluate_stage_writes_metrics(tmp_path, capsys):
    config = make_small_project(tmp_path, n_rows=300)
    shrink_decision(config)
    for stage in ["split", "train", "evaluate"]:
        assert main(["--config", str(config), stage]) == 0
    table = pd.read_csv(tmp_path / "reports" / "tables" / "test_metrics.csv")
    assert table["setup"].iloc[0] == "model at tuned threshold"
    assert (tmp_path / "reports" / "figures" / "calibration_test.png").is_file()
    assert (tmp_path / "reports" / "figures" / "roc_pr_test.png").is_file()
    assert "flag everyone" in capsys.readouterr().out


def test_evaluate_before_train_prints_error(tmp_path, capsys):
    config = make_small_project(tmp_path)
    assert main(["--config", str(config), "split"]) == 0
    code = main(["--config", str(config), "evaluate"])
    err = capsys.readouterr().err
    assert code == 1
    assert "make train" in err
    assert "Traceback" not in err


@pytest.mark.filterwarnings("ignore:Found unknown categories:UserWarning")  # see above
def test_explain_stage_writes_figures(tmp_path, capsys):
    config = make_small_project(tmp_path, n_rows=300)
    shrink_decision(config)
    for stage in ["split", "train", "explain"]:
        assert main(["--config", str(config), stage]) == 0
    assert (tmp_path / "reports" / "figures" / "shap_summary.png").is_file()
    assert (tmp_path / "reports" / "tables" / "shap_importance.csv").is_file()
    assert "Wrote" in capsys.readouterr().out
