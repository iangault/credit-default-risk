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
