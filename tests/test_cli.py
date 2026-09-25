from pathlib import Path

import pytest

from credit_risk.cli import main

ROOT = Path(__file__).resolve().parents[1]
BASE = str(ROOT / "configs" / "base.yaml")


def test_data_stage_passes_on_committed_csv(capsys):
    assert main(["--config", BASE, "data"]) == 0
    assert "OK" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("stage", "milestone"),
    [
        ("split", "M1"),
        ("experiments", "M2"),
        ("train", "M5"),
        ("evaluate", "M6"),
        ("explain", "M6"),
    ],
)
def test_pending_stages_exit_zero_and_name_milestone(stage, milestone, capsys):
    assert main(["--config", BASE, stage]) == 0
    assert milestone in capsys.readouterr().out


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
