from pathlib import Path

import pytest

from credit_risk.config import Config, Costs, Paths, load_config
from credit_risk.data import DataFileError, sha256_of, verify_raw

ROOT = Path(__file__).resolve().parents[1]
ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def make_config(raw: Path, sha: str) -> Config:
    base = raw.parent
    return Config(
        paths=Paths(
            raw=raw,
            processed_dir=base / "processed",
            reports_dir=base / "reports",
            artifacts_dir=base / "artifacts",
        ),
        raw_sha256=sha,
        seed=0,
        test_size=0.2,
        cv_folds=2,
        costs=Costs(false_negative=5.0, false_positive=1.0),
    )


def test_sha256_of_known_content(tmp_path):
    f = tmp_path / "abc.txt"
    f.write_bytes(b"abc")
    assert sha256_of(f) == ABC_SHA256


def test_verify_raw_passes_on_matching_hash(tmp_path):
    f = tmp_path / "raw.csv"
    f.write_bytes(b"abc")
    verify_raw(make_config(f, ABC_SHA256))


def test_verify_raw_raises_when_file_changed(tmp_path):
    f = tmp_path / "raw.csv"
    f.write_bytes(b"abd")
    with pytest.raises(DataFileError, match="changed"):
        verify_raw(make_config(f, ABC_SHA256))


def test_verify_raw_raises_when_file_missing(tmp_path):
    with pytest.raises(DataFileError, match="not found"):
        verify_raw(make_config(tmp_path / "missing.csv", ABC_SHA256))


def test_committed_raw_csv_matches_base_config():
    verify_raw(load_config(ROOT / "configs" / "base.yaml"))


def test_gitattributes_keeps_raw_csv_byte_exact():
    rules = (ROOT / ".gitattributes").read_text().splitlines()
    assert "data/raw/*.csv -text" in rules


def test_verify_raw_raises_data_file_error_when_unreadable(tmp_path):
    f = tmp_path / "raw.csv"
    f.write_bytes(b"abc")
    f.chmod(0o000)
    try:
        with pytest.raises(DataFileError, match="could not be read"):
            verify_raw(make_config(f, ABC_SHA256))
    finally:
        f.chmod(0o644)
