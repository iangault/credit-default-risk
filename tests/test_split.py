from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from credit_risk.config import load_config
from credit_risk.data import (
    TARGET,
    DataFileError,
    clean,
    load_raw,
    load_split,
    save_splits,
    split,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = load_config(ROOT / "configs" / "base.yaml")


def cfg_with(tmp_path, **changes):
    """The repo's base config, writing splits under tmp_path instead of data/processed."""
    paths = replace(BASE.paths, processed_dir=tmp_path / "processed")
    return replace(BASE, paths=paths, **changes)


@pytest.fixture(scope="module")
def cleaned():
    return clean(load_raw(BASE))


def test_split_sizes_follow_test_size(cleaned, tmp_path):
    train, test = split(cleaned, cfg_with(tmp_path))
    assert len(train) == 24000
    assert len(test) == 6000


def test_split_is_stratified(cleaned, tmp_path):
    train, test = split(cleaned, cfg_with(tmp_path))
    assert abs(train[TARGET].mean() - test[TARGET].mean()) < 0.002


def test_split_has_no_client_in_both(cleaned, tmp_path):
    train, test = split(cleaned, cfg_with(tmp_path))
    assert set(train["ID"]).isdisjoint(test["ID"])
    assert set(train["ID"]) | set(test["ID"]) == set(cleaned["ID"])


def test_split_is_repeatable_with_same_seed(cleaned, tmp_path):
    first = split(cleaned, cfg_with(tmp_path))
    second = split(cleaned, cfg_with(tmp_path))
    pd.testing.assert_frame_equal(first[1], second[1])


def test_split_changes_with_different_seed(cleaned, tmp_path):
    _, test_a = split(cleaned, cfg_with(tmp_path))
    _, test_b = split(cleaned, cfg_with(tmp_path, seed=1))
    assert set(test_a["ID"]) != set(test_b["ID"])


def test_sample_rows_limits_total_rows(cleaned, tmp_path):
    train, test = split(cleaned, cfg_with(tmp_path, sample_rows=1000))
    assert len(train) + len(test) == 1000
    assert len(test) == 200


def test_sample_rows_larger_than_data_uses_all_rows(cleaned, tmp_path):
    train, test = split(cleaned, cfg_with(tmp_path, sample_rows=50000))
    assert len(train) + len(test) == 30000


def test_save_and_load_round_trip(cleaned, tmp_path):
    cfg = cfg_with(tmp_path)
    train, test = split(cleaned, cfg)
    save_splits(train, test, cfg)
    pd.testing.assert_frame_equal(load_split("train", cfg), train)
    pd.testing.assert_frame_equal(load_split("test", cfg), test)


def test_load_split_before_make_split_explains_fix(tmp_path):
    with pytest.raises(DataFileError, match="make split"):
        load_split("train", cfg_with(tmp_path))


def test_load_split_rejects_unknown_name(tmp_path):
    with pytest.raises(ValueError, match="train"):
        load_split("validation", cfg_with(tmp_path))


def test_load_split_uses_project_config_by_default(monkeypatch):
    # From a subfolder with no cfg argument, load_split must look in the project's
    # data/processed. Whether `make split` has run on this machine decides which branch.
    monkeypatch.chdir(ROOT / "src")
    try:
        df = load_split("train")
    except DataFileError as exc:
        assert str(ROOT / "data" / "processed") in str(exc)
    else:
        assert TARGET in df.columns
