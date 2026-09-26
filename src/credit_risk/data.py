"""Raw data access, cleaning, and train/test splitting.

Cleaning decisions (approved in M1):
1. EDUCATION codes 0, 5, 6 are undocumented; merged into 4 ("other/unknown").
2. MARRIAGE code 0 is undocumented; merged into 3 ("other").
3. PAY_x values -2 (no card use), -1 (paid in full), 0 (paid the minimum),
   1-8 (months late) are kept as-is; how to use them is feature engineering.
4. PAY_0 is renamed PAY_1 so the six months line up with BILL_AMT1-6 and PAY_AMT1-6.
5. Negative BILL_AMT values (credit balances) are kept.
6. Rows identical apart from ID are kept (likely separate inactive clients).
7. The target is renamed to `default`.
8. ID is kept so tests can check that no client is in both train and test.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk.config import Config, default_config_path, load_config

RAW_TARGET = "default.payment.next.month"
TARGET = "default"
PAY_STATUS = [f"PAY_{i}" for i in range(1, 7)]
BILL_AMTS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMTS = [f"PAY_AMT{i}" for i in range(1, 7)]

_RAW_PAY_STATUS = ["PAY_0", *[f"PAY_{i}" for i in range(2, 7)]]
_DEMOGRAPHICS = ["ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
RAW_COLUMNS = [*_DEMOGRAPHICS, *_RAW_PAY_STATUS, *BILL_AMTS, *PAY_AMTS, RAW_TARGET]
CLEAN_COLUMNS = [*_DEMOGRAPHICS, *PAY_STATUS, *BILL_AMTS, *PAY_AMTS, TARGET]

_EDUCATION_OTHER = 4
_MARRIAGE_OTHER = 3
_ALLOWED_VALUES = {
    "SEX": {1, 2},
    "EDUCATION": {1, 2, 3, 4},
    "MARRIAGE": {1, 2, 3},
    TARGET: {0, 1},
    **{col: set(range(-2, 9)) for col in PAY_STATUS},
}

SPLITS = ("train", "test")


class DataFileError(RuntimeError):
    """Raised when a data file is missing or does not match what the pipeline expects."""


def sha256_of(path: Path) -> str:
    """Return the hex sha256 of a file, read in 1 MB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_raw(cfg: Config) -> None:
    """Check that the raw CSV exists and matches the expected sha256."""
    path = cfg.paths.raw
    if not path.is_file():
        raise DataFileError(
            f"Raw data not found at {path}. It is committed to the repo; "
            f"restore it with `git checkout -- {path.name}` from its folder."
        )
    try:
        actual = sha256_of(path)
    except OSError as exc:
        raise DataFileError(f"Raw data at {path} could not be read: {exc}") from exc
    if actual != cfg.raw_sha256:
        raise DataFileError(
            f"Raw data at {path} has changed.\n"
            f"  expected sha256: {cfg.raw_sha256}\n"
            f"  actual sha256:   {actual}\n"
            "Restore the committed file with `git checkout -- data/raw/`."
        )


def load_raw(cfg: Config) -> pd.DataFrame:
    """Check the raw CSV's hash, then read it."""
    verify_raw(cfg)
    return pd.read_csv(cfg.paths.raw)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning decisions in the module docstring. Returns a new DataFrame."""
    missing = [col for col in RAW_COLUMNS if col not in df.columns]
    if missing:
        raise DataFileError(f"Raw data is missing column(s): {missing}")

    out = df[RAW_COLUMNS].copy()
    out = out.rename(columns={"PAY_0": "PAY_1", RAW_TARGET: TARGET})
    out["EDUCATION"] = out["EDUCATION"].replace(
        {0: _EDUCATION_OTHER, 5: _EDUCATION_OTHER, 6: _EDUCATION_OTHER}
    )
    out["MARRIAGE"] = out["MARRIAGE"].replace({0: _MARRIAGE_OTHER})

    for col, allowed in _ALLOWED_VALUES.items():
        unexpected = sorted({int(v) for v in out[col].unique()} - allowed)
        if unexpected:
            raise DataFileError(
                f"Column {col} has unexpected value(s) {unexpected}; expected {sorted(allowed)}"
            )
    return out


def split(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split, after an optional stratified sample of cfg.sample_rows."""
    if cfg.sample_rows is not None and cfg.sample_rows < len(df):
        df, _ = train_test_split(
            df, train_size=cfg.sample_rows, stratify=df[TARGET], random_state=cfg.seed
        )
    train, test = train_test_split(
        df, test_size=cfg.test_size, stratify=df[TARGET], random_state=cfg.seed
    )
    return _tidy(train), _tidy(test)


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("ID").reset_index(drop=True)


def split_path(name: str, cfg: Config) -> Path:
    return cfg.paths.processed_dir / f"{name}.parquet"


def save_splits(train: pd.DataFrame, test: pd.DataFrame, cfg: Config) -> None:
    cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(split_path("train", cfg), index=False)
    test.to_parquet(split_path("test", cfg), index=False)


def load_split(name: str, cfg: Config | None = None) -> pd.DataFrame:
    """Load the train or test split. With no cfg, uses this project's configs/base.yaml."""
    if name not in SPLITS:
        raise ValueError(f"name must be one of {SPLITS}, got {name!r}")
    cfg = cfg or load_config(default_config_path())
    path = split_path(name, cfg)
    if not path.is_file():
        raise DataFileError(f"{path} not found. Run `make split` first.")
    return pd.read_parquet(path)
