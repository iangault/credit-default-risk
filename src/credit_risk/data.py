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

# Python setting
# Store type hints as text so newer hint syntax works on older Python
from __future__ import annotations

# Standard libraries
# Computes a SHA fingerprint of the raw CSV, so function can confirm file hasn't changed.
import hashlib
import json

# Handles file paths as objects
from pathlib import Path

# Third-party libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# This project: config holds the settings, and the two functions read them
from credit_risk.config import Config, default_config_path, load_config

RAW_TARGET = "default.payment.next.month"
TARGET = "default"

# List comprehension: for loop on an f-string up to 6
PAY_STATUS = [f"PAY_{i}" for i in range(1, 7)]
BILL_AMTS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMTS = [f"PAY_AMT{i}" for i in range(1, 7)]

# Listing the original columns names and the cleaned out
# _ means only for use inside this file, not for import to other files
_RAW_PAY_STATUS = ["PAY_0", *[f"PAY_{i}" for i in range(2, 7)]]
_DEMOGRAPHICS = ["ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
RAW_COLUMNS = [*_DEMOGRAPHICS, *_RAW_PAY_STATUS, *BILL_AMTS, *PAY_AMTS, RAW_TARGET]
CLEAN_COLUMNS = [*_DEMOGRAPHICS, *PAY_STATUS, *BILL_AMTS, *PAY_AMTS, TARGET]

_EDUCATION_OTHER = 4
_MARRIAGE_OTHER = 3

# Dictionary: column name -> set of values that column may contain after cleaning.
# clean() checks each column against its set and raises an error on anything else,
# so a bad code in the data stops the pipeline instead of slipping through.
# The last line builds one entry per PAY_x column, each allowing -2 to 8
# (range stops before 9); ** unpacks those entries into this dictionary.
_ALLOWED_VALUES = {
    "SEX": {1, 2},
    "EDUCATION": {1, 2, 3, 4},
    "MARRIAGE": {1, 2, 3},
    TARGET: {0, 1},
    **{col: set(range(-2, 9)) for col in PAY_STATUS},
}

# Tuple: can't be changed, read-only, order stays same
SPLITS = ("train", "test")


# Empty on purpose: inherits all behaviour from RuntimeError. The separate name lets
# the CLI catch data problems and print a short message, while bugs still show a traceback.
class DataFileError(RuntimeError):
    """Raised when a data file is missing or does not match what the pipeline expects."""


# Takes a file Path, returns its sha256 fingerprint as a str.
def sha256_of(path: Path) -> str:
    """Return the hex sha256 of a file, read in 1 MB chunks."""
    # A sha256 hash is a 64-character fingerprint of a file's contents. Changing even
    # one byte gives a different fingerprint, so verify_raw uses it to detect that the
    # raw data file is not the one the pipeline was built against.
    digest = hashlib.sha256()  # empty hasher; bytes get fed in below
    # "rb" = read binary: hash the raw bytes, with no text decoding or newline changes.
    with path.open("rb") as f:
        # Read 1 MB at a time (1 << 20 = 2**20 = 1,048,576 bytes) so a large file
        # never has to fit in memory all at once. iter(func, b"") keeps calling func
        # and stops when it returns b"" (empty bytes), which is what read() gives
        # at the end of the file.
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)  # add this chunk to the running hash
    return digest.hexdigest()  # final fingerprint as a hex string, e.g. "3f9a..."


# Takes the Config; returns nothing (None). Raises DataFileError if the raw file is bad.
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


# Takes the Config, returns the raw CSV as a DataFrame.
def load_raw(cfg: Config) -> pd.DataFrame:
    """Check the raw CSV's hash, then read it."""
    verify_raw(cfg)
    return pd.read_csv(cfg.paths.raw)


# Takes the raw DataFrame, returns a cleaned copy.
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning decisions in the module docstring. Returns a new DataFrame."""
    # List comprehension: collect every expected column that is not in the data.
    missing = [col for col in RAW_COLUMNS if col not in df.columns]
    # An empty list counts as False, so this only runs if something is missing.
    if missing:
        raise DataFileError(f"Raw data is missing column(s): {missing}")

    # Keep only the expected columns, in a fixed order. copy() makes a separate
    # DataFrame so the changes below don't touch the caller's df.
    out = df[RAW_COLUMNS].copy()
    # Rename columns using an old -> new dictionary (decisions 4 and 7).
    out = out.rename(columns={"PAY_0": "PAY_1", RAW_TARGET: TARGET})
    # replace() swaps values using an old -> new dictionary: undocumented
    # EDUCATION codes 0, 5, 6 all become 4 "other/unknown" (decision 1).
    out["EDUCATION"] = out["EDUCATION"].replace(
        {0: _EDUCATION_OTHER, 5: _EDUCATION_OTHER, 6: _EDUCATION_OTHER}
    )
    # Same idea: undocumented MARRIAGE code 0 becomes 3 "other" (decision 2).
    out["MARRIAGE"] = out["MARRIAGE"].replace({0: _MARRIAGE_OTHER})

    # Check each column against its allowed values. items() gives (key, value)
    # pairs, so col is the column name and allowed is its set of valid values.
    for col, allowed in _ALLOWED_VALUES.items():
        # unique() lists the distinct values in the column; {...} makes them a set
        # of ints; "- allowed" is set difference: values present but not allowed.
        # sorted() turns the result into an ordered list for a readable message.
        unexpected = sorted({int(v) for v in out[col].unique()} - allowed)
        # Any leftover value means the data has a code we didn't plan for: stop here.
        if unexpected:
            raise DataFileError(
                f"Column {col} has unexpected value(s) {unexpected}; expected {sorted(allowed)}"
            )
    return out


# Takes a DataFrame and the Config, returns a tuple of two DataFrames: (train, test).
def split(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split, after an optional stratified sample of cfg.sample_rows."""
    # Optional step 1: shrink the data. sample_rows is None in base.yaml (use all
    # rows) and 1000 in smoke.yaml (a fast test run). Skipped if sample_rows is not
    # set, or if it is not smaller than the data anyway.
    if cfg.sample_rows is not None and cfg.sample_rows < len(df):
        # train_test_split cuts a DataFrame into two parts. Here it is borrowed just
        # to sample: keep sample_rows rows in df, and throw the rest away (_ is the
        # conventional name for a value you don't need).
        # stratify=df[TARGET]: keep the same share of defaults in the sample as in
        # the full data (about 22%), instead of leaving it to chance.
        # random_state=cfg.seed: a fixed seed, so the same rows are picked every run.
        df, _ = train_test_split(
            df, train_size=cfg.sample_rows, stratify=df[TARGET], random_state=cfg.seed
        )
    # Step 2: the real split. test_size=0.2 puts 20% of rows in test and 80% in
    # train, with the same default rate in both (stratify) and the same rows each
    # run (random_state). The two returned parts are unpacked into train and test.
    train, test = train_test_split(
        df, test_size=cfg.test_size, stratify=df[TARGET], random_state=cfg.seed
    )
    # train_test_split shuffles rows; _tidy puts them back in ID order.
    return _tidy(train), _tidy(test)


# Takes a DataFrame, returns it sorted by ID with the index reset to 0, 1, 2, ...
def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    # sort_values("ID") orders rows by client ID. After shuffling, the row labels
    # (the index) are jumbled; reset_index renumbers them 0, 1, 2, ... and
    # drop=True discards the old labels instead of keeping them as a new column.
    return df.sort_values("ID").reset_index(drop=True)


# Takes a split name ("train"/"test") and the Config, returns the Path of its Parquet file.
def split_path(name: str, cfg: Config) -> Path:
    # With Path objects, / joins path parts, e.g. data/processed + "train.parquet"
    # -> data/processed/train.parquet (data/processed/smoke/... for smoke runs).
    # Parquet is a compact column-based file format that keeps each column's type.
    return cfg.paths.processed_dir / f"{name}.parquet"


def split_settings_path(cfg: Config) -> Path:
    return cfg.paths.processed_dir / "split_settings.json"


def _split_settings(cfg: Config) -> dict:
    """The config values that decide which rows land in train and test."""
    return {
        "raw_sha256": cfg.raw_sha256,
        "seed": cfg.seed,
        "test_size": cfg.test_size,
        "sample_rows": cfg.sample_rows,
    }


# Takes the train and test DataFrames and the Config; writes them to disk, returns None.
def save_splits(train: pd.DataFrame, test: pd.DataFrame, cfg: Config) -> None:
    # Create the output folder. parents=True also creates any missing folders above
    # it; exist_ok=True means no error if it is already there.
    cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    # Write each DataFrame to its file. index=False skips saving the 0, 1, 2, ...
    # row labels, since they carry no information.
    train.to_parquet(split_path("train", cfg), index=False)
    test.to_parquet(split_path("test", cfg), index=False)
    # Record the settings used, so load_split can refuse a split made with other ones.
    split_settings_path(cfg).write_text(json.dumps(_split_settings(cfg), indent=2))


# `cfg: Config | None = None`: cfg is either a Config object or None, and defaults to
# None. That makes it optional, so a notebook can call load_split("train") without
# building a config first. `-> pd.DataFrame` says the function returns a DataFrame.
def load_split(name: str, cfg: Config | None = None) -> pd.DataFrame:
    """Load the train or test split. With no cfg, uses this project's configs/base.yaml."""
    # Reject anything other than "train" or "test". {name!r} prints the value with
    # quotes (its repr), so a typo like "tain" is easy to spot in the message.
    if name not in SPLITS:
        raise ValueError(f"name must be one of {SPLITS}, got {name!r}")
    # If no cfg was passed (None), load the default config file instead.
    cfg = cfg or load_config(default_config_path())
    # Build the file path, e.g. data/processed/train.parquet.
    path = split_path(name, cfg)
    # If the split stage hasn't been run yet, say how to fix it.
    if not path.is_file():
        raise DataFileError(f"{path} not found. Run `make split` first.")
    # Refuse a split made with different settings (e.g. seed changed without re-splitting).
    settings_file = split_settings_path(cfg)
    saved = json.loads(settings_file.read_text()) if settings_file.is_file() else None
    if saved != _split_settings(cfg):
        raise DataFileError(
            f"The split in {cfg.paths.processed_dir} was made with different settings "
            f"({saved}) than the current config ({_split_settings(cfg)}). "
            "Run `make split` again."
        )
    # Read the Parquet file back into a DataFrame.
    return pd.read_parquet(path)
