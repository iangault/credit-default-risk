"""Raw data access and validation."""

from __future__ import annotations

import hashlib
from pathlib import Path

from credit_risk.config import Config


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
    actual = sha256_of(path)
    if actual != cfg.raw_sha256:
        raise DataFileError(
            f"Raw data at {path} has changed.\n"
            f"  expected sha256: {cfg.raw_sha256}\n"
            f"  actual sha256:   {actual}\n"
            "Restore the committed file with `git checkout -- data/raw/`."
        )
