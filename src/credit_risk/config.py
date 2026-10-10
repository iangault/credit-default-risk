"""Load and validate project configuration from YAML.

Paths in a config file are relative to the project root (the nearest folder
above the config file that holds pyproject.toml), so the same config works
from the Makefile, tests, and notebooks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed, or has invalid values."""


@dataclass(frozen=True)
class Paths:
    raw: Path
    processed_dir: Path
    reports_dir: Path
    artifacts_dir: Path


@dataclass(frozen=True)
class Costs:
    false_negative: float
    false_positive: float


@dataclass(frozen=True)
class Experiment:
    """One row of the experiments table: a name, a model name, model settings, an
    optional resampler name (from credit_risk.samplers; None means no resampling), and
    engineered feature groups to add (from credit_risk.features; empty means none).
    """

    name: str
    model: str
    params: dict[str, Any] = field(default_factory=dict)
    sampler: str | None = None
    features: tuple[str, ...] = ()


@dataclass(frozen=True)
class Config:
    paths: Paths
    raw_sha256: str
    seed: int
    test_size: float
    cv_folds: int
    costs: Costs
    sample_rows: int | None = None
    experiments: tuple[Experiment, ...] = ()


_TOP_REQUIRED = {"paths", "raw_sha256", "seed", "test_size", "cv_folds", "costs"}
_TOP_OPTIONAL = {"sample_rows", "experiments"}
_PATH_KEYS = {"raw", "processed_dir", "reports_dir", "artifacts_dir"}
_COST_KEYS = {"false_negative", "false_positive"}


def find_project_root(start: Path) -> Path:
    """Return the nearest directory at or above `start` containing pyproject.toml."""
    start = start.resolve()
    for directory in (start, *start.parents):
        if (directory / "pyproject.toml").is_file():
            return directory
    raise ConfigError(f"No pyproject.toml found at or above {start}; cannot locate project root.")


def default_config_path() -> Path:
    """Return configs/base.yaml for the project containing the current working directory."""
    return find_project_root(Path.cwd()) / "configs" / "base.yaml"


# `path: str | Path` is a type hint: path may be a plain string ("configs/base.yaml")
# or a pathlib.Path object. The `|` means "or". `-> Config` says what comes back.
# Hints are not enforced at runtime; they document intent and let editors catch mistakes.
def load_config(path: str | Path) -> Config:
    """Read a YAML config (following `extends`), validate it, and return a Config."""
    # Convert to a Path whatever was passed, so the code below handles one type.
    # resolve() turns a relative path into a full absolute one.
    path = Path(path).resolve()
    data = _read_yaml(path, frozenset())
    _check_keys("config", data, _TOP_REQUIRED, _TOP_OPTIONAL)
    root = find_project_root(path.parent)

    paths = _section(data, "paths", _PATH_KEYS)
    costs = _section(data, "costs", _COST_KEYS)
    sample_rows = data.get("sample_rows")

    return Config(
        paths=Paths(**{key: root / str(value) for key, value in paths.items()}),
        raw_sha256=_sha256(data["raw_sha256"]),
        seed=_int("seed", data["seed"], minimum=0),
        test_size=_fraction("test_size", data["test_size"]),
        cv_folds=_int("cv_folds", data["cv_folds"], minimum=2),
        costs=Costs(
            false_negative=_positive("costs.false_negative", costs["false_negative"]),
            false_positive=_positive("costs.false_positive", costs["false_positive"]),
        ),
        sample_rows=None if sample_rows is None else _int("sample_rows", sample_rows, minimum=1),
        experiments=_experiments(data["experiments"]) if "experiments" in data else (),
    )


def _read_yaml(path: Path, seen: frozenset[Path]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ConfigError(f"Config 'extends' cycle involving {path}")
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Could not read config file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a YAML mapping of keys to values")
    parent = data.pop("extends", None)
    if parent is None:
        return data
    base = _read_yaml(path.parent / str(parent), seen | {path})
    return _deep_merge(base, data)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _check_keys(
    section: str,
    data: dict[str, Any],
    required: set[str],
    optional: set[str] | frozenset[str] = frozenset(),
) -> None:
    valid = required | optional
    unknown = set(data) - valid
    if unknown:
        raise ConfigError(
            f"Unknown key(s) in {section}: {sorted(unknown)}. Valid keys: {sorted(valid)}"
        )
    missing = required - set(data)
    if missing:
        raise ConfigError(f"Missing required key(s) in {section}: {sorted(missing)}")


def _section(data: dict[str, Any], name: str, keys: set[str]) -> dict[str, Any]:
    section = data[name]
    if not isinstance(section, dict):
        raise ConfigError(f"'{name}' must be a mapping with keys {sorted(keys)}")
    _check_keys(name, section, keys)
    return section


def _experiments(value: Any) -> tuple[Experiment, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(
            "'experiments' must be a non-empty list of {name, model, params, sampler, features}"
        )
    experiments = []
    for i, item in enumerate(value):
        where = f"experiments[{i}]"
        if not isinstance(item, dict):
            raise ConfigError(f"'{where}' must be a mapping with keys name, model, params")
        _check_keys(where, item, {"name", "model"}, {"params", "sampler", "features"})
        params = item.get("params") or {}
        if not isinstance(params, dict):
            raise ConfigError(f"'{where}.params' must be a mapping, got {params!r}")
        # Only the type is checked here. Whether the name exists is checked when the
        # pipeline is built (samplers.get_sampler), the same way model names are.
        sampler = item.get("sampler")
        if sampler is not None and not isinstance(sampler, str):
            raise ConfigError(f"'{where}.sampler' must be a sampler name, got {sampler!r}")
        # A list such as [delinquency, trend]. Group names are checked when the
        # pipeline is built (features.feature_columns), like model and sampler names.
        features = item.get("features", [])
        if not isinstance(features, list) or not all(isinstance(f, str) for f in features):
            raise ConfigError(
                f"'{where}.features' must be a list of feature group names, got {features!r}"
            )
        experiments.append(
            Experiment(
                str(item["name"]), str(item["model"]), dict(params), sampler, tuple(features)
            )
        )
    names = [e.name for e in experiments]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ConfigError(f"Duplicate experiment name(s): {duplicates}")
    return tuple(experiments)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _int(key: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"'{key}' must be an integer >= {minimum}, got {value!r}")
    return value


def _fraction(key: str, value: Any) -> float:
    if not _is_number(value) or not 0 < value < 1:
        raise ConfigError(f"'{key}' must be a number between 0 and 1 (exclusive), got {value!r}")
    return float(value)


def _positive(key: str, value: Any) -> float:
    if not _is_number(value) or value <= 0:
        raise ConfigError(f"'{key}' must be a number > 0, got {value!r}")
    return float(value)


def _sha256(value: Any) -> str:
    text = str(value)
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ConfigError(f"'raw_sha256' must be 64 lowercase hex characters, got {text!r}")
    return text
