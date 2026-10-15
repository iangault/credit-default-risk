"""The app's pinned packages must match the env the committed model was saved with."""

import re
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "app" / "requirements.txt"


def pins() -> dict[str, str]:
    """Package name -> pinned version, from app/requirements.txt (comments skipped)."""
    lines = REQUIREMENTS.read_text().splitlines()
    entries = [line.split("#")[0].strip() for line in lines]
    return dict(entry.split("==") for entry in entries if entry)


def test_every_app_requirement_is_pinned_exactly():
    lines = [line.split("#")[0].strip() for line in REQUIREMENTS.read_text().splitlines()]
    assert all("==" in line for line in lines if line)


def test_model_libraries_match_the_installed_versions():
    # A saved model must be loaded with the versions it was saved with.
    for name in ("scikit-learn", "lightgbm"):
        assert pins()[name] == version(name), f"update app/requirements.txt for {name}"


def test_environment_pins_the_same_model_libraries():
    env = (ROOT / "environment.yml").read_text()
    for name in ("scikit-learn", "lightgbm"):
        assert re.search(rf"^\s*- {name}={re.escape(pins()[name])}\b", env, re.MULTILINE)
