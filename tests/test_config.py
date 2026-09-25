import textwrap
from pathlib import Path

import pytest

from credit_risk.config import Config, ConfigError, find_project_root, load_config

ROOT = Path(__file__).resolve().parents[1]

BASE = textwrap.dedent(
    f"""\
    paths:
      raw: data/raw/x.csv
      processed_dir: data/processed
      reports_dir: reports
      artifacts_dir: artifacts
    raw_sha256: {"a" * 64}
    seed: 1
    test_size: 0.2
    cv_folds: 5
    costs:
      false_negative: 5
      false_positive: 1
    """
)


@pytest.fixture
def project(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "base.yaml").write_text(BASE)
    return tmp_path.resolve()


def write(project, name, text):
    path = project / "configs" / name
    path.write_text(textwrap.dedent(text))
    return path


def test_loads_valid_config(project):
    cfg = load_config(project / "configs" / "base.yaml")
    assert isinstance(cfg, Config)
    assert cfg.seed == 1
    assert cfg.test_size == 0.2
    assert cfg.cv_folds == 5
    assert cfg.costs.false_negative == 5.0
    assert cfg.costs.false_positive == 1.0
    assert cfg.sample_rows is None


def test_paths_resolve_against_project_root_not_cwd(project, monkeypatch):
    monkeypatch.chdir(project / "configs")
    cfg = load_config("base.yaml")
    assert cfg.paths.raw == project / "data" / "raw" / "x.csv"
    assert cfg.paths.artifacts_dir == project / "artifacts"


def test_extends_overrides_only_given_keys(project):
    path = write(project, "smoke.yaml", "extends: base.yaml\nsample_rows: 100\ncv_folds: 2\n")
    cfg = load_config(path)
    assert cfg.sample_rows == 100
    assert cfg.cv_folds == 2
    assert cfg.seed == 1


def test_extends_deep_merges_nested_sections(project):
    path = write(project, "c.yaml", "extends: base.yaml\ncosts:\n  false_negative: 10\n")
    cfg = load_config(path)
    assert cfg.costs.false_negative == 10.0
    assert cfg.costs.false_positive == 1.0


def test_extends_cycle_raises(project):
    write(project, "a.yaml", "extends: b.yaml\n")
    write(project, "b.yaml", "extends: a.yaml\n")
    with pytest.raises(ConfigError, match="cycle"):
        load_config(project / "configs" / "a.yaml")


def test_missing_file_raises(project):
    with pytest.raises(ConfigError, match="not found"):
        load_config(project / "configs" / "nope.yaml")


def test_empty_file_raises(project):
    path = write(project, "empty.yaml", "")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


def test_unknown_top_level_key_names_valid_keys(project):
    path = write(project, "typo.yaml", "extends: base.yaml\ncv_fold: 3\n")
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    assert "cv_fold" in str(exc.value)
    assert "cv_folds" in str(exc.value)


def test_unknown_nested_key_names_section(project):
    path = write(project, "typo.yaml", "extends: base.yaml\npaths:\n  rwa: x.csv\n")
    with pytest.raises(ConfigError, match="paths"):
        load_config(path)


def test_missing_required_key_raises(project):
    no_seed = "\n".join(line for line in BASE.splitlines() if not line.startswith("seed"))
    path = write(project, "noseed.yaml", no_seed + "\n")
    with pytest.raises(ConfigError, match="seed"):
        load_config(path)


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ("test_size: 1.5", "test_size"),
        ("test_size: 0", "test_size"),
        ("cv_folds: 1", "cv_folds"),
        ("cv_folds: true", "cv_folds"),
        ("seed: 1.5", "seed"),
        ("costs:\n  false_negative: 0", "false_negative"),
        ("costs:\n  false_positive: -1", "false_positive"),
        ("costs:\n  false_positive: abc", "false_positive"),
        ("raw_sha256: xyz", "raw_sha256"),
        ("sample_rows: 0", "sample_rows"),
    ],
)
def test_invalid_values_raise_naming_the_key(project, override, fragment):
    path = write(project, "bad.yaml", f"extends: base.yaml\n{override}\n")
    with pytest.raises(ConfigError, match=fragment):
        load_config(path)


def test_find_project_root_walks_up(project):
    nested = project / "a" / "b"
    nested.mkdir(parents=True)
    assert find_project_root(nested) == project


def test_find_project_root_raises_without_pyproject(tmp_path):
    with pytest.raises(ConfigError, match="pyproject.toml"):
        find_project_root(tmp_path)


def test_repo_configs_load():
    base = load_config(ROOT / "configs" / "base.yaml")
    smoke = load_config(ROOT / "configs" / "smoke.yaml")
    assert base.seed == 573
    assert base.cv_folds == 5
    assert base.sample_rows is None
    assert smoke.cv_folds == 2
    assert smoke.sample_rows == 1000
    assert smoke.paths == base.paths


def test_malformed_yaml_raises_config_error_naming_file(project):
    path = write(project, "broken.yaml", "seed: [1\n")
    with pytest.raises(ConfigError, match="broken.yaml"):
        load_config(path)


def test_non_utf8_config_raises_config_error(project):
    path = project / "configs" / "binary.yaml"
    path.write_bytes(b"\xff\xfe\x00seed: 1")
    with pytest.raises(ConfigError, match="binary.yaml"):
        load_config(path)
