import textwrap
from pathlib import Path

import pytest

from credit_risk.config import (
    Config,
    ConfigError,
    Decision,
    Final,
    default_config_path,
    find_project_root,
    load_config,
)

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
    assert smoke.paths.raw == base.paths.raw
    assert smoke.paths.processed_dir != base.paths.processed_dir
    assert smoke.paths.reports_dir != base.paths.reports_dir
    assert smoke.paths.artifacts_dir != base.paths.artifacts_dir
    assert [e.name for e in base.experiments] == [
        "dummy",
        "logreg",
        "forest",
        "lgbm",
        "xgboost",
        "logreg_weighted",
        "logreg_undersample",
        "logreg_smotenc",
        "lgbm_weighted",
        "lgbm_undersample",
        "lgbm_smotenc",
        "logreg_weighted_delinquency",
        "logreg_weighted_payment",
        "logreg_weighted_utilization",
        "logreg_weighted_trend",
        "logreg_weighted_allfeatures",
        "lgbm_weighted_delinquency",
        "lgbm_weighted_payment",
        "lgbm_weighted_utilization",
        "lgbm_weighted_trend",
        "lgbm_weighted_allfeatures",
    ]
    assert base.decision is not None
    assert base.decision.candidates == ("lgbm", "lgbm_weighted", "logreg_weighted_allfeatures")
    assert base.decision.calibration == ("sigmoid", "isotonic")
    assert base.decision.false_negative_costs == (3.0, 5.0, 10.0)
    assert base.final is not None
    assert smoke.decision == base.decision
    assert smoke.final == base.final
    assert smoke.experiments == base.experiments


def test_malformed_yaml_raises_config_error_naming_file(project):
    path = write(project, "broken.yaml", "seed: [1\n")
    with pytest.raises(ConfigError, match="broken.yaml"):
        load_config(path)


def test_non_utf8_config_raises_config_error(project):
    path = project / "configs" / "binary.yaml"
    path.write_bytes(b"\xff\xfe\x00seed: 1")
    with pytest.raises(ConfigError, match="binary.yaml"):
        load_config(path)


def test_default_config_path_found_from_subfolder(monkeypatch):
    monkeypatch.chdir(ROOT / "src")
    assert default_config_path() == ROOT / "configs" / "base.yaml"


def test_experiments_parsed_in_order(project):
    path = write(
        project,
        "exp.yaml",
        """\
        extends: base.yaml
        experiments:
          - {name: base, model: logreg}
          - {name: strong_reg, model: logreg, params: {C: 0.1}}
        """,
    )
    cfg = load_config(path)
    assert [e.name for e in cfg.experiments] == ["base", "strong_reg"]
    assert cfg.experiments[0].params == {}
    assert cfg.experiments[1].params == {"C": 0.1}


def test_experiment_sampler_parsed_and_defaults_to_none(project):
    path = write(
        project,
        "exp.yaml",
        """\
        extends: base.yaml
        experiments:
          - {name: plain, model: logreg}
          - {name: resampled, model: logreg, sampler: smotenc}
        """,
    )
    cfg = load_config(path)
    assert cfg.experiments[0].sampler is None
    assert cfg.experiments[1].sampler == "smotenc"


def test_experiment_features_parsed_and_default_to_empty(project):
    path = write(
        project,
        "exp.yaml",
        """\
        extends: base.yaml
        experiments:
          - {name: plain, model: logreg}
          - {name: engineered, model: logreg, features: [delinquency, trend]}
        """,
    )
    cfg = load_config(path)
    assert cfg.experiments[0].features == ()
    assert cfg.experiments[1].features == ("delinquency", "trend")


def test_experiment_features_null_means_none(project):
    # Same as params: null and sampler: null, which are also read as "not set".
    path = write(
        project,
        "exp.yaml",
        "extends: base.yaml\nexperiments:\n  - {name: a, model: m, features: null}\n",
    )
    assert load_config(path).experiments[0].features == ()


def test_experiments_default_to_empty(project):
    assert load_config(project / "configs" / "base.yaml").experiments == ()


@pytest.mark.parametrize(
    ("block", "fragment"),
    [
        ("experiments: []", "experiments"),
        ("experiments:\n  - {name: a}", "model"),
        ("experiments:\n  - {name: a, model: m, parms: {}}", "parms"),
        ("experiments:\n  - {name: a, model: m, params: 3}", "params"),
        ("experiments:\n  - {name: a, model: m, sampler: 3}", "sampler"),
        ("experiments:\n  - {name: a, model: m, features: delinquency}", "features"),
        ("experiments:\n  - {name: a, model: m, features: [1]}", "features"),
        ("experiments:\n  - {name: a, model: m}\n  - {name: a, model: n}", "Duplicate"),
    ],
)
def test_invalid_experiments_raise(project, block, fragment):
    path = write(project, "bad.yaml", f"extends: base.yaml\n{block}\n")
    with pytest.raises(ConfigError, match=fragment):
        load_config(path)


DECISION_BLOCK = """\
extends: base.yaml
experiments:
  - {name: lr, model: logreg}
  - {name: gbm, model: lgbm}
decision:
  candidates: [lr, gbm]
  calibration: [sigmoid, isotonic]
  false_negative_costs: [3, 5.5]
final:
  experiment: gbm
  calibration: isotonic
"""


def test_decision_and_final_parsed(project):
    cfg = load_config(write(project, "dec.yaml", DECISION_BLOCK))
    assert cfg.decision == Decision(("lr", "gbm"), ("sigmoid", "isotonic"), (3.0, 5.5))
    assert cfg.final == Final("gbm", "isotonic")


def test_decision_and_final_default_to_none(project):
    cfg = load_config(project / "configs" / "base.yaml")
    assert cfg.decision is None
    assert cfg.final is None


@pytest.mark.parametrize(
    ("old", "new", "fragment"),
    [
        ("candidates: [lr, gbm]", "candidates: [lr, gmb]", "decision.candidates"),
        ("candidates: [lr, gbm]", "candidates: []", "decision.candidates"),
        ("calibration: [sigmoid, isotonic]", "calibration: [platt]", "decision.calibration"),
        ("false_negative_costs: [3, 5.5]", "false_negative_costs: [0]", "false_negative_costs"),
        ("false_negative_costs: [3, 5.5]", "false_negative_costs: 5", "false_negative_costs"),
        ("  experiment: gbm", "  experiment: forest", "final.experiment"),
        ("  calibration: isotonic", "  calibration: [isotonic]", "final.calibration"),
        ("  calibration: isotonic\n", "", "final"),
        ("candidates: [lr, gbm]", "candidates: [lr, lr]", "more than once"),
        ("calibration: [sigmoid, isotonic]", "calibration: [sigmoid, sigmoid]", "more than once"),
        ("false_negative_costs: [3, 5.5]", "false_negative_costs: [5, 5.0]", "more than once"),
    ],
)
def test_invalid_decision_and_final_raise(project, old, new, fragment):
    assert old in DECISION_BLOCK
    path = write(project, "bad.yaml", DECISION_BLOCK.replace(old, new))
    with pytest.raises(ConfigError, match=fragment):
        load_config(path)


def test_decision_without_experiments_says_so(project):
    block = DECISION_BLOCK.replace(
        "experiments:\n  - {name: lr, model: logreg}\n  - {name: gbm, model: lgbm}\n", ""
    )
    path = write(project, "noexp.yaml", block)
    with pytest.raises(ConfigError, match="no experiments are defined"):
        load_config(path)
