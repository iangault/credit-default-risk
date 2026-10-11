from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from credit_risk.config import ConfigError, Final, load_config
from credit_risk.data import TARGET, clean, load_raw
from credit_risk.decision import train_final
from credit_risk.explain import check_tree_model, inner_pipeline, pick_clients, shap_explanation

ROOT = Path(__file__).resolve().parents[1]
BASE = load_config(ROOT / "configs" / "base.yaml")


LGBM_FINAL = Final("lgbm", "sigmoid")


def small_cfg(tmp_path, final=LGBM_FINAL, **changes):
    paths = replace(
        BASE.paths, reports_dir=tmp_path / "reports", artifacts_dir=tmp_path / "artifacts"
    )
    return replace(BASE, paths=paths, cv_folds=2, final=final, **changes)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    # Samples of the cleaned raw data stand in for train and test (the split is not read).
    rows = clean(load_raw(BASE)).sample(1200, random_state=0)
    train, test = rows.iloc[:1000], rows.iloc[1000:]
    model = train_final(small_cfg(tmp_path_factory.mktemp("m")), train)
    return model, test.drop(columns=TARGET), test[TARGET]


def test_inner_pipeline_is_the_fitted_pipeline(fitted):
    model, X, _ = fitted
    pipe = inner_pipeline(model)
    assert list(pipe.named_steps) == ["scale", "encode", "model"]


def test_shap_values_add_up_to_the_model_score(fitted):
    model, X, _ = fitted
    explanation = shap_explanation(model, X)
    pipe = inner_pipeline(model)
    raw = pipe.named_steps["model"].predict(pipe[:-1].transform(X), raw_score=True)
    # SHAP's defining property: base value + the row's SHAP values = the model's score.
    total = explanation.values.sum(axis=1) + explanation.base_values
    assert np.allclose(total, raw)
    assert explanation.values.shape == (len(X), 28)


def test_shap_display_values_are_unscaled(fitted):
    model, X, _ = fitted
    explanation = shap_explanation(model, X)
    column = explanation.feature_names.index("LIMIT_BAL")
    assert np.array_equal(explanation.data[:, column], X["LIMIT_BAL"].to_numpy())


# 400 rows in 2-fold CV: a fold can lack a rare EDUCATION or MARRIAGE code, which the
# encoder sets to all zeros by design (preprocess.py) and warns about. Expected here.
@pytest.mark.filterwarnings("ignore:Found unknown categories:UserWarning")
def test_forest_explanation_keeps_the_default_class(tmp_path):
    rows = clean(load_raw(BASE)).sample(400, random_state=1)
    model = train_final(small_cfg(tmp_path, final=Final("forest", "sigmoid")), rows)
    explanation = shap_explanation(model, rows.drop(columns=TARGET).head(20))
    assert explanation.values.shape == (20, 28)  # one value per column, class 1 only


def test_non_tree_final_model_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="lgbm"):
        check_tree_model(small_cfg(tmp_path, final=Final("logreg", "sigmoid")))
    check_tree_model(small_cfg(tmp_path))  # lgbm passes


def test_pick_clients_takes_the_median_of_each_group():
    y = [1, 1, 1, 0, 1, 1, 1]
    proba = [0.9, 0.6, 0.3, 0.8, 0.1, 0.05, 0.7]
    # Flagged defaulters (>= 0.5): rows 0, 1, 6 (0.9, 0.6, 0.7), median 0.7 -> row 6.
    # Missed defaulters: rows 2, 4, 5 (0.3, 0.1, 0.05), median 0.1 -> row 4.
    assert pick_clients(y, proba, 0.5) == {"flagged": 6, "missed": 4}


def test_pick_clients_skips_an_empty_group():
    assert pick_clients([1, 0, 0], [0.9, 0.2, 0.1], 0.5) == {"flagged": 0}
    assert pick_clients([0, 0], [0.9, 0.1], 0.5) == {}
