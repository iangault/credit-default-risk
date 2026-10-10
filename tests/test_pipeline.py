from pathlib import Path

import pytest

from credit_risk.config import Experiment, load_config
from credit_risk.data import TARGET, clean, load_raw
from credit_risk.models import MODEL_NAMES
from credit_risk.pipeline import build_pipeline

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sample():
    cleaned = clean(load_raw(load_config(ROOT / "configs" / "base.yaml"))).head(1000)
    return cleaned.drop(columns=TARGET), cleaned[TARGET]


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_pipeline_fits_and_predicts_probabilities(sample, model):
    X, y = sample
    pipe = build_pipeline(Experiment(name=model, model=model), seed=0).fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_steps():
    pipe = build_pipeline(Experiment(name="lr", model="logreg"), seed=0)
    assert list(pipe.named_steps) == ["pipeline", "logisticregression"]


def test_unknown_model_fails_when_building():
    with pytest.raises(ValueError, match="lgbm"):
        build_pipeline(Experiment(name="typo", model="lgmb"), seed=0)
