from pathlib import Path

import pytest

from credit_risk.config import Experiment, load_config
from credit_risk.data import TARGET, clean, load_raw
from credit_risk.models import MODEL_NAMES
from credit_risk.pipeline import build_pipeline
from credit_risk.preprocess import CATEGORICAL, ORDINAL
from credit_risk.samplers import SAMPLER_NAMES

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
    assert list(pipe.named_steps) == ["scale", "encode", "model"]


def test_pipeline_steps_with_sampler():
    pipe = build_pipeline(Experiment(name="lr", model="logreg", sampler="smotenc"), seed=0)
    assert list(pipe.named_steps) == ["scale", "sample", "encode", "model"]


@pytest.mark.parametrize("sampler", SAMPLER_NAMES)
def test_resampling_balances_training_rows_only(sample, sampler):
    # The dummy model records the class mix it was trained on (class_prior_), which
    # shows what the resample step handed it.
    X, y = sample
    exp = Experiment(name="d", model="dummy", sampler=sampler)
    pipe = build_pipeline(exp, seed=0).fit(X, y)
    assert pipe.named_steps["model"].class_prior_ == pytest.approx([0.5, 0.5])
    # Predicting skips the resample step: one prediction per input row.
    assert len(pipe.predict_proba(X)) == len(X)


def test_no_sampler_keeps_the_real_class_mix(sample):
    X, y = sample
    pipe = build_pipeline(Experiment(name="d", model="dummy"), seed=0).fit(X, y)
    assert pipe.named_steps["model"].class_prior_[1] == pytest.approx(y.mean())


def test_smotenc_rows_have_real_codes(sample):
    X, y = sample
    pipe = build_pipeline(Experiment(name="s", model="logreg", sampler="smotenc"), seed=0)
    # pipe[:2] is the scale and sample steps on their own; fit_resample returns the
    # rows the model would be trained on.
    X_res, y_res = pipe[:2].fit_resample(X, y)
    assert len(X_res) > len(X)  # synthetic defaulters were added
    assert y_res.mean() == pytest.approx(0.5)
    for column in CATEGORICAL + ORDINAL:
        assert set(X_res[column]) <= set(X[column]), column


def test_unknown_sampler_fails_when_building():
    with pytest.raises(ValueError, match="smotenc"):
        build_pipeline(Experiment(name="typo", model="logreg", sampler="smote"), seed=0)


def test_unknown_model_fails_when_building():
    with pytest.raises(ValueError, match="lgbm"):
        build_pipeline(Experiment(name="typo", model="lgmb"), seed=0)
