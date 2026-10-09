"""Join the preprocessor and a model into one sklearn Pipeline.

Keeping both in one Pipeline means cross-validation refits the scaler and encoder
inside each training fold, so the validation fold never influences them.
M3 swaps this for imblearn's Pipeline when resampling is added.
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline, make_pipeline

from credit_risk.config import Experiment
from credit_risk.models import get_model
from credit_risk.preprocess import build_preprocessor


def build_pipeline(experiment: Experiment, seed: int) -> Pipeline:
    model = get_model(experiment.model, experiment.params, seed)
    return make_pipeline(build_preprocessor(), model)
