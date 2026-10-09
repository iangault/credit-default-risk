"""Join the preprocessor and a model into one sklearn Pipeline.

A Pipeline is a chain of steps that acts like a single model: calling `.fit(X, y)`
fits each step in order, and `.predict_proba(X)` runs X through every step. Here
the chain has two steps:

    raw columns -> preprocessor (scale, one-hot) -> model -> probability of default

Why pair them instead of preprocessing once up front (as scratch.ipynb cell 4 did):
during cross-validation, the scaler learns means and standard deviations from the
data it is fit on. If it were fit on all of train, the validation fold's numbers
would leak into it. Inside a Pipeline, cross_validate refits the scaler and encoder
on each training fold only, so the validation fold stays unseen.

M3 swaps this for imblearn's Pipeline when resampling is added.
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline, make_pipeline

from credit_risk.config import Experiment
from credit_risk.models import get_model
from credit_risk.preprocess import build_preprocessor


def build_pipeline(experiment: Experiment, seed: int) -> Pipeline:
    """Return an unfitted preprocessor + model Pipeline for one experiment from the config.

    `experiment` is one entry of the `experiments` list in configs/base.yaml, for
    example {name: lgbm, model: lgbm}. Nothing is trained here; experiments.py fits it.
    """
    # Look up the model by name (raises ValueError on a typo like "lgmb").
    model = get_model(experiment.model, experiment.params, seed)
    # make_pipeline names each step after its class in lowercase, so the steps are
    # "columntransformer" and e.g. "lgbmclassifier".
    return make_pipeline(build_preprocessor(), model)
