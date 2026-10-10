"""Join the preprocessing steps, an optional resampler, and a model into one Pipeline.

A Pipeline is a chain of steps that acts like a single model: calling `.fit(X, y)`
fits each step in order, and `.predict_proba(X)` runs X through every step. Here
the chain is:

    raw columns -> scale -> [resample] -> one-hot encode -> model -> probability of default

The resample step is there only for experiments that name a `sampler` in the config.

Why pair preprocessing with the model instead of preprocessing once up front (as
scratch.ipynb cell 4 did): during cross-validation, the scaler learns means and
standard deviations from the data it is fit on. If it were fit on all of train, the
validation fold's numbers would leak into it. Inside a Pipeline, cross_validate
refits the scaler and encoder on each training fold only, so the validation fold
stays unseen.

Why imblearn's Pipeline instead of sklearn's (M3): sklearn's Pipeline has no place
for a step that changes the rows. imblearn's version accepts resamplers and runs
them only during fit. When cross_validate scores a fold, or predict_proba is
called, the resample step is skipped, so there is one prediction per row and the
validation fold keeps its real 22% default rate. Resampling all of train before
cross-validation would instead let synthetic rows built partly from validation rows
into training.

Every experiment uses this same chain. Without a sampler it gives the same results
as the M2 pipeline.
"""

from __future__ import annotations

from imblearn.pipeline import Pipeline

from credit_risk.config import Experiment
from credit_risk.models import get_model
from credit_risk.preprocess import build_encoder, build_scaler
from credit_risk.samplers import get_sampler


def build_pipeline(experiment: Experiment, seed: int) -> Pipeline:
    """Return an unfitted pipeline for one experiment from the config.

    `experiment` is one entry of the `experiments` list in configs/base.yaml, for
    example {name: lgbm_smotenc, model: lgbm, sampler: smotenc}. The steps are named
    "scale", "sample" (only when the experiment has a sampler), "encode" and "model".
    Nothing is trained here; experiments.py fits it.
    """
    # Look up the model by name (raises ValueError on a typo like "lgmb").
    model = get_model(experiment.model, experiment.params, seed)
    steps = [("scale", build_scaler())]
    if experiment.sampler is not None:
        # Raises ValueError on a typo like "smote", before any fitting starts.
        steps.append(("sample", get_sampler(experiment.sampler, seed)))
    steps += [("encode", build_encoder()), ("model", model)]
    # Steps are named by hand (make_pipeline would name them after their classes) so
    # other code can ask for pipe.named_steps["model"] whatever the model is.
    return Pipeline(steps)
