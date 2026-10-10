"""Sampler registry: turn a sampler name from the config into an unfitted resampler.

A resampler changes the training rows before the model sees them, to even out the
78% / 22% class split (M3). Two are compared:

- "undersample": RandomUnderSampler drops randomly chosen non-default rows until
  the two classes are the same size. Only real rows, but fewer of them: a training
  fold shrinks from about 19,200 rows to about 8,500.
- "smotenc": SMOTENC adds synthetic defaulters until the classes are the same size.
  For each new row it picks a defaulter and one of its 5 nearest defaulter
  neighbours. Numeric columns get a value at a random point between the two; the
  columns listed as categorical get the most common value among the neighbours,
  so every code is one that exists. PAY_1 to PAY_6 are listed as categorical here
  even though the model treats them as numbers, so a synthetic row cannot get
  PAY_1 = 1.4.

Class weights are the third M3 technique, but they are a model setting
(`params: {class_weight: balanced}` in the config), not a resampler, so they are
not in this file.

Resamplers run only when the pipeline is fitted, not when it predicts or is scored
(see pipeline.py), so validation folds keep their real class mix.
"""

from __future__ import annotations

from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler

from credit_risk.preprocess import CATEGORICAL, ORDINAL

# name -> function that builds the resampler from the seed. A function rather than
# a ready-made object, so each call returns a fresh, unfitted one.
_SAMPLERS = {
    "undersample": lambda seed: RandomUnderSampler(random_state=seed),
    # Column names work because the scale step before it outputs a DataFrame.
    "smotenc": lambda seed: SMOTENC(categorical_features=CATEGORICAL + ORDINAL, random_state=seed),
}

SAMPLER_NAMES = tuple(_SAMPLERS)


def get_sampler(name: str, seed: int = 0):
    """Return a new, unfitted resampler for `name`, with `random_state=seed`.

    Raise ValueError naming the valid options if `name` is not in SAMPLER_NAMES.
    """
    if name not in _SAMPLERS:
        raise ValueError(f"Unknown sampler {name!r}. Valid names: {', '.join(SAMPLER_NAMES)}")
    return _SAMPLERS[name](seed)
