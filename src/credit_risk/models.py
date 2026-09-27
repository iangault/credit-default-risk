"""Model registry: turn a model name from the config into an unfitted estimator.

Written by the author in M2 (see docs/superpowers/plans/2026-09-26-m2-baseline.md, Task 5).
"""

from __future__ import annotations

# noqa: F401 keeps ruff from deleting these while get_model is still empty; remove the
# markers once get_model uses the three classes.
from lightgbm import LGBMClassifier  # noqa: F401
from sklearn.dummy import DummyClassifier  # noqa: F401
from sklearn.linear_model import LogisticRegression  # noqa: F401

MODEL_NAMES = ("dummy", "logreg", "lgbm")


def get_model(name: str, params: dict | None = None, seed: int = 0):
    """Return a new, unfitted model for `name`, with `params` applied and `random_state=seed`.

    dummy: DummyClassifier(strategy="prior")
    logreg: LogisticRegression(max_iter=1000)
    lgbm: LGBMClassifier(verbose=-1)
    Raise ValueError naming the valid options if `name` is not in MODEL_NAMES.
    """
    raise NotImplementedError("Author writes this in M2 Task 5")
