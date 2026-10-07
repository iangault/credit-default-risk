"""Model registry: turn a model name from the config into an unfitted estimator.

No training happens here. `get_model("lgbm")` returns the same object as writing
`LGBMClassifier(verbose=-1, random_state=0)` by hand. Other code fits it (both files
come in M2 Task 6): pipeline.py puts it behind the preprocessor, and experiments.py
cross-validates each experiment in configs/base.yaml and writes the scores to
reports/tables/experiments.csv.

Why this lives in src/ and not only in the scratch notebook, where the comparison
was first run: in a notebook, cells can be run out of order or edited after they ran,
there are no tests, and the results live only in the notebook's saved output. Here the
comparison reruns from one command and the results are written to a file.

Every model starts from library defaults, plus only the settings it needs to run
cleanly. That keeps M2 a fair baseline: score differences come from the model family,
not from tuning. Tuning goes through `params` in the config.

Findings from notebooks/scratch.ipynb (default settings, 5-fold CV ROC-AUC, val / train):
lgbm 0.776 / 0.893, xgboost 0.760 / 0.959, forest 0.759 / 1.000, logreg 0.722 / 0.725.
The tree models beat logreg; lgbm scores best with the smallest train/val gap.
"""

from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# name -> (class, settings that differ from the library default)
_MODELS = {
    # Predicts the training default rate for everyone: the floor a real model must beat.
    "dummy": (DummyClassifier, {"strategy": "prior"}),
    # Raised from the default 100 so the solver converges.
    "logreg": (LogisticRegression, {"max_iter": 1000}),
    # Deep, unrestricted trees averaged together; memorizes train rows at default settings.
    "forest": (RandomForestClassifier, {}),
    # Silences per-fit log output.
    "lgbm": (LGBMClassifier, {"verbose": -1}),
    "xgboost": (XGBClassifier, {}),
}

MODEL_NAMES = tuple(_MODELS)


def get_model(name: str, params: dict | None = None, seed: int = 0):
    """Return a new, unfitted model for `name`, with `params` applied and `random_state=seed`.

    Order: library defaults, then the settings in `_MODELS`, then `random_state=seed`,
    then `params`, so the config has the last word.
    Raise ValueError naming the valid options if `name` is not in MODEL_NAMES, or naming
    the bad keys if `params` has a key the model does not accept.
    """
    if name not in _MODELS:
        raise ValueError(f"Unknown model {name!r}. Valid names: {', '.join(MODEL_NAMES)}")
    cls, settings = _MODELS[name]
    model = cls(**settings, random_state=seed)

    params = params or {}
    # LightGBM and XGBoost accept any keyword without complaint, so check names here
    # rather than let a typo in the config silently fall back to the default.
    unknown = sorted(set(params) - set(model.get_params()))
    if unknown:
        raise ValueError(f"Unknown params for {name!r}: {', '.join(unknown)}")
    return model.set_params(**params)
