# M3 Imbalance Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `make experiments` adds six rows comparing class weights, random undersampling and SMOTENC on logistic regression and LightGBM, under the same stratified 5-fold CV as M2, plus a short explanation of why ROC-AUC changes little.

**Architecture:** The single-step preprocessor is split into `build_scaler()` (scale amounts, keep category and PAY codes as single columns) and `build_encoder()` (one-hot categories). A new `samplers.py` maps a sampler name to an imbalanced-learn resampler. `pipeline.py` switches to imblearn's `Pipeline` with named steps `scale -> [sample] -> encode -> model`; resamplers run only during fit, so validation folds keep their real class mix. Class weights need no new code: they are `params` entries in the config.

**Tech Stack:** Python 3.12, conda, pandas, scikit-learn 1.9, imbalanced-learn 0.14, LightGBM, pytest.

**Spec:** `docs/superpowers/specs/2026-09-25-credit-default-risk-design.md` (M3 row of the Milestones table)

## Global Constraints

- Every command runs from the repo root with `conda activate credit-risk` active (or prefixed with `conda run -n credit-risk`).
- `imbalanced-learn>=0.12` is added to `environment.yml` (already edited and installed on 2026-10-09, uncommitted; it is committed in Task 2).
- Ownership: the spec lists M3 as the author's. On 2026-10-09 the author asked Claude to write all of M3 (plan, tests, code) and to review each task and the PR themselves.
- **Annotations:** the author asked that new and changed code carry explanatory comments in the style already used in `src/` (module docstrings that say what and why, short comments on non-obvious lines). Keep existing comments; update any that the change makes wrong.
- All comparisons use the same stratified CV folds on train only (`StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)`). The test split is not read in M3.
- Imbalance techniques are applied to `logreg` and `lgbm` only (author's decision, 2026-10-09). The M2 rows stay as they are.
- SMOTENC, not plain SMOTE (author's decision): categorical columns and PAY_1 to PAY_6 are passed to SMOTENC as categorical, so synthetic rows get real codes.
- One pipeline shape for every experiment (author's decision, "option A"): rows without a sampler have no `sample` step and must reproduce the committed M2 metrics.
- `reports/tables/experiments.csv` is committed; `reports/smoke/` is gitignored. No timing columns in the table.
- Docs and comments: no em dashes, plain factual tone, avoid absolute words ("always", "never", "exact") unless literally true.
- Work on branch `m3-imbalance`; do not push or merge without asking the user.

## Review Focus

- A sampler typo in the config (e.g. `sampler: smote`): should fail before any CV runs, with a message listing valid names. Pinned in Task 4 (`test_unknown_sampler_fails_when_building`).
- SMOTENC creating codes that cannot occur (fractional one-hot values, `PAY_1 = 1.4`, an EDUCATION level outside 1 to 4): every synthetic row should have codes seen in the input. Pinned in Task 4 (`test_smotenc_rows_have_real_codes`).
- Resampling leaking into prediction or scoring: `predict_proba` should return one row per input row, and only the training data should be balanced. Pinned in Task 4 (`test_resampling_balances_training_rows_only`).
- The preprocessing refactor silently changing M2 results: columns, order and values should match the M2 preprocessor. Pinned in Task 1 (`test_preprocessor_column_order_matches_m2`) and checked against the committed table in Task 5.
- Small samples (`make smoke`, 1,000 rows and 2 folds; the CLI test, 300 rows): SMOTENC needs at least 6 defaulters per training fold and should run without error on these sizes. Pinned in Task 5 (the CLI test now runs the full experiment list).

---

### Task 1: Split the preprocessor into scale and encode steps

**Files:**
- Modify: `src/credit_risk/preprocess.py`
- Test: `tests/test_preprocess.py`

**Interfaces:**
- Consumes: `NUMERIC`, `CATEGORICAL`, `ORDINAL` (already in `preprocess.py`).
- Produces: `build_scaler() -> ColumnTransformer` (output columns `[*NUMERIC, *CATEGORICAL, *ORDINAL]`, pandas output, codes unchanged); `build_encoder() -> ColumnTransformer` (input: the scaler's output; output: the M2 columns in M2 order); `build_preprocessor() -> sklearn.pipeline.Pipeline` (the two chained).

- [ ] **Step 1: Create the branch**

```bash
git checkout -b m3-imbalance
```

The uncommitted `environment.yml` change comes along; it is committed in Task 2.

- [ ] **Step 2: Write the failing tests**

Add to `tests/test_preprocess.py` (update the import line to `from credit_risk.preprocess import CATEGORICAL, NUMERIC, ORDINAL, build_preprocessor, build_scaler`):

```python
def test_scaler_keeps_codes_as_single_columns(features):
    # The middle state SMOTENC needs: amounts scaled, codes still one column each.
    out = build_scaler().fit_transform(features)
    assert list(out.columns) == [*NUMERIC, *CATEGORICAL, *ORDINAL]
    codes = CATEGORICAL + ORDINAL
    assert out[codes].values.tolist() == features[codes].values.tolist()
    assert abs(out["LIMIT_BAL"].mean()) < 1e-9


def test_preprocessor_column_order_matches_m2(features):
    # Pins the M2 layout so the split into two steps cannot reorder columns
    # (tree models can shift slightly when column order changes).
    out = build_preprocessor().fit_transform(features)
    one_hot = ["SEX_2", "EDUCATION_1", "EDUCATION_2", "EDUCATION_3", "EDUCATION_4"]
    one_hot += ["MARRIAGE_1", "MARRIAGE_2", "MARRIAGE_3"]
    assert list(out.columns) == [*NUMERIC, *one_hot, *ORDINAL]
```

The four existing tests stay as they are; they now run through the two-step version.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_preprocess.py -v`
Expected: collection error, `ImportError: cannot import name 'build_scaler'`.

- [ ] **Step 4: Implement**

Replace `src/credit_risk/preprocess.py` with:

```python
"""Column transformations applied before any model.

- NUMERIC (credit limit, age, bill and payment amounts): standardized, so linear
  models see them on one scale. Tree models are unaffected by scaling.
- CATEGORICAL (sex, education, marriage): one-hot encoded. The codes are labels,
  not quantities, so "EDUCATION = 3" should not mean "three times EDUCATION = 1".
  A level not seen during fitting is encoded as all zeros instead of raising.
- ORDINAL (PAY_1 to PAY_6): kept as numbers, since -2 to 8 runs roughly from least
  to most risky.
- ID and any other column are dropped.

Since M3 the work is done in two steps, so a resampler can sit between them:

    build_scaler():  scale NUMERIC; keep CATEGORICAL and ORDINAL as raw code columns
    build_encoder(): one-hot CATEGORICAL; pass NUMERIC and ORDINAL through

SMOTENC (samplers.py) needs that middle state. The amounts must already be scaled,
because SMOTENC finds neighbours by distance and an unscaled LIMIT_BAL (up to
1,000,000) would outweigh AGE (20 to 79). The categories must still be one column
each, so SMOTENC can give a synthetic row a real code instead of a fractional
one-hot value such as EDUCATION_2 = 0.4.

build_preprocessor() chains the two steps with nothing between them. Its output
(columns, order and values) matches the single-step M2 version.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.data import BILL_AMTS, PAY_AMTS, PAY_STATUS

NUMERIC = ["LIMIT_BAL", "AGE", *BILL_AMTS, *PAY_AMTS]
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
ORDINAL = PAY_STATUS


def build_scaler() -> ColumnTransformer:
    """Step 1, a fresh unfitted ColumnTransformer: scale NUMERIC, pass the codes through."""
    scaler = make_column_transformer(
        (StandardScaler(), NUMERIC),
        # Left as raw codes here; build_encoder one-hot encodes CATEGORICAL later,
        # after any resampling.
        ("passthrough", CATEGORICAL + ORDINAL),
        remainder="drop",  # drops ID
        verbose_feature_names_out=False,
    )
    # pandas output keeps the column names, which SMOTENC uses to find the
    # categorical columns.
    return scaler.set_output(transform="pandas")


def build_encoder() -> ColumnTransformer:
    """Step 2, a fresh unfitted ColumnTransformer: one-hot CATEGORICAL, pass the rest through.

    Expects the output of build_scaler (or of a resampler placed after it).
    """
    encoder = make_column_transformer(
        # All three groups are listed (instead of remainder="passthrough") to keep
        # the M2 column order: amounts, one-hot columns, then PAY codes.
        ("passthrough", NUMERIC),
        (
            OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
            CATEGORICAL,
        ),
        ("passthrough", ORDINAL),
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return encoder.set_output(transform="pandas")


def build_preprocessor() -> Pipeline:
    """Both steps chained with no resampling: raw columns in, model-ready DataFrame out.

    pipeline.py uses the two steps separately; this is for notebooks and tests that
    want the whole transformation as one object.
    """
    return make_pipeline(build_scaler(), build_encoder())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_preprocess.py tests/test_pipeline.py -v`
Expected: all PASS except `test_pipeline_steps`. `pipeline.py` still calls `build_preprocessor()` until Task 4, which works because a Pipeline can be a step, but the first step's automatic name is now `"pipeline"` instead of `"columntransformer"`. Update that one assertion in `tests/test_pipeline.py` to `["pipeline", "logisticregression"]` and rerun; Task 4 replaces the test.

- [ ] **Step 6: Full suite, lint, commit**

```bash
make test lint
git add src/credit_risk/preprocess.py tests/test_preprocess.py tests/test_pipeline.py
git commit -m "Split preprocessor into scale and encode steps"
```

---

### Task 2: Sampler registry

**Files:**
- Create: `src/credit_risk/samplers.py`
- Create: `tests/test_samplers.py`
- Modify: `environment.yml` (already edited; commit only)

**Interfaces:**
- Consumes: `CATEGORICAL`, `ORDINAL` from `credit_risk.preprocess`.
- Produces: `SAMPLER_NAMES: tuple[str, ...] == ("undersample", "smotenc")`; `get_sampler(name: str, seed: int = 0)` returning a fresh unfitted `RandomUnderSampler` or `SMOTENC`, raising `ValueError` listing valid names for an unknown name.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_samplers.py`:

```python
import pytest
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler

from credit_risk.preprocess import CATEGORICAL, ORDINAL
from credit_risk.samplers import SAMPLER_NAMES, get_sampler


@pytest.mark.parametrize(
    ("name", "cls"), [("undersample", RandomUnderSampler), ("smotenc", SMOTENC)]
)
def test_each_name_builds_its_sampler_with_the_seed(name, cls):
    sampler = get_sampler(name, seed=7)
    assert isinstance(sampler, cls)
    assert sampler.random_state == 7


def test_names_cover_the_registry():
    assert SAMPLER_NAMES == ("undersample", "smotenc")


def test_smotenc_treats_categories_and_pay_codes_as_categorical():
    assert get_sampler("smotenc").categorical_features == CATEGORICAL + ORDINAL


def test_each_call_returns_a_new_object():
    # A shared object would carry fitted state from one experiment into the next.
    assert get_sampler("smotenc") is not get_sampler("smotenc")


def test_unknown_sampler_lists_valid_names():
    with pytest.raises(ValueError, match="undersample, smotenc"):
        get_sampler("smote")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_samplers.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'credit_risk.samplers'`.

- [ ] **Step 3: Implement**

Create `src/credit_risk/samplers.py`:

```python
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
    "smotenc": lambda seed: SMOTENC(
        categorical_features=CATEGORICAL + ORDINAL, random_state=seed
    ),
}

SAMPLER_NAMES = tuple(_SAMPLERS)


def get_sampler(name: str, seed: int = 0):
    """Return a new, unfitted resampler for `name`, with `random_state=seed`.

    Raise ValueError naming the valid options if `name` is not in SAMPLER_NAMES.
    """
    if name not in _SAMPLERS:
        raise ValueError(f"Unknown sampler {name!r}. Valid names: {', '.join(SAMPLER_NAMES)}")
    return _SAMPLERS[name](seed)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_samplers.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Full suite, lint, commit**

```bash
make test lint
git add environment.yml src/credit_risk/samplers.py tests/test_samplers.py
git commit -m "Add imbalanced-learn and a sampler registry"
```

---

### Task 3: Optional `sampler` in the experiment config

**Files:**
- Modify: `src/credit_risk/config.py` (`Experiment` dataclass and `_experiments`)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Experiment(name: str, model: str, params: dict = {}, sampler: str | None = None)`. YAML key `sampler` is optional; a non-string value raises `ConfigError` naming `sampler`. The name itself is checked later by `get_sampler` (same split as model names).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_config.py`:

```python
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
```

And add one case to the `test_invalid_experiments_raise` parameter list:

```python
        ("experiments:\n  - {name: a, model: m, sampler: 3}", "sampler"),
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_config.py -v -k "sampler or invalid_experiments"`
Expected: `test_experiment_sampler_parsed_and_defaults_to_none` FAILS with `ConfigError: Unknown key(s) in experiments[1]: ['sampler']`. The new invalid case already passes at this point, because "sampler" appears in the unknown-key message; it must still pass after Step 3, when the type check produces the error instead.

- [ ] **Step 3: Implement**

In `src/credit_risk/config.py`, update the dataclass:

```python
@dataclass(frozen=True)
class Experiment:
    """One row of the experiments table: a name, a model name, model settings, and an
    optional resampler name (a name from credit_risk.samplers, or None for no resampling).
    """

    name: str
    model: str
    params: dict[str, Any] = field(default_factory=dict)
    sampler: str | None = None
```

In `_experiments`, replace the loop body from `_check_keys(...)` to `experiments.append(...)` with:

```python
        _check_keys(where, item, {"name", "model"}, {"params", "sampler"})
        params = item.get("params") or {}
        if not isinstance(params, dict):
            raise ConfigError(f"'{where}.params' must be a mapping, got {params!r}")
        # Only the type is checked here. Whether the name exists is checked when the
        # pipeline is built (samplers.get_sampler), the same way model names are.
        sampler = item.get("sampler")
        if sampler is not None and not isinstance(sampler, str):
            raise ConfigError(f"'{where}.sampler' must be a sampler name, got {sampler!r}")
        experiments.append(Experiment(str(item["name"]), str(item["model"]), dict(params), sampler))
```

Also update the first error message in `_experiments` to `"'experiments' must be a non-empty list of {name, model, params, sampler}"`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: all PASS.

- [ ] **Step 5: Full suite, lint, commit**

```bash
make test lint
git add src/credit_risk/config.py tests/test_config.py
git commit -m "Allow an optional sampler per experiment in the config"
```

---

### Task 4: imblearn pipeline with an optional resample step

**Files:**
- Modify: `src/credit_risk/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `build_scaler`, `build_encoder` (Task 1); `get_sampler`, `SAMPLER_NAMES` (Task 2); `Experiment.sampler` (Task 3); `get_model` (M2).
- Produces: `build_pipeline(experiment: Experiment, seed: int) -> imblearn.pipeline.Pipeline` with steps named `["scale", "encode", "model"]`, or `["scale", "sample", "encode", "model"]` when `experiment.sampler` is set. Raises `ValueError` for an unknown model or sampler name.

- [ ] **Step 1: Write the failing tests**

In `tests/test_pipeline.py`, update the imports:

```python
from credit_risk.preprocess import CATEGORICAL, ORDINAL
from credit_risk.samplers import SAMPLER_NAMES
```

Replace `test_pipeline_steps` with:

```python
def test_pipeline_steps():
    pipe = build_pipeline(Experiment(name="lr", model="logreg"), seed=0)
    assert list(pipe.named_steps) == ["scale", "encode", "model"]


def test_pipeline_steps_with_sampler():
    pipe = build_pipeline(Experiment(name="lr", model="logreg", sampler="smotenc"), seed=0)
    assert list(pipe.named_steps) == ["scale", "sample", "encode", "model"]
```

Add:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL. `test_pipeline_steps` sees `["pipeline", "logisticregression"]`; the sampler tests fail because `build_pipeline` ignores `sampler` (dummy prior is about 0.22, not 0.5; `pipe[:2]` has no `fit_resample`; no ValueError for "smote").

- [ ] **Step 3: Implement**

Replace `src/credit_risk/pipeline.py` with:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_pipeline.py -v`
Expected: all PASS.

- [ ] **Step 5: Full suite, lint, commit**

```bash
make test lint
git add src/credit_risk/pipeline.py tests/test_pipeline.py
git commit -m "Build pipelines with imblearn and an optional resample step"
```

---

### Task 5: M3 experiments in the config and a `sampler` column in the table

**Files:**
- Modify: `configs/base.yaml` (experiments list)
- Modify: `src/credit_risk/experiments.py` (row building, module docstring)
- Test: `tests/test_experiments.py`, `tests/test_config.py`, `tests/test_cli.py`
- Regenerate: `reports/tables/experiments.csv`

**Interfaces:**
- Consumes: `Experiment.sampler` (Task 3), `build_pipeline` (Task 4).
- Produces: results table columns `experiment, model, sampler, roc_auc_mean, ...`, where `sampler` is the sampler name or `"none"`. Eleven rows: the five M2 rows, then `logreg_weighted, logreg_undersample, logreg_smotenc, lgbm_weighted, lgbm_undersample, lgbm_smotenc`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_experiments.py`, change the first assertion of `test_one_row_per_experiment_with_all_metrics` so it follows the config instead of a hardcoded list, and check the new column:

```python
    assert table["experiment"].tolist() == [e.name for e in BASE.experiments]
    assert "sampler" in table.columns
```

Add:

```python
def test_sampler_column_names_the_resampler(train, tmp_path):
    experiments = (
        Experiment("lr", "logreg"),
        Experiment("lr_under", "logreg", sampler="undersample"),
    )
    table = run_experiments(small_cfg(tmp_path, experiments=experiments), train)
    assert table["sampler"].tolist() == ["none", "undersample"]
```

In `tests/test_config.py`, in `test_repo_configs_load` (the test containing `assert [e.name for e in base.experiments] == [...]`), replace that list with:

```python
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
    ]
```

In `tests/test_cli.py`, `test_experiments_stage_writes_table`, replace the hardcoded list so the 300-row run covers every experiment, SMOTENC included:

```python
    assert table["experiment"].tolist() == [e.name for e in load_config(config).experiments]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_experiments.py tests/test_config.py -v`
Expected: FAIL. `"sampler" in table.columns` is false; the config name list has only five entries.

- [ ] **Step 3: Implement**

In `configs/base.yaml`, replace the experiments comment and list with:

```yaml
# Setups compared by `make experiments`, all on the same CV folds.
# model: a name from credit_risk.models; params: passed to that model;
# sampler (optional): a name from credit_risk.samplers, applied to training folds only.
experiments:
  # M2: each model family at library defaults.
  - {name: dummy, model: dummy}
  - {name: logreg, model: logreg}
  - {name: forest, model: forest}
  - {name: lgbm, model: lgbm}
  - {name: xgboost, model: xgboost}
  # M3: three ways to handle the 78% / 22% class split, each on logreg and lgbm.
  # class_weight: balanced counts an error on a defaulter about 3.5 times (78/22)
  # as much in the training loss. (XGBoost ignores class_weight; it would need
  # scale_pos_weight, which is why xgboost is not in this group.)
  - {name: logreg_weighted, model: logreg, params: {class_weight: balanced}}
  - {name: logreg_undersample, model: logreg, sampler: undersample}
  - {name: logreg_smotenc, model: logreg, sampler: smotenc}
  - {name: lgbm_weighted, model: lgbm, params: {class_weight: balanced}}
  - {name: lgbm_undersample, model: lgbm, sampler: undersample}
  - {name: lgbm_smotenc, model: lgbm, sampler: smotenc}
```

In `src/credit_risk/experiments.py`, change the row start in `run_experiments`:

```python
        # "none" rather than an empty cell, so the CSV reads clearly.
        row = {
            "experiment": name,
            "model": experiment.model,
            "sampler": experiment.sampler or "none",
        }
```

And in the module docstring, change step 1 of the list to:

```
1. Build a pipeline: preprocessing, an optional resampler, and a model (pipeline.py).
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_experiments.py tests/test_config.py tests/test_cli.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the real stages and check M2 rows did not move**

```bash
make experiments
make smoke
git diff reports/tables/experiments.csv
```

Expected: the five M2 rows have the same metric values as before (the diff shows only the added `sampler` column on them, value `none`), plus six new rows. A prototype on 2026-10-09 gave `lgbm` 0.7763 and `lgbm_smotenc` about 0.763 ROC-AUC. If any M2 metric changed, stop and investigate before committing (see Review Focus).

- [ ] **Step 6: Full suite, lint, commit**

```bash
make test lint
git add configs/base.yaml src/credit_risk/experiments.py tests/test_experiments.py tests/test_config.py tests/test_cli.py reports/tables/experiments.csv
git commit -m "Add M3 imbalance experiments; record sampler in results table"
```

---

### Task 6: Docs and the ROC-AUC explanation

**Files:**
- Modify: `README.md` ("How the code fits together", "Where to find things", "What the tests check")
- Modify: `Makefile` (code map in `help`)
- Modify: `docs/learning-log.md` (new M3 entry)

**Interfaces:**
- Consumes: the committed `reports/tables/experiments.csv` from Task 5.

- [ ] **Step 1: README code map**

Replace the module table in "How the code fits together" with:

```markdown
| Order | Module | What it does | Run with |
|---|---|---|---|
| | `config.py` | Reads settings from `configs/*.yaml`; used by every step | |
| 1 | `data.py` | Load, clean, and split into train/test | `make data`, `make split` |
| 2 | `preprocess.py` | Scale amounts (step 1), one-hot encode categories (step 2) | (inside the pipeline) |
| 3 | `samplers.py` | Turn a sampler name into a resampler that evens out the classes | (inside the pipeline) |
| 4 | `models.py` | Turn a model name from the config into an unfitted model | (inside the pipeline) |
| 5 | `pipeline.py` | Chain scaling, optional resampling, encoding, and a model into one object | (inside experiments) |
| 6 | `experiments.py` | Cross-validate each setup and write a results table | `make experiments` |
| | `cli.py` | The `credit-risk` command; runs one stage at a time | |
```

In "Where to find things", add after the models row:

```markdown
| How is class imbalance handled? | `src/credit_risk/samplers.py` and the M3 rows in `configs/base.yaml` |
```

In "What the tests check", add after the `test_preprocess.py` row, and replace the `test_pipeline.py` row:

```markdown
| `test_samplers.py` | Each sampler name builds the right resampler with the seed; a misspelled name fails |
| `test_pipeline.py` | Every model fits and returns probabilities; resampling balances only the training rows; SMOTENC rows have real codes |
```

Change "about 110 tests" to the count `make test` now reports, rounded down to the nearest ten.

- [ ] **Step 2: Makefile code map**

In the `help` target, replace the numbered module lines with:

```make
	@printf "%b\n" "  $(GREEN)1 data.py$(RESET)         - Load, clean, split"
	@printf "%b\n" "  $(GREEN)2 preprocess.py$(RESET)   - Scale amounts, one-hot categories"
	@printf "%b\n" "  $(GREEN)3 samplers.py$(RESET)     - Sampler name -> resampler (evens out classes)"
	@printf "%b\n" "  $(GREEN)4 models.py$(RESET)       - Model name -> unfitted model"
	@printf "%b\n" "  $(GREEN)5 pipeline.py$(RESET)     - Scale, [resample], encode, model in one object"
	@printf "%b\n" "  $(GREEN)6 experiments.py$(RESET)  - Cross-validate each setup, results table"
```

Run `make help` and check the columns line up.

- [ ] **Step 3: Learning log M3 entry (Claude drafts, author edits)**

Add after the M2 entry in `docs/learning-log.md`, following the template's heading style, `### M3: Class imbalance (draft by Claude for the author to edit, 2026-10-09)`, with these subsections. Fill every number from the committed table; do not round differently from the README.

1. **The problem:** 22% defaulters; at the 0.5 cutoff the M2 LightGBM catches 36% of them. A decision problem more than a ranking problem.
2. **The three techniques:** one short paragraph each (class weights, random undersampling, SMOTENC), including the XGBoost `scale_pos_weight` note.
3. **Why resampling sits inside the CV folds:** the imblearn Pipeline runs samplers only during fit; resampling before CV leaks validation rows into training and changes the validation class mix.
4. **Why SMOTENC and the two-step preprocessor:** fractional one-hot values and PAY codes with plain SMOTE; scale before, encode after.
5. **Results:** a table of the eight logreg and lgbm rows (experiment, ROC-AUC, PR-AUC, recall, precision) copied from `reports/tables/experiments.csv`.
6. **Why ROC-AUC changes little:** ROC-AUC measures ranking (the chance a random defaulter scores above a random non-defaulter); TPR and FPR are each computed within one class, so the class ratio does not enter; class weights and resampling mostly move scores up together, leaving the order similar. What moves is recall and precision at the fixed 0.5 cutoff, roughly what lowering the cutoff on the original model would do. That is the job of M5's cost-based threshold. Note that all three techniques push predicted probabilities up, which M5's calibration has to undo. State whether the observed numbers match this; if a technique hurt PR-AUC or ROC-AUC (for example undersampling, which trains on fewer rows), say so plainly.

- [ ] **Step 4: Lint and commit**

```bash
make lint
git add README.md Makefile docs/learning-log.md
git commit -m "Document M3: samplers, two-step preprocessing, imbalance results"
```

- [ ] **Step 5: Hand off to the author**

Ask the author to review the branch and edit the learning log draft in their own words. Ask before pushing `m3-imbalance` or opening a PR.
