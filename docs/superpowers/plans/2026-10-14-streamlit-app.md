# Streamlit App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Streamlit app (`app/app.py`, run with `make app`) where a visitor picks a real test client, tries coherent changes (payment behavior, credit limit, demographics), sees the probability of default, the decision at a cost-based cutoff and what moved the score, and sees how the cost ratio changes the counts on all 6,000 test clients.

**Architecture:** New module `src/credit_risk/showcase.py` holds all logic with no Streamlit imports: loading the model and rebuilding the test split in memory, example clients, payment scenarios, edits, scoring, SHAP contributions, plain-word tables and test-set outcomes. `app/app.py` only lays out a sidebar slider and three tabs and calls `showcase.py`. Two small changes to existing modules: `plot_tradeoff_curves` thins its curves and takes a caption for its points; `explain.py` shares its median-client rule. `app/requirements.txt` holds pinned packages for Streamlit Community Cloud.

**Tech Stack:** Python 3.12, conda, pandas, numpy, scikit-learn 1.9.1, LightGBM 4.7.0, shap, Altair 6, pytest, **new:** `streamlit` 1.65.0 (with `streamlit.testing.v1.AppTest` for app tests).

**Spec:** `docs/superpowers/specs/2026-10-14-streamlit-app-design.md` (approved 2026-10-14). Also touches the project spec `docs/superpowers/specs/2026-09-25-credit-default-risk-design.md` (test set rule).

## Global Constraints

- Every command runs from the repo root with the `credit-risk` env active (`eval "$(conda shell.bash hook)" && conda activate credit-risk`).
- Work on branch `streamlit-app` (already holds the committed model `96c7f5d` and the spec `c797c10`). Do not push or merge without asking.
- **Annotations:** new and changed code carries explanatory comments in the style already in `src/` (this is a learning project). Module docstrings say what the module is for and how it fits in.
- No em dashes, plain factual tone, no unwarranted absolute words, in code comments, UI text and docs.
- Line length 99 (ruff `E501`); the formatter does not rewrap comments. Each commit step runs `make format` and `make lint` first.
- Commit messages: plain subject line, no `Co-Authored-By` trailer (the author's settings turn attribution off).
- `showcase.py` has no `streamlit` import. `app/app.py` has no modeling logic.
- Cutoff for cost ratio N (false alarm = 1): `1 / (1 + N)`. Slider 1 to 20, step 1, default 5.
- Flagging is `proba >= cutoff` everywhere (matches the saved model and `decision_metrics`).
- Example clients are chosen once, at the saved model's cutoff (`model.best_threshold_`, 0.1697), with the median rule; keys and labels:
  `caught` "A defaulter the model caught", `missed` "A defaulter the model missed", `false_alarm` "A false alarm (paid, but flagged)", `cleared` "A low-risk client, correctly cleared".
- Scenarios, names verbatim: `"As recorded"`, `"Pays in full every month"`, `"Pays the minimum every month"`, `"Fell 2 months behind"`. Rules in Task 3. Payments never below 0; bills unchanged.
- Slider ranges from the training split: credit limit 10,000 to 500,000 (min to 99th percentile), age 21 to 79 (min to max); widened to include the current client's own value. Credit limit step 1,000.
- Only test clients are shown. Training clients are used for slider ranges only.
- Months: column 1 is September 2005, column 6 is April 2005.
- Status codes: -2 "No card use", -1 "Paid in full", 0 "Paid the minimum", 1 "1 month late", k "k months late".
- Out of scope: deploying, README GIF and live link (piece B); the website page (piece C); any change to the model or its cutoff; fairness metrics by group.

## Review Focus

- A client whose credit limit is above 500,000 (47 test clients): the limit slider must include their real value instead of raising a Streamlit range error. Pinned in Task 2 (`test_widen_includes_the_clients_own_value`) and Task 6 (`test_client_above_the_limit_range_renders`).
- Switching clients after changing controls: the controls must go back to the new client's recorded values, so "Your changes moved this client's risk" is not shown for changes the visitor did not make to this client. Pinned in Task 6 (`test_picking_another_client_resets_the_controls`).
- A negative bill (credit balance) under a scenario: payments computed from it must be 0, not negative. Pinned in Task 3 (`test_pays_in_full`, `test_pays_the_minimum`, `test_fell_two_months_behind`, all using a row with a negative bill).
- A probability exactly at the cutoff: must be flagged, as the saved model would. Pinned in Task 2 (`test_is_flagged_at_the_cutoff`).
- A precision-recall curve with more than 5,000 points (one per distinct score): Altair refuses to draw it, so the cost tab would crash. Pinned in Task 1 (`test_tradeoff_curves_draw_with_many_distinct_scores`).

---

### Task 1: Dependencies, curve thinning, and the test set rule

**Files:**
- Modify: `environment.yml` (pin scikit-learn and lightgbm; add streamlit)
- Create: `app/requirements.txt`
- Create: `tests/test_requirements.py`
- Modify: `src/credit_risk/evaluate.py` (`plot_tradeoff_curves`: thin curves, `points_note` parameter)
- Modify: `tests/test_evaluate.py`
- Modify: `pyproject.toml` (ruff per-file ignore for `app/app.py`)
- Modify: `docs/superpowers/specs/2026-09-25-credit-default-risk-design.md` (test set rule)

**Interfaces:**
- Consumes: `plot_tradeoff_curves(y, proba, marks)` (exists).
- Produces: `plot_tradeoff_curves(y, proba, marks, points_note="cutoffs tuned for each cost ratio, and 0.5") -> alt.HConcatChart`; `MAX_CURVE_POINTS = 1000` in `evaluate.py`; `app/requirements.txt` with exact `==` pins.

- [ ] **Step 1: Install streamlit into the env and pin the model libraries**

In `environment.yml`, replace the `scikit-learn` and `lightgbm` lines and add streamlit after `shap`:

```yaml
  # Pinned exactly: artifacts/model.joblib was saved with these versions, and a saved
  # model can fail or change behavior when loaded with others. Retrain to upgrade.
  - scikit-learn=1.9.1
  - imbalanced-learn>=0.12   # resamplers (SMOTE, SMOTENC, undersampling) and a Pipeline that runs them only during fit
  - pyarrow>=15
  - lightgbm=4.7.0
```

```yaml
  - streamlit=1.65.0       # the app in app/ (make app); same version as app/requirements.txt
```

Run: `conda install -n credit-risk -c conda-forge streamlit=1.65.0 scikit-learn=1.9.1 lightgbm=4.7.0 -y`
Then: `python -c "import streamlit, sklearn, lightgbm; print(streamlit.__version__, sklearn.__version__, lightgbm.__version__)"`
Expected: `1.65.0 1.9.1 4.7.0`

- [ ] **Step 2: Write `app/requirements.txt`**

```text
# Packages for Streamlit Community Cloud. It reads this file, not environment.yml at the
# repo root, because a dependency file next to the entrypoint (app/app.py) takes
# precedence. The app imports this repo's code from src/ directly (see app.py), so the
# package itself is not installed here.
#
# Versions match the credit-risk conda env. scikit-learn and lightgbm must match the
# versions artifacts/model.joblib was saved with; tests/test_requirements.py checks it.
streamlit==1.65.0
pandas==3.0.6
numpy==2.5.3
scikit-learn==1.9.1
lightgbm==4.7.0
imbalanced-learn==0.14.2
shap==0.52.0
matplotlib==3.11.2
altair==6.3.0
pyyaml==6.0.3
# models.py imports XGBoost (a candidate model). xgboost-cpu is the same library without
# the large GPU dependencies; 3.4.1 is its latest release. The final model is LightGBM,
# so this version does not affect predictions.
xgboost-cpu==3.4.1
```

- [ ] **Step 3: Write the failing tests**

Create `tests/test_requirements.py`:

```python
"""The app's pinned packages must match the env the committed model was saved with."""

import re
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "app" / "requirements.txt"


def pins() -> dict[str, str]:
    """Package name -> pinned version, from app/requirements.txt (comments skipped)."""
    lines = REQUIREMENTS.read_text().splitlines()
    entries = [line.split("#")[0].strip() for line in lines]
    return dict(entry.split("==") for entry in entries if entry)


def test_every_app_requirement_is_pinned_exactly():
    lines = [line.split("#")[0].strip() for line in REQUIREMENTS.read_text().splitlines()]
    assert all("==" in line for line in lines if line)


def test_model_libraries_match_the_installed_versions():
    # A saved model must be loaded with the versions it was saved with.
    for name in ("scikit-learn", "lightgbm"):
        assert pins()[name] == version(name), f"update app/requirements.txt for {name}"


def test_environment_pins_the_same_model_libraries():
    env = (ROOT / "environment.yml").read_text()
    for name in ("scikit-learn", "lightgbm"):
        assert re.search(rf"^\s*- {name}={re.escape(pins()[name])}\b", env, re.MULTILINE)
```

Add to `tests/test_evaluate.py` (it already imports `np`, `pd`, `plot_tradeoff_curves`):

```python
def test_tradeoff_curves_draw_with_many_distinct_scores():
    # 6,000 distinct scores give a precision-recall curve of 6,001 points; Altair refuses
    # more than 5,000 rows per chart, so the curves are thinned before drawing.
    rng = np.random.default_rng(0)
    proba = rng.uniform(0, 1, 6000)
    y = (rng.uniform(0, 1, 6000) < proba).astype(int)
    marks = pd.DataFrame({"label": ["a"], "threshold": [0.3]})
    chart = plot_tradeoff_curves(y, proba, marks)
    chart.to_dict()  # raises MaxRowsError if a curve is too long
    for panel in chart.hconcat:
        assert len(panel.layer[0].data) <= 1000


def test_tradeoff_curves_subtitle_names_the_points():
    rng = np.random.default_rng(0)
    proba = rng.uniform(0, 1, 200)
    y = (rng.uniform(0, 1, 200) < proba).astype(int)
    marks = pd.DataFrame({"label": ["a"], "threshold": [0.3]})
    default = plot_tradeoff_curves(y, proba, marks).to_dict()["title"]["subtitle"]
    assert "cutoffs tuned for each cost ratio, and 0.5" in default
    custom = plot_tradeoff_curves(y, proba, marks, points_note="the current cutoff")
    assert "Points: the current cutoff" in custom.to_dict()["title"]["subtitle"]
```

- [ ] **Step 4: Run the tests to verify the evaluate tests fail**

Run: `python -m pytest tests/test_requirements.py tests/test_evaluate.py -k "requirement or pinned or model_libraries or tradeoff_curves_draw or subtitle" -v`
Expected: the three requirements tests PASS (Steps 1 and 2 are done); `test_tradeoff_curves_draw_with_many_distinct_scores` FAILS with `MaxRowsError`; `test_tradeoff_curves_subtitle_names_the_points` FAILS with `TypeError` (unexpected keyword `points_note`).

- [ ] **Step 5: Thin the curves and add `points_note`**

In `src/credit_risk/evaluate.py`, add after `COLUMNS`:

```python
# Altair refuses charts with more than 5,000 rows, and a precision-recall curve has one
# point per distinct score (up to one per client). 1,000 evenly spaced points look the
# same at the size the curves are drawn.
MAX_CURVE_POINTS = 1000


def _thin(curve: pd.DataFrame, max_points: int = MAX_CURVE_POINTS) -> pd.DataFrame:
    """At most `max_points` evenly spaced rows of `curve`, keeping the first and last."""
    if len(curve) <= max_points:
        return curve
    keep = np.unique(np.linspace(0, len(curve) - 1, max_points).round().astype(int))
    return curve.iloc[keep]
```

Change the signature and docstring end of `plot_tradeoff_curves`:

```python
def plot_tradeoff_curves(
    y: Sequence[int],
    proba: Sequence[float],
    marks: pd.DataFrame,
    points_note: str = "cutoffs tuned for each cost ratio, and 0.5",
) -> alt.HConcatChart:
```

and add to its docstring: `` `points_note` says what the markers are, in the subtitle (the app marks one cutoff). Curves are thinned to MAX_CURVE_POINTS points each. ``

Replace the two curve frames:

```python
    roc = _thin(pd.DataFrame({"fpr": fpr, "tpr": tpr}))
    pr = _thin(pd.DataFrame({"recall": recall, "precision": precision}))
```

and the subtitle:

```python
            subtitle=(
                "Dashed lines: random guessing (ROC) and the default rate, "
                f"{default_rate:.0%} (PR). Points: {points_note}"
            ),
```

- [ ] **Step 6: Allow imports after `sys.path` in the app**

In `pyproject.toml`, under `[tool.ruff.lint.per-file-ignores]`, add:

```toml
# app.py puts src/ on sys.path before importing the package (see its comment), so its
# imports come after code.
"app/app.py" = ["E402"]
```

- [ ] **Step 7: Update the project spec's test set rule**

In `docs/superpowers/specs/2026-09-25-credit-default-risk-design.md`, Success criteria, replace:

```markdown
- The test set is read only by `evaluate` and `explain` (changed in
  M6). Neither changes the model, and no setting is changed after
  seeing their output.
```

with:

```markdown
- The test set is read only by `evaluate`, `explain` and the
  Streamlit app (changed in M6, and on 2026-10-14 for the app). None
  of them changes the model, and no setting is changed after seeing
  their output.
```

- [ ] **Step 8: Run the full suite**

Run: `make format && make lint && make test`
Expected: lint clean; all tests pass (217 before this plan, plus 5 new).

- [ ] **Step 9: Commit**

```bash
git add environment.yml app/requirements.txt tests/test_requirements.py src/credit_risk/evaluate.py tests/test_evaluate.py pyproject.toml docs/superpowers/specs/2026-09-25-credit-default-risk-design.md
git commit -m "Prepare for the Streamlit app: pinned requirements, thinner curves, test set rule"
```

---

### Task 2: `showcase.py` basics (labels, cutoff, examples, client tables)

**Files:**
- Modify: `src/credit_risk/explain.py` (add `nearest_to_median`; `pick_clients` uses it)
- Create: `src/credit_risk/showcase.py`
- Create: `tests/test_showcase.py`

**Interfaces:**
- Consumes: `PAY_STATUS`, `BILL_AMTS`, `PAY_AMTS`, `TARGET` from `credit_risk.data`.
- Produces (in `showcase.py`):
  - `MONTHS: list[str]` (6 items, `"Sep 2005"` first)
  - `SEX_LABELS`, `EDUCATION_LABELS`, `MARRIAGE_LABELS: dict[int, str]`
  - `DEMOGRAPHICS = ("SEX", "EDUCATION", "MARRIAGE", "AGE")`
  - `PLAIN_LABELS: dict[str, str]` (28 model columns)
  - `EXAMPLE_LABELS: dict[str, str]` (keys `caught`, `missed`, `false_alarm`, `cleared`)
  - `status_label(code: int) -> str`
  - `cutoff_for(ratio: float) -> float`
  - `is_flagged(proba: float, cutoff: float) -> bool`
  - `widen(low: int, high: int, value: int) -> tuple[int, int]`
  - `example_clients(y, proba, threshold: float) -> dict[str, int]` (row positions)
  - `describe_client(row: pd.DataFrame) -> pd.DataFrame` (columns `Detail`, `Value`)
  - `payment_history(row: pd.DataFrame) -> pd.DataFrame` (columns `Month`, `Status`, `Bill (NT$)`, `Payment (NT$)`)
- Produces (in `explain.py`): `nearest_to_median(proba: np.ndarray, in_group: np.ndarray) -> int | None`.
- A "row" everywhere in `showcase.py` is a one-row `pd.DataFrame` with the cleaned columns minus `TARGET` (it keeps `ID`), as passed to `model.predict_proba`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_showcase.py`:

```python
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import Final, load_config
from credit_risk.data import TARGET, clean, load_raw
from credit_risk.decision import train_final
from credit_risk.explain import inner_pipeline, nearest_to_median
from credit_risk.showcase import (
    EXAMPLE_LABELS,
    MONTHS,
    PLAIN_LABELS,
    cutoff_for,
    describe_client,
    example_clients,
    is_flagged,
    payment_history,
    status_label,
    widen,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = load_config(ROOT / "configs" / "base.yaml")


def small_cfg(tmp_path):
    paths = replace(
        BASE.paths, reports_dir=tmp_path / "reports", artifacts_dir=tmp_path / "artifacts"
    )
    return replace(BASE, paths=paths, cv_folds=2, final=Final("lgbm", "sigmoid"))


@pytest.fixture(scope="module")
def clients():
    # Samples of the cleaned raw data stand in for train and test (the split is not read).
    rows = clean(load_raw(BASE)).sample(1200, random_state=0)
    return rows.iloc[:1000], rows.iloc[1000:].drop(columns=TARGET)


@pytest.fixture(scope="module")
def model(clients, tmp_path_factory):
    train, _ = clients
    return train_final(small_cfg(tmp_path_factory.mktemp("m")), train)


@pytest.fixture
def row(clients):
    """One client with a hand-set history: a negative bill in month 3, all 1 month late."""
    _, test = clients
    one = test.iloc[[0]].copy()
    bills = [1000.0, 2000.0, -500.0, 4000.0, 5000.0, 6000.0]
    one[[f"BILL_AMT{i}" for i in range(1, 7)]] = bills
    one[[f"PAY_AMT{i}" for i in range(1, 7)]] = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    one[[f"PAY_{i}" for i in range(1, 7)]] = 1
    one[["LIMIT_BAL", "AGE", "SEX", "EDUCATION", "MARRIAGE"]] = [50000.0, 30, 2, 2, 1]
    return one


def test_cutoff_for_five_to_one_is_one_sixth():
    assert cutoff_for(5) == pytest.approx(1 / 6)
    assert cutoff_for(1) == pytest.approx(0.5)


def test_is_flagged_at_the_cutoff():
    # >=, as in the saved model's predict and decision_metrics.
    assert is_flagged(0.2, 0.2)
    assert not is_flagged(0.19, 0.2)


def test_widen_includes_the_clients_own_value():
    assert widen(10, 20, 15) == (10, 20)
    assert widen(10, 20, 25) == (10, 25)
    assert widen(10, 20, 5) == (5, 20)


def test_nearest_to_median_picks_the_middle_client_and_skips_empty_groups():
    proba = np.array([0.9, 0.8, 0.6, 0.1])
    assert nearest_to_median(proba, np.array([True, True, True, False])) == 1
    assert nearest_to_median(proba, np.zeros(4, dtype=bool)) is None


def test_example_clients_are_each_in_their_group():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])
    proba = np.array([0.9, 0.8, 0.6, 0.2, 0.7, 0.55, 0.3, 0.1, 0.2])
    picks = example_clients(y, proba, 0.5)
    assert set(picks) == set(EXAMPLE_LABELS) == {"caught", "missed", "false_alarm", "cleared"}
    assert picks["caught"] == 1  # 0.8 is the median of 0.9, 0.8, 0.6
    assert y[picks["missed"]] == 1 and proba[picks["missed"]] < 0.5
    assert y[picks["false_alarm"]] == 0 and proba[picks["false_alarm"]] >= 0.5
    assert y[picks["cleared"]] == 0 and proba[picks["cleared"]] < 0.5


def test_example_clients_skip_an_empty_group():
    picks = example_clients(np.zeros(4, dtype=int), np.array([0.1, 0.2, 0.7, 0.9]), 0.5)
    assert set(picks) == {"false_alarm", "cleared"}


def test_status_labels():
    assert [status_label(c) for c in (-2, -1, 0, 1, 2)] == [
        "No card use",
        "Paid in full",
        "Paid the minimum",
        "1 month late",
        "2 months late",
    ]


def test_describe_client_in_plain_words(row):
    table = describe_client(row).set_index("Detail")["Value"]
    assert table["Credit limit"] == "50,000 NT$"
    assert table["Age"] == "30"
    assert table["Sex"] == "Female"
    assert table["Education"] == "University"
    assert table["Marital status"] == "Married"
    assert table["Most recent payment"] == "1 month late"


def test_payment_history_has_six_months_newest_first(row):
    history = payment_history(row)
    assert history["Month"].tolist() == MONTHS
    assert MONTHS[0] == "Sep 2005"
    assert history["Status"].iloc[0] == "1 month late"
    assert history["Bill (NT$)"].iloc[2] == "-500"
    assert history["Payment (NT$)"].iloc[0] == "11"


def test_plain_labels_cover_every_model_column(model, clients):
    _, test = clients
    columns = inner_pipeline(model)[:-1].transform(test.head(2)).columns
    assert set(columns) == set(PLAIN_LABELS)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_showcase.py -v`
Expected: collection error, `ImportError: cannot import name 'nearest_to_median'` (and no `credit_risk.showcase`).

- [ ] **Step 3: Share the median rule in `explain.py`**

Add above `pick_clients`:

```python
def nearest_to_median(proba: np.ndarray, in_group: np.ndarray) -> int | None:
    """Row number of the client in a group whose probability is nearest the group's median.

    `in_group` is a True/False array the length of `proba`. None for an empty group.
    Ties go to the first such row. Shared by pick_clients and the app's example clients
    (showcase.py), so both use one fixed rule.
    """
    rows = np.flatnonzero(in_group)
    if len(rows) == 0:
        return None
    distance = np.abs(proba[rows] - np.median(proba[rows]))
    return int(rows[np.argmin(distance)])
```

Replace the loop body of `pick_clients` with:

```python
    picks = {}
    for name, in_group in groups.items():
        row = nearest_to_median(proba, in_group)
        if row is not None:  # a group with no clients is left out
            picks[name] = row
    return picks
```

- [ ] **Step 4: Create `src/credit_risk/showcase.py`**

```python
"""Logic behind the Streamlit app (app/app.py), kept here so it can be tested.

The app shows the final model on the test clients: pick a client, try a few changes,
and see the probability of default, the decision at a cost-based cutoff, and which
columns moved the score. A second view shows how the cost ratio changes the counts on
all test clients. This module has no Streamlit code; app.py only lays things out.

Nothing here changes the model or chooses a setting. Like make evaluate, the app scores
the test clients once and then counts them at different cutoffs.

A "row" is a one-row DataFrame with the cleaned columns minus the target (ID included),
the shape model.predict_proba takes.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from credit_risk.data import BILL_AMTS, PAY_AMTS, PAY_STATUS
from credit_risk.explain import nearest_to_median

# The data's six months, newest first: column 1 (PAY_1, BILL_AMT1, PAY_AMT1) is
# September 2005 and column 6 is April 2005.
MONTHS = ["Sep 2005", "Aug 2005", "Jul 2005", "Jun 2005", "May 2005", "Apr 2005"]

# Category codes after cleaning (data.py merges undocumented codes into "other").
SEX_LABELS = {1: "Male", 2: "Female"}
EDUCATION_LABELS = {1: "Graduate school", 2: "University", 3: "High school", 4: "Other or unknown"}
MARRIAGE_LABELS = {1: "Married", 2: "Single", 3: "Other"}

# Columns the fairness check lets a visitor change. Nothing else in a row depends on them.
DEMOGRAPHICS = ("SEX", "EDUCATION", "MARRIAGE", "AGE")

# The model's 28 input columns (after scaling and one-hot encoding) in plain words, for
# the "what moved this score" chart. One-hot columns name the category they stand for;
# SEX_1 is not there because the encoder drops one category of a two-value column.
PLAIN_LABELS = {
    "LIMIT_BAL": "Credit limit",
    "AGE": "Age",
    **{f"BILL_AMT{i}": f"Bill, {month}" for i, month in enumerate(MONTHS, start=1)},
    **{f"PAY_AMT{i}": f"Payment, {month}" for i, month in enumerate(MONTHS, start=1)},
    **{f"PAY_{i}": f"Payment status, {month}" for i, month in enumerate(MONTHS, start=1)},
    "SEX_2": "Sex: female",
    **{f"EDUCATION_{k}": f"Education: {v.lower()}" for k, v in EDUCATION_LABELS.items()},
    **{f"MARRIAGE_{k}": f"Marital status: {v.lower()}" for k, v in MARRIAGE_LABELS.items()},
}

# The four example clients the app offers, by the group each comes from.
EXAMPLE_LABELS = {
    "caught": "A defaulter the model caught",
    "missed": "A defaulter the model missed",
    "false_alarm": "A false alarm (paid, but flagged)",
    "cleared": "A low-risk client, correctly cleared",
}


def status_label(code: int) -> str:
    """Plain words for a repayment status code (PAY_1 to PAY_6)."""
    named = {-2: "No card use", -1: "Paid in full", 0: "Paid the minimum"}
    if code in named:
        return named[code]
    return f"{code} month{'s' if code > 1 else ''} late"


def cutoff_for(ratio: float) -> float:
    """The cutoff when a missed defaulter costs `ratio` times a false alarm.

    For a client with probability p, flagging costs (1 - p) on average (a false alarm if
    they pay) and not flagging costs ratio * p (a miss if they default). Flagging is the
    cheaper choice when p is above 1 / (1 + ratio). Section 5 of docs/results.md shows the
    search on training data lands within about 0.01 of this.
    """
    return 1.0 / (1.0 + ratio)


def is_flagged(proba: float, cutoff: float) -> bool:
    """`>=`, as in the saved model's predict and evaluate.decision_metrics."""
    return proba >= cutoff


def widen(low: int, high: int, value: int) -> tuple[int, int]:
    """A slider range that also includes `value`, e.g. a client above the usual range."""
    return min(low, value), max(high, value)


def example_clients(
    y: Sequence[int], proba: Sequence[float], threshold: float
) -> dict[str, int]:
    """Row positions of one typical client per group (keys as in EXAMPLE_LABELS).

    Groups at `threshold`: defaulters flagged (caught) or not (missed), payers flagged
    (false alarm) or not (cleared). In each, the client nearest the group's median
    probability, the rule make explain uses. A group with no clients is left out.
    """
    y, proba = np.asarray(y), np.asarray(proba)
    flagged = proba >= threshold
    groups = {
        "caught": (y == 1) & flagged,
        "missed": (y == 1) & ~flagged,
        "false_alarm": (y == 0) & flagged,
        "cleared": (y == 0) & ~flagged,
    }
    picks = {}
    for name, in_group in groups.items():
        position = nearest_to_median(proba, in_group)
        if position is not None:
            picks[name] = position
    return picks


def describe_client(row: pd.DataFrame) -> pd.DataFrame:
    """The client's details in plain words, as a two-column table."""
    client = row.iloc[0]
    details = {
        "Credit limit": f"{client['LIMIT_BAL']:,.0f} NT$",
        "Age": f"{int(client['AGE'])}",
        "Sex": SEX_LABELS[int(client["SEX"])],
        "Education": EDUCATION_LABELS[int(client["EDUCATION"])],
        "Marital status": MARRIAGE_LABELS[int(client["MARRIAGE"])],
        "Most recent payment": status_label(int(client["PAY_1"])),
    }
    return pd.DataFrame({"Detail": list(details), "Value": list(details.values())})


def payment_history(row: pd.DataFrame) -> pd.DataFrame:
    """Six months of status, bill and payment, newest first, as display text."""
    client = row.iloc[0]
    return pd.DataFrame(
        {
            "Month": MONTHS,
            "Status": [status_label(int(client[c])) for c in PAY_STATUS],
            "Bill (NT$)": [f"{client[c]:,.0f}" for c in BILL_AMTS],
            "Payment (NT$)": [f"{client[c]:,.0f}" for c in PAY_AMTS],
        }
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_showcase.py tests/test_explain.py -v`
Expected: all PASS (the existing `pick_clients` tests confirm the refactor kept its behavior).

- [ ] **Step 6: Commit**

```bash
make format && make lint
git add src/credit_risk/explain.py src/credit_risk/showcase.py tests/test_showcase.py
git commit -m "Add showcase module: labels, cutoff, example clients, client tables"
```

---

### Task 3: Payment scenarios and edits

**Files:**
- Modify: `src/credit_risk/showcase.py`
- Modify: `tests/test_showcase.py`

**Interfaces:**
- Consumes: `DEMOGRAPHICS`, `PAY_STATUS`, `BILL_AMTS`, `PAY_AMTS` (Task 2).
- Produces:
  - `SCENARIOS: tuple[str, ...]` (the four names in Global Constraints, `"As recorded"` first)
  - `MIN_PAYMENT_SHARE = 0.06`, `LATE_PAYMENT_SHARE = 0.04`
  - `apply_scenario(row: pd.DataFrame, scenario: str) -> pd.DataFrame` (new frame; raises `ValueError` for an unknown name)
  - `edit_client(row, scenario: str, limit: float, demographics: dict[str, int]) -> pd.DataFrame` (raises `ValueError` for a key not in `DEMOGRAPHICS`)
  - `is_edited(original: pd.DataFrame, edited: pd.DataFrame) -> bool`

Rules (spec table), with BILL(i) the bill in column i:

| Scenario | Statuses | Payments |
|---|---|---|
| As recorded | unchanged | unchanged |
| Pays in full every month | PAY_1 to PAY_6 = -1 | PAY_AMTi = max(BILL(i+1), 0) for i = 1 to 5; PAY_AMT6 unchanged |
| Pays the minimum every month | PAY_1 to PAY_6 = 0 | PAY_AMTi = round(0.06 * max(BILL(i+1), 0)) for i = 1 to 5; PAY_AMT6 unchanged |
| Fell 2 months behind | PAY_1, PAY_2 = 2; others unchanged | PAY_AMT1 = round(0.04 * max(BILL(2), 0)), PAY_AMT2 = round(0.04 * max(BILL(3), 0)); others unchanged |

- [ ] **Step 1: Write the failing tests**

Add to the import list in `tests/test_showcase.py`: `SCENARIOS, apply_scenario, edit_client, is_edited`. Then add:

```python
STATUS = [f"PAY_{i}" for i in range(1, 7)]
PAID = [f"PAY_AMT{i}" for i in range(1, 7)]
BILLS = [f"BILL_AMT{i}" for i in range(1, 7)]


def values(frame, columns):
    return frame[columns].iloc[0].tolist()


def test_scenario_names():
    assert SCENARIOS == (
        "As recorded",
        "Pays in full every month",
        "Pays the minimum every month",
        "Fell 2 months behind",
    )


def test_as_recorded_changes_nothing(row):
    assert apply_scenario(row, "As recorded").equals(row)


def test_pays_in_full(row):
    # Each month pays the previous month's bill; the -500 bill (credit) gives 0, not -500.
    out = apply_scenario(row, "Pays in full every month")
    assert values(out, STATUS) == [-1] * 6
    assert values(out, PAID) == [2000, 0, 4000, 5000, 6000, 16]
    assert values(out, BILLS) == values(row, BILLS)


def test_pays_the_minimum(row):
    out = apply_scenario(row, "Pays the minimum every month")
    assert values(out, STATUS) == [0] * 6
    assert values(out, PAID) == [120, 0, 240, 300, 360, 16]
    assert values(out, BILLS) == values(row, BILLS)


def test_fell_two_months_behind(row):
    out = apply_scenario(row, "Fell 2 months behind")
    assert values(out, STATUS) == [2, 2, 1, 1, 1, 1]
    assert values(out, PAID) == [80, 0, 13, 14, 15, 16]
    assert values(out, BILLS) == values(row, BILLS)


def test_scenarios_leave_the_input_row_unchanged(row):
    before = row.copy()
    for scenario in SCENARIOS:
        apply_scenario(row, scenario)
    assert row.equals(before)


def test_unknown_scenario_is_refused(row):
    with pytest.raises(ValueError, match="Unknown scenario"):
        apply_scenario(row, "Pays twice")


def test_edit_client_with_the_recorded_values_is_not_an_edit(row):
    same = {column: int(row[column].iloc[0]) for column in ("SEX", "EDUCATION", "MARRIAGE", "AGE")}
    edited = edit_client(row, "As recorded", float(row["LIMIT_BAL"].iloc[0]), same)
    assert not is_edited(row, edited)


def test_edit_client_sets_the_limit_and_demographics(row):
    edited = edit_client(
        row, "Pays in full every month", 120000, {"SEX": 1, "EDUCATION": 3, "AGE": 45}
    )
    assert edited["LIMIT_BAL"].iloc[0] == 120000
    assert values(edited, ["SEX", "EDUCATION", "MARRIAGE", "AGE"]) == [1, 3, 1, 45]
    assert values(edited, STATUS) == [-1] * 6
    assert is_edited(row, edited)


def test_edit_client_refuses_other_columns(row):
    with pytest.raises(ValueError, match="BILL_AMT1"):
        edit_client(row, "As recorded", 50000, {"BILL_AMT1": 0})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_showcase.py -v`
Expected: collection error, `ImportError: cannot import name 'SCENARIOS'`.

- [ ] **Step 3: Implement**

Add to the module docstring of `showcase.py`, after the "Nothing here changes" paragraph:

```text
Payment behavior scenarios rewrite all six months together, so an edited client stays
plausible instead of, say, "paid in full" this month with every earlier month late. The
rules use patterns measured on the training clients (2026-10-14):
- 80% of clients who paid in full paid at least the previous month's bill;
- clients paying the minimum paid a median 6% of the previous bill;
- clients 2 months late in both of the last two months paid a median of about 4%, and
  "2 then 2" is the most common history for clients 2 months late now.
Bills stay as recorded. Credit limit and demographics change alone, since no other
column depends on them.
```

Add after `EXAMPLE_LABELS`:

```python
# Payment behavior options, in the order the app lists them.
SCENARIOS = (
    "As recorded",
    "Pays in full every month",
    "Pays the minimum every month",
    "Fell 2 months behind",
)
MIN_PAYMENT_SHARE = 0.06  # median minimum payment, as a share of the previous bill
LATE_PAYMENT_SHARE = 0.04  # median payment of clients 2 months late, same basis
```

Add after `widen`:

```python
def _share_of_previous_bill(out: pd.DataFrame, month: int, share: float) -> pd.Series:
    """`share` of the bill a payment in column `month` (0-based) pays: the next column's.

    A payment in September pays the August statement, which is the next column (data
    columns run newest first). A negative bill (a credit) gives 0, never a negative
    payment.
    """
    return (share * out[BILL_AMTS[month + 1]].clip(lower=0)).round()


def apply_scenario(row: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """A copy of `row` with one of SCENARIOS applied (see the module docstring)."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; expected one of {SCENARIOS}.")
    out = row.copy()  # never change the caller's row
    if scenario == "Pays in full every month":
        out[PAY_STATUS] = -1
        for month in range(5):  # the oldest payment's bill is not in the data; keep it
            out[PAY_AMTS[month]] = _share_of_previous_bill(out, month, 1.0)
    elif scenario == "Pays the minimum every month":
        out[PAY_STATUS] = 0
        for month in range(5):
            out[PAY_AMTS[month]] = _share_of_previous_bill(out, month, MIN_PAYMENT_SHARE)
    elif scenario == "Fell 2 months behind":
        out[PAY_STATUS[:2]] = 2
        for month in range(2):
            out[PAY_AMTS[month]] = _share_of_previous_bill(out, month, LATE_PAYMENT_SHARE)
    return out


def edit_client(
    row: pd.DataFrame, scenario: str, limit: float, demographics: dict[str, int]
) -> pd.DataFrame:
    """A copy of `row` with a scenario, a credit limit and demographic values applied.

    `demographics` maps any of DEMOGRAPHICS to a new value; columns left out keep theirs.
    """
    unknown = set(demographics) - set(DEMOGRAPHICS)
    if unknown:
        raise ValueError(f"Only {DEMOGRAPHICS} can be changed, not {sorted(unknown)}.")
    out = apply_scenario(row, scenario)
    out["LIMIT_BAL"] = float(limit)  # LIMIT_BAL is stored as a float in the cleaned data
    for column, value in demographics.items():
        out[column] = value
    return out


def is_edited(original: pd.DataFrame, edited: pd.DataFrame) -> bool:
    """Whether any value differs, comparing numbers (so 50000 and 50000.0 match)."""
    return not np.allclose(original.to_numpy(dtype=float), edited.to_numpy(dtype=float))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_showcase.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
make format && make lint
git add src/credit_risk/showcase.py tests/test_showcase.py
git commit -m "Add payment scenarios and client edits to the showcase module"
```

---

### Task 4: Scoring, contributions and test-set outcomes

**Files:**
- Modify: `src/credit_risk/showcase.py`
- Modify: `tests/test_showcase.py`

**Interfaces:**
- Consumes: `shap_explanation(model, X) -> shap.Explanation` (explain.py); `decision_metrics(y, proba, threshold, costs) -> dict` with keys `caught, missed, false_alarms, expected_cost` (evaluate.py); `Costs(false_negative=..., false_positive=...)` (config.py); `PLAIN_LABELS`, `cutoff_for` (Task 2).
- Produces:
  - `score_client(model, row) -> float`
  - `top_contributions(model, row, n: int = 8) -> pd.DataFrame` (columns `feature`, `label`, `shap`, `direction`; `direction` is `"Raises risk"` or `"Lowers risk"`; sorted by absolute `shap`, largest first)
  - `plot_contributions(table: pd.DataFrame) -> alt.Chart`
  - `outcomes_at_ratio(y, proba, ratio: float) -> pd.DataFrame` (not named `test_*`, which pytest would collect as a test once imported into a test file) (rows `This cutoff`, `Cutoff 0.5`, `Flag nobody` in column `setup`; columns `setup, cutoff, caught, missed, false_alarms, cost_per_client`; `cutoff` is NaN for flag nobody)

- [ ] **Step 1: Write the failing tests**

Add to the imports in `tests/test_showcase.py`: `from credit_risk.config import Costs` (extend the existing `credit_risk.config` import), `from credit_risk.evaluate import decision_metrics`, and from `credit_risk.showcase`: `outcomes_at_ratio, plot_contributions, score_client, top_contributions`. Then add:

```python
def test_score_client_matches_predict_proba(model, row):
    assert score_client(model, row) == pytest.approx(model.predict_proba(row)[0, 1])


def test_top_contributions_are_the_largest_in_plain_words(model, row):
    table = top_contributions(model, row, n=8)
    assert len(table) == 8
    assert table["label"].notna().all()
    sizes = table["shap"].abs().to_numpy()
    assert (np.diff(sizes) <= 0).all()  # largest first
    raises = table["shap"] > 0
    assert (table.loc[raises, "direction"] == "Raises risk").all()
    assert (table.loc[~raises, "direction"] == "Lowers risk").all()


def test_plot_contributions_is_a_bar_chart(model, row):
    spec = plot_contributions(top_contributions(model, row)).to_dict()
    assert spec["mark"]["type"] == "bar"


def test_outcomes_at_ratio_match_decision_metrics():
    rng = np.random.default_rng(0)
    proba = rng.uniform(0, 1, 400)
    y = (rng.uniform(0, 1, 400) < proba).astype(int)
    table = outcomes_at_ratio(y, proba, ratio=4).set_index("setup")
    assert table.index.tolist() == ["This cutoff", "Cutoff 0.5", "Flag nobody"]
    costs = Costs(false_negative=4.0, false_positive=1.0)
    expected = decision_metrics(y, proba, 0.2, costs)
    assert table.loc["This cutoff", "cutoff"] == pytest.approx(0.2)
    assert table.loc["This cutoff", "caught"] == expected["caught"]
    assert table.loc["This cutoff", "cost_per_client"] == pytest.approx(expected["expected_cost"])
    assert table.loc["Cutoff 0.5", "cutoff"] == 0.5
    nobody = table.loc["Flag nobody"]
    assert np.isnan(nobody["cutoff"])
    assert (nobody["caught"], nobody["false_alarms"]) == (0, 0)
    assert nobody["missed"] == y.sum()
    assert nobody["cost_per_client"] == pytest.approx(4.0 * y.mean())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_showcase.py -v`
Expected: collection error, `ImportError: cannot import name 'plot_contributions'`.

- [ ] **Step 3: Implement**

Add imports to `showcase.py`:

```python
import altair as alt
from sklearn.model_selection import TunedThresholdClassifierCV

from credit_risk.config import Costs
from credit_risk.evaluate import decision_metrics
from credit_risk.explain import nearest_to_median, shap_explanation
```

Add at the end of the module:

```python
def score_client(model: TunedThresholdClassifierCV, row: pd.DataFrame) -> float:
    """Calibrated probability of default for one client (the full saved pipeline)."""
    return float(model.predict_proba(row)[0, 1])


def top_contributions(
    model: TunedThresholdClassifierCV, row: pd.DataFrame, n: int = 8
) -> pd.DataFrame:
    """The `n` model columns that moved this client's score most, in plain words.

    Values are SHAP values (see explain.py): log-odds for LightGBM, before calibration.
    Positive pushes toward default. Sorted by size, largest first.
    """
    explanation = shap_explanation(model, row)
    table = pd.DataFrame({"feature": explanation.feature_names, "shap": explanation.values[0]})
    table["label"] = table["feature"].map(PLAIN_LABELS)
    table["direction"] = np.where(table["shap"] > 0, "Raises risk", "Lowers risk")
    order = table["shap"].abs().sort_values(ascending=False).index
    return table.loc[order].head(n).reset_index(drop=True)


def plot_contributions(table: pd.DataFrame) -> alt.Chart:
    """Horizontal bars from top_contributions: right raises risk, left lowers it."""
    return (
        alt.Chart(table)
        .mark_bar()
        .encode(
            x=alt.X("shap:Q", title="Effect on the model's score (right: toward default)"),
            # sort=None keeps the table's order (largest effect at the top).
            y=alt.Y("label:N", sort=None, title=None),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Raises risk", "Lowers risk"], range=["#c0392b", "#2e86c1"]
                ),
                title=None,
            ),
            tooltip=["label:N", alt.Tooltip("shap:Q", format=".2f")],
        )
        .properties(title="What moved this client's score most")
    )


def outcomes_at_ratio(y: Sequence[int], proba: Sequence[float], ratio: float) -> pd.DataFrame:
    """Counts and cost per client on the test clients at three settings, for one ratio.

    This cutoff (1 / (1 + ratio)), the default 0.5, and flag nobody. Cost per client
    uses the same ratio for all three (a miss costs `ratio`, a false alarm 1), so the
    rows can be compared. Flag nobody uses an infinite cutoff, which flags no one.
    """
    costs = Costs(false_negative=float(ratio), false_positive=1.0)
    settings = {"This cutoff": cutoff_for(ratio), "Cutoff 0.5": 0.5, "Flag nobody": np.inf}
    rows = []
    for setup, cutoff in settings.items():
        metrics = decision_metrics(y, proba, cutoff, costs)
        rows.append(
            {
                "setup": setup,
                "cutoff": cutoff if np.isfinite(cutoff) else np.nan,
                "caught": metrics["caught"],
                "missed": metrics["missed"],
                "false_alarms": metrics["false_alarms"],
                "cost_per_client": metrics["expected_cost"],
            }
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_showcase.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
make format && make lint
git add src/credit_risk/showcase.py tests/test_showcase.py
git commit -m "Add scoring, SHAP contributions and test-set outcomes to the showcase module"
```

---

### Task 5: Load everything the app needs

**Files:**
- Modify: `src/credit_risk/showcase.py`
- Modify: `tests/test_showcase.py`

**Interfaces:**
- Consumes: `load_model(cfg)` (decision.py, raises `DataFileError` for a missing or stale model); `verify_raw(cfg)`, `load_raw(cfg)`, `clean(df)`, `split(df, cfg) -> (train, test)` (data.py); `example_clients` (Task 2).
- Produces:
  - `Ranges` dataclass: `limit_min: int, limit_max: int, age_min: int, age_max: int`
  - `Assets` dataclass: `model, test_X: pd.DataFrame (no target, index 0..n-1), test_y: np.ndarray, proba: np.ndarray, ranges: Ranges, examples: dict[str, int], saved_cutoff: float`
  - `load_assets(cfg: Config) -> Assets`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_showcase.py` imports: `load_assets` from `credit_risk.showcase`. Then:

```python
@pytest.fixture(scope="module")
def assets():
    # The committed model and the real test split, as the app loads them. Reading the
    # test set here is allowed: the app is one of its readers (project spec, 2026-10-14).
    return load_assets(BASE)


def test_load_assets_matches_make_evaluate(assets):
    assert len(assets.test_X) == 6000
    assert TARGET not in assets.test_X.columns
    assert assets.test_X.index.tolist() == list(range(6000))
    assert assets.proba.shape == (6000,)
    assert assets.saved_cutoff == pytest.approx(0.1697, abs=1e-3)
    # Same clients and probabilities as reports/tables/test_metrics.csv.
    flagged = assets.proba >= assets.saved_cutoff
    assert flagged.sum() == 2666
    assert (flagged & (assets.test_y == 1)).sum() == 1016


def test_load_assets_ranges_come_from_train(assets):
    ranges = assets.ranges
    assert (ranges.limit_min, ranges.limit_max) == (10000, 500000)
    assert (ranges.age_min, ranges.age_max) == (21, 79)


def test_load_assets_examples_are_in_their_groups(assets):
    assert set(assets.examples) == {"caught", "missed", "false_alarm", "cleared"}
    flagged = assets.proba >= assets.saved_cutoff
    y = assets.test_y
    assert y[assets.examples["caught"]] == 1 and flagged[assets.examples["caught"]]
    assert y[assets.examples["missed"]] == 1 and not flagged[assets.examples["missed"]]
    assert y[assets.examples["false_alarm"]] == 0 and flagged[assets.examples["false_alarm"]]
    assert y[assets.examples["cleared"]] == 0 and not flagged[assets.examples["cleared"]]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_showcase.py -k load_assets -v`
Expected: collection error, `ImportError: cannot import name 'load_assets'`.

- [ ] **Step 3: Implement**

Add imports to `showcase.py`:

```python
from dataclasses import dataclass

from credit_risk.config import Config, Costs
from credit_risk.data import (
    BILL_AMTS,
    PAY_AMTS,
    PAY_STATUS,
    TARGET,
    clean,
    load_raw,
    split,
    verify_raw,
)
from credit_risk.decision import load_model
```

Add after `LATE_PAYMENT_SHARE`:

```python
@dataclass(frozen=True)
class Ranges:
    """Slider ranges, from the training clients: limit min to 99th percentile (so a few
    very high limits do not stretch the slider), age min to max."""

    limit_min: int
    limit_max: int
    age_min: int
    age_max: int


@dataclass(frozen=True)
class Assets:
    """Everything the app loads once at startup. None of it changes while it runs."""

    model: TunedThresholdClassifierCV
    test_X: pd.DataFrame  # test clients without the target; index 0 to n-1
    test_y: np.ndarray  # 1 = defaulted
    proba: np.ndarray  # calibrated probability of default per test client
    ranges: Ranges
    examples: dict[str, int]  # EXAMPLE_LABELS key -> row position in test_X
    saved_cutoff: float  # the saved model's cutoff (searched on train at the config's costs)
```

Add at the end of the module:

```python
def load_assets(cfg: Config) -> Assets:
    """Load the saved model and rebuild the train/test split in memory.

    The split comes from the committed raw CSV with the config's seed, so it is the same
    6,000 test clients as make evaluate, and nothing is written to disk (data/processed
    is not in git). Raises DataFileError if the model is missing or stale, or the raw
    CSV does not match its checksum.
    """
    model = load_model(cfg)  # checks artifacts/model.json against the config
    verify_raw(cfg)
    train, test = split(clean(load_raw(cfg)), cfg)
    test_X = test.drop(columns=TARGET).reset_index(drop=True)
    test_y = test[TARGET].to_numpy()
    proba = model.predict_proba(test_X)[:, 1]  # one scoring, as in make evaluate
    ranges = Ranges(
        limit_min=int(train["LIMIT_BAL"].min()),
        limit_max=int(train["LIMIT_BAL"].quantile(0.99)),
        age_min=int(train["AGE"].min()),
        age_max=int(train["AGE"].max()),
    )
    saved_cutoff = float(model.best_threshold_)
    return Assets(
        model=model,
        test_X=test_X,
        test_y=test_y,
        proba=proba,
        ranges=ranges,
        # Chosen once at the saved cutoff, so each example stays the same person when the
        # app's cost slider moves.
        examples=example_clients(test_y, proba, saved_cutoff),
        saved_cutoff=saved_cutoff,
    )
```

(Merge these imports with the ones from Tasks 2 and 4 so each name is imported once, in ruff's order.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_showcase.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
make format && make lint
git add src/credit_risk/showcase.py tests/test_showcase.py
git commit -m "Add load_assets: the committed model and the test split, rebuilt in memory"
```

---

### Task 6: The app, `make app`, and app tests

**Files:**
- Create: `app/app.py`
- Create: `tests/test_app.py`
- Modify: `Makefile` (`app` target, help text, `.PHONY`)
- Modify: `README.md` (one line on running the app locally)

**Interfaces:**
- Consumes: everything in `showcase.py` (Tasks 2 to 5); `plot_tradeoff_curves(..., points_note=...)` (Task 1); `load_config`, `ConfigError` (config.py); `DataFileError` (data.py).
- Widget keys (used by tests): sidebar slider `ratio`; buttons `example_caught`, `example_missed`, `example_false_alarm`, `example_cleared`, `random`, `reset`; selectbox `scenario`; slider `limit`; selectboxes `SEX`, `EDUCATION`, `MARRIAGE`; slider `AGE`. Session state `client` holds the row position.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_app.py`:

```python
"""The Streamlit app runs end to end on the committed model and data (AppTest).

AppTest runs app.py as Streamlit would, without a browser. The first run loads the model
and the split (cached for the rest of the session), so these tests take a few seconds.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "app.py")


@pytest.fixture
def app():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def sidebar_text(at) -> str:
    return " ".join(m.value for m in at.sidebar.markdown)


def probability(at) -> str:
    return at.metric[0].value


def test_app_starts_with_three_tabs(app):
    assert [tab.label for tab in app.tabs] == ["About", "Try a client", "Cost ratio"]
    assert "16.7%" in sidebar_text(app)  # default ratio 5: cutoff 1/6


def test_moving_the_slider_changes_the_cutoff(app):
    app.slider(key="ratio").set_value(2).run()
    assert not app.exception
    assert "33.3%" in sidebar_text(app)


def test_each_example_client_shows_a_probability(app):
    for key in ("caught", "missed", "false_alarm", "cleared"):
        app.button(key=f"example_{key}").click().run()
        assert not app.exception
        assert probability(app).endswith("%")


def test_changing_payment_behavior_changes_the_probability(app):
    app.button(key="example_caught").click().run()
    before = probability(app)
    app.selectbox(key="scenario").set_value("Pays in full every month").run()
    assert not app.exception
    assert probability(app) != before
    assert any("Your changes moved" in m.value for m in app.markdown)


def test_picking_another_client_resets_the_controls(app):
    app.selectbox(key="scenario").set_value("Pays in full every month").run()
    app.button(key="example_missed").click().run()
    assert app.selectbox(key="scenario").value == "As recorded"
    assert not any("Your changes moved" in m.value for m in app.markdown)


def test_fairness_controls_run(app):
    app.selectbox(key="SEX").set_value(1).run()
    app.selectbox(key="SEX").set_value(2).run()
    app.slider(key="AGE").set_value(40).run()
    assert not app.exception


def test_reset_button_restores_the_client(app):
    app.selectbox(key="scenario").set_value("Fell 2 months behind").run()
    app.button(key="reset").click().run()
    assert app.selectbox(key="scenario").value == "As recorded"


def test_random_client_runs(app):
    app.button(key="random").click().run()
    assert not app.exception


def test_client_above_the_limit_range_renders(app):
    # 47 test clients have limits above the slider's usual 500,000 maximum.
    from credit_risk.config import load_config
    from credit_risk.showcase import load_assets

    assets = load_assets(load_config(ROOT / "configs" / "base.yaml"))
    position = int((assets.test_X["LIMIT_BAL"] > 500_000).to_numpy().nonzero()[0][0])
    limit = int(assets.test_X.loc[position, "LIMIT_BAL"])
    app.session_state["client"] = position
    app.session_state["limit"] = limit
    app.run()
    assert not app.exception
    assert app.slider(key="limit").value == limit
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_app.py -v`
Expected: FAIL, the app file does not exist (`AppTest.from_file` raises or `at.exception` is set).

- [ ] **Step 3: Write `app/app.py`**

```python
"""Streamlit app: try the credit default model on clients it never saw.

Layout only. The logic (example clients, payment scenarios, scoring, SHAP
contributions, test-set counts) is in src/credit_risk/showcase.py, which has tests.
Run locally with `make app`; deployed on Streamlit Community Cloud (app/requirements.txt).

Streamlit runs this whole file again after every click. Values that must survive a
rerun (which client is shown, the control values) live in st.session_state; the model
and data are loaded once per app process with st.cache_resource.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Community Cloud installs app/requirements.txt but not this repo's package, so put
# src/ on the import path. Locally the package is installed too, and this changes nothing.
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from credit_risk import showcase as sc
from credit_risk.config import ConfigError, load_config
from credit_risk.data import DataFileError
from credit_risk.evaluate import plot_tradeoff_curves

REPO = "https://github.com/iangault/credit-default-risk"
RESULTS = f"{REPO}/blob/main/docs/results.md"

st.set_page_config(page_title="Credit default risk", layout="wide")


@st.cache_resource(show_spinner="Loading the model and the test clients...")
def get_assets() -> sc.Assets:
    """Loaded once per app process and shared by every visitor (it never changes)."""
    return sc.load_assets(load_config(ROOT / "configs" / "base.yaml"))


try:
    assets = get_assets()
except (ConfigError, DataFileError) as error:
    st.error(f"The app could not start: {error}")
    st.stop()


# ---- Which client is shown, and the controls -------------------------------------


def current_row() -> pd.DataFrame:
    """The selected test client, as a one-row frame (double brackets keep it a frame)."""
    return assets.test_X.iloc[[st.session_state.client]]


def reset_controls() -> None:
    """Set every control to the current client's recorded values."""
    client = current_row().iloc[0]
    st.session_state.scenario = sc.SCENARIOS[0]
    st.session_state.limit = int(client["LIMIT_BAL"])
    for column in sc.DEMOGRAPHICS:
        st.session_state[column] = int(client[column])


def show_example(key: str) -> None:
    st.session_state.client = assets.examples[key]
    st.session_state.shown = sc.EXAMPLE_LABELS[key]
    reset_controls()


def show_random() -> None:
    st.session_state.client = int(np.random.default_rng().integers(len(assets.test_X)))
    st.session_state.shown = "A random test client"
    reset_controls()


# First run in a browser session: start on the first example (the caught defaulter).
if "client" not in st.session_state:
    show_example(next(iter(assets.examples)))


# ---- Sidebar: the cost ratio, shared by both views ---------------------------------

with st.sidebar:
    st.header("Cost of mistakes")
    ratio = st.slider("A missed defaulter costs N times a false alarm", 1, 20, 5, key="ratio")
    cutoff = sc.cutoff_for(ratio)
    st.markdown(f"Flag clients with a default risk of **{cutoff:.1%}** or more.")
    st.caption(
        "Cutoff = 1 / (1 + N), with a false alarm costing 1. The saved model's cutoff at "
        f"5x came from a search on training data: {assets.saved_cutoff:.3f}. So counts "
        "at 5x differ slightly from the write-up."
    )

about, client_tab, cost_tab = st.tabs(["About", "Try a client", "Cost ratio"])

# ---- Tab 1: About ------------------------------------------------------------------

with about:
    st.header("Who will miss next month's credit card payment?")
    st.markdown(
        "This app shows a model that predicts whether a credit card client will miss "
        "next month's payment.\n\n"
        "- **The data:** 30,000 real, anonymized clients of a Taiwanese bank in 2005, "
        "published by the UCI Machine Learning Repository.\n"
        "- **The model:** LightGBM, with probabilities calibrated so that 30% means "
        "about 30% of such clients default.\n"
        "- **The key idea:** missing a defaulter costs more than a false alarm on a "
        "client who would have paid. If a miss costs 5 times as much, the best cutoff is "
        "about 0.17, not the usual 0.5.\n"
        "- **The result:** on 6,000 clients the model never saw, it caught 77% of "
        "defaulters, and the cost per client was 29% lower than with a 0.5 cutoff.\n\n"
        "**Try a client** lets you pick a real client and change a few things. "
        "**Cost ratio** shows what the cutoff does across all 6,000 clients. The slider "
        "on the left sets how much a miss costs, for both.\n\n"
        f"[Code on GitHub]({REPO}) · [Full write-up]({RESULTS})"
    )

# ---- Tab 2: Try a client -----------------------------------------------------------

with client_tab:
    left, right = st.columns(2, gap="large")
    row = current_row()
    position = st.session_state.client

    with left:
        st.subheader("Pick a test client")
        for key, label in sc.EXAMPLE_LABELS.items():
            if key in assets.examples:
                st.button(label, key=f"example_{key}", on_click=show_example, args=(key,))
        st.button("A random test client", key="random", on_click=show_random)
        st.caption(f"Showing: {st.session_state.shown} (test client ID {int(row['ID'].iloc[0])})")

        st.subheader("Change a few things")
        st.selectbox("Payment behavior", sc.SCENARIOS, key="scenario")
        low, high = sc.widen(
            assets.ranges.limit_min, assets.ranges.limit_max, int(row["LIMIT_BAL"].iloc[0])
        )
        st.slider("Credit limit (NT$)", low, high, step=1000, key="limit")
        with st.expander("Fairness check: change only demographics"):
            st.selectbox("Sex", list(sc.SEX_LABELS), format_func=sc.SEX_LABELS.get, key="SEX")
            st.selectbox(
                "Education",
                list(sc.EDUCATION_LABELS),
                format_func=sc.EDUCATION_LABELS.get,
                key="EDUCATION",
            )
            st.selectbox(
                "Marital status",
                list(sc.MARRIAGE_LABELS),
                format_func=sc.MARRIAGE_LABELS.get,
                key="MARRIAGE",
            )
            age_low, age_high = sc.widen(
                assets.ranges.age_min, assets.ranges.age_max, int(row["AGE"].iloc[0])
            )
            st.slider("Age", age_low, age_high, key="AGE")
            st.markdown(
                "- The model was trained with these columns, so it can use them. "
                "Sex: female ranks 10th of 28 columns by average effect and lowers scores "
                "slightly.\n"
                "- In many places a lender could not legally base a decision on some of "
                "them.\n"
                "- Changing one client is a spot check, not a fairness audit. The proper "
                "checks (error rates by group, and the model with and without these "
                f"columns) are listed in the [write-up]({RESULTS}) as future work."
            )
        st.button("Reset to original", key="reset", on_click=reset_controls)
        st.caption(
            "Scenarios rewrite all six months using patterns common in the data; bills "
            "stay as recorded."
        )

    demographics = {column: st.session_state[column] for column in sc.DEMOGRAPHICS}
    edited = sc.edit_client(row, st.session_state.scenario, st.session_state.limit, demographics)

    with right:
        before = sc.score_client(assets.model, row)
        after = sc.score_client(assets.model, edited)
        st.metric("Probability of default", f"{after:.1%}")
        decision = "Flag for review" if sc.is_flagged(after, cutoff) else "Don't flag"
        st.markdown(f"**Decision at the {cutoff:.1%} cutoff:** {decision}")
        if sc.is_edited(row, edited):
            st.markdown(f"Your changes moved this client's risk from {before:.1%} to {after:.1%}.")
        actual = "Yes" if assets.test_y[position] == 1 else "No"
        st.markdown(f"**Did this client actually default?** {actual}")
        st.altair_chart(
            sc.plot_contributions(sc.top_contributions(assets.model, edited)), width="stretch"
        )
        st.caption(
            "Each bar is how much one column pushed this client's score up (toward "
            "default) or down, compared with an average client (SHAP values)."
        )
        st.dataframe(sc.describe_client(edited), hide_index=True)
        st.dataframe(sc.payment_history(edited), hide_index=True)

# ---- Tab 3: Cost ratio -------------------------------------------------------------

with cost_tab:
    outcomes = sc.outcomes_at_ratio(assets.test_y, assets.proba, ratio)
    defaulters = int(assets.test_y.sum())
    payers = len(assets.test_y) - defaulters
    st.subheader(f"All {len(assets.test_y):,} test clients at a {ratio}x cost ratio")
    table = pd.DataFrame(
        {
            "Cutoff": [
                "none flagged" if pd.isna(c) else f"{c:.3f}" for c in outcomes["cutoff"]
            ],
            f"Defaulters caught (of {defaulters:,})": outcomes["caught"],
            "Defaulters missed": outcomes["missed"],
            f"False alarms (of {payers:,})": outcomes["false_alarms"],
            "Cost per client": outcomes["cost_per_client"].round(3),
        },
    )
    table.index = outcomes["setup"]
    st.dataframe(table)
    st.caption(f"Cost per client: a miss costs {ratio}, a false alarm costs 1, averaged.")
    marks = pd.DataFrame({"label": [f"cutoff {cutoff:.3f}"], "threshold": [cutoff]})
    st.altair_chart(
        plot_tradeoff_curves(
            assets.test_y, assets.proba, marks, points_note=f"the cutoff for {ratio}x"
        )
    )
    st.markdown(
        "The model's probabilities stay the same when you move the slider. Only the line "
        "drawn through them moves: a higher cost of a miss lowers the cutoff, so more "
        "defaulters are caught and more clients who would have paid are flagged."
    )
```

- [ ] **Step 4: Run the app tests**

Run: `python -m pytest tests/test_app.py -v`
Expected: all PASS. If a Streamlit deprecation warning appears for `width="stretch"` or another argument, switch to the form the warning names and rerun.

- [ ] **Step 5: Add `make app` and a README line**

In `Makefile`: add `app` to `.PHONY`; in `help`, after the Step 4 block, add:

```make
	@printf "%b\n" "$(CYAN)Step 5 - Showcase app:$(RESET)"
	@printf "%b\n" "  $(YELLOW)make app$(RESET)          - Run the Streamlit app locally (app/app.py)"
	@printf "%b\n" ""
```

and in the code map list, after the `10 explain.py` line, add:

```make
	@printf "%b\n" "  $(GREEN)11 showcase.py$(RESET)    - Logic behind the Streamlit app (app/app.py)"
```

Add the target after the `explain:` target:

```make
app:
	@printf "%b\n" "$(CYAN)Starting the Streamlit app (Ctrl+C to stop)...$(RESET)"
	streamlit run app/app.py
```

In `README.md`, Quickstart section, add a line to the code block after `make all`:

```bash
make app                  # run the Streamlit app locally (uses the committed model)
```

- [ ] **Step 6: Run the full suite**

Run: `make format && make lint && make test`
Expected: lint clean; all tests pass.

- [ ] **Step 7: Commit**

```bash
git add app/app.py tests/test_app.py Makefile README.md
git commit -m "Add the Streamlit app, make app, and app tests"
```

---

### Task 7: Manual check with the author

**Files:** none changed unless the check finds problems.

- [ ] **Step 1: Start the app**

Run (in the background): `make app`
Expected: Streamlit prints `Local URL: http://localhost:8501`.

- [ ] **Step 2: Ask the author to click through**

Ask the author to open http://localhost:8501 and check, in this order: About text; each example button; each payment behavior; the credit limit slider; the fairness section (each control); Reset; Random; the cost ratio slider at 1, 5 and 20 on both tabs; the cost tab table and chart. Ask for anything that reads oddly or looks wrong.

- [ ] **Step 3: Fix what the author raises**

Each fix follows the same cycle: failing test (where the fix is logic), fix, `make format && make lint && make test`, commit with a plain message naming the fix.

- [ ] **Step 4: Stop the app and report**

Stop the background `make app`. Report to the author: commits on `streamlit-app`, test count, and that piece B (deploy) is next. Do not push.
