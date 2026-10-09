# Learning log

One entry per milestone I own. Written in my own words, as if explaining it in an
interview.

## Template

### Mx: <title> (<date>)

- **What I built:**
- **Key concept, explained simply:**
- **A decision I made and why:**
- **What surprised me or went wrong:**
- **What I would do differently:**

---

### M2: Baseline models (in progress, notes from 2026-10-09)

#### 1. The problem

- **Task:** binary classification. Predict whether a credit card client defaults next
  month (1) or not (0).
- **Size:** 30,000 clients, 23 features. Small by modern standards; fits in memory and
  trains in seconds.
- **Class balance:** about 78% no default, 22% default (imbalanced).
- **Tabular, mixed types:** one row per client, one column per feature, like a
  spreadsheet. "Mixed" means the columns are different kinds of data:
  - Numeric amounts: `LIMIT_BAL`, `AGE`, `BILL_AMT1-6`, `PAY_AMT1-6` (scaled).
  - Categorical codes: `SEX`, `EDUCATION`, `MARRIAGE`. The numbers are labels, not
    quantities (one-hot encoded).
  - Ordinal codes: `PAY_1-6` payment status, -2 to 8, roughly from paid early to many
    months late (kept as numbers because the order means something).

#### 2. The candidate models and why each is there

| Model | Family | Role in the comparison |
|---|---|---|
| Dummy | none | The floor: predicts the average default rate for everyone. A real model must beat it. |
| Logistic regression | linear | The credit industry standard and the benchmark to beat. |
| Random forest | tree ensemble, bagging | A robust comparison point for boosting. |
| LightGBM | tree ensemble, boosting | Usually the strongest on tabular data of this size. |
| XGBoost | tree ensemble, boosting | A second boosting implementation; mostly overlaps with LightGBM. |

**Logistic regression (linear):**
- Adds up each feature times a weight to get the log-odds of default, then turns that
  into a probability.
- "Linear" means each feature's effect is a straight line: more of X always moves risk
  the same amount in the same direction.
- Why it matters in credit: interpretable (each weight is readable), accepted by
  regulators, and its probabilities tend to be well calibrated.
- Weakness: misses curves and interactions (for example, risk jumping sharply at 2
  months late) unless they are built in as features.

**Tree ensembles:** many decision trees combined. A single tree is easy to read but
unstable and weak; combining many fixes that.
- **Bagging (random forest):** grows many deep trees independently, each on a random
  sample of rows and features, and averages their votes. Averaging cancels out the
  individual trees' mistakes.
- **Boosting (LightGBM, XGBoost):** grows small trees one after another, each one
  correcting the errors of the trees before it.
- Good for: curves, interactions and mixed feature types, with little preprocessing.
- Cost: harder to explain than logistic regression (SHAP helps, in M6).

**Other options considered, not used:** neural nets (better suited to images and text,
rarely beat boosting on small tabular data), SVM and kNN (rely on distance between
rows, which suits mixed codes poorly; SVM is slow at 30k rows and gives no native
probabilities), EBM (a GAM fit with boosting: one curve per feature, near-boosting
accuracy while staying interpretable; a possible later addition).

**Decision:** kept the lineup of dummy, logreg, forest, LightGBM, XGBoost.

#### 3. Metrics: why not accuracy

- **Accuracy is misleading here.** The dummy model predicts "no default" for everyone
  and still scores about 78% accuracy, while catching zero defaulters.
- Metrics used instead:
  - **ROC-AUC:** how well the model ranks defaulters above non-defaulters (0.5 = random,
    1.0 = perfect).
  - **PR-AUC:** like ROC-AUC but focused on the default class; more sensitive to the
    imbalance.
  - **Recall:** share of actual defaulters the model catches.
  - **Precision:** share of flagged clients who actually default.
  - **F1:** a single number balancing recall and precision.

**The train/validation gap is informative:**

Scratch notebook results (default settings, 5-fold CV ROC-AUC, validation / train):

| Model | Validation | Train | What the gap says |
|---|---|---|---|
| LightGBM | 0.776 | 0.893 | Best score, moderate gap |
| XGBoost | 0.760 | 0.959 | Larger gap, overfitting more |
| Random forest | 0.759 | 1.000 | Memorizes the training rows |
| Logistic regression | 0.722 | 0.725 | Almost no gap, but too simple (underfitting) |

- Large gap: the model learned noise specific to the training data (overfitting).
- No gap with a low score: the model is too simple to capture the pattern
  (underfitting).

#### 4. How the project pieces fit together

| Piece | Role |
|---|---|
| Notebook (`.ipynb`) | Exploring and trying ideas. Cells can run out of order, so results are hard to reproduce. |
| Module (`src/credit_risk/*.py`) | The reusable version of what worked in the notebook. |
| Tests (`tests/`) | Check that the module code does what it should. |
| CLI (`credit-risk`) | Runs one pipeline stage from the terminal. |
| Makefile | Short names for CLI commands and dev tasks. |

- **CLI (command-line interface):** a program run by typing a command in the terminal
  instead of clicking or running notebook cells. Here, `credit-risk` is defined in
  `src/credit_risk/cli.py`, and each subcommand (`data`, `split`, and later
  `experiments`) runs one stage of the pipeline.
- **Makefile as a wrapper:** `make split` runs
  `credit-risk --config configs/base.yaml split`. Make saves typing, keeps commands
  consistent, and chains stages (`make all` runs `data` then `split`).
- **Smoke run (`make smoke`):** a quick check that the whole pipeline runs end to end.
  Tests check pieces one at a time; smoke checks they work together. Uses a small
  sample (1,000 rows, 2 CV folds, via `configs/smoke.yaml`), so scores are not
  meaningful. CI runs it on every push.
- **The flow:** idea tried in a notebook → moved into a `.py` module → covered by tests
  → exposed as a CLI stage → wrapped by a `make` target. CI runs `make smoke` and the
  tests on every push, so the whole chain stays working.
- Example from M2: the model comparison was first run in `notebooks/scratch.ipynb`,
  then moved to `models.py` (`get_model`) with tests in `tests/test_models.py`. It
  becomes `make experiments` in Task 7.
