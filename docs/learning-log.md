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

#### 5. Why the project is built as a package (a lab framing)

**The short version:** the notebook is the bench notebook, where I try things. The
package (`src/credit_risk/`) is the validated protocol: the one version everyone
follows, with controls that confirm it still works.

| Software practice | Lab equivalent |
|---|---|
| Notebook (`notebooks/`) | Bench notebook: trying things, looking at results |
| Package (`src/credit_risk/`) | Validated SOP: the one version of each step everyone uses |
| Config (`configs/base.yaml`) | Protocol parameter sheet, kept separate from the steps |
| Raw data sha256 check | Checking the sample ID / lot number before starting |
| Fixed seed | A fixed plate layout, so reruns are comparable |
| Tests (`tests/`) | Positive and negative controls: confirm the assay works before trusting results |
| Dummy model | Negative control: the result with no real signal |
| `make all` | Running the full protocol start to finish |
| CI (GitHub Actions) | QC run every time the protocol is edited |
| git history | Dated notebook entries showing what changed and when |

**Why a package instead of relative paths:**
- A relative import (`sys.path.append("../src")`) depends on which folder the code runs
  from: it works in one place and breaks in another. An installed package imports the
  same way from notebooks, tests, the terminal, and CI.
- There is one copy of each function. The notebook, the tests, and `make experiments`
  call the same `get_model`, so a fix lands everywhere at once. Copy-pasted code
  drifts, like photocopies of a protocol that each get hand-edited.

**The principles behind the layout, and what each one buys:**
1. **One job per module.** Cleaning is in `data.py`, scaling in `preprocess.py`, models in
   `models.py`. I know where to look, and changing one step does not touch the others.
2. **One source of truth.** Each step is written once and imported everywhere else.
3. **Functions take inputs and return outputs, with no hidden state.**
   `run_experiments(cfg, train)` depends only on what is passed in. Same input, same
   output, so it can be tested, and notebook cell order stops mattering.
4. **Settings in config, not in code.** Changing CV folds or adding a model is a config
   edit, and the config is a record of what was run.
5. **Fail early, with a clear message.** A typo in a model name fails before 30 seconds
   of training; a changed raw file stops at step 1 instead of producing quietly
   different results.
6. **Test first.** Write the expected result before writing the code, like defining
   what a positive result looks like before running the assay. It stops me from
   deciding after the fact that whatever came out was correct.

**Where notebooks still fit:** exploration, plots, and trying ideas. They import from
the package, so they stay short. My scratch notebook already used functions
(`run_cv`, `compare`); the difference with `src/` is not tidiness but that the code
there is tested, shared, and run by one command. The path for an idea is: try it in a
notebook, then move it into `src/` with a test once it settles.

**How to find what does what:** see "Where to find things" in the README.

#### 6. How the tests work

**What they test:** the code, not the data. Each test feeds a function an input where I
already know the right answer, then checks the function gives it.

**Why fake data is useful:** to check that a guard catches a problem, I need an input
that has the problem. Example: `verify_raw` should stop the pipeline if the raw CSV
changes. I can't test that with the real CSV, because it hasn't changed. So the test
writes a tiny file (`abd` instead of `abc`) and confirms the guard trips. Like the test
button on a smoke detector: trigger the alarm on purpose to confirm it works.

- **Fake inputs** check the logic: does it catch typos, bad values, missing files?
- **Real-data tests** are the sanity check: the committed CSV matches its hash,
  cleaning gives a 22.12% default rate, the split is 24,000 / 6,000.

**How they run:** test files are not scripts. Running `python tests/test_data.py` only
defines the functions; nothing calls them. `pytest` (via `make test`) does that:
1. Finds files in `tests/` named `test_*.py`.
2. Collects every function named `test_*`. The name prefix is what makes it a test.
3. Calls each one. It passes if it finishes; it fails if an `assert` is false or an
   unexpected error is raised.

Useful commands: `pytest -v` (list every test with pass/fail), `pytest tests/test_data.py`
(one file), `pytest tests/test_data.py::test_sha256_of_known_content` (one test).

**Fixtures (where test arguments come from):** I never call a test, so who fills in its
arguments? pytest matches each parameter name to a **fixture**, something it builds and
passes in.
- `tmp_path` (built into pytest): a fresh empty folder for that test. Tests write fake
  files there, so the real `data/` folder is never touched and tests can't interfere
  with each other.
- `capsys`, `monkeypatch` (built in): capture printed output; temporarily change things
  like the current folder.
- Custom fixtures with `@pytest.fixture`, e.g. `cleaned` in `test_split.py` loads and
  cleans the real data. `scope="module"` builds it once per file instead of once per
  test.

**What the tests don't tell me:** whether the model is any good. That comes from EDA and
the experiments table. Tests confirm the plumbing does what it claims, so I can trust
the results came from the data and settings I intended.
