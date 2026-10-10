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

**Where the expected answer comes from:** it has to come from somewhere other than the
code being tested. If I ran my own function, copied its output, and asserted the output
equals that, the test would pass whether the code was right or wrong.
- Good sources: a published standard, a hand calculation, or a case small enough to
  check by eye (`EDUCATION=[0, 5, 6, 1]` should become `[4, 4, 4, 1]`).
- Example: `ABC_SHA256` in `test_data.py` is the answer published in the SHA-256
  standard for the input "abc". Every correct implementation gives it, so a bug in
  `sha256_of` would make the test fail.

**Two jobs that look alike:**
- `ABC_SHA256` checks that **the hashing code is correct**. Outside reference.
- `raw_sha256` in `base.yaml` checks that **the data hasn't changed**. It was recorded
  once by hashing the downloaded CSV and is compared on every run. It proves nothing
  about correctness, only "same file as before", which is all it needs to do.

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

---

### M3: Class imbalance (draft by Claude for me to edit, 2026-10-09)

#### 1. The problem

About 22% of clients in train defaulted. At the default 0.5 cutoff, the M2 LightGBM
catches 36% of them (recall 0.361), even though it ranks clients well (ROC-AUC 0.776).
So the weak point is the decision, not the ranking. M3 tests three common fixes, each on
its own, on logistic regression and LightGBM.

#### 2. The three techniques

- **Class weights** (`params: {class_weight: balanced}`): every row is kept, but an
  error on a defaulter counts about 3.5 times (78/22) as much in the training loss. A
  model setting, not a resampler. XGBoost ignores `class_weight` (it uses
  `scale_pos_weight`), which is one reason it was left out of M3.
- **Random undersampling** (`sampler: undersample`): randomly drops non-defaulters from
  the training rows until the classes are equal. Only real rows, but a training fold
  shrinks from about 19,200 rows to about 8,500.
- **SMOTENC** (`sampler: smotenc`): adds synthetic defaulters until the classes are
  equal. Each new row sits between a real defaulter and one of its 5 nearest defaulter
  neighbours for the numeric columns, and takes the most common neighbour value for the
  categorical ones.

A **resampler** changes which rows the model trains on. It only touches training
folds; validation folds keep their real 22% default rate.

#### 3. How the pipeline is built

Each line in the `experiments` list of `configs/base.yaml` is one recipe. An optional
`sampler:` asks for the training rows to be resampled first:

```yaml
- {name: lgbm,         model: lgbm}                     # no resampling
- {name: lgbm_smotenc, model: lgbm, sampler: smotenc}   # resample, then train
```

`build_pipeline` in `src/credit_risk/pipeline.py` turns one line into a chain of steps:

```python
steps = [("scale", build_scaler())]            # a list of (label, step) pairs
if experiment.sampler is not None:             # only if the line named a sampler
    steps.append(("sample", get_sampler(experiment.sampler, seed)))
steps += [("encode", build_encoder()), ("model", model)]
return Pipeline(steps)                         # runs the steps in list order
```

| Config line | Steps |
|---|---|
| no `sampler:` | scale -> encode -> model |
| `sampler: smotenc` | scale -> **sample** -> encode -> model |

- **Python pieces:** `[...]` is a list (ordered, can grow); `("scale", ...)` is a tuple
  (a fixed pair: a label and the step object); `None` means "no value", which is what
  `sampler` holds when the config line leaves it out; `.append` adds one item to the
  end of a list, and `+=` adds several.
- **Why one function for every experiment:** all experiments share the same chain, and
  the `if` adds the resample step only where asked. Differences in the results table
  then come from the technique, not from different code paths.
- **Why imblearn's `Pipeline`:** it runs the resample step only while training. When a
  fold is scored or `predict_proba` is called, the step is skipped, so the model is
  judged on real rows with the real 22% default rate.
- **Resampling inside the folds, not before:** resampling all of train before
  cross-validation would let synthetic rows built partly from validation rows into
  training, and would change the validation class mix, so the scores would no longer
  describe real clients.

#### 4. Why SMOTENC and a two-step preprocessor

Plain SMOTE interpolates every column. After one-hot encoding that gives values like
`EDUCATION_2 = 0.4`, and on the PAY columns values like `PAY_1 = 1.4`, neither of which
can occur. SMOTENC handles listed columns as categories instead, so the PAY columns are
listed too (only for resampling; the model still sees them as numbers).

SMOTENC needs two things at once: amounts already scaled (it uses distances, and an
unscaled `LIMIT_BAL` would outweigh `AGE`), and categories not yet one-hot encoded. So
the M2 preprocessor was split into `build_scaler()` and `build_encoder()`, with the
resampler between them. A test pins the column order, and the M2 rows in the results
table came out unchanged to four decimals.

#### 5. Results (5-fold CV on train)

| experiment | ROC-AUC | PR-AUC | recall | precision |
|---|---|---|---|---|
| logreg | 0.722 | 0.503 | 0.236 | 0.720 |
| logreg_weighted | 0.723 | 0.500 | 0.647 | 0.377 |
| logreg_undersample | 0.723 | 0.499 | 0.649 | 0.374 |
| logreg_smotenc | 0.722 | 0.496 | 0.655 | 0.355 |
| lgbm | 0.776 | 0.555 | 0.361 | 0.668 |
| lgbm_weighted | 0.775 | 0.555 | 0.617 | 0.465 |
| lgbm_undersample | 0.773 | 0.538 | 0.653 | 0.436 |
| lgbm_smotenc | 0.766 | 0.542 | 0.431 | 0.598 |

ROC-AUC varies by about 0.004 to 0.008 between folds (the `_std` columns), so
differences smaller than that cannot be told apart from fold-to-fold variation.

- No technique improved ranking. ROC-AUC stayed within about 0.01 of the plain model
  (the largest drop was lgbm_smotenc, 0.011 below lgbm), and PR-AUC did not go up.
- Recall at 0.5 rose a lot (logreg 0.24 to about 0.65; lgbm 0.36 to 0.62 to 0.65 for
  weights and undersampling), and precision fell.
- Class weights did best overall: same PR-AUC as plain lgbm, much higher recall, and the
  simplest and cheapest to run.
- Undersampling cost lgbm some PR-AUC (0.538), likely because it trains on fewer than
  half the rows.
- SMOTENC was the weakest on lgbm: lowest ROC-AUC (0.766) and a much smaller recall
  gain (0.431). A possible reason, not checked: trees can learn the synthetic rows as
  their own region, so real defaulters' scores rise less. On logreg it matched the
  other two.

#### 6. Why ROC-AUC changes little

- ROC-AUC measures ranking: the chance that a random defaulter gets a higher score than
  a random non-defaulter. It does not use the 0.5 cutoff.
- The ROC curve plots the true positive rate (computed among defaulters only) against
  the false positive rate (computed among non-defaulters only), so the class ratio the
  model trained on does not enter it.
- Class weights and resampling mostly push everyone's score up together. The order of
  clients stays about the same, so ROC-AUC stays about the same.
- What changes is which clients cross 0.5. Raising all scores with a fixed 0.5 cutoff is
  roughly the same as keeping the original scores and lowering the cutoff. Example: four
  clients scored 0.10, 0.30, 0.45, 0.70 (the last two defaulted) become 0.25, 0.55,
  0.65, 0.85. Same order, so the same ROC-AUC, but recall at 0.5 goes from 50% to 100%
  and precision drops because of one new false alarm.

#### 7. What this means for later milestones

- The cutoff is the real lever. M5 picks it from the cost of each error type instead of
  using 0.5, which can reach a similar recall/precision trade-off without resampling.
- All three techniques push predicted probabilities above the true default rate, so
  they no longer read as real probabilities. M5's calibration step has to account for
  that if a weighted or resampled model is used.
