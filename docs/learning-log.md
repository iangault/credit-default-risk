# Learning log

One entry per milestone, explaining what was built and why, as if in an interview.

## Template

### Mx: <title> (<date>)

- **What I built:**
- **Key concept, explained simply:**
- **A decision I made and why:**
- **What surprised me or went wrong:**
- **What I would do differently:**

---

### M2: Baseline models (2026-10-09)

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

### M3: Class imbalance (2026-10-09)

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

---

### M4: Feature engineering (2026-10-10)

#### 1. The idea

Feature engineering means building new columns from the existing ones, so a pattern the
model would have to work out for itself is handed to it directly. Example: the raw data
has six PAY columns; `months_late` counts how many of them show a late payment.

Logistic regression is expected to gain more than LightGBM:
- Logistic regression gives each column one weight and adds them up. It cannot count
  across columns, divide one by another, or compare months by itself.
- Trees split one column at a time. They can build something close to "late in several
  months" from a few splits, but rebuilding a ratio such as bill / limit takes many.

#### 2. The four groups

| Group | Columns | Formula | Evidence (EDA section 6) |
|---|---|---|---|
| delinquency | `months_late`, `max_delay` | count of PAY_1..6 >= 1; max of PAY_1..6 | strongest single signals (AUC 0.72 and 0.71) |
| payment | `pay_ratio`, `no_bill`, `months_zero_pay` | log1p(PAY_AMT1 / BILL_AMT2), 0 when no bill; flag for BILL_AMT2 <= 0; count of PAY_AMT1..6 == 0 | paying nothing doubles the default rate; defaulters over-represented at zero payment |
| utilization | `utilization` (+ 10 bins) | BILL_AMT1 / LIMIT_BAL | U-shaped link to default |
| trend | `util_change`, `delay_trend` | (BILL_AMT1 - BILL_AMT6) / LIMIT_BAL; mean PAY_1..3 minus mean PAY_4..6 | is debt growing, is the client falling further behind |

#### 3. What adding features changes, and how the system handles it

**Analogy:** think of the pipeline as an assembly line and each config line as an order
ticket. The ticket says which model to build (`model:`), whether to adjust the parts mix
(`sampler:`), and which extra parts to bolt on (`features:`). Before the line starts, a
parts list is written from the ticket, and every station gets a copy, so the painting
station knows to paint the new parts too and the packing station knows where they go in
the box. A ticket with no extras gets the same line as before.

What could go wrong when new columns appear, and what handles it:

| Issue | How it is handled |
|---|---|
| Later steps select columns by name and drop anything not listed (`remainder="drop"`), so new columns would be silently lost | `feature_columns(groups)` turns the group names into column names (the "parts list") from one table, `FEATURE_GROUPS` in `features.py`. `build_pipeline` passes that list to `build_scaler(extra)` and `build_encoder(extra, binned)` |
| The number of columns changes per experiment | Each experiment gets its own pipeline, built fresh. The model sees 28 columns with no features, 30 with delinquency, 39 with utilization (1 column + 10 bins), 46 with all four. A model only needs the same columns at fit and predict time, which the pipeline ensures since both go through the same steps |
| Column order | Columns are selected by name in fixed groups: amounts, engineered columns, one-hot columns, bins, PAY codes |
| A typo such as `features: [delinquncy]`, or a group listed twice | Raises an error when the pipelines are built, before any CV runs |
| Leakage between folds | The features use only the client's own row, so they learn nothing. The utilization bin edges do learn from data, so they live inside the encode step and are fitted on each training fold only |
| Divide by zero | `BILL_AMT2` is 0 or negative for about 10% of clients; their `pay_ratio` is set to 0 and `no_bill` flags them. `LIMIT_BAL` is never 0 |
| Older results changing | The `features` step is added only when the list is non-empty, so M2 and M3 rows build the same pipeline. The 11 old rows came out identical |
| SMOTENC with features | Refused with an error: it would interpolate counts such as `months_late` into values like 2.4 |

**Model choice is separate from features.** `model:`, `sampler:` and `features:` are
independent settings on the same line, so any model can be combined with any feature
list. The results table records all three for each row, so each set of numbers can be
traced to the setup that produced it.

**The `+` in the results table.** In Python, an experiment's features are a list such as
`("delinquency", "trend")`. A CSV cell holds one piece of text, so the row stores them
joined into one label:

```python
"features": "+".join(experiment.features) or "none",
```

- `"+".join(...)` puts a `+` between the items: `"delinquency+trend"`. It is a label,
  not addition; nothing reads it back to build a pipeline.
- With no features, the join gives an empty string `""`, which Python treats as false,
  so `or "none"` writes `none` instead of an empty cell (the same style as `sampler`).

**How `build_pipeline` assembles the steps** (`src/credit_risk/pipeline.py`; the Python
pieces are explained in M3, section 3). M4 adds one more `if` and passes `extra` along:

```python
extra = feature_columns(experiment.features)   # e.g. ["months_late", "max_delay"], or []
binned = [c for c in BINNED if c in extra]     # ["utilization"] when that group is on
steps = []
if extra:                                      # true only if the list is not empty
    steps.append(("features", build_feature_step(experiment.features)))
steps.append(("scale", build_scaler(extra)))   # also scale the new columns
if experiment.sampler is not None:
    steps.append(("sample", get_sampler(experiment.sampler, seed)))
steps += [("encode", build_encoder(extra, binned)), ("model", model)]  # also keep them
return Pipeline(steps)
```

| Config line | `extra` | `if extra:` | Final steps |
|---|---|---|---|
| `logreg_weighted` (no `features:`) | `[]` | skipped | scale -> encode -> model (same as M3) |
| `logreg_weighted_delinquency` | `["months_late", "max_delay"]` | adds the step | **features** -> scale -> encode -> model |

The `if` lines decide which stations are on the line for this order. Passing `extra` to
the scaler and encoder is the note to the later stations: "two new parts are coming;
keep them." Without it, the encoder would drop `months_late` with no error, and the
model would never see it.

#### 4. Why logistic regression cannot follow a U-shape

- One weight per column means a column's effect on the log-odds of default is a straight
  line: as utilization rises, predicted risk can only rise steadily or fall steadily.
- Utilization is U-shaped: default is higher for clients using almost none of their
  limit, lowest around 1% to 13%, and highest at or over the limit. A straight line can
  follow one side of the U, not both.
- **Bins** split utilization into ranges, each its own 0/1 column with its own weight.
  Worked example with three bins and made-up weights: low +0.3, middle -0.4, high +0.8.
  A client is in one bin, so only that weight applies: a low-use client gets +0.3, a
  middle one -0.4, a high one +0.8. That traces the U, which one weight cannot.
- The project uses 10 quantile bins (about the same number of clients in each) and keeps
  the raw column too.
- Trees do not need bins: they can split utilization at several points on their own.

#### 5. Results (5-fold CV on train)

| experiment | ROC-AUC | PR-AUC | recall | precision |
|---|---|---|---|---|
| logreg_weighted (no features) | 0.723 | 0.500 | 0.647 | 0.377 |
| + delinquency | 0.756 | 0.504 | 0.618 | 0.453 |
| + payment | 0.747 | 0.500 | 0.615 | 0.438 |
| + utilization | 0.746 | 0.491 | 0.621 | 0.437 |
| + trend | 0.723 | 0.499 | 0.632 | 0.383 |
| + all four | 0.762 | 0.503 | 0.616 | 0.454 |
| lgbm_weighted (no features) | 0.775 | 0.555 | 0.617 | 0.465 |
| + delinquency | 0.778 | 0.559 | 0.620 | 0.470 |
| + payment | 0.776 | 0.554 | 0.617 | 0.472 |
| + utilization | 0.776 | 0.553 | 0.616 | 0.462 |
| + trend | 0.775 | 0.553 | 0.615 | 0.466 |
| + all four | 0.777 | 0.556 | 0.615 | 0.471 |

ROC-AUC varies by about 0.005 to 0.008 between folds (the `_std` columns), so smaller
differences cannot be told apart from fold-to-fold variation.

#### 6. What it shows

- **Logistic regression gained, as expected.** All four groups lifted ROC-AUC from 0.723
  to 0.762. Delinquency alone gave most of it (+0.033); payment and utilization about
  +0.023 each; trend nothing.
- **PR-AUC for logistic regression barely moved** (0.500 to 0.503), even though ROC-AUC
  rose. A possible reason, not checked: the features improved the ordering lower down
  the list more than at the top, where PR-AUC puts most of its weight. Utilization
  alone lowered it slightly (0.491), within fold-to-fold variation.
- **Precision rose** for logistic regression (0.377 to about 0.45) with a small drop in
  recall (0.647 to about 0.62), at the same 0.5 cutoff.
- **LightGBM did not gain beyond fold-to-fold variation.** The largest change was
  +0.003 ROC-AUC (delinquency). It was already finding these patterns in the raw columns.
- **For M5:** LightGBM with class weights is still the strongest. The features narrow the
  gap to logistic regression from 0.052 to 0.015 ROC-AUC, which matters if an
  easier-to-explain model is wanted. Which one to carry forward is a decision for M5.

---

### M5: Decision layer (2026-10-11)

#### 1. The problem

Every result up to M4 turned scores into yes/no decisions with a 0.5 cutoff. That is
scikit-learn's default, not a choice. M3 showed the effect: class weights pushed scores
up, so with the cutoff fixed at 0.5, recall rose and precision fell. M5 replaces the
fixed cutoff with one chosen from what each kind of mistake costs.

#### 2. Costs, calibration, threshold

**Analogy: a weather forecast and an umbrella.**

| Piece | Weather | Credit |
|---|---|---|
| Model score | The forecaster's raw number | LightGBM's output for a client |
| Calibration | Making "30% rain" true: it rains on about 30% of such days | Making 0.30 mean about 30 in 100 such clients default |
| Costs | Getting soaked (5) vs carrying an umbrella for nothing (1) | A missed defaulter (5) vs a false alarm on a good client (1) |
| Threshold | "Take the umbrella above X%" | "Flag the client above X" |

Calibration is about whether the number is honest, costs are a business judgment, and
the threshold is the rule that combines them.

**The rule of thumb.** For a client with an honest probability p of default, flagging
costs 1 × (1 - p) (a false alarm if they would have paid) and not flagging costs 5 × p
(a miss if they default). Flag when 5p > 1 - p, so when p > 1/6, about 0.17. The code
does not use this formula; it searches for the cheapest cutoff. The formula is a check.

**Why calibrate if the threshold could just move?** For one decision, a threshold on raw
scores would give the same calls. Calibration adds two things: probabilities that can be
reported ("an 18% chance of default"), and a threshold that can be explained from the
costs instead of taken on trust.

**The threshold is not a hyperparameter in the usual sense.** Hyperparameters (tree
depth, `class_weight`) act during training and need a refit to change. The threshold
acts on the finished scores, so changing it needs no retraining. It is chosen the same
honest way: from predictions on rows the model did not train on, never from test.

#### 3. What each wrapper does

`build_decision_model` in `src/credit_risk/decision.py` stacks three layers:

```
TunedThresholdClassifierCV       outer: picks the cutoff with the lowest expected cost
  CalibratedClassifierCV         middle: maps scores to real default rates
    build_pipeline(experiment)   inner: the same pipeline as M2 to M4
```

What `.fit(X, y)` does, with 5 folds:

1. **Outer:** for each fold, fit the middle layer on the other 4, get calibrated
   probabilities for the held-out fold, and compute the cost at each of 100 cutoffs.
   Average over the folds, keep the cheapest cutoff, and refit the middle layer on all
   rows.
2. **Middle:** get out-of-fold scores from the inner pipeline (its own 5 folds), fit the
   mapping from score to default rate on them (sigmoid: a smooth S-curve; isotonic: a
   staircase that only goes up), then refit the inner pipeline on all its rows.
3. **Inner:** features (if any), scaling, encoding, model.

The threshold is therefore picked on rows that neither the calibration nor the model saw.
The inner pipeline is fitted 36 times per setup, which is why `make decide` (18 setups)
takes about 5 minutes.

Settings worth knowing:
- **`ensemble=False`:** keep one inner pipeline refit on all rows, rather than five
  averaged ones. One model keeps M6's SHAP explanations simple.
- **`greater_is_better=False`** in `cost_scorer`: scikit-learn always picks the highest
  score, so the cost is negated (-0.55 beats -0.60). Without it, the search would pick
  the most expensive cutoff with no error. A test checks the sign.
- **`store_cv_results=True`:** keeps the cost at every cutoff tried, which is what the
  cost vs threshold plot draws.

#### 4. Results (5-fold CV on train)

Expected cost per client at 5:1 (lower is better):

| Setup | Threshold | Expected cost |
|---|---|---|
| lgbm, sigmoid | 0.170 | 0.555 |
| lgbm, isotonic | 0.162 | 0.557 |
| lgbm_weighted, sigmoid | 0.176 | 0.560 |
| lgbm_weighted, isotonic | 0.162 | 0.561 |
| logreg_weighted_allfeatures, sigmoid | 0.164 | 0.573 |
| logreg_weighted_allfeatures, isotonic | 0.168 | 0.573 |
| *Flag nobody* | | 1.106 |
| *Flag everyone* | | 0.779 |

The chosen setup (lgbm, sigmoid) at each cost ratio:

| Cost of a miss | Rule of thumb | Threshold | Expected cost | Flag everyone |
|---|---|---|---|---|
| 3 | 0.25 | 0.26 | 0.407 | 0.779 |
| 5 | 0.17 | 0.17 | 0.555 | 0.779 |
| 10 | 0.09 | 0.09 | 0.718 | 0.779 |

Full table: `reports/tables/decision.csv`; curves: `reports/figures/cost_vs_threshold.png`.

#### 5. What it shows, and the choice

- **Calibration worked.** All 18 thresholds land within about 0.04 of the rule of thumb
  (0.24 to 0.29 at 3:1, 0.16 to 0.18 at 5:1, 0.09 to 0.11 at 10:1).
- **Class weights add nothing once the threshold is tuned.** Plain `lgbm` is cheapest or
  tied at every ratio; `lgbm_weighted` is 0.003 to 0.005 higher, close to fold-to-fold
  variation. The threshold does the job the weights did in M3.
- **Sigmoid and isotonic are about the same** for LightGBM (within about 0.007).
- **Logistic regression costs 2% to 3% more** per client at 3:1 and 5:1 (0.008 to 0.018),
  the price of the easier-to-audit model.
- **At 10:1 the model adds little.** The best cutoff flags most clients, and the cost
  (0.718) is not far below flagging everyone (0.779).

**Choice:** `lgbm` with sigmoid calibration, at a 5:1 cost ratio.
- Setup: lowest cost at 3:1 and 5:1, no class weights needed, and sigmoid is the less
  flexible method, so less likely to overfit the calibration. SHAP (M6) works well on
  it: `TreeExplainer` on the 28 original columns.
- Costs: 5:1 is the ratio in the documented cost matrix of the Statlog (German Credit)
  dataset, a common reference for credit scoring. `make decide` reports 3:1 and 10:1 so
  the effect of that choice is visible.
- Saved by `make train`: threshold 0.170, cross-validated expected cost 0.555 per client.

#### 6. Limitations

- **The reported cost is slightly optimistic.** Each setup's cost is the best of 100
  cutoffs, scored on the same folds used to pick it. The curves are smooth near their
  minimum, so the effect should be small. M6's test-set score is the unbiased number.
- **No fold-to-fold spread.** `TunedThresholdClassifierCV` keeps only the mean cost at
  each cutoff, so differences under about 0.005 cannot be told apart from noise.
- **The costs are a judgment, not data.** Real costs depend on balances, recovery rates
  and client value, which this dataset does not include.

---

### M6: Test and explain (2026-10-11)

#### 1. Why the test set is read only now

Every choice so far (model, features, class weights, calibration, threshold, costs) was
made on train. The test set (6,000 clients split off in M1) was not read, so its score
estimates how the model does on clients it has never seen. If test results were used to
adjust anything, the test set would become part of the tuning and its score would be
optimistic.

The spec rule, reworded in M6: the test set is read only by `evaluate` and `explain`;
neither changes the model, and no setting is changed after seeing their output. That makes
it a process commitment: a disappointing result is reported, not tuned away. A safeguard
supports it: `make train` records its settings in `artifacts/model.json` (the final
experiment's full definition, calibration, costs, folds, and split settings), and
`evaluate` and `explain` refuse a model trained under different settings. Recording the
split settings matters most: a new split could put rows the model trained on into test.

#### 2. Results on the test set (1,327 of 6,000 clients defaulted, 22.1%)

| Setup | Cost per client | ROC-AUC | PR-AUC | Recall | Precision |
|---|---|---|---|---|---|
| Model at tuned threshold (0.17) | 0.534 | 0.788 | 0.554 | 0.766 | 0.381 |
| Same model at 0.5 | 0.751 | 0.788 | 0.554 | 0.356 | 0.673 |
| Flag nobody | 1.106 | | | 0 | 0 |
| Flag everyone | 0.779 | | | 1.000 | 0.221 |
| Train CV estimate | 0.555 | 0.776 | 0.555 | | |

- **The train estimate held up.** Test cost 0.534 vs 0.555 estimated; ROC-AUC 0.788 vs
  0.776. Both slightly better on test, within sampling noise (section 6). ROC-AUC matches
  the original lab's tuned LightGBM (about 0.787).
- **The M5 threshold is worth 29%.** At 0.5 the same scores cost 0.751 per client.
- **In plain numbers:** the model flags 2,666 of 6,000 clients (44%), catches 1,016 of the
  1,327 defaulters (77%), misses 311, and raises 1,650 false alarms (35% of the clients
  who would have paid). That is what a 5:1 cost ratio implies; "flag" fits a review or a
  lower limit better than an outright rejection.

#### 3. Calibration on unseen clients

`reports/figures/calibration_test.png` groups test clients into 10 equal-size groups by
predicted probability and compares each group's average prediction with its actual default
rate. The points stay close to the diagonal: the largest gap is a group predicted at about
42% that defaulted at 47%, and at the low end about 7% predicted came out at 4.5%. So the
probabilities can be read as probabilities on new clients, not only on train.

#### 4. SHAP, explained simply

**Analogy: splitting a group bill.** Several friends share a meal and want to split the
bill by what each one added. The order they arrived in changes how much each one seems to
add (the second person to order dessert adds less to a shared platter). Shapley values
average each person's addition over every possible arrival order, which gives one fair
share each. SHAP does the same for columns: each column's share of one client's score,
averaged over every order in which the columns could be added.

- **Additivity:** base value (the average score) + the client's SHAP values = the client's
  score. A test checks that the two sides add up.
- **Units:** log-odds of the LightGBM score before calibration. +0.7 multiplies the odds of
  default by about 2. Sigmoid calibration keeps the order of clients, so "pushes risk up"
  means the same after calibration.
- **Reading the beeswarm** (`shap_summary.png`): one row per column, most important at the
  top; one dot per test client, placed left (lower risk) or right (higher risk) by its
  SHAP value, colored by the column's value (red high, blue low).
- **Reading a waterfall** (`shap_client_*.png`): one client, starting from the base value
  at the bottom; each bar is one column pushing the score up (red) or down (blue); the top
  is the client's score. The plots show original values (LIMIT_BAL = 50000), not the
  scaled ones the model sees.

#### 5. What drives the model

Top columns by mean absolute SHAP value (`reports/tables/shap_importance.csv`):

| Column | Mean \|SHAP\| | Reading |
|---|---|---|
| PAY_1 | 0.480 | Last month's repayment status; a late payment pushes risk up sharply |
| LIMIT_BAL | 0.203 | Higher credit limit, lower risk |
| BILL_AMT1 | 0.164 | Latest bill amount |
| PAY_AMT2 | 0.153 | Larger recent payments, lower risk |
| PAY_AMT1 | 0.109 | Same |

PAY_1 moves scores more than twice as much as any other column, which matches the EDA.

**The two clients** (picked by a fixed rule: the client nearest each group's median
probability):
- **Flagged defaulter, client 18048 (probability 0.46).** Paid on time last month
  (PAY_1 = -1 pulls risk down), but was 2 months late in each of the two months before and
  paid nothing in four of the last months. The model added these up correctly.
- **Missed defaulter, client 3448 (probability 0.12).** "Paid duly" in every month shown,
  a 150,000 limit, and a 34,672 payment six months ago. Only the bill amounts pushed risk
  up (+0.20 and +0.09), and not by enough. Some defaults have no warning sign in this data, and a threshold low enough to
  catch them would flag many more good clients.

**Demographics, for M7:** `SEX_2` (female) ranks 10th of 28 and nudges scores down
slightly. `MARRIAGE_1` ranks 20th, `AGE` 21st, and `EDUCATION_4` 22nd. The model uses
these columns; whether it should is the fairness question M7 takes up.

#### 6. Limitations

- **One test set gives one number.** The cost per client has a standard error of about
  0.015 (from the counts: miss cost 5, false alarm cost 1, 6,000 clients), so a rough 95%
  range is ±0.03. The gap between test (0.534) and the train estimate (0.555) is within it.
  No confidence intervals are computed beyond this.
- **SHAP explains the model, not the world.** It shows what the model uses, not what causes
  default. A column can matter to the model because it stands in for something unmeasured.
- **Two waterfalls are two clients.** They illustrate how the model reasons; the beeswarm
  and importance table are the summary.
- **Unseen categories in small samples.** The smoke run and tiny test projects can give a
  cross-validation fold no clients with a rare code (MARRIAGE = 3 is 1.2% of train). The
  encoder sets such a client to all zeros by design and warns. It does not occur at full
  size: inner training folds of about 15,000 rows hold about 180 such clients.
