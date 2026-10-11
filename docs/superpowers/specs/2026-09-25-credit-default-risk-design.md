# Credit Default Risk: Design Spec

Date: 2026-09-25

## Purpose

A portfolio project that rebuilds a DSCI 573 (UBC MDS) group lab as a
modular, reproducible ML project: predicting whether a credit card
client defaults on next month's payment. It has two goals of equal
weight:

1. **Portfolio:** show the practices of a working ML project
   (importable package, config-driven pipeline, tests, CI, honest
   experiment comparisons, business-aware decision threshold) to
   hiring managers and technical reviewers, with a README that a
   non-technical reader can follow.
2. **Learning:** the author writes the ML-heavy modules themselves,
   with guidance and review, and can explain every design decision.

The original group lab (a single notebook) is a starting point for
ideas only. Code is rewritten, not copied.

## Background: what the original lab found

- Class balance is about 78% non-default / 22% default.
- Tuned LightGBM reached a test ROC-AUC of about 0.787. Published
  results on this dataset are roughly 0.77 to 0.80, so ranking quality
  was near what these features support.
- The weak point was the decision, not the ranking: all models used
  the default 0.5 probability cutoff, so results looked like a
  recall/precision trade-off or "barely beats 78% accuracy."
- Known defects to avoid: notebook cells not executed in the committed
  version, permutation importance called on a pipeline with
  pre-transformed data, `class_weight` passed to `XGBClassifier`
  (ignored), unstratified split, categorical codes treated as numeric.

The improvement goal is therefore framed as honest, like-for-like
comparisons plus a cost-based decision layer, not as chasing a higher
ROC-AUC. A technique that does not help is a valid, reported result.

## Scope

**In scope (this spec):**

- Layer 1, core: importable package, pipeline stages run by a
  Makefile, YAML config, tests, CI, README.
- Layer 2, better modelling: correct categorical handling,
  imbalance-technique comparison, feature engineering, probability
  calibration, cost-based threshold, SHAP explanations, limitations
  and fairness note.

**Out of scope (possible later specs, each separate):**

- Layer 3: experiment tracking (e.g. MLflow).
- Layer 4: serving or demo (FastAPI or Streamlit), hosting.
- Docker, mypy, cloud compute. The dataset (30,000 rows, 24 columns)
  trains in seconds to minutes on a laptop.

## Success criteria

- `make setup && make all` runs end to end on a laptop from a fresh
  clone with no manual steps.
- GitHub Actions runs `ruff` and `pytest` (including a smoke test of
  the full pipeline on a small sample) on every push and passes.
- The test set is read by one stage only (`evaluate`), once.
- The experiments table compares each technique against the same
  baseline under the same CV folds.
- README lets a reader understand the problem, headline results, and
  trade-offs within about 2 minutes, and credits the original group.
- `docs/learning-log.md` has an entry for each milestone the author
  owned.

## Attribution

New repository with fresh history. The README states that the project
grew out of a DSCI 573 group lab with Vy and Christine, that the
pipeline, feature engineering, experiments, and write-up in this repo
are the author's own work, and links the original if it becomes
public. Lab solution code is not reposted. The author should confirm
the course's policy on publishing derived work.

## Tooling

> Changed 2026-09-25 after M0: switched from uv to conda so the author
> works with a familiar environment tool while learning the rest.
> `environment.yml` lists dependencies; `pyproject.toml` only makes the
> package installable (`pip install -e .` from inside environment.yml).

- Python 3.12, conda (conda-forge) with `environment.yml`.
- Package in `src/` layout, installed in editable mode by the
  `pip: [-e .]` entry in `environment.yml`, with a `credit-risk`
  console script.
- `ruff` (lint and format), `pre-commit`, `pytest`.
- GitHub Actions CI.
- Makefile as the runner: `setup`, `lint`, `test`, and one target per
  pipeline stage, plus `all`.
- Main libraries: pandas, scikit-learn, imbalanced-learn, lightgbm,
  xgboost, shap, pyyaml, matplotlib/altair. Parquet via pyarrow.

## Repository layout

```
credit-default-risk/
├── environment.yml, pyproject.toml, Makefile, README.md, .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── configs/
│   ├── base.yaml          # paths, seed, CV folds, test size, cost matrix,
│   │                      # experiments, decision candidates, final setup
│   └── smoke.yaml         # small sample, few folds, for the smoke test
├── data/
│   ├── raw/UCI_Credit_Card.csv   # committed (about 2.8 MB)
│   └── processed/                # gitignored
├── src/credit_risk/
│   ├── config.py, data.py, features.py, preprocess.py, models.py,
│   ├── pipeline.py, experiments.py, decision.py, evaluate.py,
│   ├── explain.py, cli.py
├── tests/
├── notebooks/             # exploration and results write-up; import from the package
├── reports/{tables,figures}/     # small outputs, committed
├── artifacts/             # saved models, gitignored
└── docs/
    ├── learning-log.md
    └── superpowers/{specs,plans}/
```

## Pipeline stages

Each stage reads files and writes files, so it can be run and
inspected on its own. Notebooks read these outputs; they are not
pipeline stages.

| Target | Input | Output |
|---|---|---|
| `make data` | `data/raw/UCI_Credit_Card.csv` | hash check passes (fails with a clear message if the file changed) |
| `make split` | raw CSV | `data/processed/train.parquet`, `test.parquet` |
| `make experiments` | train only | `reports/tables/experiments.csv` |
| `make decide` | train only | `reports/tables/decision.csv`, `reports/figures/cost_vs_threshold.png` |
| `make train` | train only | `artifacts/model.joblib` (pipeline + calibration + threshold) |
| `make evaluate` | model + test | `reports/tables/test_metrics.csv`, figures |
| `make explain` | model + test | SHAP figures in `reports/figures/` |
| `make all` | | all of the above in order |

Stage details:

- **split:** validate schema, handle undocumented codes (EDUCATION 0,
  5, 6; MARRIAGE 0; PAY_* values -2 and 0), rename `PAY_0` to
  `PAY_1`, stratified 80/20 split with a fixed seed.
- **experiments:** all comparisons use the same stratified CV folds on
  train. Metrics: ROC-AUC, PR-AUC, recall, precision, F1, fit time.
- **decide (added in M5):** for each candidate setup, calibration
  method (sigmoid, isotonic) and cost ratio listed under `decision` in
  the config, fit a calibrated model with a cost-tuned threshold and
  record the threshold and its cross-validated expected cost. The
  author reads the table and plot, then sets `final` and `costs`.
- **train:** fit the `final` setup on all of train. Calibration and the
  cost-based threshold are chosen from out-of-fold predictions on
  train, never from test. Built from scikit-learn wrappers:
  `CalibratedClassifierCV` (sigmoid or isotonic, `ensemble=False`, so
  one fitted pipeline is kept for SHAP) inside
  `TunedThresholdClassifierCV` (scored by `expected_cost`).
- **evaluate:** ROC-AUC, PR-AUC, recall and precision at the chosen
  threshold, expected cost, calibration plot. The only stage that
  reads `test.parquet`.
- **explain:** SHAP computed on transformed features against the fitted
  model step, global summary plus one or two individual clients.

## Modules

"Author" means the author writes it with guidance and review. "Claude"
means Claude writes it and explains the decisions.

| Module | Responsibility | Public interface | Owner |
|---|---|---|---|
| `config.py` | Load and validate YAML into a typed config object | `load_config(path) -> Config` | Claude |
| `data.py` | Load, validate, clean, split, load splits | `load_raw()`, `clean(df)`, `split(df, cfg)`, `load_split(name)` | Claude |
| `preprocess.py` | ColumnTransformer: scale numerics, one-hot true categoricals, keep PAY_* ordinal | `build_preprocessor(cfg)` | Claude |
| `features.py` | Engineered features as an sklearn transformer | `FeatureEngineer` (fit/transform) | Author |
| `models.py` | Registry of estimators by name, params from config | `get_model(name, params)` | Author |
| `pipeline.py` | Compose features, preprocessor, optional resampler, model into one imblearn `Pipeline` | `build_pipeline(cfg)` | Author and Claude together |
| `experiments.py` | Run CV comparisons from config | `run_experiments(cfg) -> DataFrame` | Author |
| `decision.py` | Expected cost, calibration and cost-based threshold (M5) | `expected_cost(y_true, y_pred, costs)`, `build_decision_model(...)`, `run_decision(cfg)` | Claude (author chose this in M5) |
| `evaluate.py` | Test-set metrics and calibration plot (M6) | `evaluate(model, X, y)` | Author |
| `explain.py` | SHAP global and per-client explanations | `shap_summary(...)`, `explain_client(...)` | Author |
| `cli.py` | Subcommands for each stage | `credit-risk split / experiments / train / evaluate / explain` | Claude |

Leakage rule: every fitted step (feature engineering, scaling,
resampling) lives inside the single pipeline, so CV refits it per fold
and resampling applies to training folds only.

Cost matrix: lives in `base.yaml` as the cost of a false negative
(missed defaulter) and a false positive (false alarm). Starting values
(e.g. 5 and 1) are chosen and justified in M5, and the README shows
how the threshold shifts as the ratio changes. `make decide` compares
the false negative costs listed under `decision` (3, 5, 10, with the
false positive cost fixed at 1).

## Error handling

Light, since this is not a production service:

- Validate the raw data once at the edge (`data.py`): expected
  columns and types, clear error on mismatch.
- Config is validated at load time. Unknown model or technique names
  fail with a message listing valid options.
- Each stage checks its inputs exist and names the `make` target to
  run if they do not.

## Testing

Tests check behaviour, not scores.

- **Data:** no undocumented codes after `clean()`; split preserves
  class ratio within a small tolerance; no ID overlap between train
  and test.
- **Features:** each engineered feature against a small hand-made
  DataFrame with a known answer, including edge cases (zero credit
  limit gives a finite value).
- **Pipeline:** fits and predicts on a small fixture in seconds;
  probabilities in [0, 1]; resampler only affects fit, not predict.
- **Cost:** `expected_cost()` on a toy example with a hand-worked
  answer. The threshold search itself is scikit-learn's
  (`TunedThresholdClassifierCV`); tests check that `predict` matches
  `predict_proba >= threshold`.
- **Smoke:** `make all` with `configs/smoke.yaml` (about 1,000 rows,
  2 folds) runs end to end in CI.

The author practises writing tests before implementation for the
modules they own.

## Milestones

Each milestone ends with a working, reviewable commit or PR. Only M0
gets an implementation plan now; each later milestone gets its own
plan when reached.

| # | Milestone | Owner | Done when |
|---|---|---|---|
| M0 | Scaffold: conda env, ruff, pre-commit, CI, Makefile, `config.py`, `cli.py` stubs, raw CSV, README skeleton with attribution, learning log | Claude | `make setup lint test` passes locally and in CI with stub stages |
| M1 | Data: `clean()`, stratified split, data tests; then the author's EDA notebook focused on quirks and feature ideas | Claude, then author | `make split` produces both files and tests pass |
| M2 | Baseline rebuild: `preprocess.py` (Claude), `models.py` (author), `pipeline.py` (together), minimal `experiments.py` with dummy, logistic regression, LightGBM | Mixed | CV ROC-AUC for LightGBM is near the lab's (about 0.78), confirming the setup |
| M3 | Imbalance experiments: class weights vs SMOTE vs random undersampling, PR-AUC added | Author | Comparison table plus a short explanation of why ROC-AUC changes little |
| M4 | Feature engineering: utilization, payment-to-bill ratios, months delinquent, trend features | Author | With vs without comparison under the same CV |
| M5 | Decision layer: calibration, cost-based threshold, `make decide`, `make train` | Claude (author chose this on 2026-10-11) | Comparison table and cost vs threshold plot; saved model with threshold |
| M6 | Test and explain: `make evaluate`, `make explain` | Author | Final metrics and SHAP figures |
| M7 | Write-up: README results, figures, limitations and fairness note | Author writes, Claude edits | README meets the 2-minute criterion |

## Learning process (milestones the author owns)

1. Concept brief from Claude: a short explanation plus a pointer to
   the relevant course lecture or library documentation.
2. Task and test outline: Claude states what the function must do and
   sketches cases; the author writes the tests.
3. Author explores in a notebook (with `%autoreload 2`), then moves
   the code into the module.
4. Author opens a PR on a branch; Claude reviews with questions rather
   than rewrites.
5. Author adds an entry to `docs/learning-log.md`.

## Limitations and fairness (M7)

The dataset includes SEX, MARRIAGE, and AGE. Using these in credit
decisions raises legal and ethical concerns in many jurisdictions. M7
includes a short section comparing performance with and without these
features and discussing the implications.

## Open questions

None blocking. Cost matrix values and the final setup are decided in
M5, after reading `make decide`'s output.
