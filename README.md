# credit-default-risk

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://credit-default-risk-2005-taiwan.streamlit.app/)
[![CI](https://github.com/iangault/credit-default-risk/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/iangault/credit-default-risk/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Data: UCI](https://img.shields.io/badge/data-UCI%20ML%20Repository-lightgrey.svg)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

Predicting whether a credit card client will miss next month's payment, built as a
modular, reproducible ML pipeline.

Results: [summary below](#results), full walkthrough in [docs/results.md](docs/results.md).

**Try the model:** [credit-default-risk-2005-taiwan.streamlit.app](https://credit-default-risk-2005-taiwan.streamlit.app/).
It runs on Streamlit Community Cloud's free tier, so if nobody has used it recently it
can take about 30 seconds to wake up.

## The problem

The [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
dataset has 30,000 clients from Taiwan (2005) with credit limit, demographics, six months
of repayment status, bill amounts, and payment amounts. About 22% of clients default.
The goal is to rank clients by default risk and choose a decision threshold based on the
cost of each type of error, rather than the default 0.5 cutoff.

## Results

The model flags a client when their probability of default is above a cutoff. The
cutoff comes from the cost of each mistake rather than the usual 0.5: a missed defaulter
is assumed to cost 5 times a false alarm, which puts the cutoff at about 0.17.

On 6,000 held-out clients, read once at the end:

| | Cutoff 0.17 (cost-based) | Cutoff 0.5 (default) | Flag everyone |
|---|---|---|---|
| Defaulters caught (of 1,327) | 1,016 (77%) | 472 (36%) | 1,327 |
| False alarms (of 4,673 who paid) | 1,650 (35%) | 229 (5%) | 4,673 |
| Cost per client (miss = 5, false alarm = 1) | **0.534** | 0.751 | 0.779 |

- Choosing the cutoff from the costs lowers the cost per client by 29%, with the same
  model. Catching more defaulters means flagging more clients who would have paid.
- Ranking quality (ROC-AUC 0.788) is in the range published for this dataset. Class
  weights, resampling and engineered features each changed it by about 0.01 or less.
- Last month's repayment status drives predictions more than any other column. Sex,
  marital status and age are used too, which is a fairness concern.

![Cost per client at each cutoff, for three cost ratios](reports/figures/cost_vs_threshold.png)

The full walkthrough, with all the figures, limitations and the fairness note, is in
**[docs/results.md](docs/results.md)**.

## Quickstart

Requires [conda](https://conda-forge.org/download/) (Miniforge recommended) and `make`.

```bash
make setup                # create the conda env from environment.yml, install git hooks
conda activate credit-risk
make test                 # run the test suite
make all                  # run the pipeline end to end
make app                  # run the Streamlit app locally (uses the committed model)
```

Each stage can also be run on its own (`make data`, `make split`, ...). Use a different
config with `make all CONFIG=configs/smoke.yaml`.

## Project layout

```
configs/          settings for every stage (paths, seed, CV folds, error costs)
data/raw/         the original CSV (committed; checked by sha256)
src/credit_risk/  the package: one module per pipeline responsibility
tests/            pytest suite, including a small end-to-end smoke run
notebooks/        exploration; imports from the package
docs/             results write-up, design spec, plans, and a learning log
```

## How the code fits together

The modules in `src/credit_risk/`, in the order the pipeline uses them:

| Order | Module | What it does | Run with |
|---|---|---|---|
| | `config.py` | Reads settings from `configs/*.yaml`; used by every step | |
| 1 | `data.py` | Load, clean, and split into train/test | `make data`, `make split` |
| 2 | `features.py` | Add engineered columns (late months, payment ratio, utilization, trends) | (inside the pipeline) |
| 3 | `preprocess.py` | Scale amounts (step 1), one-hot encode categories (step 2) | (inside the pipeline) |
| 4 | `samplers.py` | Turn a sampler name into a resampler that evens out the classes | (inside the pipeline) |
| 5 | `models.py` | Turn a model name from the config into an unfitted model | (inside the pipeline) |
| 6 | `pipeline.py` | Chain optional features, scaling, optional resampling, encoding, and a model into one object | (inside experiments) |
| 7 | `experiments.py` | Cross-validate each setup and write a results table | `make experiments` |
| 8 | `decision.py` | Calibrate a setup, tune its threshold to the error costs, compare candidates, save the final model | `make decide`, `make train` |
| 9 | `evaluate.py` | Score the saved model on the test set; compare with 0.5, baselines and the train estimate | `make evaluate` |
| 10 | `explain.py` | SHAP values for the saved model; beeswarm and two example clients | `make explain` |
| 11 | `showcase.py` | The logic behind the Streamlit app (`app/app.py`): example clients, payment scenarios, scoring, test-set counts | `make app` |
| | `cli.py` | The `credit-risk` command; runs one stage at a time | |

`make` targets call the `credit-risk` command, which calls the modules. Run `make help`
for the list of targets.

## Where to find things

| Question | Where to look |
|---|---|
| What settings were used (seed, CV folds, which models)? | `configs/base.yaml` |
| How is the raw data cleaned? | `clean()` in `src/credit_risk/data.py` |
| How are features scaled or encoded? | `src/credit_risk/preprocess.py` |
| Which models are compared, with which settings? | `src/credit_risk/models.py` |
| How is class imbalance handled? | `src/credit_risk/samplers.py` and the M3 rows in `configs/base.yaml` |
| Which engineered features exist, and how are they built? | `src/credit_risk/features.py` and the M4 rows in `configs/base.yaml` |
| How are models scored? | `src/credit_risk/experiments.py` |
| How is the decision threshold chosen? | `src/credit_risk/decision.py` and the `decision` / `final` sections of `configs/base.yaml` |
| Which setup was chosen, and why? | `reports/tables/decision.csv`, `reports/figures/cost_vs_threshold.png`, and the M5 learning log entry |
| How well does the final model do on unseen clients? | `reports/tables/test_metrics.csv`, `reports/figures/calibration_test.png`, `reports/figures/roc_pr_test.png` |
| What drives the model's predictions? | `reports/figures/shap_summary.png`, `reports/tables/shap_importance.csv`, and the two `shap_client_*.png` figures |
| What were the results? | `docs/results.md`; the cross-validation comparison is in `reports/tables/experiments.csv` |
| What does the data look like? | `notebooks/01-eda.ipynb` |
| What is a given function supposed to do? | Its test in `tests/test_<module>.py` |
| Why was a decision made? | `docs/learning-log.md` and the design spec |

## What the tests check

`make test` runs the suite (about 280 tests; many are one test run over several inputs).
The tests check the code, not the model's quality: each one feeds a function an input
with a known right answer, often a small made-up one, and checks the result.

| File | What it checks |
|---|---|
| `test_config.py` | Settings load correctly; typos, bad values, and broken YAML fail with a clear message |
| `test_data.py` | The raw CSV matches its recorded fingerprint (sha256); a changed or missing file stops the pipeline |
| `test_clean.py` | Undocumented codes are merged, columns renamed, impossible values rejected; real data gives a 22% default rate |
| `test_split.py` | 80/20 split, same default rate in both, no client in both, same seed gives the same split |
| `test_features.py` | Each feature matches a hand-worked example; zero or negative bills give a flag, not an error; no missing or infinite values on the real data |
| `test_preprocess.py` | Amounts are scaled, categories one-hot encoded, unseen categories don't crash |
| `test_models.py` | Each name builds the right model with the seed applied; a misspelled setting fails instead of being ignored |
| `test_samplers.py` | Each sampler name builds the right resampler with the seed; a misspelled name fails |
| `test_pipeline.py` | Every model fits and returns probabilities; resampling balances only the training rows; SMOTENC rows have real codes |
| `test_experiments.py` | One results row per model; the dummy scores chance level and real models beat it; reruns match |
| `test_decision.py` | Expected cost matches a hand-worked example; the scorer prefers cheaper predictions; predictions use the tuned threshold; the saved model loads and predicts the same |
| `test_evaluate.py` | Metrics match a hand-worked example; baselines cost what arithmetic says; the train estimate is read correctly or left blank |
| `test_explain.py` | SHAP values add up to the model's score; plots show unscaled values; example clients follow the fixed rule; non-tree models are refused |
| `test_cli.py` | Each `make` stage runs end to end on a small sample and prints readable errors |
| `test_showcase.py` | Example clients fall in the right group; payment scenarios rewrite the right columns and never give a negative payment; recorded and changed values are compared cell by cell |
| `test_app.py` | The Streamlit app starts, each tab renders, and each control changes what it should (run without a browser) |
| `test_requirements.py` | The app's pinned scikit-learn and LightGBM versions match the ones the saved model was trained with |

Why the tests use made-up inputs, and how pytest runs them, is explained in
`docs/learning-log.md` (M2, "How the tests work").

## Why it is built this way

The analysis code is an installable Python package (`src/credit_risk/`) rather than a
set of notebooks. Notebooks, tests, and the `credit-risk` command all import the same
functions, so there is one copy of each step and a fix to it applies everywhere. Each
module does one job, settings live in `configs/` rather than in the code, and each
step is covered by tests. The result is that `make all` reruns the analysis from the
raw file and, with the same data, code, and seed, gives the same numbers.

Notebooks are still where exploration happens. They import from the package, so they
stay short and focused on looking at the data and trying ideas. When an idea settles,
it moves into `src/` with a test.

## How this project was built

This is an AI-assisted ML project, built with [Claude Code](https://www.anthropic.com/claude-code)
as a learning exercise and a way to explore what the tool can do. I set the goals and
made the design and modelling decisions; Claude Code wrote much of the code from
written plans (`docs/superpowers/plans/`). I explored the data and compared models in
notebooks, checked the generated code against tests and reruns, and recorded what I
learned at each milestone in `docs/learning-log.md`.

## Acknowledgements

This project grew out of a group lab for DSCI 573 (Feature and Model Selection) in the
UBC Master of Data Science program, completed with Vy and Christine. The pipeline,
feature engineering, experiments, and write-up in this repository are a separate,
individual rebuild.

Data: Yeh, I. C., and Lien, C. H. (2009). The comparisons of data mining techniques for
the predictive accuracy of probability of default of credit card clients. *Expert
Systems with Applications*, 36(2), 2473-2480. Distributed via the UCI Machine Learning
Repository.
