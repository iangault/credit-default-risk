# credit-default-risk

Predicting whether a credit card client will miss next month's payment, built as a
modular, reproducible ML pipeline.

> Status: work in progress. The project scaffold is in place; modelling stages are
> being added milestone by milestone (see the [design spec](docs/superpowers/specs/2026-09-25-credit-default-risk-design.md)).

## The problem

The [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
dataset has 30,000 clients from Taiwan (2005) with credit limit, demographics, six months
of repayment status, bill amounts, and payment amounts. About 22% of clients default.
The goal is to rank clients by default risk and choose a decision threshold based on the
cost of each type of error, rather than the default 0.5 cutoff.

## Quickstart

Requires [conda](https://conda-forge.org/download/) (Miniforge recommended) and `make`.

```bash
make setup                # create the conda env from environment.yml, install git hooks
conda activate credit-risk
make test                 # run the test suite
make all                  # run the pipeline end to end
```

Each stage can also be run on its own (`make data`, `make split`, ...). Use a different
config with `make all CONFIG=configs/smoke.yaml`.

## Project layout

```
configs/          settings for every stage (paths, seed, CV folds, error costs)
data/raw/         the original CSV (committed; checked by sha256)
src/credit_risk/  the package: one module per pipeline responsibility
tests/            pytest suite, including a small end-to-end smoke run
notebooks/        exploration and write-up; imports from the package
docs/             design spec, plans, and a learning log
```

## How the code fits together

The modules in `src/credit_risk/`, in the order the pipeline uses them:

| Order | Module | What it does | Run with |
|---|---|---|---|
| | `config.py` | Reads settings from `configs/*.yaml`; used by every step | |
| 1 | `data.py` | Load, clean, and split into train/test | `make data`, `make split` |
| 2 | `preprocess.py` | Scale amounts, one-hot encode categories | (inside the pipeline) |
| 3 | `models.py` | Turn a model name from the config into an unfitted model | (inside the pipeline) |
| 4 | `pipeline.py` | Join the preprocessor and a model into one object | (inside experiments) |
| 5 | `experiments.py` | Cross-validate each model and write a results table | `make experiments` |
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
| How are models scored? | `src/credit_risk/experiments.py` |
| What were the results? | `reports/tables/experiments.csv` |
| What does the data look like? | `notebooks/01-eda.ipynb` |
| What is a given function supposed to do? | Its test in `tests/test_<module>.py` |
| Why was a decision made? | `docs/learning-log.md` and the design spec |

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

## Acknowledgements

This project grew out of a group lab for DSCI 573 (Feature and Model Selection) in the
UBC Master of Data Science program, completed with Vy and Christine. The pipeline,
feature engineering, experiments, and write-up in this repository are a separate,
individual rebuild.

Data: Yeh, I. C., and Lien, C. H. (2009). The comparisons of data mining techniques for
the predictive accuracy of probability of default of credit card clients. *Expert
Systems with Applications*, 36(2), 2473-2480. Distributed via the UCI Machine Learning
Repository.
