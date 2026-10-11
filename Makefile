# Makefile for the credit default risk pipeline

ENV := credit-risk
CONFIG ?= configs/base.yaml
RUN := credit-risk --config $(CONFIG)

# Colors for output (set NO_COLOR=1 to turn them off)
ifeq ($(NO_COLOR),)
GREEN := \033[0;32m
YELLOW := \033[1;33m
CYAN := \033[0;36m
DIM := \033[2m
RESET := \033[0m
endif

.PHONY: help setup lint format test data split experiments decide train evaluate all smoke
.NOTPARALLEL:

help:
	@printf "%b\n" "$(CYAN)Available commands:$(RESET)"
	@printf "%b\n" ""
	@printf "%b\n" "$(CYAN)Environment Setup & Testing:$(RESET)"
	@printf "%b\n" "  $(YELLOW)make setup$(RESET)        - Create or update the conda env and install git hooks"
	@printf "%b\n" "  $(YELLOW)make test$(RESET)         - Run the pytest suite"
	@printf "%b\n" ""
	@printf "%b\n" "$(CYAN)Developer - Python formatting:$(RESET)"
	@printf "%b\n" "  $(YELLOW)make lint$(RESET)         - Check style with ruff"
	@printf "%b\n" "  $(YELLOW)make format$(RESET)       - Auto-format and fix with ruff"
	@printf "%b\n" ""
	@printf "%b\n" "$(CYAN)Step 1 - Data:$(RESET)"
	@printf "%b\n" "  $(YELLOW)make data$(RESET)         - Check the raw CSV against its expected sha256"
	@printf "%b\n" "  $(YELLOW)make split$(RESET)        - Clean the raw data and write a stratified train/test split"
	@printf "%b\n" ""
	@printf "%b\n" "$(CYAN)Step 2 - Model comparison:$(RESET)"
	@printf "%b\n" "  $(YELLOW)make experiments$(RESET)  - Cross-validate each model in the config on train;"
	@printf "%b\n" "                      writes reports/tables/experiments.csv"
	@printf "%b\n" ""
	@printf "%b\n" "$(CYAN)Step 3 - Decision layer:$(RESET)"
	@printf "%b\n" "  $(YELLOW)make decide$(RESET)       - Compare calibrated, cost-tuned candidates on train;"
	@printf "%b\n" "                      writes reports/tables/decision.csv and a cost plot"
	@printf "%b\n" "  $(YELLOW)make train$(RESET)        - Fit the 'final' setup on train; saves artifacts/model.joblib"
	@printf "%b\n" ""
	@printf "%b\n" "$(CYAN)Step 4 - Test and explain:$(RESET)"
	@printf "%b\n" "  $(YELLOW)make evaluate$(RESET)     - Score the saved model on the test set;"
	@printf "%b\n" "                      writes reports/tables/test_metrics.csv and a calibration plot"
	@printf "%b\n" ""
	@printf "%b\n" "$(CYAN)Run everything:$(RESET)"
	@printf "%b\n" "  $(YELLOW)make all$(RESET)          - Run every step in order (data, split, experiments, decide, train)"
	@printf "%b\n" "  $(YELLOW)make smoke$(RESET)        - Run every step on a 1,000-row sample (what CI runs)"
	@printf "%b\n" ""
	@printf "%b\n" "$(CYAN)How the code fits together (src/credit_risk/, in order):$(RESET)"
	@printf "%b\n" "  $(GREEN)config.py$(RESET)         - Settings from configs/*.yaml, used by every step"
	@printf "%b\n" "  $(GREEN)1 data.py$(RESET)         - Load, clean, split"
	@printf "%b\n" "  $(GREEN)2 features.py$(RESET)     - Engineered columns (late months, ratios, trends)"
	@printf "%b\n" "  $(GREEN)3 preprocess.py$(RESET)   - Scale amounts, one-hot categories"
	@printf "%b\n" "  $(GREEN)4 samplers.py$(RESET)     - Sampler name -> resampler (evens out classes)"
	@printf "%b\n" "  $(GREEN)5 models.py$(RESET)       - Model name -> unfitted model"
	@printf "%b\n" "  $(GREEN)6 pipeline.py$(RESET)     - [Features], scale, [resample], encode, model"
	@printf "%b\n" "  $(GREEN)7 experiments.py$(RESET)  - Cross-validate each setup, results table"
	@printf "%b\n" "  $(GREEN)8 decision.py$(RESET)     - Calibrate, tune threshold to costs, save model"
	@printf "%b\n" "  $(GREEN)cli.py$(RESET)            - The credit-risk command; make targets call it"
	@printf "%b\n" ""
	@printf "%b\n" "$(DIM)Override the config with CONFIG=path, e.g. make all CONFIG=configs/smoke.yaml$(RESET)"

# ─── Environment setup ───────────────────────────────────────────────────────

setup:
	@printf "%b\n" "$(CYAN)Creating or updating the conda env from environment.yml...$(RESET)"
	conda env create -f environment.yml || conda env update -n $(ENV) -f environment.yml --prune
	conda run -n $(ENV) pre-commit install
	@printf "%b\n" "$(GREEN)Setup complete. Run: conda activate $(ENV)$(RESET)"

# ─── Developer checks ────────────────────────────────────────────────────────

lint:
	@printf "%b\n" "$(CYAN)Checking style with ruff...$(RESET)"
	ruff check .
	ruff format --check .
	@printf "%b\n" "$(GREEN)Lint checks complete.$(RESET)"

format:
	@printf "%b\n" "$(CYAN)Formatting with ruff...$(RESET)"
	ruff format .
	ruff check --fix .
	@printf "%b\n" "$(GREEN)Formatting complete.$(RESET)"

test:
	@printf "%b\n" "$(CYAN)Running tests...$(RESET)"
	pytest
	@printf "%b\n" "$(GREEN)Tests complete.$(RESET)"

# ─── Step 1: Data ────────────────────────────────────────────────────────────

data:
	@printf "%b\n" "$(CYAN)Checking the raw CSV...$(RESET)"
	$(RUN) data

split:
	@printf "%b\n" "$(CYAN)Cleaning and splitting into train/test...$(RESET)"
	$(RUN) split

# ─── Step 2: Model comparison ────────────────────────────────────────────────

experiments:
	@printf "%b\n" "$(CYAN)Cross-validating each experiment on train...$(RESET)"
	$(RUN) experiments
	@printf "%b\n" "$(GREEN)Experiments complete.$(RESET)"

# ─── Step 3: Decision layer ──────────────────────────────────────────────────

decide:
	@printf "%b\n" "$(CYAN)Comparing calibrated, cost-tuned candidates on train (about 5 minutes)...$(RESET)"
	$(RUN) decide
	@printf "%b\n" "$(GREEN)Decision table complete.$(RESET)"

train:
	@printf "%b\n" "$(CYAN)Fitting the final setup on train...$(RESET)"
	$(RUN) train
	@printf "%b\n" "$(GREEN)Model saved.$(RESET)"

# ─── Step 4: Test and explain ────────────────────────────────────────────────

evaluate:
	@printf "%b\n" "$(CYAN)Scoring the saved model on the test set...$(RESET)"
	$(RUN) evaluate
	@printf "%b\n" "$(GREEN)Evaluation complete.$(RESET)"

# ─── Run everything ──────────────────────────────────────────────────────────

# Steps in order; .NOTPARALLEL keeps them sequential even with make -j.
all: data split experiments decide train

smoke:
	@printf "%b\n" "$(CYAN)Smoke run: every step on a small sample...$(RESET)"
	$(MAKE) all CONFIG=configs/smoke.yaml
	@printf "%b\n" "$(GREEN)Smoke run complete.$(RESET)"
