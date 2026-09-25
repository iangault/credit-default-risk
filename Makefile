CONFIG ?= configs/base.yaml
RUN := uv run credit-risk --config $(CONFIG)

.PHONY: help setup lint format test data split experiments train evaluate explain all smoke

help:
	@echo "setup        install dependencies and git hooks"
	@echo "lint         check style with ruff"
	@echo "format       auto-format with ruff"
	@echo "test         run pytest"
	@echo "data         check the raw CSV hash"
	@echo "split        clean and split into train/test          (M1)"
	@echo "experiments  cross-validated comparisons on train      (M2+)"
	@echo "train        fit final model, calibration, threshold  (M5)"
	@echo "evaluate     score the test set once                   (M6)"
	@echo "explain      SHAP figures                              (M6)"
	@echo "all          run every stage in order"
	@echo "smoke        run every stage on a small sample (CI)"
	@echo "Override the config with CONFIG=path, e.g. make all CONFIG=configs/smoke.yaml"

setup:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest

data:
	$(RUN) data

split:
	$(RUN) split

experiments:
	$(RUN) experiments

train:
	$(RUN) train

evaluate:
	$(RUN) evaluate

explain:
	$(RUN) explain

all: data split experiments train evaluate explain

smoke:
	$(MAKE) all CONFIG=configs/smoke.yaml
