ENV := credit-risk
CONFIG ?= configs/base.yaml
RUN := credit-risk --config $(CONFIG)

.PHONY: help setup lint format test data split all smoke
.NOTPARALLEL:

help:
	@echo "setup        create/update the conda env and git hooks"
	@echo "lint         check style with ruff"
	@echo "format       auto-format with ruff"
	@echo "test         run pytest"
	@echo "data         check the raw CSV hash"
	@echo "split        clean and split into train/test"
	@echo "all          run every stage in order"
	@echo "smoke        run every stage on a small sample (CI)"
	@echo "Override the config with CONFIG=path, e.g. make all CONFIG=configs/smoke.yaml"

setup:
	conda env create -f environment.yml || conda env update -n $(ENV) -f environment.yml --prune
	conda run -n $(ENV) pre-commit install
	@echo "Done. Run: conda activate $(ENV)"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest

data:
	$(RUN) data

split:
	$(RUN) split

all: data split

smoke:
	$(MAKE) all CONFIG=configs/smoke.yaml
