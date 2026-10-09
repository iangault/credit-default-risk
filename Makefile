ENV := credit-risk
CONFIG ?= configs/base.yaml
RUN := credit-risk --config $(CONFIG)

.PHONY: help setup lint format test data split experiments all smoke
.NOTPARALLEL:

# Colors only when printing to a terminal; set NO_COLOR=1 to turn them off.
help:
	@if [ -t 1 ] && [ -z "$$NO_COLOR" ]; then \
	  b='\033[1m'; c='\033[36m'; g='\033[32m'; d='\033[2m'; r='\033[0m'; \
	else b=''; c=''; g=''; d=''; r=''; fi; \
	row() { printf "  $${c}%-12s$${r} %-42s $${d}%s$${r}\n" "$$1" "$$2" "$$3"; }; \
	mod() { printf "  $${g}%-17s$${r} %s\n" "$$1" "$$2"; }; \
	printf "\n$${b}Pipeline stages$${r} $${d}(run in this order)$${r}\n"; \
	row "data"       "1. check the raw CSV hash"                "data.py"; \
	row "split"      "2. clean and split into train/test"       "data.py"; \
	row "experiments" "3. cross-validate each model on train"   "experiments.py"; \
	row "all"        "run every stage in order"                 ""; \
	row "smoke"      "every stage on a small sample (CI)"       "configs/smoke.yaml"; \
	printf "\n$${b}Development$${r}\n"; \
	row "setup"      "create/update the conda env and git hooks" ""; \
	row "lint"       "check style with ruff"                    ""; \
	row "format"     "auto-format with ruff"                    ""; \
	row "test"       "run pytest"                               "tests/"; \
	printf "\n$${b}How the code fits together$${r} $${d}(src/credit_risk/, in order)$${r}\n"; \
	mod "config.py" "settings from configs/*.yaml, used by every step"; \
	mod "1 data.py" "load, clean, split"; \
	mod "2 preprocess.py" "scale amounts, one-hot categories"; \
	mod "3 models.py" "model name -> unfitted model"; \
	mod "4 pipeline.py" "preprocessor + model in one object"; \
	mod "5 experiments.py" "cross-validate each model, results table"; \
	mod "cli.py" "runs each stage; make targets call it"; \
	printf "\n$${d}Override the config with CONFIG=path, e.g. make all CONFIG=configs/smoke.yaml$${r}\n\n"

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

experiments:
	$(RUN) experiments

all: data split experiments

smoke:
	$(MAKE) all CONFIG=configs/smoke.yaml
