CONFIG ?= configs/base.yaml

.PHONY: help setup lint format test

help:
	@echo "setup   install dependencies and git hooks"
	@echo "lint    check style with ruff"
	@echo "format  auto-format with ruff"
	@echo "test    run pytest"

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
