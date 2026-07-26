# Developer workflow shortcuts. Run `make help` for the list.
.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install format lint typecheck test test-all cov docs build tiny-models merge release-check clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Editable install with dev + docs extras
	$(PY) -m pip install -e ".[dev,docs]"

format: ## Auto-format with ruff
	$(PY) -m ruff format src tests scripts examples
	$(PY) -m ruff check --fix src tests scripts examples

lint: ## Lint (ruff check + format check)
	$(PY) -m ruff check src tests scripts examples
	$(PY) -m ruff format --check src tests scripts examples

typecheck: ## Static type check with mypy
	$(PY) -m mypy src/model_merger

test: ## Fast tests (exclude slow/performance)
	$(PY) -m pytest -m "not slow and not performance" -q

test-all: ## Full test suite with coverage
	$(PY) -m pytest --cov=model_merger --cov-report=term-missing

cov: test-all ## Alias for test-all

docs: ## Build the documentation site
	$(PY) -m mkdocs build --strict

build: ## Build wheel and sdist
	$(PY) -m pip install --quiet build && $(PY) -m build

tiny-models: ## Generate tiny models under ./models
	$(PY) scripts/generate_tiny_test_models.py ./models

merge: tiny-models ## Run the uniform-soup example config
	$(PY) -m model_merger merge configs/uniform_soup.example.yaml --overwrite

release-check: ## Run every algorithm end-to-end and verify
	$(PY) scripts/verify_release.py

clean: ## Remove caches and build artifacts
	rm -rf build dist site htmlcov .pytest_cache .mypy_cache .ruff_cache \
		.coverage coverage.xml **/__pycache__ *.egg-info
