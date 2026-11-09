PYTHON ?= python3
RUFF ?= ruff

.PHONY: install test lint format sample clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	PYTHONPATH=src $(PYTHON) -m pytest

lint:
	$(RUFF) check .
	$(RUFF) format --check .

format:
	$(RUFF) format .
	$(RUFF) check --fix .

sample:
	PYTHONPATH=src $(PYTHON) scripts/generate_sample_data.py

clean:
	rm -rf build dist .pytest_cache .ruff_cache src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
