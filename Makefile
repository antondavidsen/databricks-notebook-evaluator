.PHONY: help sync mypy test ruff format format-check lint ci clean

help:  ## Show available targets
	@echo "Available targets:"
	@echo "  make sync         Install/refresh the environment with uv"
	@echo "  make mypy         Run mypy type checker (strict)"
	@echo "  make test         Run pytest with coverage"
	@echo "  make ruff         Run ruff linter"
	@echo "  make format       Auto-format code with ruff"
	@echo "  make format-check Verify formatting without rewriting files"
	@echo "  make lint         Run all linters (mypy + ruff)"
	@echo "  make ci           Full quality gate: lint + format-check + test"

sync:
	uv sync --all-extras

mypy:
	uv run mypy src tests

test:
	uv run pytest

ruff:
	uv run ruff check src tests

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

format-check:
	uv run ruff format --check src tests

lint: mypy ruff
	@echo "All linters passed"

ci: lint format-check test
	@echo "Quality gate passed"

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
