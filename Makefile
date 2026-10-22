.PHONY: setup lint format typecheck test test-all smoke clean

setup:            ## install locked environment and pre-commit hooks
	uv sync --locked
	uv run pre-commit install

lint:             ## ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

format:           ## auto-format and fix lint
	uv run ruff format .
	uv run ruff check --fix .

typecheck:        ## mypy on the package
	uv run mypy src

test:             ## fast test suite
	uv run pytest -m "not slow" -n auto

test-all:         ## full test suite including slow tests
	uv run pytest -n auto

smoke:            ## end-to-end run on a tiny config
	uv run periodic train --config configs/smoke.toml

clean:
	rm -rf runs/smoke .pytest_cache .mypy_cache .ruff_cache
