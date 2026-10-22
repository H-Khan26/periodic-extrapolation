.PHONY: setup lint format typecheck test test-all smoke suite sweeps report clean

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
	uv run periodic train --config configs/smoke.toml --run-dir runs/smoke/latest
	uv run periodic evaluate --run runs/smoke/latest

suite:            ## main target suite with baselines, 5 seeds -> reports/suite
	uv run periodic sweep --spec configs/sweeps/suite.toml

sweeps:           ## noise, sample-count, held-out and ablation sweeps -> reports/<name>
	uv run periodic sweep --spec configs/sweeps/noise.toml
	uv run periodic sweep --spec configs/sweeps/samples.toml
	uv run periodic sweep --spec configs/sweeps/heldout.toml
	uv run periodic sweep --spec configs/sweeps/ablation.toml

report:           ## re-aggregate and redraw every finished sweep under reports/
	for d in reports/*/; do if [ -f "$$d/results.csv" ]; then uv run periodic report --sweep "$$d"; fi; done

clean:
	rm -rf runs/smoke .pytest_cache .mypy_cache .ruff_cache
