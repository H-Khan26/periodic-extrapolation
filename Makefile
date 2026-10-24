.PHONY: setup lint format typecheck test smoke suite sweeps report examples clean

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

test:             ## test suite, in parallel
	uv run pytest -n auto --dist loadgroup

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

# <sweep run directory>=<example name>; each is the latest run in that directory. Evaluation is
# single-threaded so it does not depend on the BLAS thread count; it matches the sweep to the last
# bit wherever the sweep workers also ran one thread each (fewer than 16 cores at jobs = 8).
EXAMPLES = suite/runs/target-sin_seed-0=sin \
	suite/runs/target-missing_fundamental_seed-0=missing_fundamental \
	suite/runs/target-abs_sin_seed-0=abs_sin \
	suite/runs/target-square_seed-0=square \
	suite/runs/target-one_cycle_seed-0=one_cycle \
	noise/runs/target-sin_noise_rel-0.1_seed-0=sin_noise_0.1

examples:         ## re-evaluate representative sweep runs and copy them -> reports/examples/<name>
	set -e; for e in $(EXAMPLES); do \
		run=$$(ls -d reports/$${e%%=*}/*/ 2>/dev/null | tail -n 1); out=reports/examples/$${e#*=}; \
		[ -d "$$run" ] || { echo "no run under reports/$${e%%=*}: run the sweep first" >&2; exit 1; }; \
		rm -rf "$$run/figures" "$$out"; \
		OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run periodic evaluate --run "$$run"; \
		mkdir -p "$$out"; cp "$$run"/figures/*.png "$$run"/evaluation.json "$$run"/metrics.json \
			"$$run"/omega.json "$$out"/; \
	done

clean:
	rm -rf runs/smoke .pytest_cache .mypy_cache .ruff_cache
