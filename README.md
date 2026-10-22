# periodic-extrapolation

Learn a bounded periodic function from samples on `[-10π, 10π]` and predict it anywhere.

<!-- Approach, results, limitations and future work are written by the author. -->

## Reproduce

```bash
make setup                                   # locked environment (uv) + pre-commit hooks
make test                                    # fast test suite
make smoke                                   # tiny end-to-end run (train + evaluate + figures)
make suite                                   # 16 targets x 5 seeds with baselines -> reports/suite
make sweeps                                  # noise, sample-count, held-out, ablation -> reports/
uv run periodic train --config configs/base.toml --set data.target=square
uv run periodic evaluate --run runs/<run-dir>
uv run periodic predict --run runs/<run-dir> --x 1000000 -31.4
```

Every run directory holds `config.json`, `env.json` (versions, git SHA), `omega.json`, `metrics.json`,
`evaluation.json` and `figures/`. Sweeps write `results.csv`, `summary_by_target.csv` and `figures/`.
