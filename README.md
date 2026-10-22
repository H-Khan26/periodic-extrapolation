# periodic-extrapolation

Learn a bounded periodic function from samples on `[-10π, 10π]` and predict it anywhere.

<!-- Approach, results, limitations and future work are written by the author. -->

## Reproduce

```bash
make setup                                   # locked environment (uv) + pre-commit hooks
make test                                    # fast test suite
make smoke                                   # tiny end-to-end run
uv run periodic train --config configs/base.toml --set data.target=square
uv run periodic predict --run runs/<run-dir> --x 1000000 -31.4
```
