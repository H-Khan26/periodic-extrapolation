# periodic-extrapolation

Learn a bounded periodic function from samples on `[-10π, 10π]` and predict it anywhere, including at `|x| = 10⁶`.

The model is periodic by construction: it learns one fundamental frequency ω and a shape on top of it, and every input is first reduced to a phase on the circle. Whether it extrapolates well is decided entirely by how precisely ω is estimated, so most of the engineering is about finding ω and about reporting how far the estimate can be trusted.

## Approach

**Model.** `ŷ(x) = m + s · h(θ)` with `θ = ω (x − x_c) mod 2π`. The head `h` is either a Fourier series on the harmonics that carry energy (solved in closed form) or a small MLP on `[cos kθ, sin kθ]` for the first four harmonics (two SiLU layers of width 64). Because x enters only through θ, the prediction repeats exactly for any weights. Everything runs in float64: a float32 input is quantised to 0.06 at `|x| = 10⁶`, which is already a 0.06 rad phase error.

**Finding ω.** The training loss as a function of ω is a comb of local minima whose correct basin is only about `2π/L` wide (L is the window width), so gradient descent from a random start cannot find it. Instead:

1. A Lomb-Scargle periodogram on a grid from `2π/L` to the Nyquist frequency, scored by the harmonic sum `Σ_k P(kω)` so that a fundamental with no peak of its own (missing fundamental, merged peaks) still scores highest. The eight best local maxima become candidates.
2. Variable projection on each candidate: the Fourier coefficients are solved in closed form at every trial ω, which turns the fit into a one-dimensional problem. Bounded Brent, then Newton with the analytic gradient, reaches the float64 floor in about twelve evaluations.
3. A noise-aware tie-break that prefers the largest ω, then a lift to `m·ω` when all energy sits at multiples of m. This settles the period-versus-double-period ambiguity without hardcoding anything.
4. A Fisher-information standard error on ω. It feeds a predicted extrapolation horizon that is reported with every run and checked against the measured one.

**Training the head.** Each candidate head is fitted on the middle 80 % of the window and scored on the two outer 10 % blocks, since an edge block is a short extrapolation while a middle block only measures interpolation. The winner is refitted on the whole window with ω frozen. The Fourier head is closed form; the MLP head uses full-batch Adam followed by an L-BFGS polish. Joint refinement of ω with the head was tried and removed: it improved the in-window loss while moving ω by 3e-7, enough to turn a 7.5e-11 error at `10⁶` into 0.30.

**Data.** 4,096 inputs, one uniform draw per equal cell of the window (stratified jitter): no gaps and no coherent aliases, which a regular grid would have. Label noise, when enabled, is Gaussian with a standard deviation given relative to the signal amplitude.

**Loss.** Mean squared error on standardised labels throughout. The residual that variable projection minimises is the same loss, so the frequency search is part of training, not preprocessing.

## Assumptions

- At least one full period lies inside the training window, so the period is at most `20π`. A longer period cannot be identified from any data on the window. Between one and two cycles the estimate rests on parsimony rather than repetition; such runs are flagged `low_confidence`.
- The fundamental lies below the Nyquist frequency of the mean sample spacing (about 205 rad per unit x at 4,096 samples). Shorter periods need more samples.
- The black box is cheap to query, so sample count is a free choice; 4,096 is the default and 64 to 16,384 are swept.

## Results

Sixteen targets, five seeds each, noiseless, 4,096 samples. NRMSE is RMSE divided by the target's standard deviation over one period: 1 means "predict the mean", √2 is the right shape with a random phase. The horizon is the first distance band whose NRMSE exceeds 0.1.

| Targets | ω relative error (median) | NRMSE at 1e3 to 1e4 | Horizon |
| --- | --- | --- | --- |
| sin, 3 harmonics, missing fundamental, one cycle, sin 30x, sin x/8, random Fourier series, 5 + 0.01 sin x, and 10 held-out random series | 0 (exact in float64) | 4e-13 to 2e-11 | beyond 1e6 |
| Bessel spectrum, bump train | 1e-11 | 1e-7 | beyond 1e6 |
| Kinks: abs(sin x), triangle | 2e-7, 6e-8 | 3e-3, 9e-4 | 1e5, beyond 1e6 |
| Jumps: square, 20 % pulse, sawtooth | 7e-5, 7e-5, 1.4e-4 | 0.7 to 1.1 | 100, 100, 31 |
| sin x + sin √2 x (not periodic) | n/a | 1.4 | 31 |

On every smooth target the model matches an oracle that is given the true period, to about 1e-13. Under noise the ω error follows the Cramér-Rao line (7.5e-6, 2.2e-5, 7.5e-5, 2.2e-4 at 1, 3, 10, 30 % noise on `sin x`), the horizon falls from 1e4 to 1e2, and the model's own predicted horizon lands inside the measured failing band. The held-out series use a random fundamental and random harmonics drawn from a seed fixed after development.

![Main model against baselines and the oracle](reports/suite/figures/baselines.png)

Baselines, median NRMSE in the 1e3 to 1e4 band: a periodic-kernel Gaussian process initialised from our own ω estimate reaches 1e-6 to 1e-3 on smooth targets but fails on the one-cycle target and on jumps; a ReLU MLP grows linearly (NRMSE 1e3 to 1e4); Snake activations give 30 to 300; random Fourier features drift to a random phase (about 1); the constant predictor is 1.

Two ablations show why the spectral search matters. Random ω with Adam fails 12 of 12 runs and never even fits the training window. Starting from the periodogram peak shifted by one to five basins fails 10 of 12; the two successes converged to ω/2, which still extrapolates correctly.

![Why ω needs a global search: the loss is a comb](reports/examples/square/omega_landscape.png)

![Predicted against measured horizon](reports/noise/figures/calibration.png)

## Limitations and failure cases

- **Jumps are information-limited.** Each jump is located only to within one sample gap, so ω stops improving near 7e-5 and more samples barely help (5e-5 at 16,384). The reported error bar is a noise bar and misses this truncation bias, so predicted horizons for jump targets run 5 to 10 times optimistic. `reports/examples/square/overlays.png` shows the drift.
- **Fewer than two cycles.** Exact without noise, but with noise the error bar grows to the percent level.
- **Frequency ceiling.** The scan stops at the mean-spacing Nyquist frequency even though jittered sampling has no hard limit, so `sin 30x` needs roughly 600 samples or more.
- **Quasi-periodic input** fails by design and says so: in-window NRMSE 0.37, horizon 31.
- **Cost.** A run with both heads takes about 30 s; the full suite with six baselines takes about 90 minutes on 8 cores.

## What I would do with more time

- **Keep every harmonic that carries energy.** Smooth shapes with long spectra have many tiny high harmonics. Dropping them leaves a small systematic misfit in the window, the frequency fit absorbs it by nudging ω, and any ω error grows with distance. A lower significance floor, re-applied after ω is refined because the choice of harmonics depends on ω, leaves nothing for ω to compensate for.
- **Honest error bars.** The error on ω assumes the residual is white noise. On jumps and kinks it is structured ringing that repeats every period, so it carries less independent information than the sample count suggests and the error bar comes out too small. A sandwich estimate measures the residual's actual structure instead of assuming it. The predicted horizon should also start from the error already present inside the window rather than assume the fit there is perfect.
- **Model jumps directly.** For a target with jumps, nearly all the information about ω is in where the jumps fall. A Fourier series can only smear a jump into ringing, and the ringing biases ω. Detected jumps give a sharper constraint: every jump must land between the two samples that straddle it, in every cycle, and that alone bounds ω. One sawtooth feature per jump removes the ringing while keeping the model exactly periodic. Past that the limit is the sample gap itself, which active sampling can shrink by querying extra points around each detected jump.
- **Kink features.** The same idea one derivative up: a kink is a jump in slope, so smooth periodic kink features, placed jointly with ω, would replace the slowly decaying harmonic tail a kink otherwise needs.
- **Stricter evaluation.** A finer horizon grid than decades, so smaller improvements show; evaluation points dense enough to land in the narrow gaps around jumps, where the error concentrates; jump targets whose jumps are away from x = 0, so no result depends on that coincidence; and held-out shapes with jumps, kinks and noise, since the current held-out series are smooth and the Fourier head solves them exactly.

## Reproduce

```bash
make setup                                   # locked environment (uv) + pre-commit hooks
make test                                    # test suite (132 tests)
make smoke                                   # tiny end-to-end run: train, evaluate, figures
make suite                                   # 16 targets x 5 seeds with baselines -> reports/suite
make sweeps                                  # noise, sample-count, held-out, ablation -> reports/
make examples                                # figures for six representative runs -> reports/examples
uv run periodic train --config configs/base.toml --set data.target=square
uv run periodic evaluate --run runs/<run-dir>
uv run periodic predict --run runs/<run-dir> --x 1000000 -31.4
```

Every run directory holds `config.json`, `env.json` (versions, git SHA), `omega.json` (candidates, standard error), `metrics.json`, `evaluation.json` and `figures/`. Sweeps write `results.csv`, `summary_by_target.csv` and figures under `reports/<name>/`; `reports/examples/` holds figure sets for six representative runs.

## Layout

```text
src/periodic/
  targets.py    black-box functions and their periods (evaluation side only)
  data.py       sampling, noise, caching, edge-block split
  spectral.py   frequency estimation: scan, variable projection, tie-break, error bar
  features.py   float64 phase and harmonic features
  models/       harmonic regressor (Fourier and MLP heads), baselines, ablations
  train.py      head selection and refit; inference.py loads and predicts
  metrics.py    bands, horizons, phase offset; evaluate.py, sweep.py, report.py, plots.py
configs/        base.toml, smoke.toml, sweeps/*.toml
tests/          unit, property (Hypothesis) and integration tests, incl. the leakage guard
```
