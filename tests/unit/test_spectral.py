from __future__ import annotations

import numpy as np
import pytest

from periodic.config import SpectralConfig
from periodic.data import sample_x
from periodic.spectral import (
    estimate_omega,
    harmonic_energies,
    omega_stderr,
    periodogram,
    profile_fit,
    promote_fundamental,
    refine_omega,
)
from periodic.targets import get_target

X_MIN, X_MAX = -10 * np.pi, 10 * np.pi
WIDTH = X_MAX - X_MIN
RES = 2 * np.pi / WIDTH
FAST = SpectralConfig(oversample=4, n_candidates=6)


def make(
    name: str, n: int = 2048, noise: float = 0.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = sample_x(n, X_MIN, X_MAX, "stratified", rng)
    y = get_target(name)(x)
    if noise:
        y = y + noise * rng.standard_normal(n)
    return x, y


@pytest.mark.parametrize(
    ("name", "omega_true", "rel_tol"),
    [
        ("sin", 1.0, 1e-8),
        ("harmonics3", 1.0, 1e-8),
        ("missing_fundamental", 1.0, 1e-8),
        ("sin30", 30.0, 1e-8),
        ("bessel", 1.0, 1e-7),
        ("abs_sin", 2.0, 1e-5),
        ("sin_eighth", 0.125, 1e-7),
        ("one_cycle", 0.1, 1e-6),
        ("random_fourier", 0.7, 1e-5),
        ("square", 1.0, 1e-3),
        ("pwm20", 1.0, 1e-3),
    ],
)
def test_recovers_the_fundamental_noiselessly(name: str, omega_true: float, rel_tol: float) -> None:
    x, y = make(name)
    est = estimate_omega(x, y, FAST, 0.0, width=WIDTH)
    assert abs(est.omega - omega_true) / omega_true < rel_tol, est
    assert est.omega_min <= est.omega <= est.omega_max
    assert est.low_confidence == (omega_true * WIDTH / (2 * np.pi) < 2.0)


def test_noisy_estimate_is_within_its_own_error_bar_and_near_the_bound() -> None:
    n, sigma = 2048, 0.1
    errs, ses = [], []
    for seed in range(4):
        x, y = make("sin", n=n, noise=sigma, seed=seed)
        est = estimate_omega(x, y, FAST, 0.0, width=WIDTH)
        errs.append(abs(est.omega - 1.0))
        ses.append(est.stderr)
    crb = sigma * np.sqrt(24.0) / (np.sqrt(n) * WIDTH)
    assert all(e < 5 * s for e, s in zip(errs, ses, strict=True))
    assert all(0.5 * crb < s < 2.0 * crb for s in ses)


def test_stderr_matches_the_cramer_rao_bound_for_a_sine() -> None:
    n, sigma = 2048, 0.05
    x, y = make("sin", n=n, noise=sigma, seed=1)
    se = omega_stderr(x, y, 1.0, [1], 0.0)
    crb = sigma * np.sqrt(24.0) / (np.sqrt(n) * WIDTH)
    assert se == pytest.approx(crb, rel=0.25)


def test_refine_reaches_the_float64_floor_from_within_the_basin() -> None:
    # Brent alone stops near sqrt(eps); the Newton polish is what reaches the floor.
    x, y = make("sin")
    omega, m = refine_omega(x, y, 1.0 + 0.3 * RES, [1, 2, 3], 0.0, 0.5 * RES, RES)
    assert abs(omega - 1.0) < 1e-10
    assert m < 1e-18


def test_promotion_moves_a_subharmonic_to_the_fundamental() -> None:
    x, y = make("sin")
    omega, m = promote_fundamental(x, y, 0.5, 16, 0.0, 5.0)
    assert m == 2 and omega == pytest.approx(1.0)
    x, y = make("missing_fundamental")
    assert promote_fundamental(x, y, 1.0, 16, 0.0, 5.0) == (1.0, 1)
    omega, m = promote_fundamental(x, y, 0.5, 16, 0.0, 5.0)
    assert m == 2 and omega == pytest.approx(1.0)


def test_profile_fit_recovers_known_coefficients_and_energies() -> None:
    x, _ = make("sin")
    y = 0.3 + 2.0 * np.cos(x) - 0.5 * np.sin(3 * x)
    coef, r, m = profile_fit(x, y, 1.0, [1, 2, 3], 0.0)
    # design layout: [1, cos kθ ..., sin kθ ...]
    np.testing.assert_allclose(coef, [0.3, 2.0, 0.0, 0.0, 0.0, 0.0, -0.5], atol=1e-12)
    assert m < 1e-24 and np.max(np.abs(r)) < 1e-11
    np.testing.assert_allclose(harmonic_energies(coef, 3), [4.0, 0.0, 0.25], atol=1e-12)


def test_periodogram_chunking_does_not_change_values() -> None:
    x, y = make("harmonics3", n=512)
    grid = np.linspace(0.05, 60.0, 5000)
    full = periodogram(x, y - y.mean(), grid, max_elements=1 << 30)
    chunked = periodogram(x, y - y.mean(), grid, max_elements=4096)  # eight frequencies per call
    np.testing.assert_allclose(chunked, full, rtol=1e-12, atol=1e-15)
    assert full.shape == grid.shape


@pytest.mark.parametrize("n", [64, 128])
def test_small_sample_counts_keep_every_fit_overdetermined(n: int) -> None:
    x, y = make("sin", n=n)
    est = estimate_omega(x, y, FAST, 0.0, width=WIDTH)
    assert abs(est.omega - 1.0) < 1e-8
    assert np.isfinite(est.stderr)
    assert max(est.harmonics) <= (n - 1) // 4
