from __future__ import annotations

import math

import numpy as np
import pytest

from periodic.metrics import (
    BandMetrics,
    band_pairs,
    dense_window,
    evaluate_bands,
    fourier_coefficients,
    horizon,
    horizon_band,
    omega_agreement,
    phase_offset,
    predicted_horizon,
    reference_scale,
    shape_gain,
    spectrum_comparison,
)

TWO_PI = 2 * np.pi


def test_reference_scale_is_the_per_period_std() -> None:
    assert reference_scale(np.sin, TWO_PI, -1.0, 1.0) == pytest.approx(1 / np.sqrt(2), rel=1e-3)
    assert reference_scale(lambda x: np.full_like(x, 3.0), TWO_PI, -1.0, 1.0) == 1.0
    assert reference_scale(np.sin, None, 0.0, TWO_PI) == pytest.approx(1 / np.sqrt(2), rel=1e-3)


def test_band_pairs_validate_monotone_edges() -> None:
    assert band_pairs([1.0, 10.0, 100.0]) == [(1.0, 10.0), (10.0, 100.0)]
    with pytest.raises(ValueError, match="increasing"):
        band_pairs([10.0, 1.0])


def test_evaluate_bands_scores_a_perfect_and_a_shifted_model() -> None:
    rng = np.random.default_rng(0)
    scale = 1 / np.sqrt(2)
    perfect = evaluate_bands(np.sin, np.sin, [10.0, 100.0, 1000.0], 500, scale, rng)
    assert len(perfect) == 6
    assert all(b.nrmse < 1e-12 for b in perfect)
    assert {b.sign for b in perfect} == {"+", "-", "both"}
    assert all(b.lo <= abs(b.center) <= b.hi for b in perfect)

    random_phase = evaluate_bands(
        lambda x: np.sin(x + 1.7), np.sin, [10.0, 100.0], 4000, scale, rng
    )
    both = next(b for b in random_phase if b.sign == "both")
    assert both.nrmse == pytest.approx(2 * abs(np.sin(1.7 / 2)), rel=0.05)
    assert both.max_abs <= 2.0 + 1e-12


def test_horizon_is_the_first_failing_band_or_inf() -> None:
    def band(lo: float, nrmse: float) -> BandMetrics:
        return BandMetrics(lo, lo * 10, "both", 1, nrmse, nrmse, nrmse, nrmse)

    bands = [band(1e3, 0.5), band(10.0, 0.01), band(1e2, 0.2)]
    assert horizon(bands, 0.1) == 1e2
    assert horizon(bands, 0.3) == 1e3
    assert horizon(bands, 1.0) == math.inf
    assert horizon([band(10.0, math.nan)], 0.1) == 10.0
    assert horizon_band(bands, 0.1) == (1e2, 1e3)
    assert horizon_band(bands, 1.0) == (math.inf, math.inf)


def test_phase_offset_recovers_a_known_lag() -> None:
    x = dense_window(1000.0, TWO_PI, 3.0, 600)
    truth = np.sin(x)
    for lag in (0.3, -0.5, 1.2):
        pred = np.sin(x - lag)
        assert phase_offset(pred, truth, x, TWO_PI) == pytest.approx(lag, abs=0.02)
    assert math.isnan(phase_offset(np.zeros_like(x), truth, x, TWO_PI))


def test_shape_gain_and_predicted_horizon_for_a_sine() -> None:
    gain = shape_gain(np.sin, TWO_PI)
    assert gain == pytest.approx(1.0, rel=1e-3)
    assert predicted_horizon(1e-4, 0.1, gain) == pytest.approx(1e3, rel=1e-3)
    assert predicted_horizon(0.0, 0.1, gain) == math.inf
    assert predicted_horizon(1e-4, 0.1, math.inf) == math.inf


def test_omega_agreement_snaps_to_small_rationals() -> None:
    a = omega_agreement(2.0 * (1 + 1e-6), 1.0)
    assert a["nearest_ratio"] == 2.0
    assert a["rel_err_to_nearest"] == pytest.approx(1e-6, rel=1e-3)
    assert a["rel_err"] == pytest.approx(1.0 + 2e-6, rel=1e-3)
    b = omega_agreement(1.0 / 3.0, 1.0)
    assert b["nearest_ratio"] == pytest.approx(1 / 3)


def test_fourier_coefficients_and_spectrum_comparison() -> None:
    f = lambda x: np.sin(x) + 0.5 * np.cos(3 * x)  # noqa: E731
    c = fourier_coefficients(f, TWO_PI, 4)
    np.testing.assert_allclose(2 * np.abs(c), [1.0, 0.0, 0.5, 0.0], atol=1e-12)
    cmp = spectrum_comparison(f, f, TWO_PI, TWO_PI, k_max=4)
    assert cmp["k"] == [1, 2, 3, 4]
    np.testing.assert_allclose(cmp["amp_abs_err"], 0.0, atol=1e-12)


def test_square_wave_spectrum_has_no_even_harmonics() -> None:
    square = lambda x: np.where(np.sin(x) >= 0.0, 1.0, -1.0)  # noqa: E731
    amp = 2 * np.abs(fourier_coefficients(square, TWO_PI, 8))
    np.testing.assert_allclose(amp[1::2], 0.0, atol=1e-12)
    np.testing.assert_allclose(amp[0::2], 4 / (np.pi * np.array([1, 3, 5, 7])), rtol=2e-3)
