from __future__ import annotations

import numpy as np
import pytest

from periodic.features import TWO_PI, design_matrix, harmonic_design, nyquist_omega, phase


def test_phase_is_float64_and_in_range_for_float32_input() -> None:
    x = np.linspace(-1e6, 1e6, 1001).astype(np.float32)
    th = phase(x, 1.2345, x_center=0.3)
    assert th.dtype == np.float64
    assert np.all((th >= 0.0) & (th < TWO_PI))


@pytest.mark.parametrize("omega", [0.1, 1.0, 7.3, 50.0])
def test_features_repeat_exactly_at_one_million(omega: float, rng: np.random.Generator) -> None:
    period = TWO_PI / omega
    x = rng.uniform(-30.0, 30.0, 500)
    m = np.round(1e6 / period)
    ks = list(range(1, 9))
    a = design_matrix(x, omega, ks)
    b = design_matrix(x + m * period, omega, ks)
    c = design_matrix(x - m * period, omega, ks)
    np.testing.assert_allclose(a, b, atol=1e-7)
    np.testing.assert_allclose(a, c, atol=1e-7)


def test_design_shape_and_first_column() -> None:
    th = np.linspace(0, TWO_PI, 50, endpoint=False)
    d = harmonic_design(th, [1, 2, 5])
    assert d.shape == (50, 7)
    assert np.all(d[:, 0] == 1.0)
    np.testing.assert_allclose(d[:, 3], np.cos(5 * th))
    np.testing.assert_allclose(d[:, 6], np.sin(5 * th))


def test_nyquist_for_the_default_window() -> None:
    assert nyquist_omega(4096, 20 * np.pi) == pytest.approx(204.8, rel=1e-6)
