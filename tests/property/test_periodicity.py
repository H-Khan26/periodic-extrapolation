from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from periodic.features import TWO_PI, design_matrix
from periodic.models.harmonic import HarmonicRegressor


@given(
    omega=st.floats(0.05, 100.0),
    shift=st.integers(-100_000, 100_000),
    x=st.floats(-50.0, 50.0),
)
@settings(max_examples=200, deadline=None)
def test_harmonic_features_repeat_after_whole_periods(omega: float, shift: int, x: float) -> None:
    period = TWO_PI / omega
    a = design_matrix(np.array([x]), omega, [1, 2, 3, 7])
    b = design_matrix(np.array([x + shift * period]), omega, [1, 2, 3, 7])
    np.testing.assert_allclose(a, b, atol=1e-6)


@given(
    omega=st.floats(0.05, 50.0),
    coef=st.lists(st.floats(-3.0, 3.0), min_size=5, max_size=5),
    x=st.lists(st.floats(-1e6, 1e6), min_size=1, max_size=20),
)
@settings(max_examples=100, deadline=None)
def test_linear_head_output_is_bounded_by_its_coefficients(
    omega: float, coef: list[float], x: list[float]
) -> None:
    m = HarmonicRegressor(head="linear")
    m.omega, m.ks, m.coef = omega, (1, 3), np.asarray(coef)
    bound = abs(coef[0]) + sum(abs(c) for c in coef[1:])
    assert np.all(np.abs(m.predict(np.asarray(x))) <= bound + 1e-9)
