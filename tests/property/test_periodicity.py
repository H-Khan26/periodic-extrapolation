from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from periodic.features import TWO_PI, design_matrix


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
