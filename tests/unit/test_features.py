from __future__ import annotations

import numpy as np

from periodic.features import TWO_PI, phase


def test_phase_is_float64_and_in_range_for_float32_input() -> None:
    x = np.linspace(-1e6, 1e6, 1001).astype(np.float32)
    th = phase(x, 1.2345, x_center=0.3)
    assert th.dtype == np.float64
    assert np.all((th >= 0.0) & (th < TWO_PI))
