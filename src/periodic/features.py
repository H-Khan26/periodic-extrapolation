"""Phase and harmonic features. Everything here is float64 by construction.

At |x| = 1e6 a float32 input is quantised to 0.0625, which is a phase error of 0.06 rad for
ω = 1, so inputs are promoted to float64 before the phase is formed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

TWO_PI = 2.0 * np.pi


def phase(x: np.ndarray, omega: float, x_center: float = 0.0) -> np.ndarray:
    """θ = ω (x - x_center) reduced to [0, 2π)."""
    x64 = np.asarray(x, dtype=np.float64)
    return np.mod(omega * (x64 - x_center), TWO_PI)


def harmonic_design(theta: np.ndarray, ks: Sequence[int]) -> np.ndarray:
    """Design matrix [1, cos kθ ..., sin kθ ...] for the harmonics in ks."""
    k = np.asarray(ks, dtype=np.float64)
    arg = np.asarray(theta, dtype=np.float64)[:, None] * k[None, :]
    return np.hstack([np.ones((arg.shape[0], 1)), np.cos(arg), np.sin(arg)])


def design_matrix(
    x: np.ndarray, omega: float, ks: Sequence[int], x_center: float = 0.0
) -> np.ndarray:
    return harmonic_design(phase(x, omega, x_center), ks)


def nyquist_omega(n: int, width: float) -> float:
    """Angular Nyquist frequency for n samples with mean spacing width / n."""
    return np.pi * n / width


def max_harmonics(n: int) -> int:
    """Largest harmonic bank that keeps a 2K + 1 coefficient fit overdetermined on n samples.

    Four samples per harmonic leaves a margin for the frequency parameter and the mean.
    """
    return max(1, (n - 1) // 4)
