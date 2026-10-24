"""Extrapolation metrics: distance bands, horizons, period and phase error.

Evaluation is allowed to call the black-box target; inference is not. Everything here takes a
``predict`` callable and a ``truth`` callable so models and baselines are scored identically.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from periodic.features import TWO_PI

Fn = Callable[[np.ndarray], np.ndarray]


@dataclass
class BandMetrics:
    lo: float
    hi: float
    sign: str
    n: int
    rmse: float
    mae: float
    max_abs: float
    nrmse: float

    @property
    def center(self) -> float:
        return math.sqrt(self.lo * self.hi)


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    d = np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


def reference_scale(truth: Fn, period: float | None, x_min: float, x_max: float) -> float:
    """Standard deviation of the target over one period (or over the window when aperiodic).

    NRMSE divides by this, so 1 means "no better than predicting the mean" and sqrt(2) is a
    copy of the function with a random phase.
    """
    if period is not None and math.isfinite(period):
        x = np.linspace(0.0, period, 4096, endpoint=False)
    else:
        x = np.linspace(x_min, x_max, 4096, endpoint=False)
    s = float(np.std(truth(x)))
    return s if s > 0 else 1.0


def band_pairs(edges: Sequence[float]) -> list[tuple[float, float]]:
    e = [float(v) for v in edges]
    if len(e) < 2 or any(b <= a for a, b in pairwise(e)):
        raise ValueError("band edges must be increasing with at least two values")
    return list(pairwise(e))


def evaluate_bands(
    predict: Fn,
    truth: Fn,
    edges: Sequence[float],
    points_per_band: int,
    scale: float,
    rng: np.random.Generator,
) -> list[BandMetrics]:
    """Score random points in each |x| band, separately for x > 0, x < 0 and both together."""
    out: list[BandMetrics] = []
    for lo, hi in band_pairs(edges):
        pos = rng.uniform(lo, hi, points_per_band)
        neg = -rng.uniform(lo, hi, points_per_band)
        for sign, xs in (("+", pos), ("-", neg), ("both", np.concatenate([pos, neg]))):
            p, t = predict(xs), truth(xs)
            d = np.abs(p - t)
            r = rmse(p, t)
            out.append(
                BandMetrics(
                    lo, hi, sign, len(xs), r, float(np.mean(d)), float(np.max(d)), r / scale
                )
            )
    return out


def horizon_band(
    bands: Sequence[BandMetrics], eps: float, sign: str = "both"
) -> tuple[float, float]:
    """(lo, hi) of the first band whose NRMSE exceeds eps; (inf, inf) when none does.

    The true crossing lies somewhere inside that band, so lo is the conservative horizon and
    hi bounds it from above.
    """
    for b in sorted((b for b in bands if b.sign == sign), key=lambda b: b.lo):
        if not math.isfinite(b.nrmse) or b.nrmse > eps:
            return b.lo, b.hi
    return math.inf, math.inf


def horizon(bands: Sequence[BandMetrics], eps: float, sign: str = "both") -> float:
    """Lower edge of the first band whose NRMSE exceeds eps; inf when none does."""
    return horizon_band(bands, eps, sign)[0]


def dense_window(center: float, period: float, n_periods: float, n_points: int) -> np.ndarray:
    half = 0.5 * n_periods * period
    return np.linspace(center - half, center + half, n_points)


def phase_offset(pred: np.ndarray, truth: np.ndarray, x: np.ndarray, period: float) -> float:
    """Phase lag (radians) that best aligns pred with truth on a uniformly spaced window.

    Positive means the prediction lags the truth: pred(x) ~ truth(x - lag). Uses the circular
    cross-correlation peak with parabolic refinement, so the resolution is finer than the grid.
    """
    p = np.asarray(pred, dtype=np.float64) - np.mean(pred)
    t = np.asarray(truth, dtype=np.float64) - np.mean(truth)
    if np.allclose(p, 0.0) or np.allclose(t, 0.0):
        return math.nan
    n = len(x)
    corr = np.fft.irfft(np.fft.rfft(p) * np.conj(np.fft.rfft(t)), n=n)
    i = int(np.argmax(corr))
    left, mid, right = corr[(i - 1) % n], corr[i], corr[(i + 1) % n]
    denom = left - 2.0 * mid + right
    frac = 0.5 * (left - right) / denom if denom != 0.0 else 0.0
    lag_samples = i + frac
    if lag_samples > n / 2:
        lag_samples -= n
    dx = float(x[1] - x[0])
    lag = lag_samples * dx
    return float(np.mod(TWO_PI * lag / period + math.pi, TWO_PI) - math.pi)


def shape_gain(predict: Fn, period: float, x_center: float = 0.0, n: int = 8192) -> float:
    """rms(df/dθ) / std(f) over one learned period: converts a phase error into an NRMSE.

    For sin x it is 1; for shapes with sharp features it is large, which makes the predicted
    horizon conservative.
    """
    x = x_center + np.linspace(0.0, period, n, endpoint=False)
    f = predict(x)
    s = float(np.std(f))
    if s == 0.0:
        return math.inf
    dtheta = TWO_PI / n
    df = np.gradient(f, dtheta)
    return float(np.sqrt(np.mean(df * df)) / s)


def predicted_horizon(omega_stderr: float, eps_nrmse: float, gain: float) -> float:
    """Distance at which a one-sigma frequency error produces an NRMSE of eps_nrmse."""
    if omega_stderr <= 0.0 or not math.isfinite(gain) or gain <= 0.0:
        return math.inf
    return eps_nrmse / (omega_stderr * gain)


def omega_agreement(omega: float, omega_true: float) -> dict[str, float]:
    """Relative error, and the error after snapping to the nearest small rational multiple."""
    ratio = omega / omega_true
    ratios = sorted({m / n for m in range(1, 5) for n in range(1, 5)})
    nearest = min(ratios, key=lambda r: abs(math.log(ratio / r)))
    return {
        "omega": omega,
        "omega_true": omega_true,
        "rel_err": ratio - 1.0,
        "ratio": ratio,
        "nearest_ratio": nearest,
        "rel_err_to_nearest": ratio / nearest - 1.0,
    }


def bands_to_records(bands: Sequence[BandMetrics]) -> list[dict[str, Any]]:
    return [asdict(b) | {"center": b.center} for b in bands]
