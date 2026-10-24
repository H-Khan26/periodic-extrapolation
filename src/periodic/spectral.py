"""Fundamental-frequency estimation: spectral scan, variable projection, fundamental promotion.

The loss in ω is a comb of local minima whose correct basin is only about 2π/L wide (L is the
window width), so ω is found by a global scan and then refined in one dimension. The Fourier
coefficients are solved in closed form at every trial ω (variable projection), which turns the
whole fit into a 1-D problem.

Stages
1. Single-term Lomb-Scargle periodogram on a grid from ω_min to ω_max.
2. Harmonic-sum score HS(ω) = Σ_k P(kω) for k ≤ k_scan: a fundamental scores high even when the
   periodogram has no peak at ω itself (missing fundamental, merged peaks).
3. Variable-projection refinement of the top local maxima: bounded Brent, then Newton polish on the
   profiled residual with an analytic first derivative.
4. Noise-aware tie-break preferring the largest ω (smallest period), then promotion by the
   greatest common divisor of the significant harmonics (P versus 2P ambiguity).
5. Re-refinement using only the significant harmonics, and a Fisher-information standard error.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import lombscargle

from periodic.config import SpectralConfig
from periodic.features import TWO_PI, design_matrix, max_harmonics, nyquist_omega

EPS = np.finfo(np.float64).eps


@dataclass
class Candidate:
    omega0: float
    omega: float
    mse: float
    score: float


@dataclass
class OmegaEstimate:
    omega: float
    stderr: float
    harmonics: tuple[int, ...]
    mse: float
    omega_min: float
    omega_max: float
    n_cycles: float
    promoted_by: int
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def period(self) -> float:
        return TWO_PI / self.omega

    @property
    def low_confidence(self) -> bool:
        """Fewer than two cycles in the window: ω is pinned by parsimony, not by repetition."""
        return self.n_cycles < 2.0

    def phase_horizon(self, eps_rad: float) -> float:
        """Distance at which the one-sigma phase error reaches eps_rad (inf when stderr is 0)."""
        return eps_rad / self.stderr if self.stderr > 0 else math.inf

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["period"] = self.period
        d["low_confidence"] = self.low_confidence
        return d


# -- profiled residual ------------------------------------------------------------------------


def profile_fit(
    x: np.ndarray, y: np.ndarray, omega: float, ks: Sequence[int], x_center: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares Fourier coefficients at a fixed ω. Returns (coef, residual, mse)."""
    phi = design_matrix(x, omega, ks, x_center)
    coef, *_ = np.linalg.lstsq(phi, y, rcond=None)
    r = y - phi @ coef
    return coef, r, float(r @ r) / len(y)


def profile_mse(
    x: np.ndarray, y: np.ndarray, omega: float, ks: Sequence[int], x_center: float
) -> float:
    return profile_fit(x, y, omega, ks, x_center)[2]


def _design_domega(x: np.ndarray, omega: float, ks: Sequence[int], x_center: float) -> np.ndarray:
    """dΦ/dω: derivative of the design matrix with respect to ω."""
    k = np.asarray(ks, dtype=np.float64)
    u = np.asarray(x, dtype=np.float64) - x_center
    arg = (omega * u)[:, None] * k[None, :]
    ku = u[:, None] * k[None, :]
    return np.hstack([np.zeros((len(u), 1)), -np.sin(arg) * ku, np.cos(arg) * ku])


def _profile_grad_curv(
    x: np.ndarray, y: np.ndarray, omega: float, ks: Sequence[int], x_center: float
) -> tuple[float, float, float]:
    """(mse, dmse/dω, Gauss-Newton d²mse/dω²) of the profiled residual at ω.

    At the least-squares optimum Φᵀr = 0, so the coefficient derivative drops out of dmse/dω.
    """
    phi = design_matrix(x, omega, ks, x_center)
    coef, *_ = np.linalg.lstsq(phi, y, rcond=None)
    r = y - phi @ coef
    v = _design_domega(x, omega, ks, x_center) @ coef
    n = len(y)
    grad = -2.0 * float(r @ v) / n
    proj, *_ = np.linalg.lstsq(phi, v, rcond=None)
    v_perp = v - phi @ proj
    curv = 2.0 * float(v_perp @ v_perp) / n
    return float(r @ r) / n, grad, curv


def refine_omega(
    x: np.ndarray,
    y: np.ndarray,
    omega0: float,
    ks: Sequence[int],
    x_center: float,
    half_width: float,
    omega_min: float,
    max_newton: int = 30,
) -> tuple[float, float]:
    """Minimise the profiled residual near omega0: bounded Brent, then damped Newton polish.

    Brent alone stops at about sqrt(eps) relative precision; the Newton polish with the analytic
    gradient reaches the float64 floor in a few steps when the residual is small.
    """
    lo, hi = max(omega_min, omega0 - half_width), omega0 + half_width
    res = minimize_scalar(
        lambda w: profile_mse(x, y, w, ks, x_center),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-12 * max(omega0, 1.0), "maxiter": 200},
    )
    omega, best = float(res.x), float(res.fun)
    for _ in range(max_newton):
        mse, grad, curv = _profile_grad_curv(x, y, omega, ks, x_center)
        if curv <= 0.0 or not np.isfinite(grad):
            break
        step = -grad / curv
        if abs(step) <= 4.0 * EPS * abs(omega):
            break
        trial = min(max(omega + step, lo), hi)
        trial_mse = profile_mse(x, y, trial, ks, x_center)
        shrink = 0
        while trial_mse > mse and shrink < 8:
            step *= 0.5
            trial = min(max(omega + step, lo), hi)
            trial_mse = profile_mse(x, y, trial, ks, x_center)
            shrink += 1
        if trial_mse > mse:
            break
        omega, best = trial, trial_mse
        if abs(step) <= 4.0 * EPS * abs(omega):
            break
    return omega, best


# -- scan and candidates ---------------------------------------------------------------------


def periodogram(
    x: np.ndarray, y: np.ndarray, grid: np.ndarray, max_elements: int = 1 << 21
) -> np.ndarray:
    """Normalised Lomb-Scargle power on `grid`, evaluated in frequency chunks.

    scipy evaluates every frequency at once and holds about seven (n_samples x n_frequencies)
    intermediates, which is tens of GB for 16k samples on a 65k-point grid. Each chunk is
    max_elements floats, so the peak workspace is about 7 x 8 x max_elements bytes (120 MB by
    default). The result equals a single call to within floating-point rounding.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    grid = np.ascontiguousarray(grid, dtype=np.float64)
    chunk = max(1, max_elements // max(len(x), 1))
    out = np.empty(len(grid), dtype=np.float64)
    for i in range(0, len(grid), chunk):
        out[i : i + chunk] = lombscargle(x, y, grid[i : i + chunk], normalize=True)
    return out


def harmonic_sum_scan(
    x: np.ndarray,
    y: np.ndarray,
    omega_min: float,
    omega_max: float,
    resolution: float,
    oversample: int,
    k_scan: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic-sum score of a single-term periodogram on [omega_min, omega_max].

    Returns (omegas, score). Harmonics that fall beyond the periodogram grid contribute zero.
    """
    step = resolution / oversample
    grid = np.arange(0.5 * omega_min, omega_max * 1.01 + step, step)
    power = periodogram(x, y - np.mean(y), grid)
    omegas = grid[(grid >= omega_min) & (grid <= omega_max)]
    score = np.zeros_like(omegas)
    for k in range(1, k_scan + 1):
        score += np.interp(k * omegas, grid, power, right=0.0)
    return omegas, score


def periodogram_peak(
    x: np.ndarray, y: np.ndarray, oversample: int = 8, nyquist_factor: float = 2.0
) -> float:
    """ω of the single-term periodogram maximum, unrefined (used by baselines and ablations).

    The scan runs from one cycle per sample span to nyquist_factor times the mean-spacing
    Nyquist frequency: jittered samples keep a frequency above Nyquist identifiable, with only
    weak aliases below it.
    """
    width = float(np.ptp(x))
    resolution = TWO_PI / width
    omega_max = nyquist_factor * nyquist_omega(len(x), width)
    omegas, power = harmonic_sum_scan(x, y, resolution, omega_max, resolution, oversample, 1)
    return float(omegas[np.argmax(power)])


def local_maxima(values: np.ndarray) -> np.ndarray:
    """Indices of local maxima, strict on the left and non-strict on the right.

    Endpoints count when they beat their single neighbour: a fundamental at the lowest
    identifiable frequency (one cycle in the window) peaks at the first grid point.
    """
    if len(values) == 0:
        return np.array([], dtype=int)
    if len(values) == 1:
        return np.array([0], dtype=int)
    padded = np.concatenate([[-np.inf], values, [-np.inf]])
    mask = (padded[1:-1] > padded[:-2]) & (padded[1:-1] >= padded[2:])
    return np.where(mask)[0]


def harmonic_energies(coef: np.ndarray, n_harm: int) -> np.ndarray:
    """Energy a_k² + b_k² per harmonic from a [1, cos..., sin...] coefficient vector."""
    return coef[1 : n_harm + 1] ** 2 + coef[n_harm + 1 :] ** 2


def significant_harmonics(
    x: np.ndarray,
    y: np.ndarray,
    omega: float,
    k_max: int,
    x_center: float,
    margin: float,
) -> tuple[list[int], float]:
    """Harmonics whose energy stands above the residual-implied coefficient noise floor.

    Each least-squares coefficient has variance about 2·mse/N under white noise, so a harmonic
    (two coefficients) is kept when its energy exceeds margin · 2 · (2·mse/N), with a relative
    floor of 1e-14 of the total energy for the noiseless case.
    """
    ks = list(range(1, k_max + 1))
    coef, _, mse = profile_fit(x, y, omega, ks, x_center)
    energy = harmonic_energies(coef, k_max)
    total = float(energy.sum())
    floor = max(margin * 2.0 * (2.0 * mse / len(y)), 1e-14 * total)
    sig = [k for k, e in zip(ks, energy, strict=True) if e > floor]
    return (sig or [1]), mse


def promote_fundamental(
    x: np.ndarray,
    y: np.ndarray,
    omega: float,
    k_scan: int,
    x_center: float,
    margin: float,
) -> tuple[float, int]:
    """If all energy sits in harmonics divisible by m, the true fundamental is m·ω.

    A harmonic model at ω/m fits a period-2π/ω signal exactly, so the scan can land on a
    sub-multiple. The greatest m whose non-multiples are all at the noise floor wins.
    """
    ks = list(range(1, k_scan + 1))
    coef, _, mse = profile_fit(x, y, omega, ks, x_center)
    energy = harmonic_energies(coef, k_scan)
    total = float(energy.sum())
    if total <= 0.0:
        return omega, 1
    per_coef = 2.0 * mse / len(y)
    for m in range(k_scan, 1, -1):
        non = [k - 1 for k in ks if k % m]
        mult = [k - 1 for k in ks if not k % m]
        if not non or not mult:
            continue
        e_non = float(energy[non].sum())
        e_mult = float(energy[mult].sum())
        floor = max(margin * 2.0 * len(non) * per_coef, 1e-14 * total)
        if e_non < floor and e_mult > 0.5 * total:
            return omega * m, m
    return omega, 1


def omega_stderr(
    x: np.ndarray, y: np.ndarray, omega: float, ks: Sequence[int], x_center: float
) -> float:
    """Standard error of ω from the Fisher information of the fitted harmonic model.

    Uses the residual variance with a degrees-of-freedom correction; equals the Cramér-Rao bound
    for a sinusoid in white noise and generalises it to any harmonic content.
    """
    phi = design_matrix(x, omega, ks, x_center)
    coef, *_ = np.linalg.lstsq(phi, y, rcond=None)
    r = y - phi @ coef
    v = _design_domega(x, omega, ks, x_center) @ coef
    jac = np.hstack([phi, v[:, None]])
    n, p = jac.shape
    if n <= p:
        return math.inf
    s2 = float(r @ r) / (n - p)
    cov = np.linalg.pinv(jac.T @ jac)
    return float(math.sqrt(max(s2 * cov[-1, -1], 0.0)))


# -- public entry point ----------------------------------------------------------------------


def estimate_omega(
    x: np.ndarray,
    y: np.ndarray,
    cfg: SpectralConfig,
    x_center: float,
    width: float | None = None,
) -> OmegaEstimate:
    """Estimate the fundamental angular frequency of periodic samples (x, y).

    `width` is the training window length (defaults to the sample span) and sets the lowest
    identifiable frequency 2π·min_cycles/width and the periodogram resolution 2π/width.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 8:
        raise ValueError("need at least 8 samples to estimate a frequency")
    width = float(width if width is not None else np.max(x) - np.min(x))
    resolution = TWO_PI / width
    omega_min = cfg.min_cycles * resolution
    omega_max = cfg.omega_max if cfg.omega_max > 0 else nyquist_omega(n, width)
    if omega_max <= omega_min:
        raise ValueError(f"omega_max {omega_max:.4g} must exceed omega_min {omega_min:.4g}")

    # Every least-squares fit must stay overdetermined at small n.
    k_cap = max_harmonics(n)
    k_scan = min(cfg.k_scan, k_cap)
    k_omega_max = min(cfg.k_omega_max, k_cap)

    ks = list(range(1, k_scan + 1))
    omegas, score = harmonic_sum_scan(
        x, y, omega_min, omega_max, resolution, cfg.oversample, k_scan
    )
    peaks = local_maxima(score)
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(score))])
    order = peaks[np.argsort(score[peaks])[::-1][: cfg.n_candidates]]

    candidates: list[Candidate] = []
    for i in order:
        w0 = float(omegas[i])
        w, mse = refine_omega(x, y, w0, ks, x_center, 0.5 * resolution, omega_min)
        candidates.append(Candidate(w0, w, mse, float(score[i])))

    best_mse = min(c.mse for c in candidates)
    var_y = float(np.var(y))
    tol = max(cfg.tie_margin * (2 * k_scan + 1) / n * best_mse, 1e-10 * var_y)
    omega = max(c.omega for c in candidates if c.mse <= best_mse + tol)

    omega, promoted = promote_fundamental(x, y, omega, k_scan, x_center, cfg.sig_margin)
    sig, _ = significant_harmonics(x, y, omega, k_omega_max, x_center, cfg.sig_margin)
    omega, mse = refine_omega(x, y, omega, sig, x_center, 0.25 * resolution, omega_min)
    stderr = omega_stderr(x, y, omega, sig, x_center)
    return OmegaEstimate(
        omega=omega,
        stderr=stderr,
        harmonics=tuple(sig),
        mse=mse,
        omega_min=omega_min,
        omega_max=omega_max,
        n_cycles=omega * width / TWO_PI,
        promoted_by=promoted,
        candidates=candidates,
    )
