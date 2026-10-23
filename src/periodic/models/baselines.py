"""Baselines the harmonic regressor is compared against. Like every model they see only (x, y).

- constant: the training mean.
- relu_mlp: ReLU MLP on the scaled input; it extrapolates linearly by construction.
- snake: the same MLP with the Snake activation x + sin²(a x) / a (Ziyin, Hartwig and Ueda, 2020),
  a linear trend plus wiggles that does not repeat.
- rff: ridge regression on random Fourier features with a bandwidth set by the periodogram peak;
  quasi-periodic, so it drifts out of phase.
- gp_periodic: exact GP with an ExpSineSquared kernel whose periodicity starts at the spectral
  estimate.
- gp_periodic_default: the same GP started at a fixed periodicity of 1, which exposes the
  multimodal marginal likelihood.
- oracle: Fourier least squares at the true period; an evaluation reference, not a submission.
- harmonic_random_init, harmonic_offset_init: the harmonic family with ω trained by Adam from a
  log-uniform random start or from a few basins off the periodogram peak; ablations of the
  spectral initialisation.
"""

from __future__ import annotations

import math
import time
from abc import abstractmethod
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, ExpSineSquared, WhiteKernel

from periodic.config import SpectralConfig
from periodic.features import (
    TWO_PI,
    design_matrix,
    harmonic_design,
    max_harmonics,
    nyquist_omega,
    phase,
)
from periodic.models.base import MODELS, FitResult, Regressor, mse
from periodic.spectral import estimate_omega, harmonic_sum_scan

if TYPE_CHECKING:
    from periodic.config import Config

LR_FLOOR = 1e-3  # cosine decay ends at this fraction of each initial learning rate
PEAK_OVERSAMPLE = 8  # periodogram grid points per resolution 2π/width
PEAK_NYQUIST_FACTOR = 2.0  # the peak scan ends at this multiple of the Nyquist frequency
ACTIVATIONS = ("relu", "snake")
SNAKE_A0 = 5.0
SNAKE_EPS = 1e-9  # keeps sin²(a x) / a finite if a reaches zero
RFF_FEATURES = 256
RFF_SIGMA_PER_PEAK = 2.0  # bandwidth sigma_b = 2 ω_peak
RFF_SIGMA_FLOOR = 0.5  # radians per unit x
RFF_RIDGE = 1e-8
GP_MAX_POINTS = 1000
GP_RESTARTS = 2
GP_NOISE_INIT = 1e-2
GP_NOISE_BOUNDS = (1e-8, 1e1)
DEFAULT_PERIODICITY = 1.0
DEFAULT_PERIODICITY_BOUNDS = (1e-2, 1e2)
GP_HYPERPARAMETERS = ("constant_value", "length_scale", "periodicity", "noise_level")
TRAINED_OMEGA_HARMONICS = 16
OMEGA_LR_FACTOR = 1e-3  # Adam learning rate for ω, relative to |ω0|
OFFSET_RESOLUTIONS = (1.0, 5.0)  # offset of ω0 from the periodogram peak, in units of 2π/width


# -- shared helpers ---------------------------------------------------------------------------


def _flat(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).ravel()


def _center_and_half_width(x: np.ndarray) -> tuple[float, float]:
    """Midpoint and half-width of the sample span; the half-width is 1 for a single point."""
    lo, hi = float(np.min(x)), float(np.max(x))
    half = 0.5 * (hi - lo)
    return 0.5 * (lo + hi), (half if half > 0 else 1.0)


def _standardize(y: np.ndarray) -> tuple[float, float, np.ndarray]:
    """(mean, scale, z) with z = (y - mean) / scale; the scale is 1 for a constant signal."""
    mean = float(np.mean(y))
    std = float(np.std(y))
    scale = std if std > 0 else 1.0
    return mean, scale, (y - mean) / scale


def _fit_result(
    model: Regressor,
    x: np.ndarray,
    y: np.ndarray,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
    t0: float,
    details: dict[str, Any],
) -> FitResult:
    val = mse(model.predict(x_val), y_val) if x_val is not None and y_val is not None else None
    return FitResult(mse(model.predict(x), y), val, time.perf_counter() - t0, details)


def _cosine_decay(opt: torch.optim.Optimizer, epochs: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine decay of every parameter group's learning rate to LR_FLOOR times its start value."""
    total = max(epochs, 1)

    def factor(step: int) -> float:
        cos = math.cos(math.pi * min(step, total) / total)
        return LR_FLOOR + (1.0 - LR_FLOOR) * 0.5 * (1.0 + cos)

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


def _thin(history: list[float], n: int = 50) -> list[float]:
    return history[:: max(len(history) // n, 1)]


def _not_fitted() -> RuntimeError:
    return RuntimeError("model is not fitted")


def _span(x: np.ndarray) -> float:
    return float(np.max(x) - np.min(x))


def _periodogram_peak(x: np.ndarray, z: np.ndarray) -> float:
    """ω of the single-term Lomb-Scargle maximum, unrefined.

    The scan runs from one cycle per window to twice the mean-spacing Nyquist frequency:
    jittered samples keep a frequency above Nyquist identifiable, with only weak aliases below it.
    """
    width = _span(x)
    resolution = TWO_PI / width
    omega_max = PEAK_NYQUIST_FACTOR * nyquist_omega(len(x), width)
    _, _, omegas, power = harmonic_sum_scan(
        x, z, resolution, omega_max, resolution, PEAK_OVERSAMPLE, 1
    )
    return float(omegas[np.argmax(power)])


# -- constant ---------------------------------------------------------------------------------


@MODELS.register("constant")
class ConstantRegressor(Regressor):
    """Predicts the training mean everywhere."""

    name = "constant"

    def __init__(self) -> None:
        self.mean: float | None = None

    @classmethod
    def from_config(cls, cfg: Config, seed: int, **extra: Any) -> Self:
        return cls()

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> FitResult:
        t0 = time.perf_counter()
        x, y = _flat(x), _flat(y)
        self.mean = float(np.mean(y))
        return _fit_result(self, x, y, x_val, y_val, t0, {"mean": self.mean})

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise _not_fitted()
        return np.full(np.shape(x), self.mean, dtype=np.float64)

    def state(self) -> dict[str, Any]:
        if self.mean is None:
            raise _not_fitted()
        return {"mean": self.mean}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Self:
        m = cls()
        m.mean = float(state["mean"])
        return m


# -- MLPs on the raw input --------------------------------------------------------------------


class Snake(torch.nn.Module):
    """Snake activation x + sin²(a x) / a with one learnable frequency a per neuron."""

    def __init__(self, width: int, a0: float = SNAKE_A0) -> None:
        super().__init__()
        self.a = torch.nn.Parameter(torch.full((width,), a0, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.sin(self.a * x) ** 2 / (self.a + SNAKE_EPS)


class ScalarMLP(torch.nn.Module):
    """MLP from a scalar input to a scalar output with ReLU or Snake hidden layers, float64."""

    def __init__(self, width: int, depth: int, activation: str) -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}, got {activation!r}")
        layers: list[torch.nn.Module] = []
        d_in = 1
        for _ in range(depth):
            layers.append(torch.nn.Linear(d_in, width, dtype=torch.float64))
            layers.append(Snake(width) if activation == "snake" else torch.nn.ReLU())
            d_in = width
        layers.append(torch.nn.Linear(d_in, 1, dtype=torch.float64))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u[:, None]).squeeze(-1)


class _ScaledInputMLP(Regressor):
    """MLP on u = (x - x_center) / half_width, fitted to standardized y by full-batch Adam.

    MSE loss with cosine learning-rate decay. Subclasses choose the hidden activation.
    """

    activation: ClassVar[str] = "relu"

    def __init__(
        self, width: int = 64, depth: int = 2, epochs: int = 3000, lr: float = 3e-3, seed: int = 0
    ) -> None:
        self.width = width
        self.depth = depth
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.x_center = 0.0
        self.half_width = 1.0
        self.y_mean = 0.0
        self.y_scale = 1.0
        self.net: ScalarMLP | None = None

    @classmethod
    def from_config(cls, cfg: Config, seed: int, **extra: Any) -> Self:
        m = cfg.model
        return cls(m.mlp_width, m.mlp_depth, m.mlp_epochs, m.mlp_lr, seed)

    def _scaled(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy((x - self.x_center) / self.half_width)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> FitResult:
        t0 = time.perf_counter()
        x, y = _flat(x), _flat(y)
        self.x_center, self.half_width = _center_and_half_width(x)
        self.y_mean, self.y_scale, z = _standardize(y)
        torch.manual_seed(self.seed)
        net = ScalarMLP(self.width, self.depth, self.activation)
        u, target = self._scaled(x), torch.from_numpy(z)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        sched = _cosine_decay(opt, self.epochs)
        history: list[float] = []
        for _ in range(self.epochs):
            opt.zero_grad(set_to_none=True)
            loss = torch.mean((net(u) - target) ** 2)
            loss.backward()
            opt.step()
            sched.step()
            history.append(loss.item())
        self.net = net.eval()
        return _fit_result(self, x, y, x_val, y_val, t0, {"loss_history": _thin(history)})

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise _not_fitted()
        x64 = np.asarray(x, dtype=np.float64)
        with torch.no_grad():
            z = self.net(self._scaled(x64.ravel())).numpy()
        return (self.y_mean + self.y_scale * z).reshape(x64.shape)

    def state(self) -> dict[str, Any]:
        if self.net is None:
            raise _not_fitted()
        st: dict[str, Any] = {
            "width": self.width,
            "depth": self.depth,
            "epochs": self.epochs,
            "lr": self.lr,
            "seed": self.seed,
            "x_center": self.x_center,
            "half_width": self.half_width,
            "y_mean": self.y_mean,
            "y_scale": self.y_scale,
        }
        for k, v in self.net.state_dict().items():
            st[f"net.{k}"] = v.detach().cpu().numpy()
        return st

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Self:
        m = cls(
            width=int(state["width"]),
            depth=int(state["depth"]),
            epochs=int(state["epochs"]),
            lr=float(state["lr"]),
            seed=int(state["seed"]),
        )
        m.x_center = float(state["x_center"])
        m.half_width = float(state["half_width"])
        m.y_mean = float(state["y_mean"])
        m.y_scale = float(state["y_scale"])
        net = ScalarMLP(m.width, m.depth, cls.activation)
        weights = {
            k[4:]: torch.from_numpy(np.asarray(v)) for k, v in state.items() if k.startswith("net.")
        }
        net.load_state_dict(weights)
        m.net = net.eval()
        return m


@MODELS.register("relu_mlp")
class ReluMLPRegressor(_ScaledInputMLP):
    """ReLU MLP: piecewise linear, so it extrapolates along its outermost linear pieces."""

    name = "relu_mlp"
    activation = "relu"


@MODELS.register("snake")
class SnakeMLPRegressor(_ScaledInputMLP):
    """Snake MLP with a learnable per-neuron frequency a, initialised at 5."""

    name = "snake"
    activation = "snake"


# -- random Fourier features ------------------------------------------------------------------


@MODELS.register("rff")
class RandomFourierRegressor(Regressor):
    """Ridge regression on random Fourier features [cos(B u), sin(B u)], u = x - x_center.

    B holds n_features fixed frequencies drawn from N(0, sigma_b²) in radians per unit x, with the
    bandwidth sigma_b = 2 ω_peak (at least 0.5) set by the single-term periodogram peak of the
    training data. The ridge penalty is `ridge` times the mean diagonal of the Gram matrix, and
    the head is solved in closed form on standardized y. The output is a sum of incommensurate
    sinusoids: quasi-periodic, so it drifts out of phase away from the window.
    """

    name = "rff"

    def __init__(
        self, n_features: int = RFF_FEATURES, ridge: float = RFF_RIDGE, seed: int = 0
    ) -> None:
        self.n_features = n_features
        self.ridge = ridge
        self.seed = seed
        self.x_center = 0.0
        self.y_mean = 0.0
        self.y_scale = 1.0
        self.omega_peak: float | None = None
        self.sigma_b: float | None = None
        self.freqs: np.ndarray | None = None
        self.coef: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: Config, seed: int, **extra: Any) -> Self:
        return cls(seed=seed)

    def _features(self, x: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        arg = (x - self.x_center)[:, None] * freqs[None, :]
        return np.hstack([np.cos(arg), np.sin(arg)])

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> FitResult:
        t0 = time.perf_counter()
        x, y = _flat(x), _flat(y)
        self.x_center = _center_and_half_width(x)[0]
        self.y_mean, self.y_scale, z = _standardize(y)
        self.omega_peak = _periodogram_peak(x, z)
        self.sigma_b = max(RFF_SIGMA_PER_PEAK * self.omega_peak, RFF_SIGMA_FLOOR)
        rng = np.random.default_rng(self.seed)
        freqs = self.sigma_b * rng.standard_normal(self.n_features)
        phi = self._features(x, freqs)
        gram = phi.T @ phi
        lam = self.ridge * float(np.trace(gram)) / gram.shape[0]
        self.coef = np.linalg.solve(gram + lam * np.eye(gram.shape[0]), phi.T @ z)
        self.freqs = freqs
        details = {
            "omega_peak": self.omega_peak,
            "sigma_b": self.sigma_b,
            "ridge_lambda": lam,
            "max_abs_freq": float(np.max(np.abs(freqs))),
        }
        return _fit_result(self, x, y, x_val, y_val, t0, details)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.freqs is None or self.coef is None:
            raise _not_fitted()
        x64 = np.asarray(x, dtype=np.float64)
        z = self._features(x64.ravel(), self.freqs) @ self.coef
        return (self.y_mean + self.y_scale * z).reshape(x64.shape)

    def state(self) -> dict[str, Any]:
        if self.freqs is None or self.coef is None:
            raise _not_fitted()
        return {
            "n_features": self.n_features,
            "ridge": self.ridge,
            "seed": self.seed,
            "x_center": self.x_center,
            "y_mean": self.y_mean,
            "y_scale": self.y_scale,
            "omega_peak": self.omega_peak,
            "sigma_b": self.sigma_b,
            "freqs": self.freqs,
            "coef": self.coef,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Self:
        m = cls(
            n_features=int(state["n_features"]),
            ridge=float(state["ridge"]),
            seed=int(state["seed"]),
        )
        m.x_center = float(state["x_center"])
        m.y_mean = float(state["y_mean"])
        m.y_scale = float(state["y_scale"])
        m.omega_peak = float(state["omega_peak"])
        m.sigma_b = float(state["sigma_b"])
        m.freqs = np.asarray(state["freqs"], dtype=np.float64)
        m.coef = np.asarray(state["coef"], dtype=np.float64)
        return m


# -- periodic Gaussian processes --------------------------------------------------------------


def _periodic_kernel(
    constant: float,
    length_scale: float,
    periodicity: float,
    noise: float,
    periodicity_bounds: tuple[float, float],
) -> Any:
    periodic = ExpSineSquared(length_scale, periodicity, periodicity_bounds=periodicity_bounds)
    return ConstantKernel(constant) * periodic + WhiteKernel(
        noise, noise_level_bounds=GP_NOISE_BOUNDS
    )


@MODELS.register("gp_periodic")
class PeriodicGPRegressor(Regressor):
    """Exact GP on standardized y with kernel C · ExpSineSquared(l, p) + White.

    The periodicity starts at 2π/ω from :func:`periodic.spectral.estimate_omega` (with
    cfg.spectral) and is bounded to [0.5, 2] times that value. Exact GP training costs
    O(N³) time and O(N²) memory per marginal-likelihood evaluation, so at most max_points samples,
    drawn with the seeded generator, enter the GP. L-BFGS-B fits the hyperparameters with
    n_restarts random restarts; the predictor is then rebuilt at the fitted values with
    optimizer=None, the same deterministic path :meth:`from_state` takes.
    """

    name = "gp_periodic"

    def __init__(
        self,
        spectral: SpectralConfig | None = None,
        max_points: int = GP_MAX_POINTS,
        n_restarts: int = GP_RESTARTS,
        seed: int = 0,
    ) -> None:
        self.spectral = spectral or SpectralConfig()
        self.max_points = max_points
        self.n_restarts = n_restarts
        self.seed = seed
        self.x_center = 0.0
        self.y_mean = 0.0
        self.y_scale = 1.0
        self.periodicity_bounds: tuple[float, float] = DEFAULT_PERIODICITY_BOUNDS
        self.hyper: dict[str, float] = {}
        self.x_train: np.ndarray | None = None
        self.z_train: np.ndarray | None = None
        self._gp: Any = None

    @classmethod
    def from_config(cls, cfg: Config, seed: int, **extra: Any) -> Self:
        """``extra["max_points"]`` overrides the subsample size."""
        max_points = int(extra.get("max_points", GP_MAX_POINTS))
        return cls(spectral=cfg.spectral, max_points=max_points, seed=seed)

    def _initial_periodicity(
        self, x: np.ndarray, z: np.ndarray
    ) -> tuple[float, tuple[float, float]]:
        """Starting periodicity and its optimisation bounds."""
        p0 = TWO_PI / estimate_omega(x, z, self.spectral, self.x_center).omega
        return p0, (0.5 * p0, 2.0 * p0)

    def _inputs(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_center)[:, None]

    def _predictor(self) -> Any:
        """GP conditioned on the stored subsample at the stored hyperparameters."""
        assert self.x_train is not None and self.z_train is not None
        h = self.hyper
        kernel = _periodic_kernel(
            h["constant_value"],
            h["length_scale"],
            h["periodicity"],
            h["noise_level"],
            self.periodicity_bounds,
        )
        gp = GaussianProcessRegressor(kernel, optimizer=None)
        return gp.fit(self._inputs(self.x_train), self.z_train)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> FitResult:
        t0 = time.perf_counter()
        x, y = _flat(x), _flat(y)
        self.x_center = _center_and_half_width(x)[0]
        self.y_mean, self.y_scale, z = _standardize(y)
        p0, self.periodicity_bounds = self._initial_periodicity(x, z)

        idx = np.arange(len(x))
        if len(x) > self.max_points:
            rng = np.random.default_rng(self.seed)
            idx = np.sort(rng.choice(len(x), self.max_points, replace=False))
        self.x_train, self.z_train = x[idx], z[idx]

        gp = GaussianProcessRegressor(
            _periodic_kernel(1.0, 1.0, p0, GP_NOISE_INIT, self.periodicity_bounds),
            n_restarts_optimizer=self.n_restarts,
            random_state=self.seed,
        )
        gp.fit(self._inputs(self.x_train), self.z_train)
        k = gp.kernel_
        self.hyper = {
            "constant_value": float(k.k1.k1.constant_value),
            "length_scale": float(k.k1.k2.length_scale),
            "periodicity": float(k.k1.k2.periodicity),
            "noise_level": float(k.k2.noise_level),
        }
        self._gp = self._predictor()
        details: dict[str, Any] = {
            "periodicity_initial": p0,
            "periodicity_bounds": list(self.periodicity_bounds),
            "n_gp_points": len(idx),
            "log_marginal_likelihood": float(gp.log_marginal_likelihood_value_),
            **self.hyper,
        }
        return _fit_result(self, x, y, x_val, y_val, t0, details)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._gp is None:
            raise _not_fitted()
        x64 = np.asarray(x, dtype=np.float64)
        z = np.asarray(self._gp.predict(self._inputs(x64.ravel())), dtype=np.float64)
        return (self.y_mean + self.y_scale * z).reshape(x64.shape)

    @property
    def period(self) -> float | None:
        return self.hyper.get("periodicity")

    def state(self) -> dict[str, Any]:
        if self._gp is None or self.x_train is None or self.z_train is None:
            raise _not_fitted()
        return {
            "spectral": asdict(self.spectral),
            "max_points": self.max_points,
            "n_restarts": self.n_restarts,
            "seed": self.seed,
            "x_center": self.x_center,
            "y_mean": self.y_mean,
            "y_scale": self.y_scale,
            "periodicity_bounds": list(self.periodicity_bounds),
            **self.hyper,
            "x_train": self.x_train,
            "z_train": self.z_train,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Self:
        m = cls(
            spectral=SpectralConfig(**state["spectral"]),
            max_points=int(state["max_points"]),
            n_restarts=int(state["n_restarts"]),
            seed=int(state["seed"]),
        )
        m.x_center = float(state["x_center"])
        m.y_mean = float(state["y_mean"])
        m.y_scale = float(state["y_scale"])
        lo, hi = state["periodicity_bounds"]
        m.periodicity_bounds = (float(lo), float(hi))
        m.hyper = {k: float(state[k]) for k in GP_HYPERPARAMETERS}
        m.x_train = np.asarray(state["x_train"], dtype=np.float64)
        m.z_train = np.asarray(state["z_train"], dtype=np.float64)
        m._gp = m._predictor()
        return m


@MODELS.register("gp_periodic_default")
class DefaultInitGPRegressor(PeriodicGPRegressor):
    """The periodic GP started at periodicity 1 within (1e-2, 1e2), with no spectral estimate.

    The marginal likelihood in the periodicity is multimodal, so local optimisation from a fixed
    start usually settles on a wrong period.
    """

    name = "gp_periodic_default"

    def _initial_periodicity(
        self, x: np.ndarray, z: np.ndarray
    ) -> tuple[float, tuple[float, float]]:
        return DEFAULT_PERIODICITY, DEFAULT_PERIODICITY_BOUNDS


# -- oracle -----------------------------------------------------------------------------------


@MODELS.register("oracle")
class OracleRegressor(Regressor):
    """Fourier least squares at the true period: an evaluation reference, not a submission model.

    The training code passes the target's period as ``extra["period"]``. Harmonics run from 1 to
    k_max, capped by max_harmonics(n) so the fit stays overdetermined on small samples; beyond
    that the least-squares system is underdetermined or badly conditioned.
    """

    name = "oracle"

    def __init__(self, period: float | None, k_max: int = 64) -> None:
        if period is None:
            raise ValueError("oracle needs the true period")
        self.true_period = float(period)
        self.k_max = k_max
        self.x_center = 0.0
        self.ks: tuple[int, ...] = ()
        self.coef: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: Config, seed: int, **extra: Any) -> Self:
        return cls(extra.get("period"), cfg.model.k_max)

    @property
    def omega(self) -> float:
        return TWO_PI / self.true_period

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> FitResult:
        t0 = time.perf_counter()
        x, y = _flat(x), _flat(y)
        self.x_center = _center_and_half_width(x)[0]
        n_harm = min(self.k_max, max_harmonics(len(x)))
        self.ks = tuple(range(1, n_harm + 1))
        phi = design_matrix(x, self.omega, self.ks, self.x_center)
        coef, *_ = np.linalg.lstsq(phi, y, rcond=None)
        self.coef = coef
        return _fit_result(self, x, y, x_val, y_val, t0, {"harmonics": n_harm})

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef is None:
            raise _not_fitted()
        x64 = np.asarray(x, dtype=np.float64)
        phi = design_matrix(x64.ravel(), self.omega, self.ks, self.x_center)
        return (phi @ self.coef).reshape(x64.shape)

    @property
    def period(self) -> float | None:
        return self.true_period

    def state(self) -> dict[str, Any]:
        if self.coef is None:
            raise _not_fitted()
        return {
            "period": self.true_period,
            "k_max": self.k_max,
            "x_center": self.x_center,
            "ks": list(self.ks),
            "coef": self.coef,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Self:
        m = cls(float(state["period"]), int(state["k_max"]))
        m.x_center = float(state["x_center"])
        m.ks = tuple(int(k) for k in state["ks"])
        m.coef = np.asarray(state["coef"], dtype=np.float64)
        return m


# -- harmonic family with ω trained by gradient descent ---------------------------------------


class _TrainedOmegaHarmonic(Regressor):
    """Harmonic family with ω trained jointly with the coefficients; subclasses choose ω0.

    Features [1, cos kθ, sin kθ] for k = 1..n_harm with θ = ω (x - x_center), fitted to
    standardized y by full-batch Adam on the MSE, with learning rate lr for the coefficients and
    1e-3 |ω0| for ω, both under cosine decay. The output repeats exactly for any ω, but the loss
    in ω is a comb of local minima about 2π/width apart, and gradient descent stays in a tooth
    near its start.
    """

    def __init__(
        self,
        n_harm: int = TRAINED_OMEGA_HARMONICS,
        epochs: int = 3000,
        lr: float = 3e-3,
        seed: int = 0,
    ) -> None:
        self.n_harm = n_harm
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.omega: float | None = None
        self.x_center = 0.0
        self.y_mean = 0.0
        self.y_scale = 1.0
        self.coef: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: Config, seed: int, **extra: Any) -> Self:
        m = cfg.model
        return cls(TRAINED_OMEGA_HARMONICS, m.mlp_epochs, m.mlp_lr, seed)

    @abstractmethod
    def _initial_omega(
        self, x: np.ndarray, z: np.ndarray, rng: np.random.Generator
    ) -> tuple[float, dict[str, Any]]:
        """Starting ω and the details that describe how it was drawn."""

    @property
    def ks(self) -> tuple[int, ...]:
        return tuple(range(1, self.n_harm + 1))

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> FitResult:
        t0 = time.perf_counter()
        x, y = _flat(x), _flat(y)
        self.x_center = _center_and_half_width(x)[0]
        self.y_mean, self.y_scale, z = _standardize(y)
        omega0, init_details = self._initial_omega(x, z, np.random.default_rng(self.seed))

        u = torch.from_numpy(x - self.x_center)
        target = torch.from_numpy(z)
        k = torch.arange(1, self.n_harm + 1, dtype=torch.float64)
        ones = torch.ones(len(x), 1, dtype=torch.float64)
        omega = torch.nn.Parameter(torch.tensor(omega0, dtype=torch.float64))
        coef = torch.nn.Parameter(torch.zeros(2 * self.n_harm + 1, dtype=torch.float64))

        def forward() -> torch.Tensor:
            arg = omega * u[:, None] * k[None, :]
            return torch.cat([ones, torch.cos(arg), torch.sin(arg)], dim=1) @ coef

        opt = torch.optim.Adam(
            [
                {"params": [coef], "lr": self.lr},
                {"params": [omega], "lr": OMEGA_LR_FACTOR * abs(omega0)},
            ]
        )
        sched = _cosine_decay(opt, self.epochs)
        history: list[float] = []
        for _ in range(self.epochs):
            opt.zero_grad(set_to_none=True)
            loss = torch.mean((forward() - target) ** 2)
            loss.backward()
            opt.step()
            sched.step()
            history.append(loss.item())

        self.omega = float(omega.item())
        self.coef = coef.detach().numpy().copy()
        details = {
            **init_details,
            "omega_initial": omega0,
            "omega_final": self.omega,
            "loss_history": _thin(history),
        }
        return _fit_result(self, x, y, x_val, y_val, t0, details)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.omega is None or self.coef is None:
            raise _not_fitted()
        x64 = np.asarray(x, dtype=np.float64)
        theta = phase(x64.ravel(), self.omega, self.x_center)
        z = harmonic_design(theta, self.ks) @ self.coef
        return (self.y_mean + self.y_scale * z).reshape(x64.shape)

    @property
    def period(self) -> float | None:
        return TWO_PI / abs(self.omega) if self.omega else None

    def state(self) -> dict[str, Any]:
        if self.omega is None or self.coef is None:
            raise _not_fitted()
        return {
            "n_harm": self.n_harm,
            "epochs": self.epochs,
            "lr": self.lr,
            "seed": self.seed,
            "omega": self.omega,
            "x_center": self.x_center,
            "y_mean": self.y_mean,
            "y_scale": self.y_scale,
            "coef": self.coef,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> Self:
        m = cls(
            n_harm=int(state["n_harm"]),
            epochs=int(state["epochs"]),
            lr=float(state["lr"]),
            seed=int(state["seed"]),
        )
        m.omega = float(state["omega"])
        m.x_center = float(state["x_center"])
        m.y_mean = float(state["y_mean"])
        m.y_scale = float(state["y_scale"])
        m.coef = np.asarray(state["coef"], dtype=np.float64)
        return m


@MODELS.register("harmonic_random_init")
class RandomInitHarmonicRegressor(_TrainedOmegaHarmonic):
    """Naive practitioner's ablation: ω0 log-uniform over one cycle per window to Nyquist.

    The correct basin is about 2π/width wide out of a range spanning several decades, so the fit
    almost always settles in a wrong tooth of the comb.
    """

    name = "harmonic_random_init"

    def _initial_omega(
        self, x: np.ndarray, z: np.ndarray, rng: np.random.Generator
    ) -> tuple[float, dict[str, Any]]:
        width = _span(x)
        lo, hi = math.log(TWO_PI / width), math.log(nyquist_omega(len(x), width))
        return math.exp(float(rng.uniform(lo, hi))), {}


@MODELS.register("harmonic_offset_init")
class OffsetInitHarmonicRegressor(_TrainedOmegaHarmonic):
    """Ablation that starts a few basins away from the truth: periodogram peak plus an offset.

    ω0 is the single-term periodogram peak, unrefined, plus an offset of 1 to 5 resolutions
    2π/width with a random sign; the sign turns positive when a negative offset would leave
    ω0 ≤ 0. Gradient descent does not cross even these few basins.
    """

    name = "harmonic_offset_init"

    def _initial_omega(
        self, x: np.ndarray, z: np.ndarray, rng: np.random.Generator
    ) -> tuple[float, dict[str, Any]]:
        resolution = TWO_PI / _span(x)
        peak = _periodogram_peak(x, z)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        offset = sign * float(rng.uniform(*OFFSET_RESOLUTIONS))
        if peak + offset * resolution <= 0.0:
            offset = abs(offset)
        details = {"omega_peak": peak, "offset_in_resolutions": offset}
        return peak + offset * resolution, details
