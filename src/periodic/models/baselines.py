"""Baselines the harmonic regressor is compared against. Like every model they see only (x, y).

- constant: the training mean.
- relu_mlp: ReLU MLP on the scaled input; it extrapolates linearly by construction.
- snake: the same MLP with the Snake activation x + sin²(a x) / a (Ziyin, Hartwig and Ueda, 2020),
  a linear trend plus wiggles that does not repeat.
- rff: ridge regression on random Fourier features with a bandwidth set by the periodogram peak;
  quasi-periodic, so it drifts out of phase.
- gp_periodic: exact GP with an ExpSineSquared kernel whose periodicity starts at the spectral
  estimate.
- oracle: Fourier least squares at the true period; an evaluation reference, not a submission.

The ablations of the spectral initialisation live in :mod:`periodic.models.ablations`.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np
import torch

from periodic.config import SpectralConfig
from periodic.features import TWO_PI, design_matrix, max_harmonics
from periodic.models.base import (
    MODELS,
    FitResult,
    NotFittedError,
    Regressor,
    as_flat,
    cosine_decay,
    fit_result,
    standardize,
    thin,
    window_center,
)
from periodic.spectral import estimate_omega, periodogram_peak

if TYPE_CHECKING:
    from periodic.config import Config

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
GP_HYPERPARAMETERS = ("constant_value", "length_scale", "periodicity", "noise_level")


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
        x, y = as_flat(x), as_flat(y)
        self.mean = float(np.mean(y))
        return fit_result(self, x, y, x_val, y_val, t0, {"mean": self.mean})

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise NotFittedError
        return np.full(np.shape(x), self.mean, dtype=np.float64)

    def state(self) -> dict[str, Any]:
        if self.mean is None:
            raise NotFittedError
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
        x, y = as_flat(x), as_flat(y)
        self.x_center = window_center(x)
        half = 0.5 * float(np.ptp(x))
        self.half_width = half if half > 0 else 1.0
        self.y_mean, self.y_scale, z = standardize(y)
        torch.manual_seed(self.seed)
        net = ScalarMLP(self.width, self.depth, self.activation)
        u, target = self._scaled(x), torch.from_numpy(z)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        sched = cosine_decay(opt, self.epochs)
        history: list[float] = []
        for _ in range(self.epochs):
            opt.zero_grad(set_to_none=True)
            loss = torch.mean((net(u) - target) ** 2)
            loss.backward()
            opt.step()
            sched.step()
            history.append(loss.item())
        self.net = net.eval()
        return fit_result(self, x, y, x_val, y_val, t0, {"loss_history": thin(history)})

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise NotFittedError
        x64 = np.asarray(x, dtype=np.float64)
        with torch.no_grad():
            z = self.net(self._scaled(x64.ravel())).numpy()
        return (self.y_mean + self.y_scale * z).reshape(x64.shape)

    def state(self) -> dict[str, Any]:
        if self.net is None:
            raise NotFittedError
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
        x, y = as_flat(x), as_flat(y)
        self.x_center = window_center(x)
        self.y_mean, self.y_scale, z = standardize(y)
        self.omega_peak = periodogram_peak(x, z)
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
        return fit_result(self, x, y, x_val, y_val, t0, details)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.freqs is None or self.coef is None:
            raise NotFittedError
        x64 = np.asarray(x, dtype=np.float64)
        z = self._features(x64.ravel(), self.freqs) @ self.coef
        return (self.y_mean + self.y_scale * z).reshape(x64.shape)

    def state(self) -> dict[str, Any]:
        if self.freqs is None or self.coef is None:
            raise NotFittedError
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


# -- periodic Gaussian process ----------------------------------------------------------------
# scikit-learn is imported inside the methods: it doubles the import time of the inference path,
# which only needs it when a saved GP is loaded.


def _gp(
    constant: float,
    length_scale: float,
    periodicity: float,
    noise: float,
    periodicity_bounds: tuple[float, float],
    **kwargs: Any,
) -> Any:
    """GaussianProcessRegressor with kernel C · ExpSineSquared(l, p) + White."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, ExpSineSquared, WhiteKernel

    periodic = ExpSineSquared(length_scale, periodicity, periodicity_bounds=periodicity_bounds)
    kernel = ConstantKernel(constant) * periodic + WhiteKernel(
        noise, noise_level_bounds=GP_NOISE_BOUNDS
    )
    return GaussianProcessRegressor(kernel, **kwargs)


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
        self.periodicity_bounds: tuple[float, float] | None = None
        self.hyper: dict[str, float] = {}
        self.x_train: np.ndarray | None = None
        self.z_train: np.ndarray | None = None
        self._gp: Any = None

    @classmethod
    def from_config(cls, cfg: Config, seed: int, **extra: Any) -> Self:
        """``extra["max_points"]`` overrides the subsample size."""
        max_points = int(extra.get("max_points", GP_MAX_POINTS))
        return cls(spectral=cfg.spectral, max_points=max_points, seed=seed)

    def _inputs(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_center)[:, None]

    def _predictor(self) -> Any:
        """GP conditioned on the stored subsample at the stored hyperparameters."""
        assert self.x_train is not None and self.z_train is not None
        assert self.periodicity_bounds is not None
        h = self.hyper
        gp = _gp(
            h["constant_value"],
            h["length_scale"],
            h["periodicity"],
            h["noise_level"],
            self.periodicity_bounds,
            optimizer=None,
        )
        return gp.fit(self._inputs(self.x_train), self.z_train)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> FitResult:
        t0 = time.perf_counter()
        x, y = as_flat(x), as_flat(y)
        self.x_center = window_center(x)
        self.y_mean, self.y_scale, z = standardize(y)
        p0 = TWO_PI / estimate_omega(x, z, self.spectral, self.x_center).omega
        self.periodicity_bounds = (0.5 * p0, 2.0 * p0)

        idx = np.arange(len(x))
        if len(x) > self.max_points:
            rng = np.random.default_rng(self.seed)
            idx = np.sort(rng.choice(len(x), self.max_points, replace=False))
        self.x_train, self.z_train = x[idx], z[idx]

        gp = _gp(
            1.0,
            1.0,
            p0,
            GP_NOISE_INIT,
            self.periodicity_bounds,
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
        return fit_result(self, x, y, x_val, y_val, t0, details)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._gp is None:
            raise NotFittedError
        x64 = np.asarray(x, dtype=np.float64)
        z = np.asarray(self._gp.predict(self._inputs(x64.ravel())), dtype=np.float64)
        return (self.y_mean + self.y_scale * z).reshape(x64.shape)

    @property
    def period(self) -> float | None:
        return self.hyper.get("periodicity")

    def state(self) -> dict[str, Any]:
        if self._gp is None or self.x_train is None or self.z_train is None:
            raise NotFittedError
        assert self.periodicity_bounds is not None
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
        x, y = as_flat(x), as_flat(y)
        self.x_center = window_center(x)
        n_harm = min(self.k_max, max_harmonics(len(x)))
        self.ks = tuple(range(1, n_harm + 1))
        phi = design_matrix(x, self.omega, self.ks, self.x_center)
        coef, *_ = np.linalg.lstsq(phi, y, rcond=None)
        self.coef = coef
        return fit_result(self, x, y, x_val, y_val, t0, {"harmonics": n_harm})

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef is None:
            raise NotFittedError
        x64 = np.asarray(x, dtype=np.float64)
        phi = design_matrix(x64.ravel(), self.omega, self.ks, self.x_center)
        return (phi @ self.coef).reshape(x64.shape)

    @property
    def period(self) -> float | None:
        return self.true_period

    def state(self) -> dict[str, Any]:
        if self.coef is None:
            raise NotFittedError
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
