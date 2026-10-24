"""Ablations of the spectral initialisation: the harmonic family with ω trained by Adam.

- harmonic_random_init: ω0 drawn log-uniformly between one cycle per window and Nyquist.
- harmonic_offset_init: ω0 a few basins off the periodogram peak.

Both show that gradient descent on ω stays in the tooth of the loss comb it starts in.
"""

from __future__ import annotations

import math
import time
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import torch

from periodic.features import TWO_PI, harmonic_design, nyquist_omega, phase
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
from periodic.spectral import periodogram_peak

if TYPE_CHECKING:
    from periodic.config import Config

TRAINED_OMEGA_HARMONICS = 16
OMEGA_LR_FACTOR = 1e-3  # Adam learning rate for ω, relative to |ω0|
OFFSET_RESOLUTIONS = (1.0, 5.0)  # offset of ω0 from the periodogram peak, in units of 2π/width


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
        x, y = as_flat(x), as_flat(y)
        self.x_center = window_center(x)
        self.y_mean, self.y_scale, z = standardize(y)
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
        sched = cosine_decay(opt, self.epochs)
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
            "loss_history": thin(history),
        }
        return fit_result(self, x, y, x_val, y_val, t0, details)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.omega is None or self.coef is None:
            raise NotFittedError
        x64 = np.asarray(x, dtype=np.float64)
        theta = phase(x64.ravel(), self.omega, self.x_center)
        z = harmonic_design(theta, self.ks) @ self.coef
        return (self.y_mean + self.y_scale * z).reshape(x64.shape)

    @property
    def period(self) -> float | None:
        return TWO_PI / abs(self.omega) if self.omega else None

    def state(self) -> dict[str, Any]:
        if self.omega is None or self.coef is None:
            raise NotFittedError
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
        width = float(np.ptp(x))
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
        resolution = TWO_PI / float(np.ptp(x))
        peak = periodogram_peak(x, z)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        offset = sign * float(rng.uniform(*OFFSET_RESOLUTIONS))
        if peak + offset * resolution <= 0.0:
            offset = abs(offset)
        details = {"omega_peak": peak, "offset_in_resolutions": offset}
        return peak + offset * resolution, details
