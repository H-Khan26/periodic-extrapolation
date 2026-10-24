"""Periodic-by-construction regressor: one learned fundamental ω and a head on harmonic features.

The output is f(x) = mean + scale * head(theta), theta = omega * (x - x_c) mod 2*pi,
so it repeats exactly for any parameter values. ω comes from :mod:`periodic.spectral`.

Heads
- linear: a truncated Fourier series, coefficients in closed form at the estimated ω.
- mlp: a small MLP on [cos kθ, sin kθ] for the first few harmonics, trained with Adam and then
  polished with L-BFGS (strong-Wolfe line search). ω stays frozen at the spectral estimate.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from periodic.config import SpectralConfig
from periodic.features import TWO_PI, harmonic_design, phase
from periodic.models.base import (
    MODELS,
    FitResult,
    NotFittedError,
    Regressor,
    mse,
    standardize,
    thin,
    window_center,
)
from periodic.spectral import OmegaEstimate, estimate_omega, profile_fit

if TYPE_CHECKING:
    from periodic.config import Config

HEADS = ("linear", "mlp")


class PeriodicMLP(torch.nn.Module):
    """MLP on the first n_harm harmonics of θ; input and output are float64."""

    k: torch.Tensor

    def __init__(self, n_harm: int, width: int, depth: int) -> None:
        super().__init__()
        self.register_buffer("k", torch.arange(1, n_harm + 1, dtype=torch.float64))
        layers: list[torch.nn.Module] = []
        d_in = 2 * n_harm
        for _ in range(depth):
            layers += [torch.nn.Linear(d_in, width, dtype=torch.float64), torch.nn.SiLU()]
            d_in = width
        layers.append(torch.nn.Linear(d_in, 1, dtype=torch.float64))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        arg = theta[:, None] * self.k[None, :]
        feats = torch.cat([torch.cos(arg), torch.sin(arg)], dim=1)
        return self.net(feats).squeeze(-1)


@MODELS.register("harmonic")
class HarmonicRegressor(Regressor):
    name = "harmonic"

    def __init__(
        self,
        head: str = "linear",
        spectral: SpectralConfig | None = None,
        k_max: int = 64,
        mlp_harmonics: int = 4,
        mlp_width: int = 64,
        mlp_depth: int = 2,
        mlp_epochs: int = 3000,
        mlp_lr: float = 3e-3,
        mlp_lbfgs: bool = True,
        seed: int = 0,
    ) -> None:
        if head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}, got {head!r}")
        self.head = head
        self.spectral = spectral or SpectralConfig()
        self.k_max = k_max
        self.mlp_harmonics = mlp_harmonics
        self.mlp_width = mlp_width
        self.mlp_depth = mlp_depth
        self.mlp_epochs = mlp_epochs
        self.mlp_lr = mlp_lr
        self.mlp_lbfgs = mlp_lbfgs
        self.seed = seed

        self.omega: float | None = None
        self.omega_stderr: float = float("nan")
        self.x_center: float = 0.0
        self.y_mean: float = 0.0
        self.y_scale: float = 1.0
        self.ks: tuple[int, ...] = ()
        self.coef: np.ndarray | None = None
        self.mlp: PeriodicMLP | None = None
        self.estimate: OmegaEstimate | None = None

    @classmethod
    def from_config(cls, cfg: Config, seed: int, **extra: Any) -> HarmonicRegressor:
        m = cfg.model
        return cls(
            head=str(extra.get("head", m.heads[0])),
            spectral=cfg.spectral,
            k_max=m.k_max,
            mlp_harmonics=m.mlp_harmonics,
            mlp_width=m.mlp_width,
            mlp_depth=m.mlp_depth,
            mlp_epochs=m.mlp_epochs,
            mlp_lr=m.mlp_lr,
            mlp_lbfgs=m.mlp_lbfgs,
            seed=seed,
        )

    # -- fitting ------------------------------------------------------------------------------

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        *,
        omega_estimate: OmegaEstimate | None = None,
        width: float | None = None,
    ) -> FitResult:
        t0 = time.perf_counter()
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.x_center = window_center(x)
        self.y_mean, self.y_scale, z = standardize(y)

        est = omega_estimate or estimate_omega(x, z, self.spectral, self.x_center, width)
        self.estimate = est
        self.omega = est.omega
        self.omega_stderr = est.stderr

        if self.head == "linear":
            ks = [k for k in est.harmonics if k <= self.k_max] or [1]
            coef, _, _ = profile_fit(x, z, est.omega, ks, self.x_center)
            self.ks, self.coef = tuple(ks), coef
            details: dict[str, Any] = {}
        else:
            details = self._fit_mlp(x, z, est)

        train_mse = mse(self.predict(x), y)
        val_mse = (
            mse(self.predict(x_val), y_val) if x_val is not None and y_val is not None else None
        )
        details.update(
            {
                "omega": self.omega,
                "omega_stderr": est.stderr,
                "omega_initial": est.omega,
                "harmonics": list(est.harmonics),
                "n_cycles": est.n_cycles,
                "low_confidence": est.low_confidence,
                "promoted_by": est.promoted_by,
                "head_harmonics": list(self.ks) if self.head == "linear" else self.mlp_harmonics,
            }
        )
        return FitResult(train_mse, val_mse, time.perf_counter() - t0, details)

    def _fit_mlp(self, x: np.ndarray, z: np.ndarray, est: OmegaEstimate) -> dict[str, Any]:
        torch.manual_seed(self.seed)
        u = torch.from_numpy(x - self.x_center)
        target = torch.from_numpy(z)
        omega = torch.tensor(est.omega, dtype=torch.float64)
        model = PeriodicMLP(self.mlp_harmonics, self.mlp_width, self.mlp_depth)

        def loss_fn() -> torch.Tensor:
            return torch.mean((model(omega * u) - target) ** 2)

        opt = torch.optim.Adam(model.parameters(), lr=self.mlp_lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(self.mlp_epochs, 1), eta_min=self.mlp_lr * 1e-3
        )
        history: list[float] = []
        for _ in range(self.mlp_epochs):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn()
            loss.backward()
            opt.step()
            sched.step()
            history.append(loss.detach().item())
        adam_mse = loss_fn().detach().item()

        # ω stays frozen at the variable-projection estimate: it already sits at the information
        # limit, and letting the head move it only trades far-field phase for in-window fit.
        polish: dict[str, Any] = {"applied": False}
        if self.mlp_lbfgs:
            snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
            lbfgs = torch.optim.LBFGS(
                list(model.parameters()),
                lr=1.0,
                max_iter=200,
                history_size=50,
                line_search_fn="strong_wolfe",
                tolerance_grad=0.0,
                tolerance_change=0.0,
            )

            def closure() -> torch.Tensor:
                lbfgs.zero_grad(set_to_none=True)
                loss = loss_fn()
                loss.backward()
                return loss

            lbfgs.step(closure)
            post = loss_fn().detach().item()
            if np.isfinite(post) and post < adam_mse:
                polish = {"applied": True, "mse_before": adam_mse, "mse_after": post}
            else:
                model.load_state_dict(snapshot)
                polish = {"applied": False, "mse_before": adam_mse, "mse_after": post}

        model.eval()
        self.mlp = model
        return {
            "adam_final_mse": adam_mse,
            "adam_history": thin(history),
            "lbfgs": polish,
        }

    # -- prediction -----------------------------------------------------------------------------

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.omega is None:
            raise NotFittedError
        theta = phase(np.asarray(x, dtype=np.float64).ravel(), self.omega, self.x_center)
        if self.head == "linear":
            assert self.coef is not None
            z = harmonic_design(theta, self.ks) @ self.coef
        else:
            assert self.mlp is not None
            with torch.no_grad():
                z = self.mlp(torch.from_numpy(theta)).numpy()
        out = self.y_mean + self.y_scale * z
        return out.reshape(np.shape(x))

    @property
    def period(self) -> float | None:
        return TWO_PI / self.omega if self.omega else None

    # -- persistence ----------------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        if self.omega is None:
            raise NotFittedError
        st: dict[str, Any] = {
            "head": self.head,
            "omega": self.omega,
            "omega_stderr": self.omega_stderr,
            "x_center": self.x_center,
            "y_mean": self.y_mean,
            "y_scale": self.y_scale,
            "k_max": self.k_max,
            "mlp_harmonics": self.mlp_harmonics,
            "mlp_width": self.mlp_width,
            "mlp_depth": self.mlp_depth,
            "mlp_epochs": self.mlp_epochs,
            "mlp_lr": self.mlp_lr,
            "mlp_lbfgs": self.mlp_lbfgs,
            "seed": self.seed,
            "spectral": asdict(self.spectral),
            "ks": list(self.ks),
        }
        if self.coef is not None:
            st["coef"] = np.asarray(self.coef)
        if self.mlp is not None:
            for k, v in self.mlp.state_dict().items():
                st[f"mlp.{k}"] = v.detach().cpu().numpy()
        return st

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> HarmonicRegressor:
        m = cls(
            head=state["head"],
            spectral=SpectralConfig(**state["spectral"]),
            k_max=int(state["k_max"]),
            mlp_harmonics=int(state["mlp_harmonics"]),
            mlp_width=int(state["mlp_width"]),
            mlp_depth=int(state["mlp_depth"]),
            mlp_epochs=int(state["mlp_epochs"]),
            mlp_lr=float(state["mlp_lr"]),
            mlp_lbfgs=bool(state["mlp_lbfgs"]),
            seed=int(state["seed"]),
        )
        m.omega = float(state["omega"])
        m.omega_stderr = float(state["omega_stderr"])
        m.x_center = float(state["x_center"])
        m.y_mean = float(state["y_mean"])
        m.y_scale = float(state["y_scale"])
        m.ks = tuple(int(k) for k in state["ks"])
        if "coef" in state:
            m.coef = np.asarray(state["coef"], dtype=np.float64)
        weights = {
            k[4:]: torch.from_numpy(np.asarray(v)) for k, v in state.items() if k.startswith("mlp.")
        }
        if weights:
            mlp = PeriodicMLP(m.mlp_harmonics, m.mlp_width, m.mlp_depth)
            mlp.load_state_dict(weights)
            mlp.eval()
            m.mlp = mlp
        return m
