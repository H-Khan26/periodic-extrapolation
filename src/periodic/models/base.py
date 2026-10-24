"""Regressor interface shared by the main model, the baselines and the ablations.

A regressor sees only arrays. It never receives the target function, its period or any clean
labels; the import-boundary test keeps :mod:`periodic.targets` out of this package.
"""

from __future__ import annotations

import json
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch

from periodic.registry import Registry
from periodic.runs import write_json

if TYPE_CHECKING:
    from periodic.config import Config

LR_FLOOR = 1e-3  # cosine decay ends at this fraction of each initial learning rate


class NotFittedError(RuntimeError):
    """Raised when a model is used before ``fit`` or ``from_state``."""

    def __init__(self) -> None:
        super().__init__("model is not fitted")


@dataclass
class FitResult:
    train_mse: float
    val_mse: float | None
    seconds: float
    details: dict[str, Any] = field(default_factory=dict)


def mse(pred: np.ndarray, truth: np.ndarray) -> float:
    d = np.asarray(pred, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return float(np.mean(d * d))


class Regressor(ABC):
    """Fit on (x, y) arrays, predict anywhere, round-trip through ``save``/``load_state``."""

    name: ClassVar[str] = "regressor"

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: Config, seed: int, **extra: Any) -> Regressor: ...

    @abstractmethod
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> FitResult: ...

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray: ...

    @property
    def period(self) -> float | None:
        """Learned period, when the model has one."""
        return None

    @abstractmethod
    def state(self) -> dict[str, Any]:
        """Flat dict of scalars/lists (numpy scalars allowed) plus arrays; enough to predict."""

    @classmethod
    @abstractmethod
    def from_state(cls, state: dict[str, Any]) -> Regressor: ...

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = self.state()
        arrays = {k: np.asarray(v) for k, v in state.items() if isinstance(v, np.ndarray)}
        scalars = {k: v for k, v in state.items() if not isinstance(v, np.ndarray)}
        np.savez(path / "arrays.npz", **arrays)  # type: ignore[arg-type]
        write_json(path / "model.json", {"class": self.name, "state": scalars})


MODELS: Registry[type[Regressor]] = Registry("model")


def load_state(path: Path) -> tuple[str, dict[str, Any]]:
    """Read a saved model directory; returns (registry name, merged state)."""
    path = Path(path)
    meta = json.loads((path / "model.json").read_text(encoding="utf-8"))
    state: dict[str, Any] = dict(meta["state"])
    arrays_path = path / "arrays.npz"
    if arrays_path.exists():
        with np.load(arrays_path) as z:
            state.update({k: z[k] for k in z.files})
    return str(meta["class"]), state


# -- helpers shared by the regressors ---------------------------------------------------------


def as_flat(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).ravel()


def window_center(x: np.ndarray) -> float:
    return 0.5 * float(np.min(x) + np.max(x))


def standardize(y: np.ndarray) -> tuple[float, float, np.ndarray]:
    """(mean, scale, z) with z = (y - mean) / scale; the scale is 1 for a constant signal."""
    mean = float(np.mean(y))
    std = float(np.std(y))
    scale = std if std > 0 else 1.0
    return mean, scale, (y - mean) / scale


def fit_result(
    model: Regressor,
    x: np.ndarray,
    y: np.ndarray,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
    t0: float,
    details: dict[str, Any],
) -> FitResult:
    """Train and (when given) validation MSE of a fitted model, with the time since t0."""
    val = mse(model.predict(x_val), y_val) if x_val is not None and y_val is not None else None
    return FitResult(mse(model.predict(x), y), val, time.perf_counter() - t0, details)


def cosine_decay(opt: torch.optim.Optimizer, epochs: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine decay of every parameter group's learning rate to LR_FLOOR times its start value."""
    total = max(epochs, 1)

    def factor(step: int) -> float:
        cos = math.cos(math.pi * min(step, total) / total)
        return LR_FLOOR + (1.0 - LR_FLOOR) * 0.5 * (1.0 + cos)

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


def thin(history: list[float], n: int = 50) -> list[float]:
    """About n evenly spaced entries of a loss history, for the run's fit details."""
    return history[:: max(len(history) // n, 1)]
