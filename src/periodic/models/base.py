"""Regressor interface shared by the main model and the baselines.

A regressor sees only arrays. It never receives the target function, its period or any clean
labels; the import-boundary test keeps :mod:`periodic.targets` out of this package.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from periodic.registry import Registry

if TYPE_CHECKING:
    from periodic.config import Config


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
        """Flat dict of JSON scalars/lists plus numpy arrays; everything needed to predict."""

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
        (path / "model.json").write_text(
            json.dumps({"class": self.name, "state": scalars}, indent=2), encoding="utf-8"
        )


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
