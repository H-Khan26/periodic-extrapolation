"""Models: the periodic-by-construction harmonic regressor and, later, the baselines."""

from __future__ import annotations

from pathlib import Path

from periodic.models import harmonic as _harmonic  # noqa: F401  (registers "harmonic")
from periodic.models.base import MODELS, FitResult, Regressor, load_state, mse


def load_model(path: Path) -> Regressor:
    name, state = load_state(path)
    return MODELS.get(name).from_state(state)


__all__ = ["MODELS", "FitResult", "Regressor", "load_model", "load_state", "mse"]
