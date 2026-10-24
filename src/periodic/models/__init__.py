"""Models: the periodic-by-construction harmonic regressor, the baselines and the ablations."""

from __future__ import annotations

from pathlib import Path

# Importing the model modules registers their classes in MODELS.
from periodic.models import ablations as _ablations  # noqa: F401
from periodic.models import baselines as _baselines  # noqa: F401
from periodic.models import harmonic as _harmonic  # noqa: F401
from periodic.models.base import MODELS, FitResult, NotFittedError, Regressor, load_state, mse


def load_model(path: Path) -> Regressor:
    name, state = load_state(path)
    return MODELS.get(name).from_state(state)


__all__ = ["MODELS", "FitResult", "NotFittedError", "Regressor", "load_model", "load_state", "mse"]
