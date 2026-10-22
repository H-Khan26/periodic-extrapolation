"""Inference: load a saved run and predict at arbitrary x.

This module never imports :mod:`periodic.targets` or :mod:`periodic.data`; a test enforces it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from periodic.models import Regressor, load_model


def resolve_model_dir(path: Path) -> Path:
    """Accept either a run directory (containing model/) or a model directory."""
    path = Path(path)
    if (path / "model.json").exists():
        return path
    if (path / "model" / "model.json").exists():
        return path / "model"
    raise FileNotFoundError(f"no model.json under {path} or {path / 'model'}")


def load(path: Path) -> Regressor:
    return load_model(resolve_model_dir(path))


def predict(path: Path, x: np.ndarray) -> np.ndarray:
    return load(path).predict(np.asarray(x, dtype=np.float64))
