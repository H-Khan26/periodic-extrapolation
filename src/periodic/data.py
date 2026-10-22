"""Sampling, label noise, caching and the edge-block validation split."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from periodic import __version__
from periodic.config import DataConfig, config_hash
from periodic.targets import get_target

CACHE_FORMAT = 1


@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray
    y_clean: np.ndarray
    x_min: float
    x_max: float
    target: str
    sigma: float
    seed: int

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def amplitude(self) -> float:
        return signal_amplitude(self.y_clean)

    def save(self, path: Path) -> None:
        """Write atomically: a temp file in the same directory is renamed into place."""
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {k: v for k, v in asdict(self).items() if not isinstance(v, np.ndarray)}
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npz.tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                np.savez(fh, x=self.x, y=self.y, y_clean=self.y_clean, meta=json.dumps(meta))
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: Path) -> Dataset:
        with np.load(path) as z:
            meta = json.loads(str(z["meta"]))
            return cls(x=z["x"], y=z["y"], y_clean=z["y_clean"], **meta)


def signal_amplitude(y: np.ndarray) -> float:
    """Half the peak-to-peak range; 1.0 for a constant signal so noise scaling stays defined."""
    amp = 0.5 * float(np.max(y) - np.min(y))
    return amp if amp > 0 else 1.0


def sample_x(
    n: int, x_min: float, x_max: float, strategy: str, rng: np.random.Generator
) -> np.ndarray:
    """Sample n inputs on [x_min, x_max).

    stratified: one uniform draw per equal cell (no gaps, no coherent aliases).
    uniform: independent uniform draws.
    grid: cell centres (coherent aliases above the Nyquist frequency).
    """
    width = x_max - x_min
    cell = width / n
    x: np.ndarray
    if strategy == "stratified":
        x = x_min + (np.arange(n) + rng.random(n)) * cell
    elif strategy == "uniform":
        x = np.sort(rng.uniform(x_min, x_max, n))
    elif strategy == "grid":
        x = x_min + (np.arange(n) + 0.5) * cell
    else:
        raise ValueError(f"unknown sampling strategy {strategy!r}")
    return np.asarray(x, dtype=np.float64)


def make_dataset(cfg: DataConfig, seed: int, use_cache: bool = True) -> Dataset:
    """Generate (or load from cache) the training set for one config and seed.

    The cache key covers the config, the seed, the cache format and the target's definition,
    so editing a target never serves stale samples.
    """
    fn = get_target(cfg.target)
    key = config_hash(
        {
            "cfg": asdict(cfg),
            "seed": seed,
            "format": CACHE_FORMAT,
            "version": __version__,
            "target": fn.fingerprint(),
        },
        n=12,
    )
    cache = Path(cfg.cache_dir) / f"{cfg.target}_{key}.npz"
    if use_cache and cache.exists():
        return Dataset.load(cache)

    x_rng, noise_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    x = sample_x(cfg.n_samples, cfg.x_min, cfg.x_max, cfg.sampling, x_rng)
    y_clean = fn(x)
    sigma = cfg.noise_rel * signal_amplitude(y_clean)
    noise = np.asarray(noise_rng.standard_normal(x.shape[0]), dtype=np.float64)
    y = np.asarray(y_clean + sigma * noise, dtype=np.float64)
    ds = Dataset(x, y, y_clean, cfg.x_min, cfg.x_max, cfg.target, float(sigma), seed)
    if use_cache:
        ds.save(cache)
    return ds


def edge_split(
    x: np.ndarray, x_min: float, x_max: float, frac: float
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (train, val): val is the outer `frac` of the window at each end.

    An edge block is a short extrapolation; a middle block would only measure interpolation.
    """
    width = x_max - x_min
    val = (x < x_min + frac * width) | (x >= x_max - frac * width)
    return ~val, val
