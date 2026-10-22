from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from periodic.config import DataConfig
from periodic.data import Dataset, edge_split, make_dataset, sample_x, signal_amplitude


def test_stratified_puts_exactly_one_sample_in_each_cell(rng: np.random.Generator) -> None:
    n, lo, hi = 256, -2.0, 6.0
    x = sample_x(n, lo, hi, "stratified", rng)
    cells = np.floor((x - lo) / ((hi - lo) / n)).astype(int)
    assert np.array_equal(cells, np.arange(n))
    assert np.all((x >= lo) & (x < hi))
    assert np.all(np.diff(x) > 0)


@pytest.mark.parametrize("strategy", ["uniform", "grid"])
def test_other_strategies_cover_the_window(strategy: str, rng: np.random.Generator) -> None:
    x = sample_x(100, 0.0, 1.0, strategy, rng)
    assert x.shape == (100,)
    assert x.dtype == np.float64
    assert np.all((x >= 0.0) & (x < 1.0))
    assert np.max(np.diff(np.sort(x))) < 0.1


def test_unknown_strategy_raises(rng: np.random.Generator) -> None:
    with pytest.raises(ValueError, match="unknown sampling strategy"):
        sample_x(10, 0.0, 1.0, "random", rng)


def test_dataset_is_deterministic_per_seed(tmp_path: Path) -> None:
    cfg = DataConfig(target="sin", n_samples=64, cache_dir=str(tmp_path))
    a = make_dataset(cfg, seed=1, use_cache=False)
    b = make_dataset(cfg, seed=1, use_cache=False)
    c = make_dataset(cfg, seed=2, use_cache=False)
    assert np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y)
    assert not np.array_equal(a.x, c.x)
    assert np.array_equal(a.y, a.y_clean)
    assert a.sigma == 0.0


def test_noise_is_relative_to_amplitude(tmp_path: Path) -> None:
    cfg = DataConfig(target="scaled", n_samples=4096, noise_rel=0.1, cache_dir=str(tmp_path))
    ds = make_dataset(cfg, seed=0, use_cache=False)
    assert ds.amplitude == pytest.approx(0.01, rel=1e-3)
    assert ds.sigma == pytest.approx(0.001, rel=1e-3)
    assert np.std(ds.y - ds.y_clean) == pytest.approx(ds.sigma, rel=0.1)


def test_cache_round_trip(tmp_path: Path) -> None:
    cfg = DataConfig(target="square", n_samples=32, noise_rel=0.05, cache_dir=str(tmp_path))
    a = make_dataset(cfg, seed=5, use_cache=True)
    files = list(tmp_path.glob("*.npz"))
    assert len(files) == 1
    b = make_dataset(cfg, seed=5, use_cache=True)
    assert np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y)
    assert b.target == "square" and b.seed == 5 and b.sigma == a.sigma
    c = Dataset.load(files[0])
    assert np.array_equal(c.y_clean, a.y_clean)


def test_edge_split_is_disjoint_complete_and_sized(rng: np.random.Generator) -> None:
    x = sample_x(1000, -1.0, 1.0, "stratified", rng)
    train, val = edge_split(x, -1.0, 1.0, 0.1)
    assert not np.any(train & val)
    assert np.all(train | val)
    assert val.sum() == 200
    assert np.all(np.abs(x[val]) >= 0.8)
    train0, val0 = edge_split(x, -1.0, 1.0, 0.0)
    assert val0.sum() == 0 and train0.all()


def test_signal_amplitude_handles_constants() -> None:
    assert signal_amplitude(np.array([2.0, 2.0])) == 1.0
    assert signal_amplitude(np.array([-1.0, 3.0])) == 2.0
