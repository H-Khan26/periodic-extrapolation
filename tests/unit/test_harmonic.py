from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from periodic.config import SpectralConfig, load_config
from periodic.data import edge_split, sample_x
from periodic.models import load_model
from periodic.models.harmonic import HarmonicRegressor, PeriodicMLP
from periodic.targets import get_target

X_MIN, X_MAX = -10 * np.pi, 10 * np.pi
FAST = SpectralConfig(oversample=4, n_candidates=6)


def make(name: str, n: int = 2048, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    x = sample_x(n, X_MIN, X_MAX, "stratified", np.random.default_rng(seed))
    return x, get_target(name)(x)


def test_linear_head_extrapolates_to_one_million_and_round_trips(tmp_path: Path) -> None:
    t = get_target("harmonics3")
    x, y = make("harmonics3")
    m = HarmonicRegressor(head="linear", spectral=FAST)
    res = m.fit(x, y)
    assert res.train_mse < 1e-20
    xs = np.array([1e3, -1e5, 1e6, 1e6 + 1.5, -1e6 + 0.25])
    np.testing.assert_allclose(m.predict(xs), t(xs), atol=1e-6)
    assert m.period == pytest.approx(2 * np.pi, rel=1e-8)
    assert m.predict(np.float64(0.5)).shape == ()

    m.save(tmp_path / "model")
    back = load_model(tmp_path / "model")
    assert isinstance(back, HarmonicRegressor)
    assert back.omega == m.omega
    xs = np.linspace(-1e6, 1e6, 1001)
    assert np.array_equal(back.predict(xs), m.predict(xs))


@pytest.mark.parametrize("head", ["linear", "mlp"])
def test_predictions_repeat_exactly_for_arbitrary_parameters(head: str) -> None:
    rng = np.random.default_rng(3)
    m = HarmonicRegressor(head=head, mlp_harmonics=3, mlp_width=8, mlp_depth=1)
    m.omega = 2.7
    m.x_center = 0.4
    m.y_mean, m.y_scale = 0.3, 2.0
    if head == "linear":
        m.ks = (1, 2, 5)
        m.coef = rng.standard_normal(7)
    else:
        torch.manual_seed(0)
        m.mlp = PeriodicMLP(3, 8, 1).eval()
    period = 2 * np.pi / m.omega
    x = rng.uniform(-30, 30, 200)
    shift = np.round(1e6 / period) * period
    np.testing.assert_allclose(m.predict(x + shift), m.predict(x), atol=1e-8)
    np.testing.assert_allclose(m.predict(x - shift), m.predict(x), atol=1e-8)


def test_mlp_head_learns_a_square_wave_and_round_trips(tmp_path: Path) -> None:
    x, y = make("square", n=1024)
    train, val = edge_split(x, X_MIN, X_MAX, 0.1)
    m = HarmonicRegressor(
        head="mlp", spectral=FAST, mlp_harmonics=4, mlp_width=32, mlp_depth=2, mlp_epochs=300
    )
    res = m.fit(x[train], y[train], x[val], y[val])
    assert res.val_mse is not None and res.val_mse < 0.2
    assert m.omega == res.details["omega_initial"]  # the head never moves ω
    assert m.omega == pytest.approx(1.0, abs=5e-3)
    m.save(tmp_path / "mlp")
    back = load_model(tmp_path / "mlp")
    xs = np.array([0.5, 1e5 + 0.5, -1e6 + 0.5])
    assert np.array_equal(back.predict(xs), m.predict(xs))


def test_from_config_passes_hyperparameters() -> None:
    cfg = load_config(None, ["model.mlp_width=16", "model.k_max=7", "spectral.k_scan=5"])
    m = HarmonicRegressor.from_config(cfg, 4, head="mlp")
    assert m.head == "mlp" and m.mlp_width == 16 and m.k_max == 7 and m.seed == 4
    assert m.spectral.k_scan == 5
