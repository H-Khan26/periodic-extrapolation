from __future__ import annotations

import functools
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from periodic.config import load_config
from periodic.data import sample_x
from periodic.models import MODELS, FitResult, Regressor, load_model
from periodic.runs import write_json
from periodic.targets import get_target

X_MIN, X_MAX = -10 * np.pi, 10 * np.pi
CFG = load_config(None, ["model.mlp_epochs=60", "model.mlp_width=16", "model.mlp_depth=2"])
BASELINES = [
    "constant",
    "relu_mlp",
    "snake",
    "rff",
    "gp_periodic",
    "gp_periodic_default",
    "oracle",
    "harmonic_random_init",
    "harmonic_offset_init",
]
# The GPs fit a 300-point subsample, a few seconds each, which also exercises the subsampling.
EXTRA: dict[str, dict[str, Any]] = {
    "oracle": {"period": 2 * np.pi},
    "gp_periodic": {"max_points": 300},
    "gp_periodic_default": {"max_points": 300},
}
FAR = np.array([0.0, 1e3, -1e5, 1e6])


def make(name: str = "sin", n: int = 512, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    x = sample_x(n, X_MIN, X_MAX, "stratified", np.random.default_rng(seed))
    return x, get_target(name)(x)


def build(name: str, seed: int = 0) -> Regressor:
    return MODELS.get(name).from_config(CFG, seed, **EXTRA.get(name, {}))


@functools.cache
def fitted(name: str) -> tuple[Regressor, FitResult]:
    """Each baseline fitted once on sin x with 512 samples, validated on 64 fresh samples."""
    x, y = make()
    x_val, y_val = make(n=64, seed=1)
    model = build(name)
    return model, model.fit(x, y, x_val, y_val)


def test_every_registered_baseline_is_covered() -> None:
    assert set(MODELS.names()) - {"harmonic"} == set(BASELINES)


@pytest.mark.parametrize("name", BASELINES)
def test_fit_predict_and_bitwise_round_trip(name: str, tmp_path: Path) -> None:
    model, res = fitted(name)
    assert model.name == name
    assert np.isfinite(res.train_mse) and res.seconds >= 0
    assert res.val_mse is not None and np.isfinite(res.val_mse)
    write_json(tmp_path / "details.json", res.details)  # as the training code stores them

    for x in (make()[0].reshape(16, 32), FAR, np.float64(0.5)):
        pred = model.predict(x)
        assert pred.shape == np.shape(x)
        assert pred.dtype == np.float64
        assert np.all(np.isfinite(pred))

    model.save(tmp_path / name)
    back = load_model(tmp_path / name)
    assert type(back) is type(model)
    assert back.period == model.period
    assert np.array_equal(back.predict(FAR), model.predict(FAR))


@pytest.mark.parametrize("name", BASELINES)
def test_unfitted_model_raises(name: str) -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        build(name).predict(FAR)


@pytest.mark.parametrize(
    "name", ["relu_mlp", "snake", "rff", "harmonic_random_init", "harmonic_offset_init"]
)
def test_same_seed_gives_identical_predictions(name: str) -> None:
    x, y = make()
    a, b = build(name, seed=3), build(name, seed=3)
    a.fit(x, y)
    b.fit(x, y)
    assert np.array_equal(a.predict(FAR), b.predict(FAR))


def test_constant_predicts_the_training_mean() -> None:
    model, _ = fitted("constant")
    _, y = make()
    assert np.array_equal(model.predict(FAR), np.full(FAR.shape, np.mean(y)))


def test_oracle_reproduces_sine_far_away() -> None:
    model, _ = fitted("oracle")
    assert model.period == 2 * np.pi
    np.testing.assert_allclose(model.predict(np.array([1e6])), np.sin([1e6]), atol=1e-8)


def test_oracle_caps_harmonics_for_few_samples() -> None:
    x, y = make(n=64)
    model = build("oracle")
    res = model.fit(x, y)
    assert res.details["harmonics"] == 15  # (64 - 1) // 4, below k_max = 64
    np.testing.assert_allclose(model.predict(np.array([1e6])), np.sin([1e6]), atol=1e-8)


def test_oracle_needs_the_true_period() -> None:
    oracle = MODELS.get("oracle")
    with pytest.raises(ValueError, match="oracle needs the true period"):
        oracle.from_config(CFG, 0)
    with pytest.raises(ValueError, match="oracle needs the true period"):
        oracle.from_config(CFG, 0, period=None)


def test_relu_mlp_is_not_periodic_far_away() -> None:
    model, _ = fitted("relu_mlp")
    p, q = model.predict(np.array([1e6, 1e6 + 2 * np.pi]))
    assert abs(p - q) > 1e-3 or abs(p) > 2


def test_rff_bandwidth_follows_the_periodogram_and_fits_sin30() -> None:
    x, y = make("sin30")
    model = build("rff")
    res = model.fit(x, y)
    assert res.train_mse / np.var(y) < 1e-3
    assert res.details["sigma_b"] == max(2 * res.details["omega_peak"], 0.5)
    state = model.state()
    assert state["sigma_b"] == res.details["sigma_b"]
    assert state["omega_peak"] == res.details["omega_peak"]


def test_gp_periodic_recovers_the_period() -> None:
    model, _ = fitted("gp_periodic")
    assert model.period is not None
    assert abs(model.period - 2 * np.pi) < 0.05 * 2 * np.pi
    assert model.state()["x_train"].shape == (300,)


def test_gp_periodic_carries_the_spectral_config(tmp_path: Path) -> None:
    model, _ = fitted("gp_periodic")
    assert model.state()["spectral"] == asdict(CFG.spectral)
    model.save(tmp_path / "gp")
    assert load_model(tmp_path / "gp").spectral == CFG.spectral
    cfg = load_config(None, ["spectral.k_scan=5", "spectral.oversample=4"])
    custom = MODELS.get("gp_periodic").from_config(cfg, 0)
    assert custom.spectral == cfg.spectral


def _width() -> float:
    x, _ = make()
    return float(x.max() - x.min())


def test_harmonic_random_init_exposes_a_finite_period() -> None:
    model, res = fitted("harmonic_random_init")
    assert model.period is not None and np.isfinite(model.period) and model.period > 0
    assert model.period == pytest.approx(2 * np.pi / abs(res.details["omega_final"]))
    width = _width()
    assert 2 * np.pi / width <= res.details["omega_initial"] <= np.pi * 512 / width


def test_harmonic_offset_init_starts_a_few_basins_off_the_peak() -> None:
    model, res = fitted("harmonic_offset_init")
    d = res.details
    assert model.period is not None and np.isfinite(model.period) and model.period > 0
    assert model.period == pytest.approx(2 * np.pi / abs(d["omega_final"]))
    resolution = 2 * np.pi / _width()
    assert abs(d["omega_peak"] - 1.0) < resolution  # sin x peaks at ω = 1
    assert 1.0 <= abs(d["offset_in_resolutions"]) <= 5.0
    expected = d["omega_peak"] + d["offset_in_resolutions"] * resolution
    assert d["omega_initial"] == pytest.approx(expected, rel=1e-12)
