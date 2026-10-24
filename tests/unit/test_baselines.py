"""Contract tests for every registered model other than the main one: baselines and ablations."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from periodic.config import load_config
from periodic.data import sample_x
from periodic.models import MODELS, FitResult, NotFittedError, Regressor, load_model
from periodic.runs import write_json
from periodic.targets import get_target

X_MIN, X_MAX = -10 * np.pi, 10 * np.pi
CFG = load_config(None, ["model.mlp_epochs=60", "model.mlp_width=16", "model.mlp_depth=2"])
MODELS_UNDER_TEST = [
    "constant",
    "relu_mlp",
    "snake",
    "rff",
    "gp_periodic",
    "oracle",
    "harmonic_random_init",
    "harmonic_offset_init",
]
# The GP fits a 300-point subsample, a few seconds, which also exercises the subsampling.
EXTRA: dict[str, dict[str, Any]] = {
    "oracle": {"period": 2 * np.pi},
    "gp_periodic": {"max_points": 300},
}
FAR = np.array([0.0, 1e3, -1e5, 1e6])


def make(name: str = "sin", n: int = 512, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    x = sample_x(n, X_MIN, X_MAX, "stratified", np.random.default_rng(seed))
    return x, get_target(name)(x)


def build(name: str, seed: int = 0) -> Regressor:
    return MODELS.get(name).from_config(CFG, seed, **EXTRA.get(name, {}))


@functools.cache
def fitted(name: str) -> tuple[Regressor, FitResult]:
    """Each model fitted once on sin x with 512 samples, validated on 64 fresh samples."""
    x, y = make()
    x_val, y_val = make(n=64, seed=1)
    model = build(name)
    return model, model.fit(x, y, x_val, y_val)


def test_every_registered_model_is_covered() -> None:
    assert set(MODELS.names()) - {"harmonic"} == set(MODELS_UNDER_TEST)


@pytest.mark.parametrize("name", MODELS_UNDER_TEST)
def test_fit_predict_and_bitwise_round_trip(name: str, tmp_path: Path) -> None:
    with pytest.raises(NotFittedError):
        build(name).predict(FAR)

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


@pytest.mark.parametrize(
    "name", ["relu_mlp", "snake", "rff", "harmonic_random_init", "harmonic_offset_init"]
)
def test_same_seed_gives_identical_predictions(name: str) -> None:
    x, y = make()
    a, b = build(name, seed=3), build(name, seed=3)
    a.fit(x, y)
    b.fit(x, y)
    assert np.array_equal(a.predict(FAR), b.predict(FAR))


def test_oracle_reproduces_sine_far_away_even_with_few_samples() -> None:
    model, _ = fitted("oracle")
    np.testing.assert_allclose(model.predict(np.array([1e6])), np.sin([1e6]), atol=1e-8)
    x, y = make(n=64)
    small = build("oracle")
    assert small.fit(x, y).details["harmonics"] == 15  # (64 - 1) // 4, below k_max = 64
    np.testing.assert_allclose(small.predict(np.array([1e6])), np.sin([1e6]), atol=1e-8)
    with pytest.raises(ValueError, match="oracle needs the true period"):
        MODELS.get("oracle").from_config(CFG, 0)


def test_gp_periodic_recovers_the_period() -> None:
    model, _ = fitted("gp_periodic")
    assert model.period is not None
    assert abs(model.period - 2 * np.pi) < 0.05 * 2 * np.pi


def test_ablations_start_where_documented() -> None:
    """The ablation results rest on ω0: random over [2π/L, Nyquist], or 1-5 basins off the peak."""
    x, _ = make()
    width = float(x.max() - x.min())
    resolution = 2 * np.pi / width

    _, res = fitted("harmonic_random_init")
    assert resolution <= res.details["omega_initial"] <= np.pi * len(x) / width

    _, res = fitted("harmonic_offset_init")
    d = res.details
    assert abs(d["omega_peak"] - 1.0) < resolution  # sin x peaks at ω = 1
    assert 1.0 <= abs(d["offset_in_resolutions"]) <= 5.0
    expected = d["omega_peak"] + d["offset_in_resolutions"] * resolution
    assert d["omega_initial"] == pytest.approx(expected, rel=1e-12)
