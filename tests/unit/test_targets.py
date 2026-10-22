from __future__ import annotations

import numpy as np
import pytest

from periodic.registry import RegistryError
from periodic.targets import TARGETS, get_target, heldout_target

PERIODIC = [n for n in TARGETS.names() if TARGETS.get(n).period is not None]


@pytest.mark.parametrize("name", PERIODIC)
def test_declared_period_is_a_true_period(name: str, rng: np.random.Generator) -> None:
    t = get_target(name)
    assert t.period is not None
    x = rng.uniform(-10 * np.pi, 10 * np.pi, 20_000)
    for shift in (1, 7, 1000):
        d = np.abs(t(x + shift * t.period) - t(x))
        # jump targets may flip a handful of samples that sit on a discontinuity
        assert np.mean(d > 1e-8) < 1e-3, f"{name}: period {t.period} is not a period"


@pytest.mark.parametrize("name", TARGETS.names())
def test_targets_are_bounded_float64_and_shape_preserving(name: str) -> None:
    t = get_target(name)
    x = np.linspace(-1e4, 1e4, 50_001)
    y = t(x)
    assert y.dtype == np.float64
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) < 10.0


def test_quasi_periodic_target_declares_no_period() -> None:
    assert get_target("quasi").period is None


def test_heldout_targets_are_seeded_and_distinct() -> None:
    a, b, c = heldout_target(3), heldout_target(3), heldout_target(4)
    x = np.linspace(-5, 5, 101)
    assert np.array_equal(a(x), b(x))
    assert not np.allclose(a(x), c(x))
    assert a.period is not None
    assert 2 * np.pi / 5.0 <= a.period <= 2 * np.pi / 0.3
    assert get_target("heldout-3").period == a.period


def test_unknown_target_raises() -> None:
    with pytest.raises(RegistryError, match="unknown target"):
        get_target("does-not-exist")


@pytest.mark.parametrize("name", ["random_fourier", "heldout-1", "sin", "square"])
def test_targets_accept_scalars_and_2d_input(name: str) -> None:
    t = get_target(name)
    assert t(np.float64(0.5)).shape == ()
    x2 = np.linspace(-3, 3, 12).reshape(3, 4)
    assert t(x2).shape == (3, 4)
    np.testing.assert_allclose(t(x2).ravel(), t(x2.ravel()))


def test_fingerprint_distinguishes_targets() -> None:
    assert get_target("sin").fingerprint() != get_target("square").fingerprint()
    assert get_target("heldout-1").fingerprint() != get_target("heldout-2").fingerprint()
