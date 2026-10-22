"""Black-box target functions used only for data generation and evaluation.

Nothing under :mod:`periodic.inference` or :mod:`periodic.models` may import this module or
:mod:`periodic.data`; ``tests/unit/test_import_boundary.py`` enforces it. ``period`` is
evaluation metadata (period error, oracle baseline) and is never read by any model.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from periodic.registry import Registry

TWO_PI = 2.0 * np.pi
Array = np.ndarray


@dataclass(frozen=True)
class Target:
    name: str
    fn: Callable[[Array], Array]
    period: float | None
    description: str = ""

    def __call__(self, x: Array) -> Array:
        return np.asarray(self.fn(np.asarray(x, dtype=np.float64)), dtype=np.float64)

    def fingerprint(self) -> str:
        """Text that changes when the target's definition changes; used in dataset cache keys."""
        try:
            source = inspect.getsource(self.fn)
        except (OSError, TypeError):
            source = repr(self.fn)
        return f"{self.name}|{self.period}|{self.description}|{source}"


TARGETS: Registry[Target] = Registry("target")


def target(name: str, period: float | None, description: str = "") -> Callable:
    def deco(fn: Callable[[Array], Array]) -> Callable[[Array], Array]:
        TARGETS.register(name)(Target(name, fn, period, description))
        return fn

    return deco


def _frac(x: Array) -> Array:
    return np.mod(x / TWO_PI, 1.0)


# -- smooth ---------------------------------------------------------------------------------
@target("sin", TWO_PI, "sin x")
def _sin(x: Array) -> Array:
    return np.sin(x)


@target("harmonics3", TWO_PI, "sin x + 0.5 cos(2x + 0.3) + 0.25 sin(3x + 1)")
def _harmonics3(x: Array) -> Array:
    return np.sin(x) + 0.5 * np.cos(2 * x + 0.3) + 0.25 * np.sin(3 * x + 1.0)


@target("missing_fundamental", TWO_PI, "sin 2x + sin 3x: period 2π with no energy at ω = 1")
def _missing_fundamental(x: Array) -> Array:
    return np.sin(2 * x) + np.sin(3 * x)


@target("bumps", TWO_PI, "von Mises bump train, κ = 8")
def _bumps(x: Array) -> Array:
    return np.exp(8.0 * (np.cos(x) - 1.0))


@target("bessel", TWO_PI, "sin(x + 2 sin x): Bessel harmonic spectrum")
def _bessel(x: Array) -> Array:
    return np.sin(x + 2.0 * np.sin(x))


# -- kinks ----------------------------------------------------------------------------------
@target("triangle", TWO_PI, "triangle wave, amplitude 1")
def _triangle(x: Array) -> Array:
    return (2.0 / np.pi) * np.arcsin(np.sin(x))


@target("abs_sin", np.pi, "|sin x|: period π")
def _abs_sin(x: Array) -> Array:
    return np.abs(np.sin(x))


@target("clipped", TWO_PI, "clip(2 sin x, -1, 1): flat regions")
def _clipped(x: Array) -> Array:
    return np.clip(2.0 * np.sin(x), -1.0, 1.0)


# -- jumps ----------------------------------------------------------------------------------
@target("square", TWO_PI, "sign(sin x)")
def _square(x: Array) -> Array:
    return np.where(np.sin(x) >= 0.0, 1.0, -1.0)


@target("sawtooth", TWO_PI, "2 frac(x / 2π) - 1")
def _sawtooth(x: Array) -> Array:
    return 2.0 * _frac(x) - 1.0


@target("pwm20", TWO_PI, "pulse train, 20 % duty cycle")
def _pwm20(x: Array) -> Array:
    return np.where(_frac(x) < 0.2, 1.0, -1.0)


# -- frequency edge cases --------------------------------------------------------------------
@target("sin30", TWO_PI / 30.0, "sin 30x: 300 cycles in the window")
def _sin30(x: Array) -> Array:
    return np.sin(30.0 * x)


@target("sin_eighth", 16.0 * np.pi, "sin(x / 8): 1.25 cycles in the window")
def _sin_eighth(x: Array) -> Array:
    return np.sin(x / 8.0)


@target("one_cycle", 20.0 * np.pi, "sin x + sin 1.1x: period equals the window")
def _one_cycle(x: Array) -> Array:
    return np.sin(x) + np.sin(1.1 * x)


# -- scale and generic -----------------------------------------------------------------------
@target("scaled", TWO_PI, "5 + 0.01 sin x")
def _scaled(x: Array) -> Array:
    return 5.0 + 0.01 * np.sin(x)


def random_fourier_series(
    seed: int, omega: float, n_terms: int = 20, decay: float = 2.0
) -> Callable[[Array], Array]:
    """Seeded random Fourier series with 1/k**decay amplitudes and random phases."""
    rng = np.random.default_rng(seed)
    k = np.arange(1, n_terms + 1)
    amp = 1.0 / k**decay
    phase = rng.uniform(0.0, TWO_PI, n_terms)

    def fn(x: Array) -> Array:
        arg = omega * x[..., None] * k + phase
        return np.sum(amp * np.sin(arg), axis=-1)

    return fn


TARGETS.register("random_fourier")(
    Target(
        "random_fourier",
        random_fourier_series(seed=2026, omega=0.7),
        TWO_PI / 0.7,
        "seeded 20-term random Fourier series, ω = 0.7",
    )
)


@target("quasi", None, "sin x + sin √2 x: quasi-periodic, documented failure")
def _quasi(x: Array) -> Array:
    return np.sin(x) + np.sin(np.sqrt(2.0) * x)


_HELDOUT = re.compile(r"^heldout-(\d+)$")


def heldout_target(seed: int) -> Target:
    """A random Fourier series with a random fundamental, for held-out evaluation.

    The fundamental is drawn from [0.3, 5] so the window holds between 3 and 50 cycles.
    """
    rng = np.random.default_rng([seed, 7919])
    omega = float(rng.uniform(0.3, 5.0))
    n_terms = int(rng.integers(3, 21))
    fn = random_fourier_series(seed=seed, omega=omega, n_terms=n_terms)
    return Target(f"heldout-{seed}", fn, TWO_PI / omega, f"held-out random series, {n_terms} terms")


def get_target(name: str) -> Target:
    m = _HELDOUT.match(name)
    if m:
        return heldout_target(int(m.group(1)))
    return TARGETS.get(name)
