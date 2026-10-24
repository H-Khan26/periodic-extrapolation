"""Typed configuration: frozen dataclasses loaded from TOML, with dotted command-line overrides.

Unknown keys are errors. Values are coerced to the dataclass field types, so a TOML integer
is accepted for a float field and a TOML array for a tuple field.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import tomllib
import types
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

PI = math.pi


class ConfigError(ValueError):
    """Raised for unknown keys, bad types or malformed overrides."""


@dataclass(frozen=True)
class DataConfig:
    target: str = "sin"
    n_samples: int = 4096
    x_min: float = -10 * PI
    x_max: float = 10 * PI
    sampling: str = "stratified"
    noise_rel: float = 0.0
    val_edge_frac: float = 0.1
    cache_dir: str = "data/cache"


@dataclass(frozen=True)
class SpectralConfig:
    k_scan: int = 16
    k_omega_max: int = 64
    oversample: int = 8
    n_candidates: int = 8
    min_cycles: float = 1.0
    omega_max: float = 0.0
    tie_margin: float = 3.0
    sig_margin: float = 5.0


@dataclass(frozen=True)
class ModelConfig:
    heads: tuple[str, ...] = ("linear", "mlp")
    k_max: int = 64
    mlp_harmonics: int = 4
    mlp_width: int = 64
    mlp_depth: int = 2
    mlp_epochs: int = 3000
    mlp_lr: float = 3e-3
    mlp_lbfgs: bool = True
    baselines: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvalConfig:
    band_edges: tuple[float, ...] = (10 * PI, 1e2, 1e3, 1e4, 1e5, 1e6)
    points_per_band: int = 2000
    dense_centers: tuple[float, ...] = (0.0, 1e3, 1e5, 1e6)
    dense_periods: float = 3.0
    dense_points: int = 600
    horizon_eps: tuple[float, ...] = (0.1, 0.5)


@dataclass(frozen=True)
class Config:
    name: str = "run"
    seed: int = 0
    out_dir: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _coerce(value: Any, hint: Any, path: str) -> Any:
    origin = typing.get_origin(hint)
    if origin is types.UnionType or origin is typing.Union:
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if value is None:
            return None
        return _coerce(value, args[0], path)
    if origin is tuple:
        if not isinstance(value, list | tuple):
            raise ConfigError(f"{path}: expected an array, got {type(value).__name__}")
        (elem,) = (a for a in typing.get_args(hint) if a is not Ellipsis)
        return tuple(_coerce(v, elem, f"{path}[{i}]") for i, v in enumerate(value))
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected a boolean, got {value!r}")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value
    raise ConfigError(f"{path}: unsupported field type {hint!r}")


def _build(cls: type, data: dict[str, Any], path: str) -> Any:
    hints = typing.get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    for key in data:
        if key not in known:
            raise ConfigError(f"unknown key '{path}{key}'; known: {', '.join(sorted(known))}")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        hint = hints[f.name]
        value = data[f.name]
        if isinstance(hint, type) and is_dataclass(hint):
            if not isinstance(value, dict):
                raise ConfigError(f"{path}{f.name}: expected a table")
            kwargs[f.name] = _build(hint, value, f"{path}{f.name}.")
        else:
            kwargs[f.name] = _coerce(value, hint, f"{path}{f.name}")
    return cls(**kwargs)


def parse_override(item: str) -> tuple[list[str], Any]:
    """Split 'a.b=value' into (['a', 'b'], value); the value is a TOML literal or a bare string."""
    if "=" not in item:
        raise ConfigError(f"override '{item}' must look like section.key=value")
    key, raw = item.split("=", 1)
    keys = [k for k in key.strip().split(".") if k]
    if not keys:
        raise ConfigError(f"override '{item}' has an empty key")
    try:
        value = tomllib.loads(f"v = {raw.strip()}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw.strip()
    return keys, value


def _set_nested(data: dict[str, Any], keys: list[str], value: Any) -> None:
    node = data
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise ConfigError(f"cannot descend into non-table key '{k}'")
    node[keys[-1]] = value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load a TOML config file (or defaults when path is None) and apply dotted overrides."""
    data: dict[str, Any] = {}
    if path is not None:
        with Path(path).open("rb") as fh:
            data = tomllib.load(fh)
    for item in overrides or []:
        keys, value = parse_override(item)
        _set_nested(data, keys, value)
    cfg = _build(Config, data, "")
    validate(cfg)
    return cfg


def config_from_dict(data: dict[str, Any]) -> Config:
    """Rebuild a Config from the dict written by :func:`to_dict` (a run's config.json)."""
    cfg = _build(Config, data, "")
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    d = cfg.data
    if d.x_max <= d.x_min:
        raise ConfigError("data.x_max must exceed data.x_min")
    if d.n_samples < 8:
        raise ConfigError("data.n_samples must be at least 8")
    if d.sampling not in {"stratified", "uniform", "grid"}:
        raise ConfigError(f"data.sampling must be stratified, uniform or grid, got {d.sampling!r}")
    if not 0.0 <= d.val_edge_frac < 0.5:
        raise ConfigError("data.val_edge_frac must be in [0, 0.5)")
    if d.noise_rel < 0:
        raise ConfigError("data.noise_rel must be non-negative")
    bad = set(cfg.model.heads) - {"linear", "mlp"}
    if bad:
        raise ConfigError(f"model.heads contains unknown heads: {sorted(bad)}")
    if cfg.spectral.k_scan < 1 or cfg.spectral.k_omega_max < cfg.spectral.k_scan:
        raise ConfigError("spectral.k_omega_max must be >= spectral.k_scan >= 1")


def to_dict(cfg: Config) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


def config_hash(obj: Any, n: int = 8) -> str:
    if is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    payload = json.dumps(obj, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:n]
