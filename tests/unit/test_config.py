from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from periodic.config import (
    Config,
    ConfigError,
    config_from_dict,
    config_hash,
    load_config,
    to_dict,
)


def test_defaults_load_without_a_file() -> None:
    cfg = load_config(None)
    assert isinstance(cfg, Config)
    assert cfg.data.n_samples == 4096
    assert math.isclose(cfg.data.x_max, 10 * math.pi)
    assert cfg.model.heads == ("linear", "mlp")


def test_toml_file_and_overrides(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text('name = "t"\n[data]\ntarget = "square"\nn_samples = 1024\n')
    cfg = load_config(
        path, ["data.n_samples=512", 'model.heads=["linear"]', "model.mlp_lr=1e-2", "seed=3"]
    )
    assert cfg.name == "t"
    assert cfg.data.target == "square"
    assert cfg.data.n_samples == 512
    assert cfg.model.heads == ("linear",)
    assert cfg.model.mlp_lr == pytest.approx(1e-2)
    assert cfg.seed == 3


def test_bare_string_override_is_accepted() -> None:
    cfg = load_config(None, ["data.target=square"])
    assert cfg.data.target == "square"


def test_int_is_coerced_to_float_field() -> None:
    cfg = load_config(None, ["data.noise_rel=1"])
    assert isinstance(cfg.data.noise_rel, float)


def test_unknown_key_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text("[data]\nnsamples = 10\n")
    with pytest.raises(ConfigError, match=re.escape("unknown key 'data.nsamples'")):
        load_config(path)


@pytest.mark.parametrize(
    "override",
    [
        "data.n_samples=abc",
        "data.n_samples=1.5",
        "model.mlp_lbfgs=1",
        "model.heads=linear",
        "nokey",
    ],
)
def test_bad_values_are_errors(override: str) -> None:
    with pytest.raises(ConfigError):
        load_config(None, [override])


@pytest.mark.parametrize(
    "override",
    [
        "data.noise_rel=-0.1",
        "data.val_edge_frac=0.6",
        "data.sampling=random",
        'model.heads=["cnn"]',
        "spectral.k_omega_max=4",
        "data.x_max=-100.0",
    ],
)
def test_validation_rejects_inconsistent_configs(override: str) -> None:
    with pytest.raises(ConfigError):
        load_config(None, [override])


def test_to_dict_round_trips_through_config_from_dict() -> None:
    cfg = load_config(None, ["data.target=square", 'model.heads=["mlp"]', "eval.dense_points=7"])
    back = config_from_dict(to_dict(cfg))
    assert back == cfg
    with pytest.raises(ConfigError, match="unknown key"):
        config_from_dict({"nope": 1})


def test_hash_is_stable_and_content_sensitive() -> None:
    a = load_config(None)
    b = load_config(None, ["data.n_samples=4096"])
    c = load_config(None, ["data.n_samples=4095"])
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(c)
    assert to_dict(a)["data"]["n_samples"] == 4096
