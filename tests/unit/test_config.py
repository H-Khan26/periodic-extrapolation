from __future__ import annotations

from pathlib import Path

import pytest

from periodic.config import ConfigError, config_from_dict, config_hash, load_config, to_dict


def test_toml_file_and_overrides(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text('name = "t"\n[data]\ntarget = "sin"\nn_samples = 1024\n')
    cfg = load_config(
        path,
        [
            "data.n_samples=512",
            "data.target=square",  # a bare string is accepted
            "data.noise_rel=1",  # an integer is coerced to a float field
            'model.heads=["linear"]',
            "model.mlp_lr=1e-2",
            "seed=3",
        ],
    )
    assert cfg.name == "t"
    assert cfg.data.target == "square"
    assert cfg.data.n_samples == 512
    assert isinstance(cfg.data.noise_rel, float) and cfg.data.noise_rel == 1.0
    assert cfg.model.heads == ("linear",)
    assert cfg.model.mlp_lr == pytest.approx(1e-2)
    assert cfg.seed == 3


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ("data.nsamples=10", "unknown key 'data.nsamples'"),
        ("data.n_samples=1.5", "expected an integer"),
        ("model.mlp_lbfgs=1", "expected a boolean"),
        ("model.heads=linear", "expected an array"),
        ("nokey", "must look like"),
        ("data.noise_rel=-0.1", "noise_rel"),
        ("data.val_edge_frac=0.6", "val_edge_frac"),
        ("data.sampling=random", "sampling"),
        ('model.heads=["cnn"]', "unknown heads"),
        ("spectral.k_omega_max=4", "k_omega_max"),
        ("data.x_max=-100.0", "x_max"),
    ],
)
def test_invalid_configs_are_rejected(override: str, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        load_config(None, [override])


def test_to_dict_round_trips_through_config_from_dict() -> None:
    cfg = load_config(None, ["data.target=square", 'model.heads=["mlp"]', "eval.dense_points=7"])
    assert config_from_dict(to_dict(cfg)) == cfg
    with pytest.raises(ConfigError, match="unknown key"):
        config_from_dict({"nope": 1})


def test_hash_is_stable_and_content_sensitive() -> None:
    a = load_config(None)
    b = load_config(None, ["data.n_samples=4096"])
    c = load_config(None, ["data.n_samples=4095"])
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(c)
