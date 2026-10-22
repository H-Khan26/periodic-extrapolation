from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from periodic import inference, targets
from periodic.cli import main
from periodic.config import load_config
from periodic.runs import read_json
from periodic.targets import TARGETS, Target
from periodic.train import train

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("smoke")
    posix = str(tmp).replace("\\", "/")
    cfg = load_config(
        REPO / "configs" / "smoke.toml",
        [
            f"data.cache_dir='{posix}/cache'",
            f"out_dir='{posix}/runs'",
            'model.heads=["linear","mlp"]',
        ],
    )
    return train(cfg, verbose=False)


def test_run_directory_is_complete(smoke_run: Path) -> None:
    for name in ("config", "env", "data", "omega", "selection", "fit", "metrics"):
        assert (smoke_run / f"{name}.json").exists(), name
    assert (smoke_run / "model" / "model.json").exists()
    metrics = read_json(smoke_run / "metrics.json")
    assert metrics["omega"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["head"] in ("linear", "mlp")
    assert metrics["selection"] == "edge-block validation"
    assert len(read_json(smoke_run / "selection.json")) == 2
    assert read_json(smoke_run / "omega.json")["train_split"] is not None


def test_predict_cli_matches_the_truth_far_away(
    smoke_run: Path, capsys: pytest.CaptureFixture
) -> None:
    assert main(["predict", "--run", str(smoke_run), "--x", "1000000", "3.14"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    x0, y0 = (float(v) for v in lines[0].split("\t"))
    assert x0 == 1e6
    assert y0 == pytest.approx(np.sin(1e6), abs=1e-3)


def test_inference_does_not_touch_the_black_box(
    smoke_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xs = np.array([0.0, 1e3, -1e5, 1e6])
    before = inference.predict(smoke_run, xs)

    def boom(_x: np.ndarray) -> np.ndarray:
        raise AssertionError("inference called the target function")

    for name in TARGETS.names():
        monkeypatch.setitem(TARGETS._items, name, Target(name, boom, None))
    monkeypatch.setattr(targets, "get_target", lambda name: Target(name, boom, None))

    after = inference.predict(smoke_run, xs)
    assert np.array_equal(before, after)


def test_train_cli_with_explicit_run_dir(tmp_path: Path) -> None:
    posix = str(tmp_path).replace("\\", "/")
    code = main(
        [
            "train",
            "--config",
            str(REPO / "configs" / "smoke.toml"),
            "--set",
            f"data.cache_dir='{posix}/cache'",
            "--set",
            "data.n_samples=256",
            "--run-dir",
            f"{posix}/run",
        ]
    )
    assert code == 0
    assert (tmp_path / "run" / "metrics.json").exists()
