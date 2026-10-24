from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from periodic import inference, targets
from periodic.cli import main
from periodic.runs import read_json
from periodic.targets import TARGETS, Target

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.xdist_group("train_smoke")  # one worker, so the fixture trains once


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One run trained through the CLI with both heads, written to an explicit --run-dir."""
    posix = str(tmp_path_factory.mktemp("smoke")).replace("\\", "/")
    code = main(
        [
            "train",
            "--config",
            str(REPO / "configs" / "smoke.toml"),
            "--set",
            f"data.cache_dir='{posix}/cache'",
            "--set",
            'model.heads=["linear","mlp"]',
            "--run-dir",
            f"{posix}/run",
        ]
    )
    assert code == 0
    return Path(posix) / "run"


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
