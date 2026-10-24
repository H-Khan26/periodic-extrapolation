from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from periodic.cli import main
from periodic.config import load_config
from periodic.evaluate import evaluate_run, load_run_config
from periodic.runs import read_json
from periodic.train import train

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.xdist_group("evaluate")  # one worker, so the fixture trains once


@pytest.fixture(scope="module")
def evaluated_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    tmp = tmp_path_factory.mktemp("eval")
    posix = str(tmp).replace("\\", "/")
    cfg = load_config(
        REPO / "configs" / "smoke.toml",
        [
            f"data.cache_dir='{posix}/cache'",
            f"out_dir='{posix}/runs'",
            "data.target='harmonics3'",
            "model.baselines=['constant','oracle']",
        ],
    )
    run_dir = train(cfg, verbose=False)
    return run_dir, evaluate_run(run_dir, verbose=False)


def test_evaluation_files_and_headline_numbers(evaluated_run: tuple[Path, dict]) -> None:
    run_dir, ev = evaluated_run
    assert (run_dir / "evaluation.json").exists() and (run_dir / "evaluation.npz").exists()
    assert ev["target"] == "harmonics3"
    assert abs(ev["omega"]["rel_err"]) < 1e-6
    assert ev["period"]["true"] == pytest.approx(2 * np.pi)
    both = [b for b in ev["bands"] if b["sign"] == "both"]
    assert len(both) == 5
    assert all(b["nrmse"] < 1e-3 for b in both)
    assert ev["horizons"]["0.1"] == math.inf
    assert ev["horizons_upper"]["0.1"] == math.inf
    assert ev["window"]["width"] == pytest.approx(20 * np.pi)
    assert ev["amplitude"] > 0
    assert ev["train_window"]["nrmse"] < 1e-6
    assert set(ev["baselines"]) == {"constant", "oracle"}
    assert ev["baselines"]["constant"]["horizons"]["0.1"] == 10 * np.pi
    assert all(abs(p) < 0.05 for p in (d["phase_offset_rad"] for d in ev["dense"]))


def test_evaluation_arrays_cover_windows_and_landscape(evaluated_run: tuple[Path, dict]) -> None:
    run_dir, ev = evaluated_run
    with np.load(run_dir / "evaluation.npz") as z:
        keys = set(z.files)
        assert {"landscape_omega", "landscape_mse", "scan_omega", "scan_score"} <= keys
        assert "dense_pos_0_x" in keys and "dense_neg_1e+06_pred" in keys
        # Candidates are scan-grid peaks, so the true one lies within a grid step of ω.
        step = 2 * np.pi / ev["window"]["width"] / load_run_config(run_dir).spectral.oversample
        assert np.min(np.abs(z["scan_candidates"] - ev["omega"]["omega"])) <= step
        omega = z["landscape_omega"]
        mse = z["landscape_mse"]
        i = int(np.argmin(mse))
        assert abs(omega[i] - ev["omega"]["omega"]) < 2e-3


def test_evaluate_cli(evaluated_run: tuple[Path, dict], capsys: pytest.CaptureFixture) -> None:
    run_dir, _ = evaluated_run
    assert main(["evaluate", "--run", str(run_dir)]) == 0
    assert "horizon" in capsys.readouterr().out
    assert read_json(run_dir / "evaluation.json")["target"] == "harmonics3"
