from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from periodic.runs import capture_env, make_run_dir, read_json, write_json


def test_make_run_dir_creates_a_named_directory_under_base(tmp_path: Path) -> None:
    d = make_run_dir(tmp_path / "runs", "sin", "abc123")
    assert d.is_dir()
    assert d.parent == tmp_path / "runs"
    assert d.name.endswith("_sin_abc123")
    assert len(d.name.split("_")[0]) == len("20260101-000000")


def test_json_round_trip_handles_numpy_types(tmp_path: Path) -> None:
    obj = {
        "i": np.int64(3),
        "f": np.float64(2.5),
        "arr": np.arange(3),
        "nan": math.nan,
        "path": tmp_path,
    }
    p = tmp_path / "sub" / "m.json"
    write_json(p, obj)
    back = read_json(p)
    assert back["i"] == 3 and back["f"] == 2.5 and back["arr"] == [0, 1, 2]
    assert math.isnan(back["nan"])
    assert back["path"] == str(tmp_path)


def test_capture_env_reports_versions_and_git_state() -> None:
    env = capture_env()
    assert env["numpy"] == np.__version__
    assert env["torch"] is None or isinstance(env["torch"], str)
    assert env["python"].count(".") >= 1
    assert env["git_sha"] is None or len(env["git_sha"]) == 40
    assert env["git_dirty"] in (None, True, False)
    assert env["started_utc"].endswith("Z")
