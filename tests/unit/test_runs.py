from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from periodic.runs import read_json, write_json


def test_json_round_trip_handles_numpy_types(tmp_path: Path) -> None:
    obj = {
        "i": np.int64(3),
        "f": np.float64(2.5),
        "arr": np.arange(3),
        "nan": math.nan,
        "inf": math.inf,
        "path": tmp_path,
    }
    p = tmp_path / "sub" / "m.json"
    write_json(p, obj)
    back = read_json(p)
    assert back["i"] == 3 and back["f"] == 2.5 and back["arr"] == [0, 1, 2]
    assert math.isnan(back["nan"]) and back["inf"] == math.inf
    assert back["path"] == str(tmp_path)
