from __future__ import annotations

import math
from pathlib import Path

import pytest

from periodic.config import ConfigError
from periodic.report import format_table, read_results, report, summarize
from periodic.runs import write_csv
from periodic.sweep import expand, flatten_row, load_sweep, settings_from_overrides


def test_load_sweep_and_expand(tmp_path: Path) -> None:
    p = tmp_path / "s.toml"
    p.write_text(
        'name = "t"\noverrides = ["model.heads=[\'linear\']"]\n'
        '[grid]\n"data.target" = ["sin", "square"]\nseed = [0, 1]\n"data.noise_rel" = [0.1]\n'
    )
    spec = load_sweep(p)
    assert spec.name == "t" and spec.jobs == 1 and spec.evaluate
    combos = expand(spec)
    assert len(combos) == 4
    assert combos[0] == ['data.target="sin"', "seed=0", "data.noise_rel=0.1"]


def test_sweep_rejects_unknown_keys_and_empty_grids(tmp_path: Path) -> None:
    p = tmp_path / "bad.toml"
    p.write_text('name = "x"\nfoo = 1\n')
    with pytest.raises(ConfigError, match="unknown sweep keys"):
        load_sweep(p)
    p.write_text('name = "x"\n[grid]\nseed = []\n')
    with pytest.raises(ConfigError, match="non-empty"):
        load_sweep(p)


def _metrics() -> dict:
    return {
        "seed": 0,
        "n_samples": 100,
        "sigma": 0.0,
        "head": "linear",
        "omega": 1.0,
        "omega_stderr": 1e-6,
        "n_cycles": 10.0,
        "low_confidence": False,
        "train_mse": 1e-20,
        "fit_seconds": 1.0,
    }


def _evaluation() -> dict:
    return {
        "noise_rel": 0.0,
        "omega": {"omega_true": 1.0, "rel_err": 1e-9, "ratio": 1.0, "rel_stderr": 1e-6},
        "shape_gain": 1.0,
        "train_window": {"nrmse": 1e-10},
        "phase_horizon_0.1rad": 1e5,
        "horizons": {"0.1": math.inf, "0.5": math.inf},
        "predicted_horizons": {"0.1": 1e5, "0.5": 5e5},
        "bands": [
            {"lo": 100.0, "hi": 1000.0, "sign": "both", "nrmse": 1e-8},
            {"lo": 100.0, "hi": 1000.0, "sign": "+", "nrmse": 1e-8},
        ],
        "baselines": {
            "oracle": {
                "horizons": {"0.1": math.inf},
                "train_window": {"nrmse": 0.0},
                "bands": [{"lo": 100.0, "hi": 1000.0, "sign": "both", "nrmse": 0.0}],
            }
        },
    }


def test_flatten_row_csv_round_trip_and_summary(tmp_path: Path) -> None:
    rows = []
    # (seed, relative error, nearest small rational, error to that rational, horizon)
    cases = [
        (0, 1e-9, 1.0, 1e-9, math.inf),
        (1, -2e-9, 1.0, -2e-9, math.inf),
        (2, 0.5, 1.5, 0.0, 100.0),  # locked onto 3/2 of the true frequency
        (3, 5e-3, 1.0, 5e-3, 1000.0),  # right fundamental, imprecise
        (4, -3e-9, 1.0, -3e-9, math.inf),
    ]
    for seed, err, nearest, to_nearest, h in cases:
        m = _metrics() | {"seed": seed}
        ev = _evaluation()
        ev["omega"].update(
            {
                "rel_err": err,
                "ratio": 1.0 + err,
                "nearest_ratio": nearest,
                "rel_err_to_nearest": to_nearest,
            }
        )
        ev["horizons"]["0.1"] = h
        rows.append(flatten_row(m, ev, "sin", "runs/x"))
    rows.append(
        {"target": "sin", "overrides": "data.target=sin seed=5", "error": "RuntimeError: boom"}
    )
    assert rows[0]["nrmse_1e+02"] == 1e-8
    assert rows[0]["oracle_horizon_0.1"] == math.inf
    assert rows[0]["horizon_0.1"] == math.inf

    path = tmp_path / "results.csv"
    write_csv(rows, path)
    back = read_results(path)
    assert len(back) == 6
    assert back[0]["omega_true"] == 1.0
    assert back[0]["horizon_0.1"] == math.inf
    assert back[5]["error"] == "RuntimeError: boom"

    summary = summarize(back)
    assert len(summary) == 1
    s = summary[0]
    assert s["target"] == "sin" and s["n_runs"] == 6 and s["n_errors"] == 1
    assert s["n_locked_wrong"] == 1 and s["n_imprecise"] == 1
    assert s["n_horizon_beyond_1e6"] == 3
    assert s["horizon_0.1_median"] == math.inf  # three of five runs never fail: median is inf
    assert s["omega_rel_err_median"] == pytest.approx(3e-9)
    table = format_table(summary)
    assert "sin" in table and "wrong lock" in table and ">1e6" in table

    # the error row has no noise_rel; it must not make the report group by noise level
    grouped = report(tmp_path, verbose=False)
    assert len(grouped) == 1 and grouped[0]["target"] == "sin"
    assert (tmp_path / "summary_by_target.csv").exists()


def test_flatten_row_without_evaluation_has_core_columns() -> None:
    row = flatten_row(_metrics(), None, "sin", "r")
    assert row["target"] == "sin" and "horizon_0.1" not in row


def test_settings_from_overrides_keep_grid_values() -> None:
    s = settings_from_overrides(['data.target="sin"', "data.noise_rel=0.1", "seed=3", "x=abc"])
    assert s == {"target": "sin", "noise_rel": 0.1, "seed": 3, "x": "abc"}


def test_error_rows_are_grouped_under_their_own_setting() -> None:
    rows = []
    for noise in (0.1, 0.3):
        for seed in (0, 1):
            ev = _evaluation() | {"noise_rel": noise}
            rows.append(flatten_row(_metrics() | {"seed": seed}, ev, "sin", "r"))
    rows.append({"target": "sin", "noise_rel": 0.1, "seed": 2, "error": "RuntimeError: boom"})
    summary = summarize(rows, ("target", "noise_rel"))
    assert [(s["target"], s["noise_rel"]) for s in summary] == [("sin", 0.1), ("sin", 0.3)]
    assert summary[0]["n_runs"] == 3 and summary[0]["n_errors"] == 1
    assert summary[1]["n_runs"] == 2 and summary[1]["n_errors"] == 0
