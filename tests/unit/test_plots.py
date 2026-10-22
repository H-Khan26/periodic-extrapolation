from __future__ import annotations

import itertools
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from periodic import plots
from periodic.plots import plot_run, plot_sweep
from periodic.runs import write_json
from periodic.sweep import flatten_row, write_csv

EDGES = (10 * math.pi, 1e2, 1e3, 1e4, 1e5, 1e6)
RUN_FIGURES = {
    "error_vs_distance.png",
    "overlays.png",
    "phase_offset.png",
    "spectrum.png",
    "omega_landscape.png",
    "train_fit.png",
}


@pytest.fixture(autouse=True)
def _low_dpi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Render test figures at low resolution; the tests check which files exist, not pixels."""
    monkeypatch.setattr(plots, "DPI", 40)


def _bands(nrmse_at: dict[float, float]) -> list[dict]:
    out = []
    for lo, hi in itertools.pairwise(EDGES):
        for sign in ("+", "-", "both"):
            out.append(
                {
                    "lo": lo,
                    "hi": hi,
                    "sign": sign,
                    "nrmse": nrmse_at[lo],
                    "center": math.sqrt(lo * hi),
                }
            )
    return out


def _model_result(nrmse_at: dict[float, float]) -> dict:
    return {
        "bands": _bands(nrmse_at),
        "horizons": {"0.1": math.inf},
        "train_window": {"rmse": 1e-12, "nrmse": 1e-12, "n": 256},
        "period": 2 * math.pi,
    }


def _write_run(run_dir: Path, spectrum: bool = True, new_keys: bool = True) -> None:
    """Synthetic run; new_keys=False writes the older schema without window, scan, upper."""
    stderr = 1e-9
    main = {lo: 1e-12 * lo for lo in EDGES[:-1]}
    main[EDGES[0]] = 0.0  # exercises the 1e-16 floor
    dense = [
        {"center": 0.0, "tag": "pos_0", "rmse": 1e-13, "nrmse": 1e-13, "phase_offset_rad": 0.0},
        {"center": 1e3, "tag": "pos_1000", "rmse": 1e-9, "nrmse": 1e-9, "phase_offset_rad": 2e-7},
        {"center": -1e3, "tag": "neg_1000", "rmse": 1e-9, "nrmse": 1e-9, "phase_offset_rad": -3e-7},
    ]
    ev = {
        "target": "sin",
        "sigma": 0.01,
        "noise_rel": 0.01,
        "scale": 1 / math.sqrt(2),
        "head": "linear",
        "period": {"learned": 2 * math.pi, "true": 2 * math.pi},
        "omega": {
            "omega": 1.0 + 1e-10,
            "stderr": stderr,
            "rel_stderr": stderr,
            "omega_true": 1.0,
            "rel_err": 1e-10,
        },
        "shape_gain": 1.0,
        "predicted_horizons": {"0.1": 0.1 / stderr},
        "dense": dense,
        "spectrum": None,
        "baselines": {
            "constant": _model_result(dict.fromkeys(EDGES[:-1], 1.0)),
            "oracle": _model_result(dict.fromkeys(EDGES[:-1], 1e-15)),
        },
        **_model_result(main),
    }
    if spectrum:
        k = np.arange(1, 9)
        ev["spectrum"] = {
            "k": k.tolist(),
            "amp_true": np.where(k == 1, 1.0, 0.0).tolist(),
            "amp_pred": np.where(k == 1, 1.0, 1e-14).tolist(),
            "amp_abs_err": np.full(8, 1e-14).tolist(),
        }
    if new_keys:
        ev["window"] = {"x_min": -10 * math.pi, "x_max": 10 * math.pi, "width": 20 * math.pi}
        ev["amplitude"] = 1.0
        ev["horizons_upper"] = {"0.1": math.inf}
    write_json(run_dir / "evaluation.json", ev)

    arrays: dict[str, np.ndarray] = {}
    for row in dense:
        x = np.linspace(row["center"] - 3 * math.pi, row["center"] + 3 * math.pi, 120)
        arrays[f"dense_{row['tag']}_x"] = x
        arrays[f"dense_{row['tag']}_truth"] = np.sin(x)
        arrays[f"dense_{row['tag']}_pred"] = np.sin(x - row["phase_offset_rad"])
    grid = np.linspace(0.7, 1.3, 121)
    arrays["landscape_omega"] = grid
    arrays["landscape_mse"] = 1.0 - np.sinc((grid - 1.0) * 10.0) ** 2
    rng = np.random.default_rng(0)
    xs = np.sort(rng.uniform(-10 * math.pi, 10 * math.pi, 256))
    arrays["train_x"] = xs
    arrays["train_y"] = np.sin(xs) + 0.01 * rng.standard_normal(xs.size)
    arrays["train_pred"] = np.sin(xs)
    if new_keys:
        scan_grid = np.arange(0.05, 20.0, 0.025)
        arrays["scan_grid"] = scan_grid
        arrays["scan_power"] = np.exp(-(((scan_grid - 1.0) / 0.03) ** 2))
        arrays["scan_omega"] = scan_grid[scan_grid >= 0.1]
        arrays["scan_score"] = sum(
            np.exp(-(((arrays["scan_omega"] - 1.0 / k) / 0.03) ** 2)) / k for k in (1, 2, 3)
        )
        arrays["scan_candidates"] = np.array([1.0, 0.5, 1 / 3])
    np.savez(run_dir / "evaluation.npz", **arrays)
    write_json(run_dir / "metrics.json", {"omega_stderr": stderr, "harmonics": [1], "seed": 0})


def test_plot_run_writes_six_figures(tmp_path: Path) -> None:
    _write_run(tmp_path)
    paths = plot_run(tmp_path)
    assert {p.name for p in paths} == RUN_FIGURES
    assert all(p.parent == tmp_path / "figures" and p.stat().st_size > 0 for p in paths)


def test_plot_run_old_schema_without_spectrum(tmp_path: Path) -> None:
    _write_run(tmp_path, spectrum=False, new_keys=False)
    names = {p.name for p in plot_run(tmp_path)}
    assert names == RUN_FIGURES - {"spectrum.png"}


def _strip_npz_and_omega(run_dir: Path) -> None:
    (run_dir / "evaluation.npz").unlink()
    ev_path = run_dir / "evaluation.json"
    ev = json.loads(ev_path.read_text())
    del ev["omega"]
    ev_path.write_text(json.dumps(ev))


def test_plot_run_without_npz_or_omega_keeps_json_figures(tmp_path: Path) -> None:
    _write_run(tmp_path)
    _strip_npz_and_omega(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # missing inputs skip figures quietly, nothing raises
        names = {p.name for p in plot_run(tmp_path)}
    assert names == {"error_vs_distance.png", "phase_offset.png", "spectrum.png"}


def test_plot_run_skips_a_failing_figure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _spectrum(run: object) -> None:  # stands in for plots._spectrum
        raise KeyError("amp_true")

    _write_run(tmp_path)
    (tmp_path / "evaluation.npz").unlink()  # fewer figures to draw
    monkeypatch.setattr(plots, "_spectrum", _spectrum)
    with pytest.warns(UserWarning, match="skipped figure spectrum: KeyError"):
        names = {p.name for p in plot_run(tmp_path)}
    assert names == {"error_vs_distance.png", "phase_offset.png"}


def _metrics(seed: int, n_samples: int) -> dict:
    return {
        "seed": seed,
        "n_samples": n_samples,
        "sigma": 0.0,
        "head": "linear",
        "omega": 1.0,
        "omega_stderr": 1e-9,
        "n_cycles": 10.0,
        "low_confidence": False,
        "train_mse": 1e-20,
        "fit_seconds": 1.0,
    }


def _evaluation(seed: int, target: str, noise_rel: float, new_keys: bool = True) -> dict:
    periodic = target == "sin"
    horizon = math.inf if seed < 2 else 1e5
    ev = {
        "noise_rel": noise_rel,
        "omega": {"rel_stderr": 1e-9 * (1 + seed)},
        "shape_gain": 1.0,
        "train_window": {"nrmse": 1e-10},
        "phase_horizon_0.1rad": 1e8,
        "horizons": {"0.1": horizon if periodic else 1e2, "0.5": math.inf},
        "predicted_horizons": {
            "0.1": (1e8 if seed < 2 else 3e4) if periodic else math.inf,
            "0.5": math.inf,
        },
        "bands": [
            {"lo": lo, "hi": 10 * lo, "sign": "both", "nrmse": 1e-12 * lo} for lo in EDGES[1:-1]
        ],
        "baselines": {
            "constant": {
                "horizons": {"0.1": EDGES[0]},
                "train_window": {"nrmse": 1.0},
                "bands": [{"lo": 1e3, "hi": 1e4, "sign": "both", "nrmse": 1.0}],
            }
        },
    }
    if periodic:
        ev["omega"] |= {
            "omega_true": 1.0,
            "rel_err": (-1) ** seed * 1e-10 * (1 + seed),
            "ratio": 1.0,
        }
    if new_keys:
        ev["window"] = {"x_min": -10 * math.pi, "x_max": 10 * math.pi, "width": 20 * math.pi}
        ev["amplitude"] = 1.0 if periodic else 0.5
        ev["horizons_upper"] = {eps: 10 * h for eps, h in ev["horizons"].items()}
    if seed == 0:  # a run without the oracle leaves empty cells in its columns
        ev["baselines"]["oracle"] = {
            "horizons": {"0.1": math.inf},
            "train_window": {"nrmse": 0.0},
            "bands": [{"lo": 1e3, "hi": 1e4, "sign": "both", "nrmse": 0.0}],
        }
    return ev


@pytest.mark.parametrize(
    ("vary", "extra"),
    [("none", set()), ("noise_rel", {"noise_sweep.png"}), ("n_samples", {"samples_sweep.png"})],
)
def test_plot_sweep_writes_applicable_figures(tmp_path: Path, vary: str, extra: set[str]) -> None:
    rows = []
    for target in ("sin", "aperiodic"):
        for seed in range(3):
            noise = (0.0, 0.01, 0.1)[seed] if vary == "noise_rel" else 0.0
            n = (256, 1024, 4096)[seed] if vary == "n_samples" else 4096
            ev = _evaluation(seed, target, noise, new_keys=vary != "none")
            rows.append(flatten_row(_metrics(seed, n), ev, target, f"runs/{target}-{seed}"))
    if vary == "none":  # an older results.csv without the config-derived columns
        for row in rows:
            for col in ("window_width", "eval_max", "amplitude"):
                row.pop(col, None)
    write_csv(rows, tmp_path / "results.csv")
    assert "nan" in (tmp_path / "results.csv").read_text()

    paths = plot_sweep(tmp_path)
    base = {
        "horizons_by_target.png",
        "calibration.png",
        "omega_error_by_target.png",
        "baselines.png",
    }
    assert {p.name for p in paths} == base | extra
    assert all(p.stat().st_size > 0 for p in paths)


def test_plot_sweep_skips_figures_without_columns(tmp_path: Path) -> None:
    rows = [flatten_row(_metrics(s, 4096), None, "sin", "r") for s in range(2)]
    write_csv(rows, tmp_path / "results.csv")
    assert plot_sweep(tmp_path) == []
