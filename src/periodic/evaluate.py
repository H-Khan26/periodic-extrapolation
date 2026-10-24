"""Evaluate a saved run against the black-box target far outside the training window.

Writes ``evaluation.json`` (scalars, bands, horizons, baselines) and ``evaluation.npz`` (arrays
for figures: dense windows and the ω loss landscape) into the run directory.
Evaluation may call the target; inference never does.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from periodic import inference
from periodic.config import Config, config_from_dict
from periodic.data import make_dataset
from periodic.features import TWO_PI, max_harmonics
from periodic.metrics import (
    bands_to_records,
    dense_window,
    evaluate_bands,
    horizon,
    horizon_band,
    omega_agreement,
    phase_offset,
    predicted_horizon,
    reference_scale,
    rmse,
    shape_gain,
)
from periodic.models import Regressor
from periodic.runs import read_json, write_json
from periodic.spectral import harmonic_sum_scan, profile_mse
from periodic.targets import Target, get_target


def load_run_config(run_dir: Path) -> Config:
    return config_from_dict(read_json(run_dir / "config.json"))


def _score_model(
    model: Regressor,
    target: Target,
    cfg: Config,
    scale: float,
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    bands = evaluate_bands(
        model.predict, target, cfg.eval.band_edges, cfg.eval.points_per_band, scale, rng
    )
    pred_train = model.predict(x_train)
    r = rmse(pred_train, y_train)
    return {
        "bands": bands_to_records(bands),
        "horizons": {str(eps): horizon(bands, eps) for eps in cfg.eval.horizon_eps},
        "horizons_upper": {str(eps): horizon_band(bands, eps)[1] for eps in cfg.eval.horizon_eps},
        "train_window": {"rmse": r, "nrmse": r / scale, "n": len(x_train)},
        "period": model.period,
    }


def _dense_windows(
    model: Regressor, target: Target, cfg: Config, period: float, scale: float
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for center in cfg.eval.dense_centers:
        signs = (1.0,) if center == 0.0 else (1.0, -1.0)
        for s in signs:
            c = s * center
            x = dense_window(c, period, cfg.eval.dense_periods, cfg.eval.dense_points)
            t, p = target(x), model.predict(x)
            r = rmse(p, t)
            tag = f"{'neg' if s < 0 else 'pos'}_{center:g}"
            arrays[f"dense_{tag}_x"] = x
            arrays[f"dense_{tag}_truth"] = t
            arrays[f"dense_{tag}_pred"] = p
            rows.append(
                {
                    "center": c,
                    "tag": tag,
                    "rmse": r,
                    "nrmse": r / scale,
                    "phase_offset_rad": phase_offset(p, t, x, period),
                }
            )
    return rows, arrays


def _omega_landscape(
    x: np.ndarray,
    y: np.ndarray,
    omega: float,
    x_center: float,
    width: float,
    ks: list[int],
    omega_min: float,
    omega_max: float,
    k_scan: int,
) -> dict[str, np.ndarray]:
    """Loss landscape in ω: a local zoom of the profiled MSE and the global harmonic-sum scan.

    The zoom shows the comb of local minima around the estimate; the scan shows the competing
    peaks (sub-harmonics, merged peaks) that make a global search necessary.
    """
    res = TWO_PI / width
    grid = omega + np.linspace(-3.0 * res, 3.0 * res, 601)
    grid = grid[grid > 0]
    z = (y - np.mean(y)) / (np.std(y) or 1.0)
    mse = np.array([profile_mse(x, z, w, ks, x_center) for w in grid])
    omegas, score = harmonic_sum_scan(x, z, omega_min, omega_max, res, oversample=4, k_scan=k_scan)
    return {
        "landscape_omega": grid,
        "landscape_mse": mse,
        "scan_omega": omegas,
        "scan_score": score,
    }


def evaluate_run(run_dir: Path, seed: int = 1234, verbose: bool = True) -> dict[str, Any]:
    run_dir = Path(run_dir)
    cfg = load_run_config(run_dir)
    metrics = read_json(run_dir / "metrics.json")
    target = get_target(cfg.data.target)
    model = inference.load(run_dir)
    ds = make_dataset(cfg.data, cfg.seed)

    scale = reference_scale(target, target.period, ds.x_min, ds.x_max)
    learned_period = model.period
    if learned_period is None:
        raise RuntimeError("the main model has no learned period")
    window_period = target.period if target.period is not None else learned_period

    main = _score_model(model, target, cfg, scale, ds.x, ds.y, seed)
    dense_rows, arrays = _dense_windows(model, target, cfg, window_period, scale)
    gain = shape_gain(model.predict, learned_period)
    stderr = float(metrics["omega_stderr"])
    omega_hat = TWO_PI / learned_period

    omega_info: dict[str, Any] = {
        "omega": omega_hat,
        "stderr": stderr,
        "rel_stderr": stderr / omega_hat,
        "n_cycles": metrics["n_cycles"],
        "low_confidence": metrics["low_confidence"],
    }
    if target.period is not None:
        omega_info.update(omega_agreement(omega_hat, TWO_PI / target.period))

    ks = [int(k) for k in metrics["harmonics"]][:16] or [1]
    omega_json = read_json(run_dir / "omega.json")["full"]
    arrays.update(
        _omega_landscape(
            ds.x,
            ds.y,
            omega_hat,
            0.5 * (ds.x_min + ds.x_max),
            ds.width,
            ks,
            float(omega_json["omega_min"]),
            float(omega_json["omega_max"]),
            min(cfg.spectral.k_scan, max_harmonics(ds.n)),  # the bank the estimator used
        )
    )
    arrays["scan_candidates"] = np.array([c["omega"] for c in omega_json["candidates"]])

    # Only the baselines this run's config asked for are scored, so a reused run directory
    # cannot leak stale baselines into the report; a skipped oracle simply has no directory.
    baselines: dict[str, Any] = {}
    for name in cfg.model.baselines:
        sub = run_dir / "baselines" / name
        if (sub / "model.json").exists():
            bm = inference.load(sub)
            baselines[name] = _score_model(bm, target, cfg, scale, ds.x, ds.y, seed)

    result: dict[str, Any] = {
        **main,  # bands, horizons, train_window; the main model's period is restated below
        "target": cfg.data.target,
        "description": target.description,
        "seed": cfg.seed,
        "n_samples": ds.n,
        "sigma": ds.sigma,
        "noise_rel": cfg.data.noise_rel,
        "amplitude": ds.amplitude,
        "scale": scale,
        "window": {"x_min": ds.x_min, "x_max": ds.x_max, "width": ds.width},
        "head": metrics["head"],
        "period": {"learned": learned_period, "true": target.period},
        "omega": omega_info,
        "shape_gain": gain,
        "predicted_horizons": {
            str(eps): predicted_horizon(stderr, eps, gain) for eps in cfg.eval.horizon_eps
        },
        "phase_horizon_0.1rad": metrics["phase_horizon_0.1rad"],
        "dense": dense_rows,
        "baselines": baselines,
    }
    write_json(run_dir / "evaluation.json", result)
    np.savez(run_dir / "evaluation.npz", **arrays)  # type: ignore[arg-type]
    if verbose:
        h = result["horizons"]
        print(
            f"[{cfg.data.target}] nrmse@1e3..1e4={_band_nrmse(result['bands'], 1e3):.3g} "
            f"horizon(0.1)={h.get('0.1', math.nan):.3g} "
            f"predicted={result['predicted_horizons'].get('0.1', math.nan):.3g} "
            f"omega_rel_err={omega_info.get('rel_err', math.nan):.3g}"
        )
    return result


def _band_nrmse(bands: list[dict[str, Any]], lo: float) -> float:
    for b in bands:
        if b["sign"] == "both" and math.isclose(b["lo"], lo):
            return float(b["nrmse"])
    return math.nan
