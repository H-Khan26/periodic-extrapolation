"""Training orchestration.

data → ω estimate on the training split → one fit per candidate head → edge-block selection →
ω estimate on the full window → refit of the chosen head → saved run directory.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from periodic.config import Config, config_hash, to_dict
from periodic.data import Dataset, edge_split, make_dataset
from periodic.models import MODELS, Regressor
from periodic.models.harmonic import HarmonicRegressor
from periodic.runs import capture_env, make_run_dir, write_json
from periodic.spectral import OmegaEstimate, estimate_omega
from periodic.targets import get_target


@dataclass
class HeadResult:
    head: str
    train_mse: float
    val_mse: float | None
    seconds: float
    details: dict[str, Any]


def _score(r: HeadResult) -> float:
    """Selection score: validation MSE, with NaN or missing values never winning."""
    v = r.val_mse if r.val_mse is not None else r.train_mse
    return v if math.isfinite(v) else math.inf


def select_head(results: list[HeadResult]) -> HeadResult:
    return min(results, key=_score)


def fit_candidates(cfg: Config, ds: Dataset, est_train: OmegaEstimate) -> list[HeadResult]:
    train_mask, val_mask = edge_split(ds.x, ds.x_min, ds.x_max, cfg.data.val_edge_frac)
    results = []
    for head in cfg.model.heads:
        model = HarmonicRegressor.from_config(cfg, cfg.seed, head=head)
        res = model.fit(
            ds.x[train_mask],
            ds.y[train_mask],
            ds.x[val_mask],
            ds.y[val_mask],
            omega_estimate=est_train,
        )
        results.append(HeadResult(head, res.train_mse, res.val_mse, res.seconds, res.details))
    return results


def train(cfg: Config, run_dir: Path | None = None, verbose: bool = True) -> Path:
    t0 = time.perf_counter()
    ds = make_dataset(cfg.data, cfg.seed)
    x_center = 0.5 * (ds.x_min + ds.x_max)
    train_mask, val_mask = edge_split(ds.x, ds.x_min, ds.x_max, cfg.data.val_edge_frac)

    est_full = estimate_omega(ds.x, ds.y, cfg.spectral, x_center, width=ds.width)

    # Head selection fits each candidate on the middle 80 % and scores the edge blocks. The
    # split estimate keeps the full-window frequency floor and resolution; when the window holds
    # fewer than two cycles the split cannot pin ω at all, so selection is skipped.
    selection: list[HeadResult] = []
    est_train: OmegaEstimate | None = None
    selection_note = "single head"
    if len(cfg.model.heads) > 1 and val_mask.any() and not est_full.low_confidence:
        est_train = estimate_omega(
            ds.x[train_mask], ds.y[train_mask], cfg.spectral, x_center, width=ds.width
        )
        selection = fit_candidates(cfg, ds, est_train)
        best_head = select_head(selection).head
        selection_note = "edge-block validation"
    else:
        best_head = cfg.model.heads[0]
        if est_full.low_confidence and len(cfg.model.heads) > 1:
            selection_note = "skipped: fewer than two cycles in the window"

    final = HarmonicRegressor.from_config(cfg, cfg.seed, head=best_head)
    fit = final.fit(ds.x, ds.y, omega_estimate=est_full, width=ds.width)

    run_dir = run_dir or make_run_dir(cfg.out_dir, cfg.data.target, config_hash(cfg))
    final.save(run_dir / "model")

    baselines: dict[str, Any] = {}
    for name in cfg.model.baselines:
        baselines[name] = _fit_baseline(name, cfg, ds, run_dir)

    metrics = {
        "target": cfg.data.target,
        "seed": cfg.seed,
        "n_samples": ds.n,
        "sigma": ds.sigma,
        "head": best_head,
        "selection": selection_note,
        "train_mse": fit.train_mse,
        "omega": final.omega,
        "omega_stderr": est_full.stderr,
        "period": final.period,
        "harmonics": list(est_full.harmonics),
        "n_cycles": est_full.n_cycles,
        "low_confidence": est_full.low_confidence,
        "phase_horizon_0.1rad": est_full.phase_horizon(0.1),
        "fit_seconds": fit.seconds,
        "total_seconds": time.perf_counter() - t0,
        "baselines": {k: v["train_mse"] for k, v in baselines.items()},
    }
    write_json(run_dir / "config.json", to_dict(cfg))
    write_json(run_dir / "env.json", capture_env())
    write_json(
        run_dir / "data.json",
        {
            "target": ds.target,
            "n": ds.n,
            "x_min": ds.x_min,
            "x_max": ds.x_max,
            "sigma": ds.sigma,
            "amplitude": ds.amplitude,
            "seed": ds.seed,
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
        },
    )
    write_json(
        run_dir / "omega.json",
        {
            "full": est_full.to_dict(),
            "train_split": est_train.to_dict() if est_train else None,
        },
    )
    write_json(run_dir / "selection.json", [asdict(r) for r in selection])
    write_json(run_dir / "fit.json", fit.details)
    write_json(run_dir / "baselines.json", baselines)
    write_json(run_dir / "metrics.json", metrics)
    if verbose:
        print(
            f"[{cfg.data.target}] head={best_head} omega={final.omega:.12g} "
            f"stderr={est_full.stderr:.3g} train_mse={fit.train_mse:.3g} "
            f"({metrics['total_seconds']:.1f}s) -> {run_dir}"
        )
    return run_dir


def _fit_baseline(name: str, cfg: Config, ds: Dataset, run_dir: Path) -> dict[str, Any]:
    """Fit one registered baseline on the full window. The oracle receives the true period."""
    extra: dict[str, Any] = {}
    if name.startswith("oracle"):
        extra["period"] = get_target(cfg.data.target).period
    model: Regressor = MODELS.get(name).from_config(cfg, cfg.seed, **extra)
    t0 = time.perf_counter()
    res = model.fit(ds.x, ds.y)
    model.save(run_dir / "baselines" / name)
    return {
        "train_mse": res.train_mse,
        "seconds": time.perf_counter() - t0,
        "period": model.period,
        "details": res.details,
    }
