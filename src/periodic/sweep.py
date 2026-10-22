"""Run a grid of configurations: train and evaluate each one, collect one CSV row per run.

A sweep file is TOML::

    name = "noise"
    base = "configs/base.toml"      # optional; defaults when omitted
    jobs = 4                        # worker processes
    overrides = ["model.baselines=['oracle']"]   # constant overrides
    [grid]                          # dotted config keys -> lists; the product is run
    "data.target" = ["sin", "square"]
    "data.noise_rel" = [0.01, 0.1]
    seed = [0, 1, 2]

A run that raises is recorded as a row with an ``error`` column; the other runs still count.
results.csv is rewritten after every finished run, so a crash loses nothing that completed.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import os
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from periodic.config import ConfigError, load_config
from periodic.evaluate import evaluate_run
from periodic.runs import read_json, write_json
from periodic.train import train


@dataclass
class SweepSpec:
    name: str
    base: str | None = None
    jobs: int = 1
    evaluate: bool = True
    overrides: list[str] = field(default_factory=list)
    grid: dict[str, list[Any]] = field(default_factory=dict)


def load_sweep(path: Path) -> SweepSpec:
    with Path(path).open("rb") as fh:
        raw = tomllib.load(fh)
    known = {"name", "base", "jobs", "evaluate", "overrides", "grid"}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"unknown sweep keys: {sorted(unknown)}")
    if "name" not in raw:
        raise ConfigError("sweep file needs a name")
    grid = raw.get("grid", {})
    for k, v in grid.items():
        if not isinstance(v, list) or not v:
            raise ConfigError(f"grid.{k} must be a non-empty array")
    return SweepSpec(
        name=str(raw["name"]),
        base=raw.get("base"),
        jobs=int(raw.get("jobs", 1)),
        evaluate=bool(raw.get("evaluate", True)),
        overrides=[str(o) for o in raw.get("overrides", [])],
        grid={str(k): list(v) for k, v in grid.items()},
    )


def _literal(value: Any) -> str:
    """Format a Python value as a TOML literal for a dotted override."""
    return json.dumps(value)


def expand(spec: SweepSpec) -> list[list[str]]:
    keys = list(spec.grid)
    combos = itertools.product(*(spec.grid[k] for k in keys))
    return [[f"{k}={_literal(v)}" for k, v in zip(keys, combo, strict=True)] for combo in combos]


def _run_name(overrides: list[str]) -> str:
    parts = []
    for o in overrides:
        k, v = o.split("=", 1)
        parts.append(f"{k.split('.')[-1]}-{v.strip(chr(34)).strip(chr(39))}")
    return "_".join(parts).replace("/", "-").replace(" ", "")[:120]


def run_one(job: tuple[str | None, list[str], str, bool]) -> dict[str, Any]:
    """Train and evaluate one configuration; safe to call in a worker process.

    Never raises: a failure becomes a row whose ``error`` column holds the exception.
    """
    torch.set_num_threads(1)
    base, overrides, out_dir, do_eval = job
    t0 = time.perf_counter()
    row: dict[str, Any] = {"overrides": " ".join(overrides), "error": ""}
    try:
        posix_out = out_dir.replace("\\", "/")
        cfg = load_config(base, [*overrides, f"out_dir='{posix_out}'"])
        row["target"] = cfg.data.target
        row["seed"] = cfg.seed
        run_dir = train(cfg, verbose=False)
        metrics = read_json(run_dir / "metrics.json")
        ev = evaluate_run(run_dir, verbose=False) if do_eval else None
        row.update(flatten_row(metrics, ev, cfg.data.target, str(run_dir)))
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = time.perf_counter() - t0
    return row


def flatten_row(
    metrics: dict[str, Any], ev: dict[str, Any] | None, target: str, run_dir: str
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "target": target,
        "seed": metrics["seed"],
        "n_samples": metrics["n_samples"],
        "sigma": metrics["sigma"],
        "head": metrics["head"],
        "omega": metrics["omega"],
        "omega_stderr": metrics["omega_stderr"],
        "n_cycles": metrics["n_cycles"],
        "low_confidence": metrics["low_confidence"],
        "train_mse": metrics["train_mse"],
        "fit_seconds": metrics["fit_seconds"],
        "phase_horizon_0.1rad": metrics.get("phase_horizon_0.1rad", math.nan),
        "run_dir": run_dir,
    }
    if ev is None:
        return row
    om = ev["omega"]
    row.update(
        {
            "noise_rel": ev["noise_rel"],
            "amplitude": ev.get("amplitude", math.nan),
            "omega_true": om.get("omega_true", math.nan),
            "omega_rel_err": om.get("rel_err", math.nan),
            "omega_ratio": om.get("ratio", math.nan),
            "omega_nearest_ratio": om.get("nearest_ratio", math.nan),
            "omega_rel_err_to_nearest": om.get("rel_err_to_nearest", math.nan),
            "omega_rel_stderr": om["rel_stderr"],
            "shape_gain": ev["shape_gain"],
            "train_nrmse": ev["train_window"]["nrmse"],
            "window_width": ev.get("window", {}).get("width", math.nan),
            "eval_max": max((b["hi"] for b in ev["bands"]), default=math.nan),
        }
    )
    for eps, h in ev["horizons"].items():
        row[f"horizon_{eps}"] = h
    for eps, h in ev.get("horizons_upper", {}).items():
        row[f"horizon_{eps}_upper"] = h
    for eps, h in ev["predicted_horizons"].items():
        row[f"predicted_horizon_{eps}"] = h
    for b in ev["bands"]:
        if b["sign"] == "both":
            row[f"nrmse_{b['lo']:.0e}"] = b["nrmse"]
    for name, res in ev["baselines"].items():
        for eps, h in res["horizons"].items():
            row[f"{name}_horizon_{eps}"] = h
        row[f"{name}_train_nrmse"] = res["train_window"]["nrmse"]
        for b in res["bands"]:
            if b["sign"] == "both":
                row[f"{name}_nrmse_{b['lo']:.0e}"] = b["nrmse"]
    return row


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _limit_blas_threads(jobs: int) -> None:
    """Share the cores between worker processes instead of letting each BLAS take them all."""
    per = max(1, (os.cpu_count() or jobs) // jobs)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(per))


def _describe(n: int, total: int, r: dict[str, Any]) -> str:
    if r.get("error"):
        return f"[{n}/{total}] {r.get('target')} seed={r.get('seed')} ERROR {r['error']}"
    return (
        f"[{n}/{total}] {r.get('target')} seed={r.get('seed')} "
        f"omega_rel_err={r.get('omega_rel_err', math.nan):.2e} "
        f"horizon0.1={r.get('horizon_0.1', math.nan):.3g} ({r['seconds']:.0f}s)"
    )


def run_sweep(spec_path: Path, out_root: Path = Path("reports"), verbose: bool = True) -> Path:
    spec = load_sweep(spec_path)
    out = Path(out_root) / spec.name
    (out / "runs").mkdir(parents=True, exist_ok=True)
    combos = expand(spec)
    jobs = [
        (spec.base, [*spec.overrides, *c], str(out / "runs" / _run_name(c)), spec.evaluate)
        for c in combos
    ]
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    results_path = out / "results.csv"

    def collect(row: dict[str, Any]) -> None:
        rows.append(row)
        write_csv(rows, results_path)
        if verbose:
            print(_describe(len(rows), len(jobs), row), flush=True)

    if spec.jobs > 1:
        _limit_blas_threads(spec.jobs)
        with ProcessPoolExecutor(max_workers=spec.jobs) as pool:
            for fut in as_completed([pool.submit(run_one, j) for j in jobs]):
                collect(fut.result())
    else:
        for job in jobs:
            collect(run_one(job))

    rows.sort(key=lambda r: (str(r.get("target")), float(r.get("seed", 0)), r["overrides"]))
    write_csv(rows, results_path)
    n_err = sum(1 for r in rows if r.get("error"))
    write_json(
        out / "summary.json",
        {
            "spec": asdict(spec),
            "n_runs": len(rows),
            "n_errors": n_err,
            "seconds": time.perf_counter() - t0,
        },
    )
    if verbose:
        print(f"{len(rows)} runs ({n_err} errors) -> {results_path}")
    return out
