"""Aggregate a sweep's results.csv across seeds: medians, interquartile ranges, failure counts."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from periodic.runs import write_json

LOCK_TOL = 1e-3  # |omega/omega_true - 1| above this counts as an imprecise fundamental
LOCK_NEAR = 1e-2  # within this of a wrong small rational multiple counts as a harmonic lock


def read_results(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        conv: dict[str, Any] = {}
        for k, v in r.items():
            if v is None or v == "":
                conv[k] = math.nan
                continue
            try:
                conv[k] = float(v)
            except ValueError:
                conv[k] = {"True": True, "False": False}.get(v, v)
        out.append(conv)
    return out


def _quantiles(values: list[float]) -> dict[str, float]:
    """Median and quartiles using the nearest data value, so infinite horizons stay infinite.

    Only NaN (missing) values are dropped: a run whose horizon is inf never failed out to the
    last band and must count as the best possible outcome, not be discarded.
    """
    v = np.array([x for x in values if isinstance(x, float) and not math.isnan(x)])
    if v.size == 0:
        return {"median": math.nan, "q25": math.nan, "q75": math.nan, "n": 0}
    return {
        "median": float(np.quantile(v, 0.5, method="nearest")),
        "q25": float(np.quantile(v, 0.25, method="nearest")),
        "q75": float(np.quantile(v, 0.75, method="nearest")),
        "n": int(v.size),
    }


def _count_inf(values: list[float]) -> int:
    return sum(1 for x in values if isinstance(x, float) and math.isinf(x))


def _is_error(r: dict[str, Any]) -> bool:
    e = r.get("error")
    return isinstance(e, str) and bool(e)


def _lock_counts(grp: list[dict[str, Any]]) -> tuple[int, int]:
    """(wrong harmonic locks, imprecise fundamentals).

    A wrong lock sits close to a small rational multiple of the true frequency other than 1
    (for example 1/2 or 3); an imprecise fundamental is off by more than LOCK_TOL but is not
    near any such multiple.
    """
    locked = imprecise = 0
    for r in grp:
        ratio = r.get("omega_ratio", math.nan)
        if not isinstance(ratio, float) or not math.isfinite(ratio):
            continue
        nearest = r.get("omega_nearest_ratio", 1.0)
        to_nearest = r.get("omega_rel_err_to_nearest", ratio - 1.0)
        wrong_multiple = isinstance(nearest, float) and math.isfinite(nearest) and nearest != 1.0
        if wrong_multiple and isinstance(to_nearest, float) and abs(to_nearest) < LOCK_NEAR:
            locked += 1
        elif abs(ratio - 1.0) > LOCK_TOL:
            imprecise += 1
    return locked, imprecise


def summarize(
    rows: list[dict[str, Any]], group_keys: tuple[str, ...] = ("target",)
) -> list[dict[str, Any]]:
    """One summary row per group: medians with IQR for the headline metrics, plus counts."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in group_keys)].append(r)
    numeric = [
        "omega_rel_err",
        "omega_rel_stderr",
        "train_nrmse",
        "horizon_0.1",
        "predicted_horizon_0.1",
        "phase_horizon_0.1rad",
        "fit_seconds",
    ]
    numeric += sorted({k for r in rows for k in r if k.startswith("nrmse_")})
    out = []
    for key, grp in sorted(groups.items(), key=lambda kv: str(kv[0])):
        row: dict[str, Any] = dict(zip(group_keys, key, strict=True))
        ok = [r for r in grp if not _is_error(r)]
        row["n_runs"] = len(grp)
        row["n_errors"] = len(grp) - len(ok)
        row["n_locked_wrong"], row["n_imprecise"] = _lock_counts(ok)
        row["n_horizon_beyond_1e6"] = _count_inf([r.get("horizon_0.1", math.nan) for r in ok])
        for m in numeric:
            vals = [abs(r[m]) if m == "omega_rel_err" else r[m] for r in ok if m in r]
            q = _quantiles(vals)
            row[f"{m}_median"] = q["median"]
            row[f"{m}_q25"] = q["q25"]
            row[f"{m}_q75"] = q["q75"]
        out.append(row)
    return out


def write_summary(summary: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "summary_by_target.json", summary)
    keys: list[str] = []
    for r in summary:
        for k in r:
            if k not in keys:
                keys.append(k)
    with (out_dir / "summary_by_target.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in summary:
            w.writerow(r)


def _fmt(value: Any, fmt: str) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "-"
    if isinstance(value, float) and math.isinf(value):
        return ">1e6"
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return str(value)


def format_table(summary: list[dict[str, Any]]) -> str:
    cols = [
        ("target", "target", "{}"),
        ("n_runs", "runs", "{:d}"),
        ("n_errors", "err", "{:d}"),
        ("omega_rel_err_median", "|dw/w| med", "{:.2e}"),
        ("omega_rel_stderr_median", "SE/w med", "{:.2e}"),
        ("train_nrmse_median", "in-window NRMSE", "{:.2e}"),
        ("nrmse_1e+03_median", "NRMSE 1e3-1e4", "{:.2e}"),
        ("nrmse_1e+05_median", "NRMSE 1e5-1e6", "{:.2e}"),
        ("horizon_0.1_median", "horizon(0.1)", "{:.3g}"),
        ("predicted_horizon_0.1_median", "predicted", "{:.3g}"),
        ("n_locked_wrong", "wrong lock", "{:d}"),
        ("n_imprecise", "imprecise", "{:d}"),
    ]
    extra = [k for k in ("noise_rel", "n_samples") if summary and k in summary[0]]
    cols = [(k, k, "{:g}") for k in extra] + cols
    header = " | ".join(h for _, h, _ in cols)
    lines = [header, "-" * len(header)]
    for r in summary:
        lines.append(" | ".join(_fmt(r.get(key, math.nan), fmt) for key, _, fmt in cols))
    return "\n".join(lines)


def report(sweep_dir: Path, verbose: bool = True) -> list[dict[str, Any]]:
    rows = read_results(Path(sweep_dir) / "results.csv")
    ok = [r for r in rows if not _is_error(r)]  # error rows have no settings to group by
    group: tuple[str, ...] = ("target",)
    if len({r.get("noise_rel") for r in ok}) > 1:
        group = ("target", "noise_rel")
    elif len({r.get("n_samples") for r in ok}) > 1:
        group = ("target", "n_samples")
    summary = summarize(rows, group)
    write_summary(summary, Path(sweep_dir))
    if verbose:
        print(format_table(summary))
    return summary
