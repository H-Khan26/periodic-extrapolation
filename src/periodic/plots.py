"""Figures for an evaluated run (``plot_run``) and for a sweep's ``results.csv`` (``plot_sweep``).

Colours come from one validated categorical palette in fixed slot order. Models and targets keep
their slot across every figure of a run or sweep, and each slot also carries a marker shape so
identity never rests on colour alone. PNGs are drawn on an opaque light surface, so they read the
same on light and dark pages.

Every input key is optional: a figure whose inputs are missing is skipped, and a figure whose
inputs are malformed is skipped with a warning without affecting the others.
"""

from __future__ import annotations

import functools
import math
import warnings
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator

from periodic.report import read_results
from periodic.runs import read_json

DPI = 150
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<")
MAIN = "model"
# Models take palette slots in this order (only those present), so a model keeps its colour
# across every figure drawn from the same run or sweep. Unknown names follow, sorted.
KNOWN_MODELS = (
    MAIN,
    "oracle",
    "harmonic_random_init",
    "constant",
    "relu_mlp",
    "snake",
    "rff",
    "gp_periodic",
)
MODEL_LABELS = {
    MAIN: "main model",
    "oracle": "oracle (true ω)",
    "harmonic_random_init": "ablation: random ω start",
    "harmonic_offset_init": "ablation: ω start 1-5 basins off",
    "constant": "constant (mean)",
    "relu_mlp": "ReLU MLP",
    "snake": "Snake MLP",
    "rff": "random Fourier features",
    "gp_periodic": "periodic-kernel GP",
}
# Short display names, matching the README's results table; held-out series are labelled apart.
TARGET_LABELS = {
    "sin": "sin x",
    "harmonics3": "3 harmonics",
    "missing_fundamental": "missing fundamental",
    "bumps": "bump train",
    "bessel": "Bessel spectrum",
    "triangle": "triangle",
    "abs_sin": "abs(sin x)",
    "clipped": "clipped sine",
    "square": "square",
    "sawtooth": "sawtooth",
    "pwm20": "20 % pulse",
    "sin30": "sin 30x",
    "sin_eighth": "sin x/8",
    "one_cycle": "one cycle",
    "scaled": "5 + 0.01 sin x",
    "random_fourier": "random Fourier series",
    "quasi": "sin x + sin √2 x",
}
NRMSE_FLOOR = 1e-16
CAP_FACTOR = 3.0  # values beyond the evaluated range are drawn at CAP_FACTOR * eval_max
PREFERRED_BAND_LO = 1e3  # the baselines figure uses the band whose lower edge is closest to this
RANDOM_PHASE = math.sqrt(2.0)
CRB_LABEL = "best possible for a pure sine: Cramér\N{EN DASH}Rao bound, as a median"
MEDIAN_ABS_NORMAL = 0.6745  # median of |e| / sigma for Gaussian e
DOT: dict[str, Any] = {"s": 36, "edgecolors": SURFACE, "linewidths": 1.0, "zorder": 3}
# Malformed inputs (missing keys, nulls, ragged arrays) raise these; they skip one figure only.
SKIPPABLE = (KeyError, IndexError, TypeError, ValueError, FileNotFoundError)

Style = tuple[str, str]  # (colour, marker)


@functools.cache
def _rc() -> dict[Any, Any]:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    sans = [f for f in ("Segoe UI", "Helvetica Neue", "Helvetica", "Arial") if f in installed]
    return {
        "font.family": [*sans[:1], "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "text.color": INK,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    }


@contextmanager
def _figure(path: Path, title: str, **subplot_kw: Any) -> Iterator[tuple[Figure, Any]]:
    """Create a styled figure with a title, yield it for drawing, then save to path and close."""
    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(layout="constrained", **subplot_kw)
        try:
            fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
            yield fig, axes
            fig.savefig(path, dpi=DPI)
        finally:
            plt.close(fig)


def _draw_all[T](figures: Iterable[Callable[[T], Path | None]], data: T) -> list[Path]:
    """Draw each figure on its own: one with malformed inputs is skipped with a warning."""
    paths: list[Path] = []
    for fn in figures:
        try:
            path = fn(data)
        except SKIPPABLE as exc:
            name = fn.__name__.lstrip("_")
            warnings.warn(f"skipped figure {name}: {type(exc).__name__}: {exc}", stacklevel=3)
            continue
        if path is not None:
            paths.append(path)
    return paths


def _legend(fig: Figure, axes: Iterable[Any], ncol: int = 3) -> None:
    """One figure legend below the plots, built from every axis and deduplicated by label."""
    seen: dict[str, Any] = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            seen.setdefault(label, handle)
    if seen:
        fig.legend(
            list(seen.values()), list(seen), loc="outside lower center", ncol=min(ncol, len(seen))
        )


def _styles(order: Iterable[str]) -> dict[str, Style]:
    """Colour and marker per category in slot order; past eight slots the marker shifts too."""
    n = len(PALETTE)
    return {
        name: (PALETTE[i % n], MARKERS[(i + i // n) % len(MARKERS)]) for i, name in enumerate(order)
    }


def model_styles(names: Iterable[str]) -> dict[str, Style]:
    """Style per model (the main model plus the named baselines), in a fixed canonical order."""
    present = {MAIN, *names}
    known = [m for m in KNOWN_MODELS if m in present]
    return _styles(known + sorted(present - set(known)))


def _model_label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def _target_label(name: str) -> str:
    if name.startswith("heldout-"):
        return "held-out " + name.removeprefix("heldout-")
    return TARGET_LABELS.get(name, name)


def _fmt(v: float) -> str:
    """Compact number for labels: 1e6 / 3e6 style from 1e3 up, otherwise three digits."""
    if math.isfinite(v) and abs(v) >= 1e3:
        mantissa, _, exponent = f"{v:.2g}".partition("e")
        if exponent:
            return f"{mantissa}e{int(exponent)}"
    return f"{v:.3g}"


def _as_float(v: Any) -> float:
    return float(v) if isinstance(v, int | float) and not isinstance(v, bool) else math.nan


def _num(d: Any, key: str) -> float:
    """d[key] as a float; NaN when d is not a dict or the value is missing or not a number."""
    return _as_float(d.get(key)) if isinstance(d, dict) else math.nan


def _dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _positive(v: float, default: float) -> float:
    return v if math.isfinite(v) and v > 0 else default


def _median(v: np.ndarray) -> float:
    v = v[~np.isnan(v)]
    return float(np.median(v)) if v.size else math.nan


def _reference_lines(ax: Any) -> None:
    ax.axhline(1.0, color=INK_2, lw=0.9, ls="--", label="NRMSE 1: no better than the mean")
    ax.axhline(
        RANDOM_PHASE, color=INK_2, lw=0.9, ls="-.", label="NRMSE √2: right shape, random phase"
    )


def _category_axis(ax: Any, labels: list[str]) -> None:
    rotate = len(labels) > 4
    ax.set_xticks(
        np.arange(len(labels)),
        labels,
        rotation=40 if rotate else 0,
        ha="right" if rotate else "center",
        rotation_mode="anchor",
    )
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.grid(False, axis="x")


# --------------------------------------------------------------------------- run figures


@dataclass
class _Run:
    ev: dict[str, Any]
    arrays: dict[str, np.ndarray]
    metrics: dict[str, Any]
    styles: dict[str, Style]
    out: Path

    @property
    def color(self) -> str:
        return self.styles[MAIN][0]

    @property
    def omega(self) -> dict[str, Any]:
        return _dict(self.ev.get("omega"))

    @property
    def stderr(self) -> float:
        return _num(self.omega, "stderr")

    @property
    def baselines(self) -> dict[str, Any]:
        return _dict(self.ev.get("baselines"))


def _read_dict(path: Path) -> dict[str, Any]:
    """JSON object at path; empty when the file is missing."""
    return _dict(read_json(path)) if path.exists() else {}


def _read_arrays(path: Path) -> dict[str, np.ndarray]:
    """Arrays of an .npz file; empty when the file is missing."""
    if not path.exists():
        return {}
    with np.load(path) as npz:
        return {k: npz[k] for k in npz.files}


def plot_run(run_dir: Path) -> list[Path]:
    """Write the figures for one evaluated run into run_dir/figures and return their paths."""
    run_dir = Path(run_dir)
    ev = _read_dict(run_dir / "evaluation.json")
    arrays = _read_arrays(run_dir / "evaluation.npz")
    metrics = _read_dict(run_dir / "metrics.json")
    out = run_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    run = _Run(ev, arrays, metrics, model_styles(_dict(ev.get("baselines"))), out)
    return _draw_all((_error_vs_distance, _overlays, _omega_landscape), run)


def _band_curve(bands: Any) -> tuple[np.ndarray, np.ndarray]:
    """(band centre, floored NRMSE) of the sign "both" rows, ordered by centre."""
    rows = [b for b in bands or [] if isinstance(b, dict) and b.get("sign") == "both"]
    lo = np.array([_num(b, "lo") for b in rows])
    hi = np.array([_num(b, "hi") for b in rows])
    centre = np.array([_num(b, "center") for b in rows])
    centre = np.where(np.isfinite(centre), centre, np.sqrt(np.abs(lo * hi)))
    nrmse = np.maximum(np.array([_num(b, "nrmse") for b in rows]), NRMSE_FLOOR)
    order = np.argsort(centre)
    return centre[order], nrmse[order]


def _frequency_error_line(
    run: _Run, ax: Any, xlim: tuple[float, float], ylim: tuple[float, float]
) -> None:
    """NRMSE a one-sigma frequency error predicts at distance |x|: sigma_omega * |x| * gain."""
    gain = _num(run.ev, "shape_gain")
    if not (math.isfinite(run.stderr) and run.stderr > 0 and math.isfinite(gain) and gain > 0):
        return
    x = np.geomspace(*xlim, 200)
    y = run.stderr * x * gain
    y[(y < ylim[0]) | (y > ylim[1])] = np.nan
    if np.isfinite(y).any():
        label = r"frequency-error line $\sigma_\omega\,|x|\,g$"
        ax.plot(x, y, color=run.color, lw=1.2, ls=":", label=label)


def _error_vs_distance(run: _Run) -> Path | None:
    ev = run.ev
    bands = [b for b in ev.get("bands") or [] if isinstance(b, dict)]
    edges = np.array([_num(b, k) for b in bands for k in ("lo", "hi")])
    edges = edges[np.isfinite(edges) & (edges > 0)]
    if edges.size < 2:
        return None
    xlim = (float(edges.min()), float(edges.max()))
    curves = {MAIN: _band_curve(bands)}
    curves |= {n: _band_curve(_dict(b).get("bands")) for n, b in run.baselines.items()}
    ys = np.concatenate([y for _, y in curves.values()])
    ys = ys[np.isfinite(ys)]
    noise = _num(ev, "sigma") / _positive(_num(ev, "scale"), 1.0)
    lows = [*ys, noise] if noise > 0 else [*ys, 1.0]
    ylim = (min(lows) / 3.0, max(RANDOM_PHASE, *ys) * 3.0)

    path = run.out / "error_vs_distance.png"
    target = _target_label(str(ev.get("target", "")))
    title = f"Extrapolation error against distance from the origin: {target}"
    with _figure(path, title, figsize=(7.5, 5.2)) as (fig, ax):
        for name, (x, y) in curves.items():
            color, marker = run.styles[name]
            ax.plot(
                x, y, color=color, marker=marker, mec=SURFACE, mew=1.0, label=_model_label(name)
            )
        _reference_lines(ax)
        _frequency_error_line(run, ax, xlim, ylim)
        if noise > 0:
            ax.axhline(noise, color=MUTED, lw=0.9, ls=":", label=r"noise level $\sigma$ / scale")
        ax.set(xscale="log", yscale="log", xlim=xlim, ylim=ylim)
        ax.set_xlabel("|x| (band centre √(lo·hi), both signs pooled)")
        ax.set_ylabel("NRMSE (RMSE / std of the target; floored at 1e-16)")
        _legend(fig, [ax])
    return path


def _overlays(run: _Run) -> Path | None:
    parts = ("x", "truth", "pred")
    # Windows at x >= 0 only: the -x side tells the same story with the phase error mirrored.
    rows = [
        r
        for r in run.ev.get("dense") or []
        if isinstance(r, dict)
        and _num(r, "center") >= 0
        and all(f"dense_{r.get('tag')}_{p}" in run.arrays for p in parts)
    ]
    if not rows:
        return None
    rows.sort(key=lambda r: _num(r, "center"))
    ncols = min(2, len(rows))
    nrows = math.ceil(len(rows) / ncols)

    path = run.out / "overlays.png"
    target = _target_label(str(run.ev.get("target", "")))
    title = f"Truth and prediction near and far from the training data: {target}"
    size = (8.5 if ncols == 2 else 5.0, 2.4 * nrows + 1.2)
    with _figure(
        path, title, nrows=nrows, ncols=ncols, sharey=True, squeeze=False, figsize=size
    ) as (fig, axes):
        for ax, row in zip(axes.flat, rows, strict=False):
            _overlay_panel(run, ax, row)
        for ax in axes.flat[len(rows) :]:
            ax.set_visible(False)
        for col in range(ncols):
            visible = [ax for ax in axes[:, col] if ax.get_visible()]
            if visible:
                visible[-1].set_xlabel("x - window centre")
        _legend(fig, axes.flat, ncol=2)
    return path


def _overlay_panel(run: _Run, ax: Any, row: dict[str, Any]) -> None:
    tag, c = row.get("tag"), _num(row, "center")
    x = run.arrays[f"dense_{tag}_x"] - c
    ax.plot(x, run.arrays[f"dense_{tag}_truth"], color=INK_2, lw=1.5, label="truth")
    ax.plot(
        x, run.arrays[f"dense_{tag}_pred"], color=run.color, lw=1.5, ls="--", label="prediction"
    )
    nrmse, phase = _num(row, "nrmse"), _num(row, "phase_offset_rad")
    nrmse_txt = f"{nrmse:.2g}" if math.isfinite(nrmse) else "n/a"
    phase_txt = f"{phase:+.2g} rad" if math.isfinite(phase) else "n/a"
    window = _dict(run.ev.get("window"))
    inside = _num(window, "x_min") <= c <= _num(window, "x_max")
    where = f"x ≈ {_fmt(c)}" + (" (training data)" if inside else "")
    ax.set_title(f"{where}   NRMSE {nrmse_txt}   phase error {phase_txt}")


def _omega_landscape(run: _Run) -> Path | None:
    scan, zoom = _scan_data(run), _zoom_data(run)
    panels = [(draw, data) for draw, data in ((_scan_panel, scan), (_zoom_panel, zoom)) if data]
    if not panels:
        return None

    path = run.out / "omega_landscape.png"
    title = "Finding ω: global harmonic-sum scan, then the training loss around the estimate"
    size = (7.5, 3.4 * len(panels) + 1.2)
    with _figure(path, title, nrows=len(panels), squeeze=False, figsize=size) as (fig, axes):
        for ax, (draw, data) in zip(axes[:, 0], panels, strict=True):
            draw(run, ax, *data)
            _omega_lines(run, ax)
        if scan and zoom:
            axes[0, 0].axvspan(
                zoom[0].min(), zoom[0].max(), color=GRID, lw=0, zorder=0, label="range of the zoom"
            )
        _legend(fig, axes[:, 0])
    return path


def _scan_data(run: _Run) -> tuple[np.ndarray, np.ndarray] | None:
    omega, score = run.arrays.get("scan_omega"), run.arrays.get("scan_score")
    if omega is None or score is None or omega.size == 0:
        return None
    return omega, score


def _zoom_data(run: _Run) -> tuple[np.ndarray, np.ndarray] | None:
    omega, mse = run.arrays.get("landscape_omega"), run.arrays.get("landscape_mse")
    if omega is None or mse is None or omega.size == 0:
        return None
    y = np.where(mse > 0, mse, np.nan)
    return (omega, y) if np.isfinite(y).any() else None


def _omega_lines(run: _Run, ax: Any) -> None:
    omega_hat = _num(run.omega, "omega")
    if math.isfinite(omega_hat):
        ax.axvline(omega_hat, color=INK, lw=1.0, label=f"estimated ω = {omega_hat:.12g}")
    true = _num(run.omega, "omega_true")
    if math.isfinite(true):
        err = _num(run.omega, "rel_err")
        ax.axvline(true, color=INK_2, lw=1.0, ls="--", label=f"true ω (relative error {err:.1e})")


def _scan_panel(run: _Run, ax: Any, omega: np.ndarray, score: np.ndarray) -> None:
    ax.plot(omega, score, color=run.color, lw=1.0, label="harmonic-sum score")
    cand = run.arrays.get("scan_candidates")
    if cand is not None and cand.size:
        ax.scatter(
            cand,
            np.interp(cand, omega, score),
            marker="o",
            s=44,
            facecolors="none",
            edgecolors=INK,
            linewidths=1.2,
            zorder=4,
            label=f"candidates: the {cand.size} highest scan peaks",
        )
    ax.set_xscale("log")
    ax.set_xlim(omega.min(), omega.max())
    ax.set_ylim(bottom=0.0)
    ax.set_title("global scan: harmonic-sum score over the whole search range")
    ax.set_xlabel("ω (rad per unit x, log scale)")
    ax.set_ylabel("harmonic-sum score")


def _zoom_panel(run: _Run, ax: Any, omega: np.ndarray, mse: np.ndarray) -> None:
    k = min(len(run.metrics.get("harmonics") or []), 16)
    label = f"profiled MSE ({k} harmonic{'s' if k != 1 else ''})" if k else "profiled MSE"
    ax.plot(omega, mse, color=run.color, lw=1.2, label=label)
    ax.set_yscale("log")
    bottom, peak = float(np.nanmin(mse)) / 3.0, float(np.nanmax(mse))
    ax.set_ylim(bottom, peak * (peak / bottom) ** 0.25)  # curve below 80 % height, bar above it
    ax.ticklabel_format(axis="x", useOffset=False)
    _basin_bar(run, ax)
    ax.set_title("zoom: the loss basin around the estimate is no wider than about ±2π/L")
    ax.set_xlabel("ω (rad per unit x)")
    ax.set_ylabel("profiled MSE (standardised target)")


def _basin_bar(run: _Run, ax: Any) -> None:
    """Horizontal bar over ±2π/L around the estimated ω (L: training window width)."""
    omega_hat = _num(run.omega, "omega")
    width = _num(_dict(run.ev.get("window")), "width")
    if not (math.isfinite(omega_hat) and math.isfinite(width) and width > 0):
        return
    res = 2.0 * math.pi / width
    coords = ("data", "axes fraction")
    ax.annotate(
        "",
        xy=(omega_hat - res, 0.9),
        xytext=(omega_hat + res, 0.9),
        xycoords=coords,
        textcoords=coords,
        arrowprops={"arrowstyle": "|-|", "color": INK_2, "lw": 1.0, "shrinkA": 0, "shrinkB": 0},
    )
    ax.annotate(
        f"±2π/L = ±{res:.3g}",
        xy=(omega_hat + res, 0.9),
        xycoords=coords,
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8,
        color=INK_2,
    )


# --------------------------------------------------------------------------- sweep figures


@dataclass
class _Sweep:
    rows: list[dict[str, Any]]
    out: Path

    def __post_init__(self) -> None:
        names = [str(r.get("target")) for r in self.rows]
        self.target_col = np.array(names)
        self.targets = list(dict.fromkeys(names))
        self.styles = _styles(self.targets)
        self.eval_max = _median(self.col("eval_max"))
        self.cap = CAP_FACTOR * self.eval_max
        self.capped_note = f"beyond {_fmt(self.eval_max)} drawn at {_fmt(self.cap)}"

    def has(self, *cols: str) -> bool:
        return all(c in self.rows[0] for c in cols)

    def col(self, key: str) -> np.ndarray:
        return np.array([_as_float(r.get(key)) for r in self.rows], dtype=float)

    def varies(self, key: str) -> bool:
        v = self.col(key)
        return np.unique(v[np.isfinite(v)]).size > 1

    def is_target(self, name: str) -> np.ndarray:
        return self.target_col == name


def plot_sweep(sweep_dir: Path) -> list[Path]:
    """Write the sweep figures from sweep_dir/results.csv into sweep_dir/figures."""
    sweep_dir = Path(sweep_dir)
    csv_path = sweep_dir / "results.csv"
    rows = read_results(csv_path) if csv_path.exists() else []
    if not rows:
        return []
    out = sweep_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures = (_calibration, _noise_sweep, _samples_sweep, _baselines)
    return _draw_all(figures, _Sweep(rows, out))


def _beyond_region(ax: Any, s: _Sweep, axis: str = "x", label: str | None = None) -> None:
    span = ax.axvspan if axis == "x" else ax.axhspan
    span(s.eval_max, s.cap * 2.0, color=GRID, alpha=0.6, lw=0, zorder=0, label=label)


def _calibration(s: _Sweep) -> Path | None:
    if not s.has("target", "horizon_0.1", "predicted_horizon_0.1"):
        return None
    meas, pred = s.col("horizon_0.1"), s.col("predicted_horizon_0.1")
    upper = s.col("horizon_0.1_upper")
    capped = (meas >= s.cap) | (pred >= s.cap)
    banded = np.isfinite(upper) & (upper > meas) & (meas > 0) & (meas < s.cap)
    # A run sits at the centre of its failing band, so it lands inside that band's square on the
    # diagonal exactly when the predicted horizon falls inside the measured band.
    x = np.where(banded, np.sqrt(np.abs(meas * upper)), np.minimum(meas, s.cap))
    pred = np.minimum(pred, s.cap)
    ok = np.isfinite(x) & np.isfinite(pred) & (x > 0) & (pred > 0)
    if not ok.any():
        return None
    lo, hi = min(np.where(banded, meas, x)[ok].min(), pred[ok].min()) / 3.0, s.cap * 2.0
    squares = sorted(set(zip(meas[ok & banded], upper[ok & banded], strict=True)))
    if (ok & (meas >= s.cap) & (pred >= s.eval_max)).any():
        squares.append((s.eval_max, hi))
    n = len(s.targets)
    step = min(0.06, 0.3 / max(n - 1, 1))  # decades between targets inside one band

    path = s.out / "calibration.png"
    title = "Predicted against measured extrapolation horizon (NRMSE 0.1), one point per run"
    with _figure(path, title, figsize=(7.5, 6.8)) as (fig, ax):
        _beyond_region(ax, s, "x")
        _beyond_region(ax, s, "y")
        square_label = "measured failing band: a run inside its square is calibrated"
        for j, (a, b) in enumerate(squares):
            label = square_label if j == 0 else None
            ax.add_patch(
                Rectangle((a, a), b - a, b - a, fc="none", ec=INK_2, lw=0.9, zorder=1, label=label)
            )
        ax.plot(
            [lo, hi], [lo, hi], color=MUTED, lw=0.9, ls="--", label="identity: predicted = measured"
        )
        for i, name in enumerate(s.targets):
            m = ok & s.is_target(name)
            px = x * 10.0 ** (step * (i - (n - 1) / 2))
            color, marker = s.styles[name]
            ax.scatter(
                px[m & ~capped],
                pred[m & ~capped],
                color=color,
                marker=marker,
                label=_target_label(name),
                **DOT,
            )
            ax.scatter(
                px[m & capped],
                pred[m & capped],
                marker=marker,
                s=40,
                facecolors="none",
                edgecolors=color,
                linewidths=1.3,
                zorder=3,
            )
        if (ok & capped).any():
            ax.scatter(
                [],
                [],
                marker="o",
                s=40,
                facecolors="none",
                edgecolors=INK_2,
                label=f"hollow: capped at {_fmt(s.cap)}",
            )
        note = {"fontsize": 8, "color": INK_2}
        above = "above the squares: optimistic, fails sooner than predicted"
        below = "below the squares: conservative, holds longer than predicted"
        ax.text(lo * 1.6, s.eval_max / 1.6, above, va="top", **note)
        ax.text(s.eval_max / 1.6, lo * 1.6, below, ha="right", **note)
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
        ax.set_xlabel(
            "measured horizon |x|: first band with NRMSE > 0.1, dot at its centre"
            f" ({s.capped_note})"
        )
        ax.set_ylabel(f"predicted horizon |x| from the error bar on ω ({s.capped_note})")
        _legend(fig, [ax])
    return path


def _median_lines(
    ax: Any, s: _Sweep, x: np.ndarray, y: np.ndarray, xpos: Callable[[np.ndarray], np.ndarray]
) -> None:
    """Per target: median of y at each x with interquartile bars, dodged slightly in x.

    The line joins positive x only, so an x = 0 column (noiseless runs) stands apart. A median of
    zero has no place on a log axis and is left out; callers say so where it matters.
    """
    n = len(s.targets)
    for i, name in enumerate(s.targets):
        m = s.is_target(name) & np.isfinite(x) & ~np.isnan(y)
        xs = np.unique(x[m])
        if xs.size == 0:
            continue
        q = np.array(
            [np.quantile(y[m & (x == v)], [0.25, 0.5, 0.75], method="nearest") for v in xs]
        )
        px = xpos(xs) * 10.0 ** (0.025 * (i - (n - 1) / 2))
        color, marker = s.styles[name]
        shown = q[:, 1] > 0
        px, xs, q = px[shown], xs[shown], q[shown]
        low = np.where(q[:, 0] > 0, q[:, 0], q[:, 1])
        bars = [q[:, 1] - low, q[:, 2] - q[:, 1]]
        ax.errorbar(px, q[:, 1], yerr=bars, fmt="none", ecolor=color, elinewidth=1.0, capsize=2.5)
        style = {"color": color, "marker": marker, "mec": SURFACE, "mew": 1.0}
        ax.plot(px[xs > 0], q[xs > 0, 1], label=_target_label(name), **style)
        ax.plot(px[xs <= 0], q[xs <= 0, 1], ls="none", **style)


def _value_ticks(ax: Any, values: np.ndarray, xpos: Callable[[np.ndarray], np.ndarray]) -> None:
    v = np.unique(values[np.isfinite(values)])
    ax.set_xticks(xpos(v), [f"{t:g}" for t in v])
    ax.xaxis.set_minor_locator(NullLocator())


def _crb_line(ax: Any, s: _Sweep, noise: np.ndarray) -> None:
    """Median relative ω error of an estimator at the Cramér-Rao bound for a pure sinusoid.

    The bound on the standard deviation is sqrt(24) sigma / (A sqrt(N) L omega) with
    sigma = noise_rel * A (how the data noise is scaled), so the amplitude cancels. It is drawn
    as the median of |error| it implies, which is what the plotted points are. Targets with
    harmonics carry more Fisher information about ω and can sit below this line: it is a
    reference, not a per-target bound.
    """
    pos = noise[noise > 0]
    n = _median(s.col("n_samples"))
    width = _median(s.col("window_width"))
    if pos.size == 0 or not (math.isfinite(n) and n > 0 and math.isfinite(width) and width > 0):
        return
    omega = _positive(_median(s.col("omega_true")), 1.0)
    xs = np.geomspace(pos.min(), pos.max(), 50)
    crb = MEDIAN_ABS_NORMAL * xs * math.sqrt(24.0) / (math.sqrt(n) * width * omega)
    ax.plot(xs, crb, color=INK_2, lw=1.2, ls=":", label=CRB_LABEL)


def _exact_note(ax: Any, s: _Sweep, x: np.ndarray, y: np.ndarray, at: float) -> None:
    """Name the targets whose median at x = 0 is exactly zero, which a log axis cannot show."""
    exact = [_target_label(t) for t in s.targets if _median(y[s.is_target(t) & (x == 0)]) == 0]
    if exact:
        text = "noiseless:\n" + "\n".join(exact) + "\nexact (0)"
        ax.text(
            at,
            0.04,
            text,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK_2,
        )


def _noise_sweep(s: _Sweep) -> Path | None:
    if not (s.has("target", "noise_rel", "omega_rel_err") and s.varies("noise_rel")):
        return None
    noise = s.col("noise_rel")
    pos = noise[noise > 0]
    zero_at = float(pos.min()) / 10.0 if pos.size else 1e-3

    def xpos(v: np.ndarray) -> np.ndarray:
        return np.where(v > 0, v, zero_at)

    err = np.abs(s.col("omega_rel_err"))
    path = s.out / "noise_sweep.png"
    title = (
        "Frequency error and extrapolation horizon against observation noise"
        "\nmedian over seeds; bars span the middle half of the seeds"
    )
    with _figure(path, title, ncols=2, figsize=(10.5, 4.8)) as (fig, (left, right)):
        _median_lines(left, s, noise, err, xpos)
        _crb_line(left, s, noise)
        _exact_note(left, s, noise, err, zero_at)
        left.set_title("error in the learned frequency")
        left.set_ylabel(r"relative frequency error $|\hat\omega/\omega - 1|$")
        if s.has("horizon_0.1"):
            h = s.col("horizon_0.1")
            beyond = f"no failure up to {_fmt(s.eval_max)} (drawn at {_fmt(s.cap)})"
            _beyond_region(right, s, "y", label=beyond)
            _median_lines(right, s, noise, np.where(np.isposinf(h), s.cap, h), xpos)
            right.set_title("extrapolation horizon: start of the first band with NRMSE above 0.1")
            right.set_ylabel("horizon |x|")
        else:
            right.set_visible(False)
        for ax in (left, right):
            ax.set(xscale="log", yscale="log")
            _value_ticks(ax, noise, xpos)
            ax.set_xlabel(r"noise level ($\sigma$ / signal amplitude; 0 = noiseless)")
        _legend(fig, (left, right), ncol=3)
    return path


def _samples_sweep(s: _Sweep) -> Path | None:
    if not (s.has("target", "n_samples", "omega_rel_err") and s.varies("n_samples")):
        return None
    n = s.col("n_samples")
    err = np.maximum(np.abs(s.col("omega_rel_err")), NRMSE_FLOOR)

    def xpos(v: np.ndarray) -> np.ndarray:
        return v

    path = s.out / "samples_sweep.png"
    title = "Relative frequency error against the number of training samples"
    with _figure(path, title, figsize=(7.5, 4.8)) as (fig, ax):
        _median_lines(ax, s, n, err, xpos)
        ax.set(xscale="log", yscale="log")
        _value_ticks(ax, n, xpos)
        ax.set_xlabel("training samples N")
        ax.set_ylabel(r"$|\hat\omega/\omega - 1|$, median and IQR over seeds (floored at 1e-16)")
        _legend(fig, [ax], ncol=4)
    return path


def _band_columns(s: _Sweep) -> dict[float, str]:
    """Main-model band columns nrmse_<lo>, keyed by the lower edge parsed from the name."""
    out: dict[float, str] = {}
    for c in s.rows[0]:
        lo = _parse_float(c.removeprefix("nrmse_")) if c.startswith("nrmse_") else math.nan
        if math.isfinite(lo) and lo > 0:
            out[lo] = c
    return dict(sorted(out.items()))


def _parse_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return math.nan


def _band_text(s: _Sweep, bands: dict[float, str], lo: float) -> str:
    """'1e3 ≤ |x| < 1e4' using the next band's lower edge, or eval_max for the last band."""
    edges = list(bands)
    i = edges.index(lo)
    hi = edges[i + 1] if i + 1 < len(edges) else s.eval_max
    if math.isfinite(hi) and hi > lo:
        return f"{_fmt(lo)} ≤ |x| < {_fmt(hi)}"
    return f"the band starting at |x| = {_fmt(lo)}"


def _baselines(s: _Sweep) -> Path | None:
    bands = _band_columns(s)
    if not bands or not s.has("target"):
        return None
    lo = min(bands, key=lambda v: abs(math.log10(v / PREFERRED_BAND_LO)))
    key = bands[lo]
    suffix = f"_{key}"
    cols = {MAIN: key} | {c[: -len(suffix)]: c for c in s.rows[0] if c.endswith(suffix)}
    styles = model_styles(n for n in cols if n != MAIN)
    width = 0.8 / len(cols)
    main = np.maximum(s.col(key), NRMSE_FLOOR)
    main_med = {t: _median(main[s.is_target(t)]) for t in s.targets}
    targets = sorted(s.targets, key=lambda t: (math.isnan(main_med[t]), main_med[t]))

    path = s.out / "baselines.png"
    title = (
        f"Median NRMSE over {_band_text(s, bands, lo)} per target, main model and baselines"
        "\ntargets sorted by the main model's error"
    )
    size = (min(11.0, max(7.5, 0.18 * len(cols) * len(targets) + 3.0)), 5.6)
    with _figure(path, title, figsize=size) as (fig, ax):
        for j, name in enumerate(styles):
            vals = np.maximum(s.col(cols[name]), NRMSE_FLOOR)
            med = [_median(vals[s.is_target(t)]) for t in targets]
            x = np.arange(len(targets)) + (j - (len(cols) - 1) / 2) * width
            color, marker = styles[name]
            ax.scatter(x, med, color=color, marker=marker, label=_model_label(name), **DOT)
        for edge in np.arange(len(targets) - 1) + 0.5:
            ax.axvline(edge, color=GRID, lw=0.8, zorder=0)
        _reference_lines(ax)
        _category_axis(ax, [_target_label(t) for t in targets])
        ax.set_yscale("log")
        ax.set_ylabel("median NRMSE over seeds (floored at 1e-16)")
        _legend(fig, [ax], ncol=5 if size[0] >= 10.0 else 3)
    return path
