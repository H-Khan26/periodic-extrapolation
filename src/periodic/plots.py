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
from matplotlib.ticker import MaxNLocator, NullLocator

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
NRMSE_FLOOR = 1e-16
AMP_FLOOR = 1e-12
# Fallbacks for sweeps whose results.csv predates the eval_max / window_width columns.
DEFAULT_EVAL_MAX = 1e6
DEFAULT_WINDOW_WIDTH = 20.0 * math.pi
CAP_FACTOR = 3.0  # values beyond the evaluated range are drawn at CAP_FACTOR * eval_max
PREFERRED_BAND_LO = 1e3  # the baselines figure uses the band whose lower edge is closest to this
RANDOM_PHASE = math.sqrt(2.0)
CRB_LABEL = "Cramér\N{EN DASH}Rao bound, pure sinusoid (reference)"
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
    return "main model" if name == MAIN else name


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


def _spread(n: int, half_width: float) -> np.ndarray:
    """Deterministic offsets that fan n overlapping points across a category band."""
    return np.linspace(-half_width, half_width, n) if n > 1 else np.zeros(n)


def _median(v: np.ndarray) -> float:
    v = v[~np.isnan(v)]
    return float(np.median(v)) if v.size else math.nan


def _reference_lines(ax: Any) -> None:
    ax.axhline(1.0, color=INK_2, lw=0.9, ls="--", label="NRMSE 1: no better than the mean")
    ax.axhline(
        RANDOM_PHASE, color=INK_2, lw=0.9, ls="-.", label="NRMSE √2: right shape, random phase"
    )


def _category_axis(ax: Any, labels: list[str], axis: str = "x") -> None:
    ticks = np.arange(len(labels))
    if axis == "y":
        ax.set_yticks(ticks, labels)
        ax.set_ylim(len(labels) - 0.4, -0.6)
        ax.grid(False, axis="y")
        return
    rotate = len(labels) > 4
    ax.set_xticks(
        ticks,
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
    figures = (
        _error_vs_distance,
        _overlays,
        _phase_offset,
        _spectrum,
        _omega_landscape,
        _train_fit,
    )
    return _draw_all(figures, run)


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
    title = f"Extrapolation error against distance from the origin: {ev.get('target', '')}"
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


def _dense_rows(run: _Run) -> list[dict[str, Any]]:
    """Dense-window rows with a finite centre."""
    rows = run.ev.get("dense") or []
    return [r for r in rows if isinstance(r, dict) and math.isfinite(_num(r, "center"))]


def _overlays(run: _Run) -> Path | None:
    parts = ("x", "truth", "pred")
    rows = [
        r for r in _dense_rows(run) if all(f"dense_{r.get('tag')}_{p}" in run.arrays for p in parts)
    ]
    if not rows:
        return None
    ncols = 2 if any(_num(r, "center") < 0 for r in rows) else 1
    grid: dict[float, list[dict[str, Any] | None]] = {}
    for r in rows:
        c = _num(r, "center")
        grid.setdefault(abs(c), [None] * ncols)[0 if ncols == 1 or c < 0 else 1] = r

    path = run.out / "overlays.png"
    title = "Truth and prediction in short windows near and far from the training data"
    if ncols == 2:
        title += "\nleft: windows at -x, right: windows at +x"
    size = (8.5 if ncols == 2 else 5.0, 1.9 * len(grid) + 1.3)
    with _figure(
        path, title, nrows=len(grid), ncols=ncols, sharey=True, squeeze=False, figsize=size
    ) as (
        fig,
        axes,
    ):
        for panel_row, cells in zip(axes, grid.values(), strict=True):
            for ax, cell in zip(panel_row, cells, strict=True):
                if cell is None:
                    ax.set_visible(False)
                    continue
                _overlay_panel(run, ax, cell)
                if cells[0] is None:
                    ax.yaxis.set_tick_params(labelleft=True)
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
    ax.set_title(f"centre {_fmt(c)}   NRMSE {nrmse_txt}   phase {phase_txt}")


def _phase_offset(run: _Run) -> Path | None:
    rows = [
        r
        for r in _dense_rows(run)
        if _num(r, "center") != 0 and math.isfinite(_num(r, "phase_offset_rad"))
    ]
    if not rows:
        return None
    centre = np.array([_num(r, "center") for r in rows])
    phase = np.array([_num(r, "phase_offset_rad") for r in rows])
    dist = np.abs(centre)
    sigma = run.stderr if math.isfinite(run.stderr) else 0.0
    xs = np.geomspace(dist.min() / 2.0, dist.max() * 2.0, 200)
    tiny = [v for v in (sigma * dist.min(), *np.abs(phase)) if v > 0]
    floor = max(min(tiny) / 2.0, 1e-15) if tiny else 1e-12
    linthresh = 10.0 ** math.floor(math.log10(floor))  # a decade, so ticks sit at its edges
    top = max(float(np.abs(phase).max()), 3.0 * sigma * dist.max(), 10.0 * linthresh)
    top = min(1.6 * top, math.pi)

    path = run.out / "phase_offset.png"
    title = "Phase offset of the prediction against distance, inside the model's own ω error bar"
    with _figure(path, title, figsize=(7.5, 5.0)) as (fig, ax):
        if sigma > 0:
            band = sigma * xs
            ax.fill_between(
                xs,
                -3 * band,
                3 * band,
                color=run.color,
                alpha=0.08,
                lw=0,
                label=r"$\pm 3\sigma_\omega|x|$",
            )
            ax.fill_between(
                xs, -band, band, color=run.color, alpha=0.18, lw=0, label=r"$\pm \sigma_\omega|x|$"
            )
        ax.axhline(0.0, color=AXIS, lw=0.8)
        for sign, marker, label in ((1.0, "o", "window at +x"), (-1.0, "s", "window at -x")):
            m = np.sign(centre) == sign
            if m.any():
                ax.scatter(dist[m], phase[m], color=run.color, marker=marker, label=label, **DOT)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.set(xlim=(xs[0], xs[-1]), ylim=(-top, top))
        ax.set_xlabel("|x| (window centre)")
        ax.set_ylabel("phase offset (rad, symmetric log; > 0: prediction lags)")
        _legend(fig, [ax], ncol=4)
    return path


def _spectrum(run: _Run) -> Path | None:
    sp = _dict(run.ev.get("spectrum"))
    k, truth, learned = (
        np.asarray(sp.get(key) or [], dtype=float) for key in ("k", "amp_true", "amp_pred")
    )
    if k.size == 0 or not k.size == truth.size == learned.size:
        return None
    truth, learned = np.maximum(truth, AMP_FLOOR), np.maximum(learned, AMP_FLOOR)

    path = run.out / "spectrum.png"
    title = "Harmonic amplitudes of the learned function and of the truth"
    with _figure(path, title, figsize=(7.5, 4.6)) as (fig, ax):
        ax.plot(
            k, truth, ls="none", marker="o", ms=9, mfc="none", mec=INK_2, mew=1.3, label="truth"
        )
        ax.plot(
            k,
            learned,
            ls="none",
            marker="o",
            ms=4.5,
            color=run.color,
            mec=run.color,
            label="learned",
        )
        top = float(np.nanmax(np.concatenate([truth, learned])))
        ax.set_yscale("log")
        ax.set_ylim(AMP_FLOOR / 3.0, _positive(top, AMP_FLOOR) * 3.0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("harmonic k (multiple of the fundamental)")
        ax.set_ylabel("amplitude 2|c_k| (floored at 1e-12)")
        _legend(fig, [ax], ncol=2)
    return path


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
            label="refined candidates",
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
    ax.set_ylim(float(np.nanmin(mse)) / 3.0, float(np.nanmax(mse)) * 30.0)  # bar headroom
    ax.ticklabel_format(axis="x", useOffset=False)
    _resolution_bar(run, ax)
    ax.set_title("zoom: the loss basin spans only about ±2π/L around the estimate")
    ax.set_xlabel("ω (rad per unit x)")
    ax.set_ylabel("profiled MSE (standardised target)")


def _window_width(run: _Run) -> float:
    """Training window width L from evaluation.json, else the span of the training x."""
    width = _num(_dict(run.ev.get("window")), "width")
    if math.isfinite(width) and width > 0:
        return width
    x = run.arrays.get("train_x")
    return float(np.ptp(x)) if x is not None and x.size > 1 else 0.0


def _resolution_bar(run: _Run, ax: Any) -> None:
    """Horizontal bar of width 2π/L centred on the estimated ω."""
    omega_hat, width = _num(run.omega, "omega"), _window_width(run)
    if not math.isfinite(omega_hat) or width <= 0:
        return
    res = 2.0 * math.pi / width
    coords = ("data", "axes fraction")
    ax.annotate(
        "",
        xy=(omega_hat - res / 2, 0.9),
        xytext=(omega_hat + res / 2, 0.9),
        xycoords=coords,
        textcoords=coords,
        arrowprops={"arrowstyle": "|-|", "color": INK_2, "lw": 1.0, "shrinkA": 0, "shrinkB": 0},
    )
    ax.annotate(
        f"resolution 2π/L = {res:.3g}",
        xy=(omega_hat + res / 2, 0.9),
        xycoords=coords,
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8,
        color=INK_2,
    )


def _train_fit(run: _Run) -> Path | None:
    a = run.arrays
    if not all(k in a for k in ("train_x", "train_y", "train_pred")):
        return None
    x, y, p = a["train_x"], a["train_y"], a["train_pred"]
    order = np.argsort(x)

    path = run.out / "train_fit.png"
    title = "Fit inside the training window, with residuals"
    with _figure(path, title, nrows=2, sharex=True, height_ratios=[2, 1], figsize=(8.0, 5.4)) as (
        fig,
        (top, bottom),
    ):
        top.scatter(x, y, s=4, color=MUTED, lw=0, label="training samples")
        top.plot(x[order], p[order], color=run.color, lw=1.2, label="prediction")
        top.set_ylabel("y")
        bottom.axhline(0.0, color=AXIS, lw=0.8)
        bottom.scatter(x, y - p, s=4, color=INK_2, lw=0, label="residual y - prediction")
        bottom.set_xlabel("x (training window)")
        bottom.set_ylabel("y - prediction")
        _legend(fig, (top, bottom))
    return path


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
        self.window_width = _positive(_median(self.col("window_width")), DEFAULT_WINDOW_WIDTH)
        self.eval_max = _positive(_median(self.col("eval_max")), DEFAULT_EVAL_MAX)
        self.cap = CAP_FACTOR * self.eval_max
        self.beyond = f"beyond {_fmt(self.eval_max)}"
        self.capped_note = f"{self.beyond} drawn at {_fmt(self.cap)}"

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
    figures = (
        _horizons_by_target,
        _calibration,
        _omega_error_by_target,
        _noise_sweep,
        _samples_sweep,
        _baselines,
    )
    return _draw_all(figures, _Sweep(rows, out))


def _beyond_region(ax: Any, s: _Sweep, axis: str = "x") -> None:
    span = ax.axvspan if axis == "x" else ax.axhspan
    span(s.eval_max, s.cap * 2.0, color=GRID, alpha=0.6, lw=0, zorder=0)


def _horizons_by_target(s: _Sweep) -> Path | None:
    if not s.has("target", "horizon_0.1"):
        return None
    h = s.col("horizon_0.1")
    if np.isnan(h).all():
        return None
    finite = h[np.isfinite(h) & (h > 0)]
    left = finite.min() / 3.0 if finite.size else s.eval_max / 10.0

    path = s.out / "horizons_by_target.png"
    title = "Extrapolation horizon per target and seed: first |x| band with NRMSE above 0.1"
    with _figure(path, title, figsize=(7.5, 0.42 * len(s.targets) + 2.2)) as (fig, ax):
        _beyond_region(ax, s)
        for i, name in enumerate(s.targets):
            v = h[s.is_target(name) & ~np.isnan(h)]
            y = i + _spread(v.size, 0.22)
            inf = np.isposinf(v)
            ax.scatter(v[~inf], y[~inf], color=PALETTE[0], label="one seed", **DOT)
            if inf.any():
                ax.scatter(
                    np.full(int(inf.sum()), s.cap),
                    y[inf],
                    marker=">",
                    s=44,
                    facecolors="none",
                    edgecolors=PALETTE[0],
                    linewidths=1.3,
                    zorder=3,
                    label=f"{s.beyond} (no band exceeded 0.1)",
                )
            if v.size:
                med = min(float(np.median(v)), s.cap)
                ax.plot(
                    [med],
                    [i],
                    ls="none",
                    marker="|",
                    ms=16,
                    mew=2,
                    color=INK,
                    zorder=4,
                    label="median over seeds",
                )
        ax.annotate(
            s.beyond,
            xy=(s.cap, 1.0),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK_2,
        )
        _category_axis(ax, s.targets, axis="y")
        ax.set(xscale="log", xlim=(left, s.cap * 2.0))
        ax.set_xlabel("horizon |x| (lower edge of the first band with NRMSE > 0.1)")
        _legend(fig, [ax])
    return path


def _calibration(s: _Sweep) -> Path | None:
    if not s.has("target", "horizon_0.1", "predicted_horizon_0.1"):
        return None
    meas, pred = s.col("horizon_0.1"), s.col("predicted_horizon_0.1")
    capped = (meas >= s.cap) | (pred >= s.cap)
    meas, pred = np.minimum(meas, s.cap), np.minimum(pred, s.cap)
    ok = np.isfinite(meas) & np.isfinite(pred) & (meas > 0) & (pred > 0)
    if not ok.any():
        return None
    lo, hi = min(meas[ok].min(), pred[ok].min()) / 3.0, s.cap * 2.0
    upper = np.minimum(s.col("horizon_0.1_upper"), s.cap)  # all NaN when the column is absent
    banded = ok & np.isfinite(upper) & (upper > meas)

    path = s.out / "calibration.png"
    title = "Predicted against measured extrapolation horizon (NRMSE 0.1), one point per run"
    with _figure(path, title, figsize=(7.5, 6.6)) as (fig, ax):
        _beyond_region(ax, s, "x")
        _beyond_region(ax, s, "y")
        ax.plot(
            [lo, hi], [lo, hi], color=INK_2, lw=0.9, ls="--", label="identity: predicted = measured"
        )
        for name in s.targets:
            m = ok & s.is_target(name)
            color, marker = s.styles[name]
            b = m & banded
            ax.hlines(pred[b], meas[b], upper[b], color=color, lw=1.4, alpha=0.6, zorder=2)
            ax.scatter(
                meas[m & ~capped], pred[m & ~capped], color=color, marker=marker, label=name, **DOT
            )
            ax.scatter(
                meas[m & capped],
                pred[m & capped],
                marker=marker,
                s=40,
                facecolors="none",
                edgecolors=color,
                linewidths=1.3,
                zorder=3,
            )
        if banded.any():
            ax.plot(
                [],
                [],
                color=INK_2,
                lw=1.4,
                alpha=0.6,
                label="line: rest of the failing band",
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
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
        ax.set_xlabel(
            "measured horizon |x|: first band with NRMSE > 0.1, dot at its lower edge"
            f" ({s.capped_note})"
        )
        cap = _fmt(s.cap)
        ax.set_ylabel(
            rf"predicted horizon 0.1 / ($\sigma_\omega$ · gain) (≥ {cap} and inf drawn at {cap})"
        )
        _legend(fig, [ax])
    return path


def _omega_error_by_target(s: _Sweep) -> Path | None:
    if not s.has("target", "omega_rel_err"):
        return None
    raw = s.col("omega_rel_err")
    if np.isnan(raw).all():
        return None
    err = np.maximum(np.abs(raw), NRMSE_FLOOR)
    se = np.maximum(s.col("omega_rel_stderr"), NRMSE_FLOOR)  # all NaN when the column is absent

    path = s.out / "omega_error_by_target.png"
    title = "Achieved relative frequency error per run, beside the model's own standard error"
    with _figure(path, title, figsize=(max(6.5, 0.55 * len(s.targets) + 2.5), 4.8)) as (fig, ax):
        for i, name in enumerate(s.targets):
            m = s.is_target(name)
            x = i + _spread(int(m.sum()), 0.22)
            ax.scatter(
                x, err[m], color=PALETTE[0], label=r"achieved $|\hat\omega/\omega - 1|$", **DOT
            )
            if not np.isnan(se[m]).all():
                ax.scatter(
                    x,
                    se[m],
                    marker="_",
                    s=90,
                    color=INK_2,
                    linewidths=1.6,
                    zorder=4,
                    label=r"model standard error SE($\hat\omega$)/$\omega$",
                )
        _category_axis(ax, s.targets)
        ax.set_yscale("log")
        ax.set_ylabel("relative frequency error (floored at 1e-16)")
        _legend(fig, [ax], ncol=2)
    return path


def _median_lines(
    ax: Any, s: _Sweep, x: np.ndarray, y: np.ndarray, xpos: Callable[[np.ndarray], np.ndarray]
) -> None:
    """Per target: median of y at each x with interquartile whiskers, dodged slightly in x."""
    n = len(s.targets)
    for i, name in enumerate(s.targets):
        m = s.is_target(name) & np.isfinite(x) & ~np.isnan(y)
        xs = np.unique(x[m])
        if xs.size == 0:
            continue
        q = np.array(
            [np.quantile(y[m & (x == v)], [0.25, 0.5, 0.75], method="nearest") for v in xs]
        )
        px = xpos(xs) * 10.0 ** (0.012 * (i - (n - 1) / 2))
        color, marker = s.styles[name]
        ax.vlines(px, q[:, 0], q[:, 2], color=color, lw=1.2)
        ax.plot(px, q[:, 1], color=color, marker=marker, mec=SURFACE, mew=1.0, label=name)


def _value_ticks(ax: Any, values: np.ndarray, xpos: Callable[[np.ndarray], np.ndarray]) -> None:
    v = np.unique(values[np.isfinite(values)])
    ax.set_xticks(xpos(v), [f"{t:g}" for t in v])
    ax.xaxis.set_minor_locator(NullLocator())


def _crb_line(ax: Any, s: _Sweep, noise: np.ndarray) -> None:
    """Reference Cramér-Rao bound on the relative ω error of a pure sinusoid.

    sqrt(24) sigma / (A sqrt(N) L omega) with sigma = noise_rel * A (how the data noise is
    scaled), so the amplitude cancels. Targets with harmonics carry more Fisher information about
    ω and can sit below this line: it is a reference, not a per-target bound.
    """
    pos = noise[noise > 0]
    n = _median(s.col("n_samples"))
    if pos.size == 0 or not (math.isfinite(n) and n > 0):
        return
    omega = _positive(_median(s.col("omega_true")), 1.0)
    xs = np.geomspace(pos.min(), pos.max(), 50)
    crb = xs * math.sqrt(24.0) / (math.sqrt(n) * s.window_width * omega)
    ax.plot(xs, crb, color=INK_2, lw=1.2, ls=":", label=CRB_LABEL)


def _noise_sweep(s: _Sweep) -> Path | None:
    if not (s.has("target", "noise_rel", "omega_rel_err") and s.varies("noise_rel")):
        return None
    noise = s.col("noise_rel")
    pos = noise[noise > 0]
    zero_at = float(pos.min()) / 10.0 if pos.size else 1e-3

    def xpos(v: np.ndarray) -> np.ndarray:
        return np.where(v > 0, v, zero_at)

    err = np.maximum(np.abs(s.col("omega_rel_err")), NRMSE_FLOOR)
    path = s.out / "noise_sweep.png"
    title = "Frequency error and extrapolation horizon against observation noise"
    with _figure(path, title, ncols=2, figsize=(10.5, 4.6)) as (fig, (left, right)):
        _median_lines(left, s, noise, err, xpos)
        _crb_line(left, s, noise)
        left.set_title("relative frequency error, median and IQR over seeds")
        left.set_ylabel(r"$|\hat\omega/\omega - 1|$ (floored at 1e-16)")
        if s.has("horizon_0.1"):
            h = s.col("horizon_0.1")
            _beyond_region(right, s, "y")
            _median_lines(right, s, noise, np.where(np.isposinf(h), s.cap, h), xpos)
            right.set_title("horizon at NRMSE 0.1, median and IQR over seeds")
            right.set_ylabel(f"horizon |x| ({s.capped_note})")
        else:
            right.set_visible(False)
        for ax in (left, right):
            ax.set(xscale="log", yscale="log")
            _value_ticks(ax, noise, xpos)
            ax.set_xlabel(r"noise_rel ($\sigma$ / amplitude)")
        _legend(fig, (left, right), ncol=4)
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
    hi = edges[i + 1] if i + 1 < len(edges) else _median(s.col("eval_max"))
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

    path = s.out / "baselines.png"
    title = f"Median NRMSE over {_band_text(s, bands, lo)} per target, main model and baselines"
    size = (min(12.0, max(7.5, 0.18 * len(cols) * len(s.targets) + 3.0)), 5.2)
    with _figure(path, title, figsize=size) as (fig, ax):
        for j, name in enumerate(styles):
            vals = np.maximum(s.col(cols[name]), NRMSE_FLOOR)
            med = [_median(vals[s.is_target(t)]) for t in s.targets]
            x = np.arange(len(s.targets)) + (j - (len(cols) - 1) / 2) * width
            color, marker = styles[name]
            ax.scatter(x, med, color=color, marker=marker, label=_model_label(name), **DOT)
        _reference_lines(ax)
        _category_axis(ax, s.targets)
        ax.set_yscale("log")
        ax.set_ylabel("median NRMSE over seeds (floored at 1e-16)")
        _legend(fig, [ax])
    return path
