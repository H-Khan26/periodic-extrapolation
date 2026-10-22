"""Command-line entry point: ``periodic <command> [options]``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from periodic.config import load_config


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", "-c", default=None, help="TOML config file (defaults if omitted)")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set data.target=square",
    )


def cmd_generate(args: argparse.Namespace) -> int:
    from periodic.data import make_dataset

    cfg = load_config(args.config, args.overrides)
    ds = make_dataset(cfg.data, cfg.seed, use_cache=not args.no_cache)
    print(
        f"target={ds.target} n={ds.n} x in [{ds.x_min:.4f}, {ds.x_max:.4f}) "
        f"sigma={ds.sigma:.4g} amplitude={ds.amplitude:.4g}"
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from pathlib import Path

    from periodic.train import train

    cfg = load_config(args.config, args.overrides)
    train(cfg, run_dir=Path(args.run_dir) if args.run_dir else None)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from pathlib import Path

    import numpy as np

    from periodic.inference import predict

    if args.x_file:
        path = Path(args.x_file)
        xs = np.load(path) if path.suffix == ".npy" else np.loadtxt(path)
    else:
        xs = np.asarray([float(v) for v in args.x], dtype=np.float64)
    ys = predict(Path(args.run), np.atleast_1d(xs))
    if args.out:
        np.save(args.out, ys)
    else:
        for xv, yv in zip(np.atleast_1d(xs), ys, strict=True):
            print(f"{float(xv)!r}\t{float(yv)!r}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from pathlib import Path

    from periodic.evaluate import evaluate_run

    run_dir = Path(args.run)
    evaluate_run(run_dir, seed=args.seed)
    if not args.no_figures:
        from periodic.plots import plot_run

        for p in plot_run(run_dir):
            print(p)
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    from pathlib import Path

    from periodic.report import report
    from periodic.sweep import run_sweep

    out = run_sweep(Path(args.spec), Path(args.out_root))
    report(out)
    if not args.no_figures:
        from periodic.plots import plot_sweep

        for p in plot_sweep(out):
            print(p)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from pathlib import Path

    from periodic.report import report

    report(Path(args.sweep))
    if not args.no_figures:
        from periodic.plots import plot_sweep

        for p in plot_sweep(Path(args.sweep)):
            print(p)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="periodic", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="generate (and cache) a training set")
    _add_config_args(p)
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("train", help="estimate ω, fit and select a head, save a run directory")
    _add_config_args(p)
    p.add_argument("--run-dir", default=None, help="write here instead of a timestamped dir")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="predict with a saved run at the given inputs")
    p.add_argument("--run", required=True, help="run directory (or its model/ subdirectory)")
    p.add_argument("--x", nargs="*", default=[], help="input values")
    p.add_argument("--x-file", default=None, help=".npy or text file of inputs")
    p.add_argument("--out", default=None, help="write predictions to this .npy instead of stdout")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("evaluate", help="score a run far outside the window and draw figures")
    p.add_argument("--run", required=True, help="run directory written by train")
    p.add_argument("--seed", type=int, default=1234, help="seed for the evaluation points")
    p.add_argument("--no-figures", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("sweep", help="train + evaluate a grid of configs, collect results.csv")
    p.add_argument("--spec", required=True, help="sweep TOML (see configs/sweeps/)")
    p.add_argument("--out-root", default="reports", help="where <name>/ is written")
    p.add_argument("--no-figures", action="store_true")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("report", help="re-aggregate a finished sweep and redraw its figures")
    p.add_argument("--sweep", required=True, help="sweep output directory (reports/<name>)")
    p.add_argument("--no-figures", action="store_true")
    p.set_defaults(func=cmd_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
