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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="periodic", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="generate (and cache) a training set")
    _add_config_args(p)
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(func=cmd_generate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
