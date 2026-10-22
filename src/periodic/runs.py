"""Run directories and environment capture."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from periodic import __version__

_PACKAGE_DIR = Path(__file__).resolve().parent


def _git(*args: str) -> str | None:
    """Run git inside the package's own repository, whatever the current directory is."""
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
            cwd=_PACKAGE_DIR,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def capture_env() -> dict[str, Any]:
    try:
        torch_version: str | None = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = None
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain") if sha else None
    dirty = bool(status) if status is not None else None
    return {
        "package_version": __version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch_version,
        "platform": platform.platform(),
        "git_sha": sha,
        "git_dirty": dirty,
        "argv": sys.argv,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def make_run_dir(base: str | Path, name: str, key: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    path = Path(base) / f"{stamp}_{name}_{key}"
    path.mkdir(parents=True, exist_ok=False)
    return path


class _Encoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, cls=_Encoder, allow_nan=True))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())
