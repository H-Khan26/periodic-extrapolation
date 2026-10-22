"""Inference and model code must never import the black-box targets, directly or indirectly.

Run in a fresh interpreter so indirect imports are caught: importing any module that a model
or the inference path may use must not pull periodic.targets or periodic.data into sys.modules.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "periodic"
FORBIDDEN = {"periodic.targets", "periodic.data"}

# Modules that sit on the model / inference side of the boundary. Extended as modules land.
MODEL_SIDE_MODULES = ["periodic.config", "periodic.features", "periodic.registry", "periodic.runs"]
MODEL_SIDE_MODULES += [
    f"periodic.models.{p.stem}"
    for p in sorted((SRC / "models").glob("*.py"))
    if p.stem != "__init__"
]
if (SRC / "models" / "__init__.py").exists():
    MODEL_SIDE_MODULES.append("periodic.models")
if (SRC / "inference.py").exists():
    MODEL_SIDE_MODULES.append("periodic.inference")
if (SRC / "spectral.py").exists():
    MODEL_SIDE_MODULES.append("periodic.spectral")


@pytest.mark.parametrize("module", MODEL_SIDE_MODULES)
def test_model_side_module_never_loads_targets(module: str) -> None:
    code = (
        f"import sys, importlib; importlib.import_module({module!r}); "
        f"print(sorted(m for m in sys.modules if m in {sorted(FORBIDDEN)!r}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    assert out.stdout.strip() == "[]", f"{module} loaded {out.stdout.strip()}"


def _imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


MODEL_SIDE_FILES = sorted((SRC / "models").glob("*.py")) + [
    p for p in (SRC / "inference.py", SRC / "spectral.py", SRC / "features.py") if p.exists()
]


@pytest.mark.parametrize("path", MODEL_SIDE_FILES, ids=lambda p: p.name)
def test_model_side_file_has_no_static_import_of_targets(path: Path) -> None:
    assert not (_imports_of(path) & FORBIDDEN), f"{path.name} imports a forbidden module"
