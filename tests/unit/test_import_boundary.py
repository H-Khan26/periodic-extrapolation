"""Inference and model code must never import the black-box targets, directly or indirectly.

The runtime check imports every model-side module in one fresh interpreter, so indirect imports
are caught. It names the top-level module, or the models package as a whole, that first pulls in
a forbidden one: importing any models submodule runs the package __init__, which imports them
all. The static check names the offending file, and also catches imports inside functions.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "periodic"
FORBIDDEN = {"periodic.targets", "periodic.data"}
# Top-level modules before the models package, so a violation outside models/ is blamed on the
# module that introduces it.
MODEL_SIDE_FILES = [
    *(SRC / f"{name}.py" for name in ("config", "registry", "runs", "features", "spectral")),
    *sorted((SRC / "models").glob("*.py")),
    SRC / "inference.py",
]

_PROBE = """
import importlib, json, sys
forbidden, seen, culprits = set(sys.argv[1].split(",")), set(), {}
for module in sys.argv[2:]:
    importlib.import_module(module)
    new = (forbidden & set(sys.modules)) - seen
    if new:
        culprits[module] = sorted(new)
        seen |= new
print(json.dumps(culprits))
"""


def _module_name(path: Path) -> str:
    parts = path.relative_to(SRC.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def test_model_side_modules_never_load_targets() -> None:
    modules = [_module_name(p) for p in MODEL_SIDE_FILES]
    out = subprocess.run(
        [sys.executable, "-c", _PROBE, ",".join(sorted(FORBIDDEN)), *modules],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    assert json.loads(out.stdout.strip().splitlines()[-1]) == {}


def _imports_of(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)  # from periodic import x
    return names


def test_model_side_files_have_no_static_import_of_targets() -> None:
    offenders = {p.name: sorted(_imports_of(p) & FORBIDDEN) for p in MODEL_SIDE_FILES}
    assert {k: v for k, v in offenders.items() if v} == {}
