from __future__ import annotations

from pathlib import Path

import pytest

from periodic.cli import build_parser, main


def test_generate_writes_a_cached_dataset_and_reports(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "generate",
            "--set",
            f"data.cache_dir={str(tmp_path).replace(chr(92), '/')!r}",
            "--set",
            "data.n_samples=64",
            "--set",
            "data.target=square",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "target=square" in out and "n=64" in out
    assert len(list(tmp_path.glob("square_*.npz"))) == 1


def test_parser_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
