from __future__ import annotations

import math
from pathlib import Path

import pytest

from periodic.cli import main
from periodic.report import read_results
from periodic.runs import read_json

REPO = Path(__file__).resolve().parents[2]


def test_sweep_and_report_cli_produce_results_and_summary(tmp_path: Path, capsys) -> None:
    posix = str(tmp_path).replace("\\", "/")
    spec = tmp_path / "tiny.toml"
    spec.write_text(
        f'name = "tiny"\nbase = "{(REPO / "configs" / "smoke.toml").as_posix()}"\n'
        f"overrides = [\"data.cache_dir='{posix}/cache'\"]\n"
        '[grid]\n"data.target" = ["sin", "missing_fundamental"]\nseed = [0]\n'
    )
    code = main(["sweep", "--spec", str(spec), "--out-root", f"{posix}/reports", "--no-figures"])
    assert code == 0
    out = tmp_path / "reports" / "tiny"
    rows = read_results(out / "results.csv")
    assert len(rows) == 2
    assert {r["target"] for r in rows} == {"sin", "missing_fundamental"}
    assert all(abs(r["omega_rel_err"]) < 1e-6 for r in rows)
    assert all(r["horizon_0.1"] == float("inf") for r in rows)
    assert read_json(out / "summary.json")["n_runs"] == 2
    assert (out / "summary_by_target.csv").exists()
    assert "wrong lock" in capsys.readouterr().out

    assert main(["report", "--sweep", str(out), "--no-figures"]) == 0
    summary = read_json(out / "summary_by_target.json")
    assert {s["target"] for s in summary} == {"sin", "missing_fundamental"}
    assert all(s["n_locked_wrong"] == 0 for s in summary)


def test_a_failing_run_becomes_an_error_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from periodic import sweep as sweep_mod

    real_train = sweep_mod.train

    def flaky_train(cfg, **kw):  # type: ignore[no-untyped-def]
        if cfg.data.target == "square":
            raise RuntimeError("boom")
        return real_train(cfg, **kw)

    monkeypatch.setattr(sweep_mod, "train", flaky_train)
    posix = str(tmp_path).replace("\\", "/")
    spec = tmp_path / "flaky.toml"
    spec.write_text(
        f'name = "flaky"\nbase = "{(REPO / "configs" / "smoke.toml").as_posix()}"\n'
        f"overrides = [\"data.cache_dir='{posix}/cache'\"]\nevaluate = false\n"
        '[grid]\n"data.target" = ["sin", "square"]\n'
    )
    out = sweep_mod.run_sweep(spec, tmp_path / "reports", verbose=False)
    rows = read_results(out / "results.csv")
    by_target = {r["target"]: r for r in rows if isinstance(r.get("target"), str)}
    assert by_target["square"]["error"] == "RuntimeError: boom"
    assert math.isnan(by_target["sin"]["error"])
    assert abs(by_target["sin"]["omega"] - 1.0) < 1e-6
    assert read_json(out / "summary.json")["n_errors"] == 1
