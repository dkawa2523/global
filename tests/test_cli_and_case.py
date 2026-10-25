from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from plasma_global import cli
from plasma_global.audit import AuditReport
from plasma_global.core.result import SimulationResult
from plasma_global.input._schema_output import OutputConfig
from plasma_global.output import ResultPaths


def _result(*, success: bool = True) -> SimulationResult:
    from plasma_global.core.result import SimulationStatus

    return SimulationResult(
        time_s=np.array([0.0, 1.0]),
        state=np.array([[1.0], [2.0]]),
        state_labels=("density",),
        status=SimulationStatus(success, "completed" if success else "failed"),
    )


def test_cli_validate_reports_its_scope_and_assessment(capsys) -> None:
    case = Path(__file__).parent / "fixtures" / "minimal" / "case.yaml"
    assert cli.main(["validate", str(case)]) == 0
    captured = capsys.readouterr()
    assert "integration has not run" in captured.out
    assert "Missing neutral momentum-transfer data" in captured.err


def test_cli_run_uses_three_public_operations(tmp_path, monkeypatch) -> None:
    output = tmp_path / "run"
    case = SimpleNamespace(output=OutputConfig())
    result = _result()
    calls: list[tuple[SimulationResult, Path]] = []
    monkeypatch.setattr(cli, "load_case", lambda _path: case)
    monkeypatch.setattr(
        cli, "simulate", lambda value: result if value is case else None
    )

    def fake_write(value, directory):
        calls.append((value, directory))
        return ResultPaths(
            directory, directory / "result.h5", directory / "summary.yaml"
        )

    monkeypatch.setattr(cli, "write_result", fake_write)

    assert cli.main(["run", str(tmp_path / "case.yaml"), "--output", str(output)]) == 0
    assert calls == [(result, output)]


def test_cli_run_preserves_failed_solver_status(tmp_path, monkeypatch) -> None:
    output = tmp_path / "run"
    result = _result(success=False)
    monkeypatch.setattr(
        cli, "load_case", lambda _path: SimpleNamespace(output=OutputConfig())
    )
    monkeypatch.setattr(cli, "simulate", lambda _case: result)
    monkeypatch.setattr(
        cli,
        "write_result",
        lambda _result, directory: ResultPaths(
            directory,
            directory / "result.h5",
            directory / "summary.yaml",
        ),
    )

    assert cli.main(["run", "case.yaml", "--output", str(output)]) == 1


def test_cli_audit_export_and_plot_use_canonical_syntax(tmp_path, monkeypatch) -> None:
    case_path = tmp_path / "case.yaml"
    result_path = tmp_path / "result.h5"
    exported = tmp_path / "result.csv"
    plot_dir = tmp_path / "figures"
    calls: dict[str, object] = {}

    case = object()

    def fake_audit(value):
        calls["audit"] = value
        return AuditReport()

    def fake_export(source, destination):
        calls["export"] = (source, destination)
        return destination

    def fake_plot(source, destination, **options):
        calls["plot"] = (source, destination, options)
        return (destination / "density.png",)

    monkeypatch.setattr(
        cli, "load_case", lambda path: case if path == case_path else None
    )
    monkeypatch.setattr(cli, "audit_case", fake_audit)
    monkeypatch.setattr(cli, "export_result_csv", fake_export)
    monkeypatch.setattr(cli, "plot_result_h5", fake_plot)

    assert cli.main(["audit", str(case_path)]) == 0
    assert cli.main(["export", str(result_path), "--csv", str(exported)]) == 0
    assert (
        cli.main(
            [
                "plot",
                str(result_path),
                "density",
                "--output",
                str(plot_dir),
            ]
        )
        == 0
    )
    assert calls["export"] == (result_path, exported)
    assert calls["audit"] is case
    assert calls["plot"][1] == plot_dir
    assert calls["plot"][2]["series"] == ["density"]


def test_cli_models_lists_schema_v3_discriminator_ids(capsys) -> None:
    assert cli.main(["models", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["electrons"] == ["energy_balance", "local_field"]
    assert "experimental.icp" in payload["power"]
    assert "bohm" in payload["wall_transport"]
