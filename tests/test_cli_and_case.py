from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import plasma_global
from plasma_global import cli
from plasma_global.audit import AuditReport
from plasma_global.core.result import SimulationResult
from plasma_global.output import ResultPaths


def _result(*, success: bool = True) -> SimulationResult:
    from plasma_global.core.result import SimulationStatus

    return SimulationResult(
        time_s=np.array([0.0, 1.0]),
        state=np.array([[1.0], [2.0]]),
        state_labels=("density",),
        status=SimulationStatus(success, "completed" if success else "failed"),
    )


def test_top_level_api_exposes_only_three_operations() -> None:
    assert plasma_global.__all__ == ["load_case", "simulate", "write_result"]


def test_cli_validate_is_monkeypatchable_without_build(tmp_path, monkeypatch) -> None:
    case = object()
    compiled: list[object] = []
    monkeypatch.setattr(cli, "load_case", lambda _path: case)
    monkeypatch.setattr(cli, "_compile_case", compiled.append)

    assert cli.main(["validate", str(tmp_path / "case.yaml")]) == 0
    assert compiled == [case]


def test_cli_run_uses_three_public_operations(tmp_path, monkeypatch) -> None:
    output = tmp_path / "run"
    case = object()
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


def test_cli_migrate_v2_calls_the_real_migration_boundary(
    tmp_path, monkeypatch
) -> None:
    migration = import_module("plasma_global.input.migrate_v2")

    source = tmp_path / "v2.yaml"
    destination = tmp_path / "v3.yaml"
    called: list[tuple[Path, Path]] = []

    def fake_migrate(source_path, destination_path, **options):
        called.append((source_path, destination_path))
        assert options == {
            "boundary_products": None,
            "cross_section_segments": None,
        }
        return SimpleNamespace(
            report=SimpleNamespace(warnings=("review me",), unused_keys=("old.key",))
        )

    monkeypatch.setattr(migration, "migrate_v2_to_yaml", fake_migrate)

    assert cli.main(["migrate-v2", str(source), "--output", str(destination)]) == 0
    assert called == [(source, destination)]


def test_cli_models_lists_schema_v3_discriminator_ids(capsys) -> None:
    assert cli.main(["models", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["electrons"] == [
        "maxwellian",
        "table",
        "experimental.approximate_two_term",
    ]
    assert "experimental.icp" in payload["power"]
    assert "bohm" in payload["wall_transport"]
