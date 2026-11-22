from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from plasma_global import cli
from plasma_global.audit import AuditReport
from plasma_global.core.result import SimulationResult
from plasma_global.output import ResultPaths
from tools.generate_schema_docs import minimal_case_document


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


def test_cli_validate_requires_ambipolar_diffusion_length(tmp_path, capsys) -> None:
    document = minimal_case_document()
    document["reactor"]["surfaces"] = {
        "wall": {
            "zone_id": "plasma",
            "area_m2": 0.1,
            "temperature_K": 300.0,
            "wall_transport": {
                "kind": "ambipolar",
                "diffusion_coefficient_m2_s": 0.2,
            },
        }
    }
    case = tmp_path / "case.yaml"
    case.write_text(yaml.safe_dump(document), encoding="utf-8")

    assert cli.main(["validate", str(case)]) == 2
    assert "wall_transport.ambipolar.diffusion_length_m" in capsys.readouterr().err


def test_cli_run_reuses_the_publicly_prepared_case(tmp_path, monkeypatch) -> None:
    output = tmp_path / "run"
    case = SimpleNamespace(output_plan=SimpleNamespace(metadata={}))
    result = _result()
    calls: list[tuple[SimulationResult, Path]] = []
    monkeypatch.setattr(cli, "load_case", lambda _path: case)
    monkeypatch.setattr(cli, "compile_case", lambda value: value)
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
        cli,
        "load_case",
        lambda _path: SimpleNamespace(output_plan=SimpleNamespace(metadata={})),
    )
    monkeypatch.setattr(cli, "compile_case", lambda value: value)
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
    result_path = tmp_path / "result.h5"
    exported = tmp_path / "result.csv"
    plot_dir = tmp_path / "figures"
    calls: dict[str, object] = {}

    def fake_audit(value):
        calls["audit"] = value
        return AuditReport()

    def fake_export(source, destination):
        calls["export"] = (source, destination)
        return destination

    def fake_plot(source, destination, **options):
        calls["plot"] = (source, destination, options)
        return (destination / "density.png",)

    monkeypatch.setattr(cli, "audit_result_h5", lambda path, **_kw: fake_audit(path))
    monkeypatch.setattr(cli, "export_result_csv", fake_export)
    monkeypatch.setattr(cli, "plot_result_h5", fake_plot)

    assert cli.main(["audit", str(result_path)]) == 0
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
    assert calls["audit"] == result_path
    assert calls["plot"][1] == plot_dir
    assert calls["plot"][2]["series"] == ["density"]


def test_cli_models_lists_current_discriminator_ids(capsys) -> None:
    assert cli.main(["models", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert list(payload["electrons"]) == ["energy_balance", "local_field"]
    assert "experimental.icp" in payload["power"]
    assert "bohm" in payload["wall_transport"]


def test_init_lists_installed_inputs_without_an_output_directory(
    tmp_path, capsys
) -> None:
    assert cli.main(["init", "--list"]) == 0
    assert "benchmarks/analytic-decay" in capsys.readouterr().out
    assert cli.main(["init"]) == 2
    assert "--output" in capsys.readouterr().err
    target = tmp_path / "not-created"
    assert cli.main(["init", "--list", "--output", str(target)]) == 2
    assert cli.main(["init", "argon_maxwellian", "--list"]) == 2
    assert cli.main(["init", "unknown-case", "--output", str(target)]) == 2
    assert "unknown example" in capsys.readouterr().err
    assert not target.exists()


def test_model_catalog_uses_canonical_contracts_and_installed_example_paths() -> None:
    from plasma_global import list_models
    from plasma_global.chemistry._rate_models import RATE_DEFINITIONS
    from plasma_global.input._schema_run import ExperimentalConfig
    from plasma_global.input.schema import (
        ModelsDefinition,
        PowerDefinition,
        SurfaceDefinition,
    )

    catalog = list_models()
    models = ModelsDefinition.model_json_schema()["properties"]
    for category in ("electrons", "electron_density", "gas_energy"):
        assert set(catalog[category]) == set(
            models[category]["discriminator"]["mapping"]
        )
    power = PowerDefinition.model_json_schema()["properties"]["model"]
    wall = SurfaceDefinition.model_json_schema()["properties"]["wall_transport"]
    assert set(catalog["power"]) == set(power["discriminator"]["mapping"])
    assert set(catalog["wall_transport"]) == set(wall["discriminator"]["mapping"])
    assert set(catalog["accumulators"]) == set(ExperimentalConfig.model_fields)
    assert set(catalog["reaction_rates"]) == set(RATE_DEFINITIONS)
    for kind, definition in RATE_DEFINITIONS.items():
        entry = catalog["reaction_rates"][kind]
        assert set(entry["required_inputs"]) == {"kind", *definition.required}
        assert set(entry["fields"]) == {
            "kind",
            *definition.required,
            *definition.optional,
        }
    package = Path(__file__).parents[1] / "plasma_global"
    for entries in catalog.values():
        for entry in entries.values():
            assert entry["purpose"]
            assert entry["applicability"]
            assert entry["configuration_path"]
            assert entry["fields"]
            for example in entry["examples"]:
                assert (package / example["file"]).is_file()
                assert example["relationship"] in {"uses_model", "template"}
    assert "electrode_gap_m" in catalog["power"]["experimental.ccp"]["required_inputs"]
    assert "A_m2_s" in catalog["reaction_rates"]["langmuir_hinshelwood"]["fields"]


def test_cli_and_python_share_filtered_static_model_details(
    capsys, monkeypatch
) -> None:
    from plasma_global import list_models

    def must_not_compile(*_args, **_kwargs):
        raise AssertionError("model discovery must not compile or integrate a case")

    monkeypatch.setattr(cli, "compile_case", must_not_compile)
    assert cli.main(["models", "power", "experimental.ccp", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == list_models("power", "experimental.ccp")
    assert cli.main(["models", "transport"]) == 0
    assert "gas_inlets" in capsys.readouterr().out
    assert cli.main(["models", "reaction_rates", "langmuir_hinshelwood"]) == 0
    assert "A_m2_s" in capsys.readouterr().out
    assert cli.main(["models", "unknown"]) == 2
    assert "unknown model category" in capsys.readouterr().err
    assert cli.main(["models", "power", "unknown"]) == 2
    assert "unknown power model" in capsys.readouterr().err
