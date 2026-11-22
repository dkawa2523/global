from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from plasma_global import case_from_mapping, compile_case, simulate, write_result
from plasma_global.cli import main
from plasma_global.output import inspect_result_h5, read_result_h5
from tools.generate_schema_docs import minimal_case_document


def _case(tmp_path: Path, output: dict | None = None):
    document = minimal_case_document()
    if output is not None:
        document["output"] = output
    return case_from_mapping(document, base_directory=tmp_path)


def test_basic_defaults_store_defined_physics_and_explain_omissions(
    tmp_path: Path,
) -> None:
    result = simulate(_case(tmp_path))
    catalog = result.metadata["series_catalog"]
    assert "electron_density_m3[plasma]" in result.observables
    assert "mean_energy_eV[plasma]" in result.observables
    assert "absorbed_power_W[plasma]" in result.observables
    missing = catalog["reduced_field_Td[plasma]"]
    assert missing["available"] is False
    assert missing["stored"] is False
    assert missing["unavailable_intervals"][0]["reason"]
    assert (
        result.metadata["default_output_omissions"][0]["series"]
        == "reduced_field_Td[plasma]"
    )
    assert catalog["electron_density_m3[plasma]"]["quantity"] == "electron_density_m3"
    assert catalog["electron_density_m3[plasma]"]["region"] == "plasma"


def test_minimal_profile_saves_only_selected_observables(tmp_path: Path) -> None:
    case = _case(
        tmp_path, {"profile": "minimal", "observables": ["pressure_Pa[plasma]"]}
    )
    result = simulate(compile_case(case))
    assert set(result.observables) == {"pressure_Pa[plasma]"}
    catalog = result.metadata["series_catalog"]
    assert catalog["mean_energy_eV[plasma]"]["available"] is True
    assert catalog["mean_energy_eV[plasma]"]["stored"] is False
    assert catalog["pressure_Pa[plasma]"]["unit"] == "Pa"
    paths = write_result(result, tmp_path / "run")
    assert read_result_h5(paths.result_h5).metadata["series_catalog"] == catalog
    with h5py.File(paths.result_h5) as handle:
        assert handle.attrs["format_version"] == 4


def test_outputs_remains_available_after_semantic_selection_error(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    document = minimal_case_document()
    document["output"] = {"observables": ["pressure_Pa[wrong_zone]"]}
    path = tmp_path / "case.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    def no_solve(*_args, **_kwargs):
        pytest.fail("catalog construction must not integrate")

    monkeypatch.setattr("plasma_global.build.solve_compiled_model", no_solve)
    assert main(["outputs", str(path), "--json"]) == 0
    captured = capsys.readouterr()
    assert "pressure_Pa[plasma]" in json.loads(captured.out)
    assert "OUTPUT SELECTION ERROR" in captured.err
    assert main(["validate", str(path)]) == 2


def test_saved_inspection_and_audit_do_not_load_case_or_integrate(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    paths = write_result(simulate(_case(tmp_path)), tmp_path / "run")

    def forbidden(*_args, **_kwargs):
        pytest.fail("saved-result operations must not load or integrate a case")

    monkeypatch.setattr("plasma_global.cli.load_case", forbidden)
    monkeypatch.setattr("plasma_global.cli.simulate", forbidden)
    assert main(["inspect", str(paths.result_h5), "--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["status"]["success"] is True
    assert document["series_catalog"]["pressure_Pa[plasma]"]["unit"] == "Pa"
    assert main(["audit", str(paths.result_h5)]) == 0
    assert inspect_result_h5(paths.result_h5)["provenance"]


def test_numerical_success_and_plot_failure_have_separate_cli_outcomes(
    tmp_path: Path,
    capsys,
) -> None:
    document = minimal_case_document()
    document["chemistry"]["species"].append(
        {
            "id": "Ar_star",
            "mass_amu": 39.948,
            "elements": {"Ar": 1.0},
        }
    )
    document["chemistry"]["gas_reactions"] = [
        {
            "id": "relax",
            "equation": "Ar_star -> Ar",
            "rate": {"kind": "first_order", "rate_s_inv": 2.0},
        }
    ]
    document["reactor"]["zones"]["plasma"]["initial"]["mole_fractions"] = {
        "Ar": 0.9,
        "Ar_star": 0.1,
    }
    document["output"] = {
        "csv": True,
        "plots": {
            "invalid": {
                "series": ["net_production_m3_s[plasma,Ar_star]"],
                "yscale": "log",
            },
            "valid": {"series": ["pressure_Pa[plasma]"], "format": "svg"},
        },
    }
    source = tmp_path / "case.yaml"
    source.write_text(yaml.safe_dump(document), encoding="utf-8")
    destination = tmp_path / "run"
    assert main(["run", str(source), "--output", str(destination)]) == 3
    assert "ARTIFACT ERROR" in capsys.readouterr().err
    result = read_result_h5(destination / "result.h5")
    assert result.status.success is True
    manifest = yaml.safe_load(
        (destination / "artifacts.yaml").read_text(encoding="utf-8")
    )
    assert [item["status"] for item in manifest["requests"]] == [
        "completed",
        "failed",
        "completed",
    ]
    assert (destination / "csv" / "result.csv").is_file()
    assert (destination / "plots" / "valid" / "figures" / "valid.svg").is_file()
    assert not (destination / "plots" / "invalid").exists()
    assert np.isfinite(result.state).all()


def test_partial_integration_result_is_saved_by_cli_and_attached_to_api_error(
    tmp_path: Path,
    capsys,
) -> None:
    from plasma_global import load_case
    from plasma_global.errors import IntegrationError

    table = tmp_path / "coefficients.csv"
    table.write_text("mean_energy_eV\n1\n4\n", encoding="utf-8")
    document = minimal_case_document()
    document["models"]["electrons"]["coefficients"] = {"file": table.name}
    document["reactor"]["power_ports"] = {
        "heat": {
            "zone_id": "plasma",
            "model": {
                "kind": "prescribed_power",
                "absorbed_power_W": 1e-6,
                "electron_fraction": 1.0,
                "gas_fraction": 0.0,
            },
        }
    }
    document["run"] = {"duration_s": 0.01}
    document["solver"] = {"sample_interval_s": 0.005}
    source = tmp_path / "case.yaml"
    source.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(IntegrationError) as caught:
        simulate(load_case(source))
    partial = caught.value.partial_result
    assert partial is not None
    assert partial.status.success is False
    assert partial.time_s[0] == 0.0
    assert 0.0 < partial.time_s[-1] < 0.005
    assert np.isfinite(partial.state).all()
    assert np.max(partial.series("mean_energy_eV[plasma]")) <= 4.0
    destination = tmp_path / "failed-run"
    assert main(["run", str(source), "--output", str(destination)]) == 1
    loaded = read_result_h5(destination / "result.h5")
    assert loaded.status.success is False
    assert loaded.metadata["requested_end_s"] == 0.01
    assert loaded.metadata["reached_end_s"] == float(loaded.time_s[-1])
    summary = yaml.safe_load((destination / "summary.yaml").read_text(encoding="utf-8"))
    assert summary["time"]["requested_end_s"] == 0.01
    assert summary["time"]["reached_end_s"] == float(loaded.time_s[-1])
    assert summary["time"]["failure_time_s"] >= summary["time"]["reached_end_s"]
    assert "INTEGRATION ERROR" in capsys.readouterr().err
    assert main(["audit", str(destination / "result.h5")]) == 1


def test_manifest_write_failure_preserves_numeric_data_and_reports_cli_exit_three(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    import plasma_global.output as output

    document = minimal_case_document()
    source = tmp_path / "case.yaml"
    source.write_text(yaml.safe_dump(document), encoding="utf-8")
    result = simulate(_case(tmp_path))
    original = yaml.safe_dump

    def fail_after_partial_write(data, stream=None, **options):
        if Path(getattr(stream, "name", "")).name == "artifacts.yaml":
            stream.write("status: ")
            raise OSError("manifest storage exhausted")
        return original(data, stream, **options)

    monkeypatch.setattr(output.yaml, "safe_dump", fail_after_partial_write)
    paths = write_result(result, tmp_path / "api-run")
    assert paths.artifacts.status == "failed"
    assert paths.artifacts.manifest is None
    assert paths.artifacts.requests[-1].kind == "manifest"
    assert "storage exhausted" in paths.artifacts.requests[-1].message
    assert read_result_h5(paths.result_h5).status.success is True
    assert not (paths.directory / "artifacts.yaml").exists()
    assert not list(paths.directory.glob(".artifacts-*"))
    destination = tmp_path / "cli-run"
    assert main(["run", str(source), "--output", str(destination)]) == 3
    captured = capsys.readouterr()
    assert str(destination / "result.h5") in captured.out
    assert "manifest storage exhausted" in captured.err
    assert read_result_h5(destination / "result.h5").status.success is True
    assert not (destination / "artifacts.yaml").exists()
