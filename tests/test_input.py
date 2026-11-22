from __future__ import annotations

from copy import deepcopy
from io import StringIO
from pathlib import Path

import numpy as np
import pytest
import yaml

from plasma_global import load_case, simulate
from plasma_global._yaml import YamlLoadError, load_unique_yaml
from plasma_global.build import compile_case
from plasma_global.examples import initialize_example
from plasma_global.input.load import CaseLoadError, case_from_mapping
from tools.generate_schema_docs import minimal_case_document


def _write(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_one_yaml_constant_case_has_explicit_initial_state(tmp_path: Path) -> None:
    document = minimal_case_document()
    case = load_case(_write(tmp_path / "case.yaml", document))
    result = simulate(case)
    assert result.status.success is True
    assert len(case.run.normalized_steps()) == 1
    assert case.run.total_duration_s == 0.001
    np.testing.assert_allclose(result.series("n[plasma,Ar_plus]"), 1e12)
    assert "schema_version: 5" in result.metadata["effective_case_yaml"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 3),
        ("schema_version", "4"),
        ("include", "base.yaml"),
        ("unknown", 1),
    ],
)
def test_only_one_public_schema_without_inheritance(
    tmp_path: Path, field: str, value: object
) -> None:
    document = minimal_case_document()
    document[field] = value
    with pytest.raises(CaseLoadError):
        load_case(_write(tmp_path / "invalid.yaml", document))


def test_run_and_output_selection_require_no_recipe_or_series_syntax(
    tmp_path: Path,
) -> None:
    document = minimal_case_document()
    document["output"] = {
        "quantities": ["electron_density_m3", "electron_temperature_eV"],
        "zones": ["plasma"],
    }
    case = load_case(_write(tmp_path / "case.yaml", document))
    result = simulate(case)
    assert set(result.observables) >= {
        "electron_density_m3[plasma]",
        "electron_temperature_eV[plasma]",
    }
    document["output"]["zones"] = ["typo"]
    with pytest.raises(ValueError, match="unknown output zones"):
        compile_case(load_case(_write(tmp_path / "invalid.yaml", document)))


def test_each_step_resolves_against_static_power_and_rejects_kind(
    tmp_path: Path,
) -> None:
    document = minimal_case_document()
    document["reactor"]["power_ports"] = {
        "source": {
            "zone_id": "plasma",
            "model": {
                "kind": "prescribed_power",
                "electron_fraction": 1.0,
                "gas_fraction": 0.0,
                "absorbed_power_W": 0.1,
            },
        }
    }
    document["run"] = {
        "steps": [
            {
                "duration_s": 1e-4,
                "commands": {"power_ports": {"source": {"absorbed_power_W": 0.2}}},
            },
            {"duration_s": 1e-4},
        ]
    }
    case = load_case(_write(tmp_path / "case.yaml", document))
    compiled = compile_case(case)
    assert [
        segment.port_commands["source"].power_W for segment in compiled.segments
    ] == [0.2, 0.1]
    document["run"]["steps"][0]["commands"]["power_ports"]["source"]["kind"] = (
        "prescribed_power"
    )
    with pytest.raises(CaseLoadError, match="determined"):
        load_case(_write(tmp_path / "invalid.yaml", document))


def test_table_reaction_does_not_require_cross_section_files(tmp_path: Path) -> None:
    document = minimal_case_document()
    document["chemistry"]["gas_reactions"] = [
        {
            "id": "ionize",
            "equation": "e + Ar -> 2 e + Ar_plus",
            "rate": {"kind": "electron_table", "rate_id": "ionize"},
            "electron_energy_transfer_eV": -15.76,
        }
    ]
    document["models"]["electrons"]["coefficients"] = {"file": "rates.csv"}
    document["chemistry"]["rate_models"] = {
        "unused": {"kind": "electron_table", "rate_id": "not_consumed"}
    }
    document["run"]["duration_s"] = 1e-6
    (tmp_path / "rates.csv").write_text(
        "mean_energy_eV,k:ionize,momentum:Ar\n0,1e-20,1e-15\n10,1e-20,1e-15\n",
        encoding="utf-8",
    )
    case = load_case(_write(tmp_path / "case.yaml", document))
    compiled = compile_case(case)
    assert not compiled.chemistry.cross_sections
    assert compiled.model.elastic_heating_evaluator is not None
    assert (
        compiled.metadata["model_ids"]["elastic_heating"] == "tabulated_momentum_rates"
    )
    result = simulate(case)
    assert result.status.success is True
    assert (
        result.series("n[plasma,Ar_plus]")[-1] > result.series("n[plasma,Ar_plus]")[0]
    )
    invalid = deepcopy(document)
    del invalid["chemistry"]["gas_reactions"][0]["electron_energy_transfer_eV"]
    with pytest.raises(ValueError, match="electron_energy_transfer_eV"):
        compile_case(load_case(_write(tmp_path / "invalid.yaml", invalid)))


def test_energy_balance_dc_can_use_explicit_mobility_without_table_mobility(
    tmp_path: Path,
) -> None:
    document = minimal_case_document()
    document["models"]["electrons"]["coefficients"] = {"file": "rates.csv"}
    document["reactor"]["power_ports"] = {
        "source": {
            "zone_id": "plasma",
            "model": {
                "kind": "dc_series",
                "ballast_resistance_ohm": 1e5,
                "gap_m": 0.1,
                "electrode_area_m2": 0.01,
                "source_voltage_V": 1e-4,
                "electron_mobility_m2_V_s": 0.5,
            },
        }
    }
    (tmp_path / "rates.csv").write_text("mean_energy_eV\n0\n10\n", encoding="utf-8")
    result = simulate(load_case(_write(tmp_path / "case.yaml", document)))
    assert result.status.success is True
    (tmp_path / "rates.csv").write_text(
        "mean_energy_eV,reduced_mobility_1_V_m_s\n0,1e20\n10,1e20\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="choose table mobility"):
        compile_case(load_case(tmp_path / "case.yaml"))
    document["models"]["electrons"]["kind"] = "local_field"
    document["reactor"]["zones"]["plasma"]["initial"].pop("mean_energy_eV")
    with pytest.raises(CaseLoadError, match="uses table mobility"):
        load_case(_write(tmp_path / "invalid.yaml", document))


def test_model_selects_closure_and_required_initial_energy(tmp_path: Path) -> None:
    document = minimal_case_document()
    document["models"]["electrons"] = {
        "kind": "local_field",
        "coefficients": {"file": "rates.csv"},
    }
    with pytest.raises(CaseLoadError, match=r"initial\.mean_energy_eV"):
        load_case(_write(tmp_path / "invalid.yaml", document))
    document["models"]["electron_closure"] = {"kind": "local_field"}
    with pytest.raises(CaseLoadError, match="electron_closure"):
        load_case(_write(tmp_path / "invalid.yaml", document))


@pytest.mark.parametrize(
    ("name", "csv_count"),
    [
        ("argon_maxwellian", 1),
        ("zdplaskin_example2", 1),
        ("benchmarks/analytic-decay", 0),
        ("benchmarks/argon-rtol-1e-6", 1),
    ],
)
def test_working_example_copies_only_needed_inputs(
    tmp_path: Path, name: str, csv_count: int
) -> None:
    path = initialize_example(name, tmp_path / "input")
    case = load_case(path)
    result = simulate(case)
    assert result.status.success is True
    inputs = result.metadata["provenance"]["input_files"]["files"]
    assert all(Path(source).is_relative_to(path.parent) for source in inputs)
    assert not tuple(path.parent.rglob("*.h5"))
    assert len(tuple(path.parent.rglob("*.csv"))) == csv_count
    with pytest.raises(FileExistsError):
        initialize_example(name, path.parent)


def test_yaml_identifiers_and_numbers_are_unambiguous() -> None:
    values = load_unique_yaml(
        StringIO(
            "kind: off\nnumber: 2.0e20\ncount: 012\nclock: 1:20\n"
            "date: 2026-10-25\nflag: false\n"
        )
    )
    assert values == {
        "kind": "off",
        "number": 2e20,
        "count": 12,
        "clock": "1:20",
        "date": "2026-10-25",
        "flag": False,
    }
    with pytest.raises(YamlLoadError, match="duplicate"):
        load_unique_yaml(StringIO("a: 1\na: 2\n"))


def test_labels_are_optional_descriptions_not_unique_ids(tmp_path: Path) -> None:
    document = minimal_case_document()
    document["run"] = {
        "steps": [
            {"label": "hold", "duration_s": 1e-6},
            {"label": "hold", "duration_s": 1e-6},
        ]
    }
    case = load_case(_write(tmp_path / "case.yaml", document))
    assert simulate(case).status.success is True
    assert [step.label for step in case.run.normalized_steps()] == ["hold", "hold"]
    document["output"] = {"zones": ["plasma"]}
    with pytest.raises(CaseLoadError, match=r"requires output\.quantities"):
        load_case(_write(tmp_path / "invalid.yaml", document))


def test_pulse_clock_is_explicit_across_steps(tmp_path: Path) -> None:

    document = minimal_case_document()
    document["reactor"]["power_ports"] = {
        "source": {
            "zone_id": "plasma",
            "model": {
                "kind": "prescribed_power",
                "electron_fraction": 1.0,
                "gas_fraction": 0.0,
                "absorbed_power_W": 0.1,
            },
        }
    }
    pulse = {"kind": "square_pulse", "repetition_Hz": 1000.0, "duty_cycle": 0.5}
    document["run"] = {
        "start_time_s": 0.02,
        "steps": [
            {"duration_s": 0.00075},
            {
                "duration_s": 0.0005,
                "commands": {"power_ports": {"source": {"waveform": pulse}}},
            },
        ],
    }
    case = load_case(_write(tmp_path / "case.yaml", document))
    compiled = compile_case(case)
    assert compiled.segments[1].port_commands["source"].kind == "off"
    restored = case_from_mapping(case.to_dict(), base_directory=tmp_path)
    assert [
        (segment.start_s, segment.end_s) for segment in compile_case(restored).segments
    ] == [(segment.start_s, segment.end_s) for segment in compiled.segments]
    pulse["time_reference"] = "step_start"
    shifted = compile_case(load_case(_write(tmp_path / "case.yaml", document)))
    assert shifted.segments[1].port_commands["source"].kind == "power"


def test_external_csv_time_is_relative_to_run_start(tmp_path: Path) -> None:
    document = minimal_case_document()
    (tmp_path / "power.csv").write_text(
        "time_s,electron_power_W\n0,0.1\n0.001,0.2\n", encoding="utf-8"
    )
    document["reactor"]["power_ports"] = {
        "source": {
            "zone_id": "plasma",
            "model": {"kind": "external_table", "file": "power.csv"},
        }
    }
    document["run"]["start_time_s"] = 2.0
    compiled = compile_case(load_case(_write(tmp_path / "case.yaml", document)))
    state = compiled.model.initial_state(compiled.initial_state)
    result = compiled.model.evaluate(2.0, state, compiled.segments[0])
    assert result.ledger_by_zone["plasma"].absorbed_power_J_m3_s == pytest.approx(10.0)


def test_prepared_case_uses_captured_bytes_after_source_changes(tmp_path: Path) -> None:
    from hashlib import sha256

    from plasma_global import case_from_mapping

    document = minimal_case_document()
    document["models"]["electrons"]["coefficients"] = {"file": "rates.csv"}
    table = tmp_path / "rates.csv"
    content = b"mean_energy_eV,reduced_field_Td\n0,0\n10,100\n"
    table.write_bytes(content)
    case = case_from_mapping(document, base_directory=tmp_path)
    prepared = compile_case(case)
    case.reactor.zones["plasma"].initial.seed_densities_m3["Ar_plus"] = 2e12
    table.unlink()
    first = simulate(prepared)
    second = simulate(prepared)
    np.testing.assert_array_equal(first.state, second.state)
    np.testing.assert_allclose(first.series("reduced_field_Td[plasma]"), 30.0)
    assert first.series("electron_density_m3[plasma]")[0] == pytest.approx(1e12)
    record = first.metadata["provenance"]["input_files"]["files"][str(table)]
    assert record["sha256"] == sha256(content).hexdigest()
    assert record["size_bytes"] == len(content)


def test_public_yaml_rebase_preserves_annotations_and_input_origin(
    tmp_path: Path,
) -> None:
    from plasma_global import case_from_mapping

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    table = inputs / "rates.csv"
    table.write_text("mean_energy_eV\n0\n10\n", encoding="utf-8")
    document = minimal_case_document()
    document["models"]["electrons"]["coefficients"] = {
        "file": "rates.csv",
        "provenance": {"file": "notes/source.csv"},
    }
    original = case_from_mapping(document, base_directory=inputs)
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    path = relocated / "case.yaml"
    path.write_text(original.to_yaml(relative_to=relocated), encoding="utf-8")
    reread = load_case(path)
    assert reread.models.electrons.coefficients.provenance["file"] == "notes/source.csv"
    a, b = simulate(original), simulate(reread)
    np.testing.assert_array_equal(a.time_s, b.time_s)
    np.testing.assert_allclose(a.state, b.state)


def test_mapping_factory_requires_an_explicit_input_origin() -> None:
    from plasma_global import case_from_mapping

    with pytest.raises(TypeError, match="base_directory"):
        case_from_mapping(minimal_case_document())


@pytest.mark.parametrize("identity", ["z,part", " z ", "x[y]", "x\x00y"])
def test_structural_ids_are_unambiguous_before_integration(
    tmp_path: Path, identity: str
) -> None:
    document = minimal_case_document()
    document["reactor"]["zones"] = {identity: document["reactor"]["zones"]["plasma"]}
    with pytest.raises(CaseLoadError):
        load_case(_write(tmp_path / "case.yaml", document))


def test_prescribed_density_does_not_invent_a_positive_ion_seed(tmp_path: Path) -> None:
    document = minimal_case_document()
    document["reactor"]["zones"]["plasma"]["initial"]["seed_densities_m3"] = {}
    profile = tmp_path / "density.csv"
    profile.write_text("time_s,n[e]\n0,1e12\n0.001,2e12\n", encoding="utf-8")
    document["models"]["electron_density"] = {
        "kind": "experimental.prescribed_profile",
        "file": "density.csv",
        "zone_columns": {"plasma": "n[e]"},
    }
    result = simulate(load_case(_write(tmp_path / "case.yaml", document)))
    np.testing.assert_array_equal(result.series("n[plasma,Ar_plus]"), 0.0)
    assert result.series("electron_density_m3[plasma]")[0] == pytest.approx(1e12)
    assert result.series("electron_density_m3[plasma]")[-1] == pytest.approx(2e12)
    np.testing.assert_allclose(
        result.series("charge_residual_m3[plasma]"),
        -result.series("electron_density_m3[plasma]"),
    )


@pytest.mark.parametrize("run_start", [0.0, 1000.0])
def test_run_and_profile_share_exact_clock_endpoints(
    tmp_path: Path, run_start: float
) -> None:
    document = minimal_case_document()
    document["run"] = {
        "start_time_s": run_start,
        "steps": [{"duration_s": 0.1}, {"duration_s": 0.2}],
    }
    document["models"]["electron_density"] = {
        "kind": "experimental.prescribed_profile",
        "file": "density.csv",
        "bounds": "error",
    }
    profile = tmp_path / "density.csv"
    profile.write_text("time_s,plasma\n0,1e12\n0.3,1e12\n", encoding="utf-8")
    document["solver"] = {"sample_interval_s": 0.1}
    case = load_case(_write(tmp_path / "case.yaml", document))
    prepared = compile_case(case)
    result = simulate(prepared)
    assert result.status.success
    assert result.time_s[-1] == case.run.end_time_s == prepared.segments[-1].end_s
    np.testing.assert_array_equal(
        result.metadata["run_step_boundaries_s"],
        [
            case.run.start_time_s,
            *(end for _, end in case.run.step_bounds()),
        ],
    )
    np.testing.assert_allclose(result.series("electron_density_m3[plasma]"), 1e12)
    # A real table coverage gap remains an error, rather than silently holding its edge.
    profile.write_text("time_s,plasma\n0,1e12\n0.299,1e12\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        compile_case(load_case(tmp_path / "case.yaml"))


def test_null_step_file_inherits_the_declared_power_table(tmp_path: Path) -> None:
    document = minimal_case_document()
    (tmp_path / "power.csv").write_text(
        "time_s,electron_power_W\n0,1e-6\n0.001,1e-6\n", encoding="utf-8"
    )
    document["reactor"]["power_ports"] = {
        "drive": {
            "zone_id": "plasma",
            "model": {"kind": "external_table", "file": "power.csv"},
        }
    }
    document["run"] = {
        "steps": [
            {
                "duration_s": 0.001,
                "commands": {"power_ports": {"drive": {"file": None}}},
            }
        ]
    }
    case = load_case(_write(tmp_path / "case.yaml", document))
    rebased = yaml.safe_load(case.to_yaml(relative_to=tmp_path / "elsewhere"))
    assert (
        rebased["run"]["steps"][0]["commands"]["power_ports"]["drive"]["file"] is None
    )
    prepared = compile_case(case)
    assert simulate(prepared).status.success
    assert not any(path.name == "None" for path in prepared.input_sources)
