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
from plasma_global.input.load import CaseLoadError
from plasma_global.input.migrate_v3 import migrate_v3
from plasma_global.input.schema import CaseSpec
from tools.generate_schema_docs import minimal_case_document


def _write(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_one_yaml_constant_case_has_explicit_initial_state(tmp_path: Path) -> None:
    document = minimal_case_document()
    case = load_case(_write(tmp_path / "case.yaml", document))
    result = simulate(case)
    assert result.status.success is True
    assert len(case.recipe.steps) == 1
    assert case.recipe.duration_s == 0.001
    np.testing.assert_allclose(result.series("n[plasma,Ar_plus]"), 1e12)
    assert "schema_version: 4" in result.metadata["effective_case_yaml"]


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
    assert set(result.observables) == {
        "electron_density_m3[plasma]",
        "electron_temperature_eV[plasma]",
    }
    document["output"]["zones"] = ["typo"]
    with pytest.raises(CaseLoadError, match="unknown zones"):
        load_case(_write(tmp_path / "invalid.yaml", document))


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
    with pytest.raises(CaseLoadError, match="remove configured mobility"):
        load_case(_write(tmp_path / "invalid.yaml", document))


def test_model_selects_closure_and_required_initial_energy(tmp_path: Path) -> None:
    document = minimal_case_document()
    document["models"]["electrons"] = {
        "kind": "local_field",
        "coefficients": {"file": "rates.csv"},
    }
    with pytest.raises(CaseLoadError, match="initial_mean_energy_eV"):
        load_case(_write(tmp_path / "invalid.yaml", document))
    document["models"]["electron_closure"] = {"kind": "local_field"}
    with pytest.raises(CaseLoadError, match="electron_closure"):
        load_case(_write(tmp_path / "invalid.yaml", document))


def test_working_example_copies_only_needed_inputs(tmp_path: Path) -> None:
    path = initialize_example("argon_maxwellian", tmp_path / "argon")
    case = load_case(path)
    assert simulate(case).status.success is True
    assert not tuple(path.parent.rglob("*.h5"))
    assert len(tuple(path.parent.rglob("*.csv"))) == 1
    with pytest.raises(FileExistsError):
        initialize_example("argon_maxwellian", path.parent)


def test_v3_conversion_is_self_contained_and_preserves_state(tmp_path: Path) -> None:
    source = Path(__file__).parent / "fixtures/migration_v3/case.yaml"
    target = migrate_v3(source, tmp_path / "converted")
    migrated = load_case(target)
    assert migrated.recipe.duration_s == 1e-7
    assert simulate(migrated).status.success is True
    assert set(
        CaseSpec.model_validate(
            load_unique_yaml(StringIO(target.read_text()))
        ).reactor.zones
    ) == {"plasma"}
    with pytest.raises(FileExistsError):
        migrate_v3(source, target.parent)


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
    assert [step.label for step in case.recipe.steps] == ["hold", "hold"]
    document["output"] = {"zones": ["plasma"]}
    with pytest.raises(CaseLoadError, match=r"requires output\.quantities"):
        load_case(_write(tmp_path / "invalid.yaml", document))


def test_pulse_clock_is_explicit_across_steps(tmp_path: Path) -> None:
    from plasma_global.input._serialize import case_document

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
    restored = CaseSpec.model_validate(case_document(case)).resolve(tmp_path)
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


def test_v3_hdf5_profile_and_clock_are_migrated_without_inference_at_runtime(
    tmp_path: Path,
) -> None:
    import h5py

    source = Path(__file__).parent / "fixtures/migration_v3/case.yaml"
    document = load_unique_yaml(StringIO(source.read_text()))
    document["chemistry"]["manifest"] = str(source.parent / "chemistry.yaml")
    table = tmp_path / "legacy.h5"
    with h5py.File(table, "w") as handle:
        handle["mean_energy_eV"] = [1.0, 100.0]
        handle["effective_field_Td"] = [10.0, 1000.0]
        handle["mobility_m2_V_s"] = [0.5, 0.1]
        handle.create_group("rate_coefficients")
    document["models"]["electrons"] = {
        "kind": "table",
        "file": str(table),
        "lookup": "mean_energy",
    }
    power = tmp_path / "power.csv"
    power.write_text("time_s,electron_power_W\n2,0.1\n2.001,0.2\n", encoding="utf-8")
    profile = tmp_path / "density.csv"
    profile.write_text(
        "time_s,electron_density_m3\n2,1e12\n2.001,1e12\n", encoding="utf-8"
    )
    document["models"]["electron_density"] = {
        "kind": "experimental.prescribed_profile",
        "file": str(profile),
    }
    document["reactor"]["power_ports"][0]["model"] = {
        "kind": "external_table",
        "file": str(power),
    }
    document["recipe"]["start_time_s"] = 2.0
    document["recipe"]["steps"][0]["commands"]["power_ports"] = {
        "source": {"kind": "external_table", "file": str(power)}
    }
    base = _write(tmp_path / "base.yaml", document)
    legacy = _write(
        tmp_path / "case-v3.yaml", {"include": base.name, "case": {"name": "migrated"}}
    )
    target = migrate_v3(legacy, tmp_path / "converted")
    case = load_case(target)
    compiled = compile_case(case)
    assert case.models.electrons.mobility_reference_neutral_density_m3 == 2e20
    notes = yaml.safe_load((target.parent / "migration.yaml").read_text())["notes"]
    assert any("generation conditions remain unverified" in note for note in notes)
    state = compiled.model.initial_state(compiled.initial_state)
    evaluated = compiled.model.evaluate(2.0, state, compiled.segments[0])
    assert evaluated.ledger_by_zone["plasma"].absorbed_power_J_m3_s == pytest.approx(
        10.0
    )
    assert simulate(case).status.success is True
    assert not list(target.parent.rglob("*.h5"))
