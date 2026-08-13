from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from plasma_global import simulate, write_result
from plasma_global.input.load import CaseLoadError, load_case
from plasma_global.input.migrate_v2 import (
    migrate_v2,
    migrate_v2_to_yaml,
    write_v3_case,
)
from plasma_global.input.schema import CaseSpec
from plasma_global.output import read_result_h5

ROOT = Path(__file__).resolve().parents[1]
V2_FIXTURES = ROOT / "tests" / "fixtures" / "v2"
V2_CONFIGS = V2_FIXTURES / "configs"


def _yaml_mapping(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _input_paths(value: object) -> tuple[Path, ...]:
    if isinstance(value, Path):
        return (value,)
    if isinstance(value, dict):
        return tuple(path for item in value.values() for path in _input_paths(item))
    if isinstance(value, (list, tuple)):
        return tuple(path for item in value for path in _input_paths(item))
    return ()


ARGON_LXCAT_REACTIONS = (
    "G_AR_ION",
    *(f"G_AR_EXC_{index:02d}" for index in range(1, 31)),
)

ZDP_REACTIONS = (
    "ZDP_AR_ION",
    "ZDP_AR_EXC",
    "ZDP_ARSTAR_DEEXC",
    "ZDP_ARSTAR_ION",
    "ZDP_AR2PLUS_DISS_RECOMB",
    "ZDP_AR2PLUS_CONVERSION",
    "ZDP_ARSTAR_POOLING",
    "ZDP_ARPLUS_THREE_BODY_RECOMB",
    "ZDP_ARSTAR_THREE_BODY_QUENCH",
    "ZDP_ARPLUS_CLUSTERING",
)


def _case() -> dict:
    return {
        "schema_version": 3,
        "case": {"name": "argon_0d", "description": "minimal v3 case"},
        "chemistry": {"manifest": "chemistry/manifest.yaml"},
        "reactor": {
            "chamber_id": "r0",
            "zones": [
                {
                    "zone_id": "plasma",
                    "volume_m3": 0.01,
                    "pressure_Pa": 10.0,
                    "gas_temperature_K": 300.0,
                    "initial_densities_m3": {
                        "Ar": 2.4e21,
                        "Ar_plus": 1.0e12,
                    },
                    "initial_mean_energy_eV": 3.0,
                }
            ],
            "gas_inlets": [
                {
                    "inlet_id": "feed",
                    "zone_id": "plasma",
                    "flow_sccm": {"Ar": 10.0},
                    "temperature_K": 300.0,
                }
            ],
            "power_ports": [
                {
                    "port_id": "source",
                    "zone_id": "plasma",
                    "model": {
                        "kind": "prescribed_power",
                        "electron_fraction": 0.85,
                        "gas_fraction": 0.15,
                    },
                }
            ],
        },
        "recipe": {
            "recipe_id": "pulse",
            "start_time_s": 0.0,
            "steps": [
                {
                    "step_id": "on",
                    "duration_s": 0.001,
                    "commands": {
                        "gas_inlets": {"feed": {"flow_sccm": {"Ar": 10.0}}},
                        "power_ports": {
                            "source": {
                                "kind": "prescribed_power",
                                "absorbed_power_W": 100.0,
                            }
                        },
                    },
                }
            ],
        },
        "models": {
            "electrons": {"kind": "maxwellian"},
            "electron_closure": {"kind": "electron_energy"},
            "electron_density": {"kind": "quasineutral"},
            "gas_energy": {"kind": "fixed"},
        },
        "solver": {
            "method": "BDF",
            "rtol": 1.0e-6,
            "atol": 1.0e-12,
            "sample_interval_s": 1.0e-5,
        },
        "output": {},
    }


def _write(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_load_case_resolves_paths_and_exposes_recipe_duration(tmp_path: Path) -> None:
    source = _write(tmp_path / "case.yaml", _case())

    case = load_case(source)

    assert case.schema_version == 3
    assert case.chemistry.manifest == (tmp_path / "chemistry/manifest.yaml").resolve()
    assert case.recipe.duration_s == pytest.approx(1.0e-3)
    assert case.recipe.end_time_s == pytest.approx(1.0e-3)
    assert case.reactor.power_ports[0].model.kind == "prescribed_power"
    assert case.source_path == source.resolve()


def test_include_merges_mappings_and_resolves_paths_at_declaration(
    tmp_path: Path,
) -> None:
    base_data = _case()
    _write(tmp_path / "base" / "base.yaml", base_data)
    child = {
        "include": "../base/base.yaml",
        "case": {"description": "child"},
        "chemistry": {"manifest": "child-chemistry.yaml"},
        "recipe": {
            "steps": [
                {
                    "step_id": "replacement",
                    "duration_s": 0.002,
                    "commands": {
                        "power_ports": {
                            "source": {
                                "kind": "prescribed_power",
                                "absorbed_power_W": 50.0,
                            }
                        }
                    },
                }
            ]
        },
        "solver": {"rtol": 2.0e-6},
    }
    source = _write(tmp_path / "child" / "case.yaml", child)

    case = load_case(source)

    assert case.case.name == "argon_0d"
    assert case.case.description == "child"
    assert case.solver.rtol == 2.0e-6
    assert [step.step_id for step in case.recipe.steps] == ["replacement"]
    assert (
        case.chemistry.manifest == (tmp_path / "child/child-chemistry.yaml").resolve()
    )
    assert case.included_files == ((tmp_path / "base/base.yaml").resolve(),)


@pytest.mark.parametrize("include", [["base.yaml"], {"path": "base.yaml"}, 3])
def test_include_is_one_scalar_path(tmp_path: Path, include: object) -> None:
    data = _case()
    data["include"] = include
    source = _write(tmp_path / "case.yaml", data)

    with pytest.raises(CaseLoadError, match="include must be one path string"):
        load_case(source)


def test_include_cycle_and_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("include: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("include: a.yaml\n", encoding="utf-8")
    with pytest.raises(CaseLoadError, match="include cycle"):
        load_case(tmp_path / "a.yaml")

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("schema_version: 3\nschema_version: 3\n", encoding="utf-8")
    with pytest.raises(CaseLoadError, match="duplicate key"):
        load_case(duplicate)


def test_paths_do_not_expand_environment_or_tilde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PG_V3_OUTPUT", str(tmp_path / "expanded"))
    data = _case()
    data["chemistry"]["manifest"] = "~/manifest.yaml"
    data["models"]["electron_density"] = {
        "kind": "experimental.prescribed_profile",
        "file": "$PG_V3_OUTPUT/electrons.csv",
    }

    case = load_case(_write(tmp_path / "case.yaml", data))

    assert case.chemistry.manifest == (tmp_path / "~/manifest.yaml").resolve()
    assert (
        case.models.electron_density.file
        == (tmp_path / "$PG_V3_OUTPUT/electrons.csv").resolve()
    )


def test_strict_schema_forbids_coercion_and_unknown_fields(tmp_path: Path) -> None:
    wrong_version = _case()
    wrong_version["schema_version"] = "3"
    with pytest.raises(CaseLoadError):
        load_case(_write(tmp_path / "version.yaml", wrong_version))

    legacy_flag = _case()
    legacy_flag["output"]["hdf5"] = True
    with pytest.raises(CaseLoadError, match="hdf5"):
        load_case(_write(tmp_path / "output.yaml", legacy_flag))

    removed_directory = _case()
    removed_directory["output"]["directory"] = "unused"
    with pytest.raises(CaseLoadError, match="directory"):
        load_case(_write(tmp_path / "directory.yaml", removed_directory))

    invalid_summary = _case()
    invalid_summary["output"]["summary_series"] = ["n[plasma,Ar]", "n[plasma,Ar]"]
    with pytest.raises(CaseLoadError, match="must not contain duplicates"):
        load_case(_write(tmp_path / "summary.yaml", invalid_summary))


def test_command_kind_must_match_its_port_model() -> None:
    data = _case()
    command = data["recipe"]["steps"][0]["commands"]["power_ports"]["source"]
    command.clear()
    command.update({"kind": "experimental.icp", "delivered_power_W": 100.0})

    with pytest.raises(ValueError, match="does not match power port"):
        CaseSpec.model_validate(data)


@pytest.mark.parametrize(
    ("command_group", "payload", "message"),
    [
        ("gas_inlets", {"missing": {"flow_sccm": {"Ar": 1.0}}}, "gas inlet"),
        (
            "power_ports",
            {
                "missing": {
                    "kind": "prescribed_power",
                    "absorbed_power_W": 1.0,
                }
            },
            "power port",
        ),
        ("surfaces", {"missing": {"temperature_K": 300.0}}, "surface"),
    ],
)
def test_recipe_commands_reject_unknown_reactor_references(
    command_group: str, payload: dict[str, object], message: str
) -> None:
    data = _case()
    data["recipe"]["steps"][0]["commands"][command_group] = payload

    with pytest.raises(ValueError, match=f"unknown {message}"):
        CaseSpec.model_validate(data)


def test_case_cross_references_validate_closure_and_save_range() -> None:
    missing_energy = _case()
    missing_energy["reactor"]["zones"][0]["initial_mean_energy_eV"] = None
    with pytest.raises(ValueError, match="requires initial_mean_energy_eV"):
        CaseSpec.model_validate(missing_energy)

    local_field_with_energy = _case()
    local_field_with_energy["models"]["electron_closure"] = {"kind": "local_field"}
    with pytest.raises(ValueError, match="incompatible with the local_field"):
        CaseSpec.model_validate(local_field_with_energy)

    outside_recipe = _case()
    del outside_recipe["solver"]["sample_interval_s"]
    outside_recipe["solver"]["save_at_s"] = [0.0, 0.002]
    with pytest.raises(ValueError, match="within the recipe time interval"):
        CaseSpec.model_validate(outside_recipe)


def test_initial_state_closure_and_sampling_are_explicit() -> None:
    missing_density = _case()
    missing_density["reactor"]["zones"][0]["initial_densities_m3"] = {}
    with pytest.raises(ValueError, match="initial_densities_m3"):
        CaseSpec.model_validate(missing_density)

    local_field = _case()
    local_field["models"]["electrons"] = {
        "kind": "table",
        "file": Path("rates.h5"),
        "lookup": "local_field",
    }
    local_field["models"]["electron_closure"] = {"kind": "local_field"}
    local_field["reactor"]["zones"][0]["initial_mean_energy_eV"] = None
    assert (
        CaseSpec.model_validate(local_field).models.electron_closure.kind
        == "local_field"
    )

    mean_energy_table = _case()
    mean_energy_table["models"]["electrons"] = {
        "kind": "table",
        "file": Path("rates.h5"),
        "lookup": "mean_energy",
    }
    assert (
        CaseSpec.model_validate(mean_energy_table).models.electrons.lookup
        == "mean_energy"
    )

    both_sampling_modes = _case()
    both_sampling_modes["solver"]["save_at_s"] = [0.0, 0.001]
    with pytest.raises(ValueError, match="at most one"):
        CaseSpec.model_validate(both_sampling_modes)

    native_sampling = _case()
    del native_sampling["solver"]["sample_interval_s"]
    assert CaseSpec.model_validate(native_sampling).solver.sample_interval_s is None

    invalid_rtol = _case()
    invalid_rtol["solver"]["rtol"] = 1.0
    with pytest.raises(ValueError, match="less than 1"):
        CaseSpec.model_validate(invalid_rtol)

    tiny_rtol = _case()
    tiny_rtol["solver"]["rtol"] = 1.0e-20
    with pytest.raises(ValueError, match="greater than or equal"):
        CaseSpec.model_validate(tiny_rtol)

    invalid_steps = _case()
    invalid_steps["solver"]["first_step_s"] = 2.0e-4
    invalid_steps["solver"]["max_step_s"] = 1.0e-4
    with pytest.raises(ValueError, match="must not exceed"):
        CaseSpec.model_validate(invalid_steps)

    experimental_stop = _case()
    experimental_stop["experimental"] = {
        "stop_when_quasi_steady": {
            "relative_rhs_norm_s_inv": 1.0e-3,
            "min_time_s": 1.0e-6,
        }
    }
    assert (
        CaseSpec.model_validate(
            experimental_stop
        ).experimental.stop_when_quasi_steady.min_time_s
        == 1.0e-6
    )

    removed_standard_stop = _case()
    removed_standard_stop["solver"]["stop_when_quasi_steady"] = {
        "relative_rhs_norm_s_inv": 1.0e-3
    }
    with pytest.raises(ValueError, match="stop_when_quasi_steady"):
        CaseSpec.model_validate(removed_standard_stop)


def test_evolved_gas_wall_relaxation_is_keyed_by_known_zone() -> None:
    data = _case()
    data["models"]["gas_energy"] = {
        "kind": "evolved",
        "wall_energy_relaxation_s_inv_by_zone": {"plasma": 25.0},
    }
    case = CaseSpec.model_validate(data)
    assert case.models.gas_energy.wall_energy_relaxation_s_inv_by_zone["plasma"] == 25.0

    data["models"]["gas_energy"]["wall_energy_relaxation_s_inv_by_zone"] = {
        "missing": 1.0
    }
    with pytest.raises(ValueError, match="unknown zones"):
        CaseSpec.model_validate(data)


def test_wall_transport_is_a_typed_union() -> None:
    data = _case()
    data["reactor"]["surfaces"] = [
        {
            "surface_id": "wall",
            "zone_id": "plasma",
            "area_m2": 0.1,
            "temperature_K": 300.0,
            "wall_transport": {
                "kind": "prescribed_frequency",
                "frequency_s_inv": 1000.0,
            },
        }
    ]
    case = CaseSpec.model_validate(data)
    assert case.reactor.surfaces[0].wall_transport.kind == "prescribed_frequency"

    data["reactor"]["surfaces"][0]["wall_transport"] = {"kind": "prescribed_frequency"}
    with pytest.raises(ValueError, match="frequency_s_inv"):
        CaseSpec.model_validate(data)


def test_experimental_features_are_enabled_by_optional_submodel_presence() -> None:
    data = _case()
    data["experimental"] = {"film": {}, "wall_inventory": None}
    case = CaseSpec.model_validate(data)
    assert case.experimental is not None
    assert case.experimental.film is not None
    assert case.experimental.wall_inventory is None

    data["experimental"]["film"] = {"enabled": True}
    with pytest.raises(ValueError, match="enabled"):
        CaseSpec.model_validate(data)

    data = _case()
    data["experimental"] = {}
    with pytest.raises(ValueError, match="at least one explicit feature"):
        CaseSpec.model_validate(data)


def test_v2_migration_produces_valid_standalone_v3_yaml(tmp_path: Path) -> None:
    result = migrate_v2(V2_CONFIGS / "case_smoke.yaml")

    assert result.case.schema_version == 3
    assert "physics.mode" in result.report.unused_keys
    assert any("initial_densities" in warning for warning in result.report.warnings)
    assert result.case.experimental is not None
    assert result.case.experimental.wall_inventory is not None
    assert result.case.experimental.wall_inventory.initial_by_surface["wafer"] == {
        "F_reservoir": 0.0
    }
    assert "wafer:*" not in result.case.reactor.surfaces[0].initial_coverages

    destination = write_v3_case(result, tmp_path / "migrated.yaml")
    reloaded = load_case(destination)
    assert reloaded.case.name == "experimental_smoke_maxwell"
    assert reloaded.models.electrons.kind == "maxwellian"


def test_v2_migration_refactor_is_deterministic() -> None:
    source = V2_CONFIGS / "case_smoke.yaml"

    first = migrate_v2(source)
    second = migrate_v2(source)

    assert first.case.model_dump(mode="python") == second.case.model_dump(mode="python")
    assert first.report == second.report


def test_v2_migration_preserves_an_explicit_legacy_field_grid(
    tmp_path: Path,
) -> None:
    source = _yaml_mapping(V2_CONFIGS / "case_argon_lxcat.yaml")
    source["include"] = str((V2_CONFIGS / "base_case.yaml").resolve())
    source["files"] = {
        "chamber": str((V2_CONFIGS / "chamber_argon_icp.yaml").resolve()),
        "recipe": str((V2_CONFIGS / "recipe_argon_lxcat.yaml").resolve()),
        "chemistry": {
            "manifest": str(
                (
                    V2_FIXTURES / "chemistry_argon_lxcat" / "chemistry_manifest.yaml"
                ).resolve()
            )
        },
    }
    swarm = dict(source["swarm"])
    swarm["boltzmann_2term"] = {
        "reduced_field_grid_Td": {"min": 0.2, "max": 2500.0, "n": 48}
    }
    source["swarm"] = swarm
    explicit = tmp_path / "explicit-grid.yaml"
    explicit.write_text(yaml.safe_dump(source), encoding="utf-8")

    result = migrate_v2(explicit)
    electrons = result.case.models.electrons

    assert electrons.kind == "experimental.approximate_two_term"
    assert electrons.reduced_field_grid.min_Td == 0.2
    assert electrons.reduced_field_grid.max_Td == 2500.0
    assert electrons.reduced_field_grid.n == 48
    assert not any(
        "legacy default reduced-field grid" in warning
        for warning in result.report.warnings
    )


def test_v2_initial_density_migration_preserves_explicit_and_derived_paths() -> None:
    derived = migrate_v2(V2_CONFIGS / "case_smoke.yaml")
    densities_by_zone = {
        zone.zone_id: zone.initial_densities_m3 for zone in derived.case.reactor.zones
    }
    for densities in densities_by_zone.values():
        assert densities["Ar_plus"] == 1.0e13
        assert densities["O_minus"] == 0.0
        assert densities["Ar"] / densities["CF4"] == pytest.approx(5.0)
        assert densities["CF4"] / densities["O2"] == pytest.approx(10.0)
    density_warnings = [
        warning
        for warning in derived.report.warnings
        if "positive-ion seeds" in warning or "initial_densities_m3" in warning
    ]
    assert len(density_warnings) == 4

    explicit = migrate_v2(V2_CONFIGS / "case_crane_two_reaction_argon.yaml")
    assert explicit.case.reactor.zones[0].initial_densities_m3 == {
        "Ar": 2.5e25,
        "Ar_plus": 1.0e6,
    }
    assert not any(
        "positive-ion seeds" in warning for warning in explicit.report.warnings
    )


def test_v2_power_command_migration_preserves_control_specific_setpoints() -> None:
    result = migrate_v2(V2_CONFIGS / "case_rf_envelope_calibration.yaml")

    models = {
        port.port_id: port.model.model_dump(mode="python", exclude_none=True)
        for port in result.case.reactor.power_ports
    }
    assert models == {
        "source_rf": {
            "kind": "experimental.rf_envelope",
            "role": "source",
            "frequency_Hz": 13.56e6,
            "control": "absorbed_power",
            "effective_impedance_ohm": 50.0,
            "coupling_efficiency": 0.65,
            "base_reduced_field_Td": 25.0,
            "reduced_field_per_sqrt_W_Td": 0.8,
            "self_bias_fraction": 0.35,
            "plasma_potential_offset_V": 0.0,
            "plasma_potential_per_sqrt_W": 0.0,
        },
        "wafer_bias": {
            "kind": "experimental.rf_envelope",
            "role": "bias",
            "frequency_Hz": 2.0e6,
            "control": "voltage",
            "effective_impedance_ohm": 200.0,
            "coupling_efficiency": 0.2,
            "base_reduced_field_Td": 0.0,
            "reduced_field_per_sqrt_W_Td": 0.0,
            "self_bias_fraction": 0.35,
            "plasma_potential_offset_V": 0.0,
            "plasma_potential_per_sqrt_W": 0.0,
        },
    }
    commands = [
        step.model_dump(mode="python", exclude_none=True)["commands"]["power_ports"]
        for step in result.case.recipe.steps
    ]
    assert commands == [
        {
            "source_rf": {
                "kind": "experimental.rf_envelope",
                "absorbed_power_W": 500.0,
                "waveform": {"kind": "continuous"},
            },
            "wafer_bias": {
                "kind": "experimental.rf_envelope",
                "voltage_rms_V": 50.0,
                "waveform": {"kind": "continuous"},
            },
        },
        {
            "source_rf": {
                "kind": "experimental.rf_envelope",
                "absorbed_power_W": 1500.0,
                "waveform": {"kind": "continuous"},
            },
            "wafer_bias": {
                "kind": "experimental.rf_envelope",
                "voltage_rms_V": 120.0,
                "waveform": {"kind": "continuous"},
            },
        },
        {
            "source_rf": {
                "kind": "experimental.rf_envelope",
                "absorbed_power_W": 0.0,
                "waveform": {"kind": "continuous"},
            },
            "wafer_bias": {
                "kind": "experimental.rf_envelope",
                "voltage_rms_V": 0.0,
                "waveform": {"kind": "continuous"},
            },
        },
    ]
    assert not any("waveform='false'" in warning for warning in result.report.warnings)


def test_v2_dc_command_migration_keeps_static_circuit_data_on_the_port() -> None:
    result = migrate_v2(V2_CONFIGS / "case_zdplaskin_example2.yaml")

    port = result.case.reactor.power_ports[0]
    command = result.case.recipe.steps[0].commands.power_ports["dc_series_drive"]
    assert port.model.model_dump(mode="python", exclude_none=True) == {
        "kind": "dc_series",
        "ballast_resistance_ohm": 100_000.0,
        "gap_m": 0.004,
        "electrode_area_m2": 5.0265482457e-05,
        "source_voltage_V": 1000.0,
        "power_absorption_fraction": 1.0,
        "electron_mobility_m2_V_s": 0.45,
    }
    assert command.model_dump(mode="python", exclude_none=True) == {
        "kind": "dc_series",
        "off_voltage_V": 0.0,
        "waveform": {"kind": "continuous"},
    }


@pytest.mark.parametrize(
    (
        "source_name",
        "case_name",
        "electron_kind",
        "power_kinds",
        "reaction_ids",
        "boundary_mapping",
        "segment_mapping",
    ),
    [
        pytest.param(
            "case_smoke.yaml",
            "experimental_smoke_maxwell",
            "maxwellian",
            ("experimental.icp", "experimental.ccp"),
            ("G0000", "G0001", "G0002", "G0003", "G0004"),
            None,
            None,
            id="smoke",
        ),
        pytest.param(
            "case_argon_lxcat.yaml",
            "experimental_argon_lxcat_trend",
            "experimental.approximate_two_term",
            ("experimental.icp", "experimental.ccp"),
            ARGON_LXCAT_REACTIONS,
            None,
            "argon_lxcat_cross_section_segments.yaml",
            id="argon-lxcat",
        ),
        pytest.param(
            "case_crane_two_reaction_argon.yaml",
            "crane_two_reaction_argon",
            "maxwellian",
            (),
            ("CRANE_AR_ION", "CRANE_ARPLUS_THREE_BODY_RECOMB"),
            None,
            None,
            id="crane",
        ),
        pytest.param(
            "case_rf_envelope_calibration.yaml",
            "experimental_argon_rf_envelope_calibration",
            "experimental.approximate_two_term",
            ("experimental.rf_envelope", "experimental.rf_envelope"),
            ARGON_LXCAT_REACTIONS,
            None,
            "argon_lxcat_cross_section_segments.yaml",
            id="rf-envelope",
        ),
        pytest.param(
            "case_zdplaskin_example2.yaml",
            "zdplaskin_example2_dc_series",
            "table",
            ("dc_series",),
            ZDP_REACTIONS,
            "zdplaskin_boundary_products.yaml",
            None,
            id="zdplaskin",
        ),
    ],
)
def test_all_v2_examples_migrate_to_standalone_compilable_v3_bundles(
    tmp_path: Path,
    source_name: str,
    case_name: str,
    electron_kind: str,
    power_kinds: tuple[str, ...],
    reaction_ids: tuple[str, ...],
    boundary_mapping: str | None,
    segment_mapping: str | None,
) -> None:
    destination = tmp_path / Path(source_name).stem / "case.yaml"
    mapping_root = ROOT / "tools" / "importers"
    result = migrate_v2_to_yaml(
        V2_CONFIGS / source_name,
        destination,
        boundary_products=(
            _yaml_mapping(mapping_root / boundary_mapping) if boundary_mapping else None
        ),
        cross_section_segments=(
            {
                str(key): int(value)
                for key, value in _yaml_mapping(mapping_root / segment_mapping).items()
            }
            if segment_mapping
            else None
        ),
    )

    assert result.case.chemistry.manifest.is_file()
    assert destination.with_name("case.migration.yaml").is_file()
    reloaded = load_case(destination)
    assert reloaded.solver.atol == 1.0e-14
    assert "numerics.atol" in result.report.unused_keys
    from plasma_global.build import compile_case

    compiled = compile_case(reloaded)
    simulated = simulate(reloaded)
    paths = write_result(simulated, destination.parent / "roundtrip")
    restored = read_result_h5(paths.result_h5)
    assert reloaded.case.name == case_name
    assert reloaded.models.electrons.kind == electron_kind
    assert (
        tuple(port.model.kind for port in reloaded.reactor.power_ports) == power_kinds
    )
    assert compiled.chemistry.reaction_ids == reaction_ids
    assert simulated.status.success
    assert restored.status == simulated.status
    assert restored.state_labels == simulated.state_labels
    assert restored.time_s.tolist() == simulated.time_s.tolist()
    assert restored.state.tolist() == simulated.state.tolist()
    if electron_kind == "experimental.approximate_two_term":
        field_grid = reloaded.models.electrons.reduced_field_grid
        assert field_grid.min_Td == 1.0
        assert field_grid.max_Td == 100.0
        assert any(
            "validates a power-balance root" in warning
            for warning in result.report.warnings
        )
    if case_name == "experimental_smoke_maxwell":
        assert any("cv_over_kb" in warning for warning in result.report.warnings)
        assert any(
            "positive-ion seeds" in warning for warning in result.report.warnings
        )

    # Every runtime path must resolve inside the generated bundle.  The source
    # fixture can therefore be archived or removed without breaking the result.
    bundle_root = destination.parent.resolve()
    runtime_paths = _input_paths(reloaded.model_dump(mode="python"))
    assert runtime_paths
    assert all(path.is_file() for path in runtime_paths)
    assert all(path.resolve().is_relative_to(bundle_root) for path in runtime_paths)
    assert str(V2_FIXTURES.resolve()) not in destination.read_text(encoding="utf-8")


def test_v2_migration_reports_removed_dc_tuning_and_output_directory() -> None:
    result = migrate_v2(V2_CONFIGS / "case_zdplaskin_example2.yaml")

    assert "files.output_dir" in result.report.unused_keys
    assert (
        "reactor.power_ports[0].parameters.mobility_source" in result.report.unused_keys
    )
    assert (
        "reactor.power_ports[0].parameters.conductance_multiplier"
        in result.report.unused_keys
    )
    model = result.case.reactor.power_ports[0].model
    assert model.kind == "dc_series"
    assert model.electron_mobility_m2_V_s == pytest.approx(0.45)


def test_v2_quasi_steady_event_migrates_only_to_experimental(tmp_path: Path) -> None:
    source = _yaml_mapping(V2_CONFIGS / "case_crane_two_reaction_argon.yaml")
    source["include"] = str((V2_CONFIGS / "base_case.yaml").resolve())
    files = dict(source["files"])
    files["chamber"] = str(
        (V2_CONFIGS / "chamber_crane_two_reaction_argon.yaml").resolve()
    )
    files["recipe"] = str(
        (V2_CONFIGS / "recipe_crane_two_reaction_argon.yaml").resolve()
    )
    files["chemistry"] = {
        "manifest": str(
            (
                V2_FIXTURES
                / "chemistry_crane_two_reaction_argon"
                / "chemistry_manifest.yaml"
            ).resolve()
        )
    }
    source["files"] = files
    numerics = dict(source["numerics"])
    numerics["events"] = {
        "steady_state": {
            "enabled": True,
            "relative_rhs_norm_s_inv": 2.0e-4,
            "min_step_time_s": 3.0e-8,
        }
    }
    source["numerics"] = numerics
    migrated_source = tmp_path / "old.yaml"
    migrated_source.write_text(yaml.safe_dump(source), encoding="utf-8")

    result = migrate_v2(migrated_source)

    assert result.case.experimental is not None
    stop = result.case.experimental.stop_when_quasi_steady
    assert stop is not None
    assert stop.relative_rhs_norm_s_inv == 2.0e-4
    assert stop.min_time_s == 3.0e-8
    assert "stop_when_quasi_steady" not in result.case.solver.model_fields_set
