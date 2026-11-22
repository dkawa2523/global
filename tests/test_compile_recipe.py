from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from plasma_global.build import compile_case
from plasma_global.core.transport import SCCM_TO_PARTICLES_PER_S
from plasma_global.errors import CaseValidationError
from plasma_global.input.load import case_from_mapping, load_case

FIXTURE = Path(__file__).parent / "fixtures" / "minimal" / "case.yaml"


def test_compiled_steps_preserve_forcing_temperature_and_command_order() -> None:
    data = load_case(FIXTURE).model_dump(mode="python")
    data["reactor"]["power_ports"]["source"]["model"]["absorbed_power_W"] = 0.0
    data["reactor"]["gas_inlets"] = {
        "feed": {
            "zone_id": "plasma",
            "flow_sccm": {"Ar": 1.0},
            "temperature_K": 350.0,
        }
    }
    first_step = data["run"]["steps"][0]
    first_step["commands"]["gas_inlets"] = {"feed": {"flow_sccm": {"Ar": 4.0}}}
    data["run"]["steps"].append(
        {
            "label": "idle",
            "duration_s": 2.0e-7,
            "commands": {},
        }
    )

    compiled = compile_case(case_from_mapping(data, base_directory=FIXTURE.parent))

    assert [segment.segment_id for segment in compiled.segments] == [
        "step_1",
        "step_2",
    ]
    assert [segment.start_s for segment in compiled.segments] == pytest.approx(
        [0.0, 1.0e-7]
    )
    assert [segment.end_s for segment in compiled.segments] == pytest.approx(
        [1.0e-7, 3.0e-7]
    )
    powered, idle = compiled.segments
    assert powered.port_commands["source"].kind == "power"
    assert idle.port_commands["source"].kind == "power"
    assert idle.port_commands["source"].power_W == 0.0
    assert powered.surface_temperature_K_by_surface == {}
    assert powered.wall_temperature_K_by_zone == {"plasma": 300.0}
    assert idle.surface_temperature_K_by_surface == {}
    assert idle.wall_temperature_K_by_zone == {"plasma": 300.0}
    assert powered.prescribed_electron_density_m3_by_zone == {}

    assert powered.transport is not None
    assert idle.transport is not None
    assert powered.transport.particle_source_m3_s.shape == (1, 2)
    assert idle.transport.particle_source_m3_s.shape == (1, 2)
    expected_powered = SCCM_TO_PARTICLES_PER_S * 4.0 / 0.01
    expected_idle = SCCM_TO_PARTICLES_PER_S * 1.0 / 0.01
    assert powered.transport.particle_source_m3_s[0] == pytest.approx(
        [expected_powered, 0.0]
    )
    assert idle.transport.particle_source_m3_s[0] == pytest.approx([expected_idle, 0.0])
    assert np.array_equal(powered.transport.inlet_heavy_energy_J_m3_s, [0.0])
    assert np.array_equal(idle.transport.inlet_heavy_energy_J_m3_s, [0.0])


def test_compile_rejects_a_step_without_a_required_power_setpoint() -> None:
    data = load_case(FIXTURE).model_dump(mode="python")
    data["reactor"]["power_ports"]["source"]["model"].pop("absorbed_power_W", None)
    data["run"]["steps"][0]["commands"]["power_ports"] = {}
    with pytest.raises(CaseValidationError, match="requires a power command"):
        compile_case(case_from_mapping(data, base_directory=FIXTURE.parent))


@pytest.mark.parametrize("interpolation", ["linear", "previous"])
def test_compile_rejects_external_table_without_full_recipe_coverage(
    tmp_path: Path, interpolation: str
) -> None:
    path = tmp_path / "partial.csv"
    path.write_text("time_s,electron_power_W\n0,1\n1e-8,2\n", encoding="utf-8")
    data = load_case(FIXTURE).model_dump(mode="python")
    data["reactor"]["power_ports"]["source"]["model"] = {
        "kind": "external_table",
        "file": path,
        "interpolation": interpolation,
    }
    data["run"]["steps"][0]["commands"]["power_ports"] = {}
    with pytest.raises(CaseValidationError, match="does not cover recipe interval"):
        compile_case(case_from_mapping(data, base_directory=FIXTURE.parent))


def test_only_effective_table_bindings_supply_field_in_each_step(
    tmp_path: Path,
) -> None:
    field = tmp_path / "field.csv"
    field.write_text(
        "time_s,electron_power_W,reduced_field_Td\n0,0,2\n1,0,2\n", encoding="utf-8"
    )
    power = tmp_path / "power.csv"
    power.write_text("time_s,electron_power_W\n0,0\n1,0\n", encoding="utf-8")
    data = load_case(FIXTURE).model_dump(mode="python")
    data["reactor"]["power_ports"] = {
        name: {
            "zone_id": "plasma",
            "model": {"kind": "external_table", "file": field},
        }
        for name in ("first", "second")
    }
    steps = data["run"]["steps"]
    steps[0]["commands"]["power_ports"] = {"second": {"file": power}}
    steps.append(
        {
            "label": "next",
            "duration_s": 1e-7,
            "commands": {"power_ports": {"first": {"file": power}}},
        }
    )
    compiled = compile_case(case_from_mapping(data, base_directory=FIXTURE.parent))
    assert [
        [
            name
            for name, command in segment.port_commands.items()
            if command.produces_reduced_field
        ]
        for segment in compiled.segments
    ] == [["first"], ["second"]]


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (
            {
                "kind": "prescribed_power",
                "electron_fraction": 1.0,
                "gas_fraction": 0.0,
                "absorbed_power_W": 2.0,
            },
            ("power", 2.0, None),
        ),
        (
            {
                "kind": "dc_series",
                "ballast_resistance_ohm": 1.0,
                "gap_m": 0.1,
                "electrode_area_m2": 0.01,
                "electron_mobility_m2_V_s": 0.5,
                "source_voltage_V": 2.0,
            },
            ("voltage", None, 2.0),
        ),
        (
            {"kind": "experimental.rf_envelope", "absorbed_power_W": 2.0},
            ("power", 2.0, None),
        ),
        (
            {
                "kind": "experimental.ccp",
                "control": "voltage",
                "voltage_rms_V": 2.0,
                "electrode_gap_m": 0.1,
                "powered_area_m2": 0.01,
                "grounded_area_m2": 0.02,
                "dominant_ion": "Ar_plus",
            },
            ("voltage", None, 2.0),
        ),
        (
            {"kind": "experimental.icp", "delivered_power_W": 2.0},
            ("power", 2.0, None),
        ),
    ],
)
def test_omitted_recipe_commands_resolve_to_the_model_default(
    model: dict, expected: tuple
) -> None:
    data = load_case(FIXTURE).model_dump(mode="python")
    data["reactor"]["power_ports"]["source"]["model"] = model
    data["run"]["steps"][0]["commands"]["power_ports"] = {}
    compiled = compile_case(case_from_mapping(data, base_directory=FIXTURE.parent))
    command = compiled.segments[0].port_commands["source"]
    assert (command.kind, command.power_W, command.voltage_V) == expected


@pytest.mark.parametrize("target", ["process", "plasma"])
def test_public_icp_downstream_deposition_matches_yaml_and_python(tmp_path, target):
    data = load_case(FIXTURE).to_dict()
    data["reactor"]["zones"]["process"] = deepcopy(data["reactor"]["zones"]["plasma"])
    data["reactor"]["zones"]["process"]["volume_m3"] = 0.02
    data["reactor"]["power_ports"] = {
        "icp": {
            "zone_id": "plasma",
            "model": {
                "kind": "experimental.icp",
                "delivered_power_W": 300.0,
                "downstream_zone_id": target,
                "downstream_fraction": 0.25,
            },
        }
    }
    data["run"]["steps"][0]["commands"]["power_ports"] = {}
    case = case_from_mapping(data, base_directory=FIXTURE.parent)
    path = tmp_path / "icp.yaml"
    path.write_text(case.to_yaml(relative_to=tmp_path), encoding="utf-8")
    python = compile_case(case)
    yaml = compile_case(load_case(path))
    results = []
    for prepared in (python, yaml):
        model = prepared.model
        state = model.initial_state(prepared.initial_state)
        evaluation = model.evaluate(0.0, state, prepared.segments[0])
        coupling = evaluation.power_coupling
        assert coupling is not None
        port = coupling.port_results["icp"]
        absorbed = port.observables["absorbed_power_W"]
        assert sum(coupling.electron_power_W_by_zone.values()) == pytest.approx(
            absorbed
        )
        assert absorbed + port.observables["reflected_power_W"] == pytest.approx(300.0)
        expected = absorbed if target == "plasma" else 0.75 * absorbed
        assert coupling.electron_power_W_by_zone["plasma"] == pytest.approx(expected)
        assert coupling.electron_power_W_by_zone["process"] == pytest.approx(
            0.0 if target == "plasma" else 0.25 * absorbed
        )
        results.append(evaluation.derivative)
    np.testing.assert_allclose(*results, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("gap_m", [0.01, 0.02])
def test_public_ccp_uses_explicit_geometry_and_selected_ion(gap_m: float) -> None:
    data = load_case(FIXTURE).to_dict()
    data["reactor"]["power_ports"]["source"]["model"] = {
        "kind": "experimental.ccp",
        "control": "voltage",
        "voltage_rms_V": 100.0,
        "electrode_gap_m": gap_m,
        "powered_area_m2": 0.03,
        "grounded_area_m2": 0.12,
        "dominant_ion": "Ar_plus",
    }
    data["run"]["steps"][0]["commands"]["power_ports"] = {}
    prepared = compile_case(case_from_mapping(data, base_directory=FIXTURE.parent))
    model = prepared.model
    state = model.initial_state(prepared.initial_state)
    evaluation = model.evaluate(0.0, state, prepared.segments[0])
    coupling = evaluation.power_coupling
    assert coupling is not None
    result = coupling.port_results["source"]
    expected = (
        result.observables["rf_current_rms_A"]
        * result.observables["bulk_resistance_ohm"]
        / gap_m
        / 2e20
        / 1e-21
    )
    assert result.reduced_field_Td == pytest.approx(expected)
    assert model.power_coordinator is not None
    port = model.power_coordinator.ports[0]
    assert port.electrode_gap_m == gap_m
    assert port.powered_area_m2 == 0.03
    assert port.grounded_area_m2 == 0.12
    assert port.dominant_ion_mass_kg == pytest.approx(39.948 * 1.66053906660e-27)
