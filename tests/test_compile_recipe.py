from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plasma_global.build import compile_case
from plasma_global.core.transport import SCCM_TO_PARTICLES_PER_S
from plasma_global.errors import CaseValidationError
from plasma_global.input._compile_recipe_power import _resolved_port_command
from plasma_global.input._resolved_case import (
    DCSeriesCommand,
    DCSeriesModel,
    ExperimentalCCPCommand,
    ExperimentalCCPModel,
    ExperimentalICPCommand,
    ExperimentalICPModel,
    ExperimentalRFEnvelopeCommand,
    ExperimentalRFEnvelopeModel,
    ExternalTableCommand,
    ExternalTableModel,
    PowerModel,
    PowerPortCommand,
    PowerPortConfig,
    PrescribedPowerCommand,
    PrescribedPowerModel,
    ResolvedCase,
)
from plasma_global.input.load import load_case
from plasma_global.models.external_table import ExternalTableStore

FIXTURE = Path(__file__).parent / "fixtures" / "minimal" / "case.yaml"


@pytest.mark.parametrize(
    ("model", "command", "enabled", "expected"),
    [
        (
            PrescribedPowerModel(
                kind="prescribed_power",
                electron_fraction=1.0,
                gas_fraction=0.0,
            ),
            PrescribedPowerCommand(kind="prescribed_power", absorbed_power_W=12.0),
            True,
            ("power", 12.0, None),
        ),
        (
            PrescribedPowerModel(
                kind="prescribed_power",
                electron_fraction=1.0,
                gas_fraction=0.0,
            ),
            PrescribedPowerCommand(kind="prescribed_power", absorbed_power_W=12.0),
            False,
            ("off", None, None),
        ),
        (
            PrescribedPowerModel(
                kind="prescribed_power",
                electron_fraction=1.0,
                gas_fraction=0.0,
            ),
            PrescribedPowerCommand(kind="prescribed_power"),
            True,
            None,
        ),
        (
            DCSeriesModel(
                kind="dc_series",
                ballast_resistance_ohm=1.0,
                gap_m=1.0,
                electrode_area_m2=1.0,
                source_voltage_V=100.0,
            ),
            DCSeriesCommand(kind="dc_series"),
            True,
            ("voltage", None, 100.0),
        ),
        (
            DCSeriesModel(
                kind="dc_series",
                ballast_resistance_ohm=1.0,
                gap_m=1.0,
                electrode_area_m2=1.0,
            ),
            DCSeriesCommand(kind="dc_series", off_voltage_V=-5.0),
            False,
            ("voltage", None, -5.0),
        ),
        (
            ExperimentalRFEnvelopeModel(
                kind="experimental.rf_envelope",
                control="voltage",
            ),
            ExperimentalRFEnvelopeCommand(
                kind="experimental.rf_envelope",
                voltage_rms_V=22.0,
            ),
            True,
            ("voltage", None, 22.0),
        ),
        (
            ExperimentalCCPModel(
                kind="experimental.ccp",
                control="absorbed_power",
            ),
            ExperimentalCCPCommand(
                kind="experimental.ccp",
                absorbed_power_W=9.0,
            ),
            True,
            ("power", 9.0, None),
        ),
        (
            ExperimentalICPModel(kind="experimental.icp"),
            ExperimentalICPCommand(
                kind="experimental.icp",
                delivered_power_W=8.0,
            ),
            True,
            ("power", 8.0, None),
        ),
        (
            ExperimentalICPModel(kind="experimental.icp"),
            ExperimentalICPCommand(
                kind="experimental.icp",
                delivered_power_W=8.0,
            ),
            False,
            ("off", None, None),
        ),
    ],
)
def test_resolved_port_command_preserves_each_model_contract(
    model: PowerModel,
    command: PowerPortCommand,
    enabled: bool,
    expected: tuple[str, float | None, float | None] | None,
) -> None:
    port = PowerPortConfig(port_id="source", zone_id="plasma", model=model)

    result = _resolved_port_command(
        port,
        command,
        enabled=enabled,
        external_tables=ExternalTableStore(),
    )

    if expected is None:
        assert result is None
    else:
        assert result is not None
        assert (result.kind, result.power_W, result.voltage_V) == expected


def test_resolved_port_command_keeps_model_mismatch_error() -> None:
    port = PowerPortConfig(
        port_id="source",
        zone_id="plasma",
        model=PrescribedPowerModel(
            kind="prescribed_power",
            electron_fraction=1.0,
            gas_fraction=0.0,
        ),
    )
    command = ExperimentalICPCommand(
        kind="experimental.icp",
        delivered_power_W=1.0,
    )

    with pytest.raises(
        CaseValidationError,
        match="power port 'source' command does not match prescribed_power",
    ):
        _resolved_port_command(
            port,
            command,
            enabled=True,
            external_tables=ExternalTableStore(),
        )


def test_resolved_external_table_command_keeps_bound_table(tmp_path: Path) -> None:
    table_path = tmp_path / "power.csv"
    table_path.write_text(
        "time_s,electron_power_W\n0.0,1.0\n1.0,2.0\n",
        encoding="utf-8",
    )
    port = PowerPortConfig(
        port_id="source",
        zone_id="plasma",
        model=ExternalTableModel(kind="external_table", file=table_path),
    )

    result = _resolved_port_command(
        port,
        ExternalTableCommand(kind="external_table"),
        enabled=False,
        external_tables=ExternalTableStore(),
    )

    assert result is not None
    assert result.kind == "external_table"
    assert result.external_table is not None
    assert result.time_dependent


def test_compiled_steps_preserve_forcing_temperature_and_command_order() -> None:
    data = load_case(FIXTURE).model_dump(mode="python")
    data["reactor"]["power_ports"][0]["model"]["default_absorbed_power_W"] = 0.0
    data["reactor"]["gas_inlets"] = [
        {
            "inlet_id": "feed",
            "zone_id": "plasma",
            "flow_sccm": {"Ar": 1.0},
            "temperature_K": 350.0,
        }
    ]
    first_step = data["recipe"]["steps"][0]
    first_step["commands"]["gas_inlets"] = {"feed": {"flow_sccm": {"Ar": 4.0}}}
    data["recipe"]["steps"].append(
        {
            "step_id": "idle",
            "duration_s": 2.0e-7,
            "commands": {},
        }
    )

    compiled = compile_case(ResolvedCase.model_validate(data))

    assert [segment.segment_id for segment in compiled.segments] == [
        first_step["step_id"],
        "idle",
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
    data["reactor"]["power_ports"][0]["model"].pop("default_absorbed_power_W", None)
    data["recipe"]["steps"][0]["commands"]["power_ports"] = {}
    with pytest.raises(CaseValidationError, match="requires a power command"):
        compile_case(ResolvedCase.model_validate(data))


@pytest.mark.parametrize("interpolation", ["linear", "previous"])
def test_compile_rejects_external_table_without_full_recipe_coverage(
    tmp_path: Path, interpolation: str
) -> None:
    path = tmp_path / "partial.csv"
    path.write_text("time_s,electron_power_W\n0,1\n1e-8,2\n", encoding="utf-8")
    data = load_case(FIXTURE).model_dump(mode="python")
    data["reactor"]["power_ports"][0]["model"] = {
        "kind": "external_table",
        "file": path,
        "interpolation": interpolation,
    }
    data["recipe"]["steps"][0]["commands"]["power_ports"] = {}
    with pytest.raises(CaseValidationError, match="does not cover recipe interval"):
        compile_case(ResolvedCase.model_validate(data))


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (
            {
                "kind": "prescribed_power",
                "electron_fraction": 1.0,
                "gas_fraction": 0.0,
                "default_absorbed_power_W": 2.0,
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
            {"kind": "experimental.rf_envelope", "default_absorbed_power_W": 2.0},
            ("power", 2.0, None),
        ),
        (
            {
                "kind": "experimental.ccp",
                "control": "voltage",
                "default_voltage_rms_V": 2.0,
            },
            ("voltage", None, 2.0),
        ),
        (
            {"kind": "experimental.icp", "default_delivered_power_W": 2.0},
            ("power", 2.0, None),
        ),
    ],
)
def test_omitted_recipe_commands_resolve_to_the_model_default(
    model: dict, expected: tuple
) -> None:
    data = load_case(FIXTURE).model_dump(mode="python")
    data["reactor"]["power_ports"][0]["model"] = model
    data["recipe"]["steps"][0]["commands"]["power_ports"] = {}
    compiled = compile_case(ResolvedCase.model_validate(data))
    command = compiled.segments[0].port_commands["source"]
    assert (command.kind, command.power_W, command.voltage_V) == expected
