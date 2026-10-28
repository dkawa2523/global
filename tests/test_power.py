from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from plasma_global.errors import CaseValidationError, ModelDomainError
from plasma_global.models.external_table import (
    ExternalTableBinding,
    load_external_table,
)
from plasma_global.models.kinetics import TabulatedElectronKinetics
from plasma_global.models.power import (
    CompiledPowerCommand,
    DCSeriesPort,
    ExternalTablePowerPort,
    PowerCoordinator,
    PowerState,
    PrescribedPowerPort,
)

STATE = PowerState(
    electron_density_m3=1.0e16,
    neutral_density_m3=2.0e20,
    electron_temperature_eV=3.0,
    electron_mobility_m2_V_s=0.5,
)


def test_compiled_power_command_rejects_ambiguous_payloads() -> None:
    with pytest.raises(CaseValidationError, match="one finite nonnegative power_W"):
        CompiledPowerCommand(kind="power", power_W=1.0, voltage_V=2.0)
    with pytest.raises(CaseValidationError, match="must not carry a value"):
        CompiledPowerCommand(kind="off", power_W=0.0)


def test_rejects_multiple_reduced_field_ports_in_one_zone() -> None:
    ports = tuple(
        DCSeriesPort(
            f"dc_{index}",
            "plasma",
            ballast_resistance_ohm=1.0e6,
            gap_m=0.1,
            electrode_area_m2=0.01,
            configured_mobility_m2_V_s=0.5,
            default_voltage_V=10.0,
        )
        for index in range(2)
    )

    with pytest.raises(CaseValidationError, match="multiple E/N-producing"):
        PowerCoordinator(ports, ("plasma",))


def _external_port(path: Path, *, port_id: str, zone_id: str) -> ExternalTablePowerPort:
    return ExternalTablePowerPort(
        port_id,
        zone_id,
        ExternalTableBinding(
            data=load_external_table(path),
            interpolation="linear",
            bounds="error",
            power_scale=1.0,
            voltage_scale=1.0,
            current_scale=1.0,
            gap_m=None,
            total_density_m3=None,
            plasma_potential_V=0.0,
        ),
    )


def test_prescribed_power_only_partitions_power() -> None:
    port = PrescribedPowerPort(
        "source", "plasma", electron_fraction=0.8, gas_fraction=0.2
    )
    result = port.evaluate(
        0.0, STATE, CompiledPowerCommand(kind="power", power_W=100.0)
    )
    assert result.electron_power_W == 80.0
    assert result.gas_power_W == 20.0
    assert result.reduced_field_Td is None
    assert "self_bias_V" not in result.observables


def test_prescribed_power_rejects_an_unaccounted_partition() -> None:
    with pytest.raises(CaseValidationError, match="must equal one"):
        PrescribedPowerPort(
            "source",
            "plasma",
            electron_fraction=0.2,
            gas_fraction=0.2,
        )


def test_prescribed_power_coordinator_does_not_require_mobility() -> None:
    port = PrescribedPowerPort(
        "source", "plasma", electron_fraction=1.0, gas_fraction=0.0
    )
    result = PowerCoordinator((port,), ("plasma",)).evaluate(
        time_s=0.0,
        commands={"source": CompiledPowerCommand(kind="power", power_W=25.0)},
        electron_density_m3_by_zone={"plasma": 1.0e15},
        neutral_density_m3_by_zone={"plasma": 1.0e20},
        mean_energy_eV_by_zone={"plasma": 3.0},
    )

    assert result.electron_power_W_by_zone["plasma"] == 25.0
    assert result.iterations_by_zone["plasma"] == 1


def test_coordinator_keeps_zone_aggregation_and_off_commands_independent() -> None:
    coordinator = PowerCoordinator(
        (
            PrescribedPowerPort(
                "source", "source_zone", electron_fraction=0.75, gas_fraction=0.25
            ),
            PrescribedPowerPort("bias", "process_zone", electron_fraction=1.0),
        ),
        ("source_zone", "process_zone"),
    )

    result = coordinator.evaluate(
        time_s=0.0,
        commands={
            "source": CompiledPowerCommand(kind="power", power_W=40.0),
            "bias": CompiledPowerCommand(kind="off"),
        },
        electron_density_m3_by_zone={"source_zone": 1.0e15, "process_zone": 2.0e15},
        neutral_density_m3_by_zone={"source_zone": 1.0e20, "process_zone": 2.0e20},
        mean_energy_eV_by_zone={"source_zone": 3.0, "process_zone": 6.0},
        prescribed_electron_power_W_by_zone={
            "source_zone": 2.0,
            "process_zone": 3.0,
        },
        prescribed_reduced_field_Td_by_zone={"process_zone": 12.0},
    )

    assert dict(result.electron_power_W_by_zone) == {
        "source_zone": 32.0,
        "process_zone": 3.0,
    }
    assert dict(result.gas_power_W_by_zone) == {
        "source_zone": 10.0,
        "process_zone": 0.0,
    }
    assert dict(result.reduced_field_Td_by_zone) == {"process_zone": 12.0}
    assert dict(result.iterations_by_zone) == {"source_zone": 1, "process_zone": 1}
    assert result.power_state_by_zone["source_zone"].electron_temperature_eV == 2.0
    assert result.power_state_by_zone["process_zone"].electron_temperature_eV == 4.0
    assert result.port_results["bias"].electron_power_W == 0.0


def test_dc_series_obeys_voltage_and_power_balance() -> None:
    port = DCSeriesPort(
        "dc",
        "plasma",
        ballast_resistance_ohm=1.0e5,
        gap_m=4.0e-3,
        electrode_area_m2=5.0e-5,
    )
    command = CompiledPowerCommand(kind="voltage", voltage_V=1_000.0)
    result = port.evaluate(0.0, STATE, command)
    current = result.observables["current_A"]
    voltage = result.observables["plasma_voltage_V"]
    assert result.electron_power_W == pytest.approx(current * voltage)
    assert result.reduced_field_Td is not None
    assert result.reduced_field_Td > 0.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"configured_mobility_m2_V_s": float("nan")},
        {"configured_mobility_m2_V_s": float("inf")},
        {"default_voltage_V": float("nan")},
        {"default_voltage_V": float("inf")},
    ],
)
def test_dc_series_rejects_nonfinite_optional_parameters(
    overrides: dict[str, float],
) -> None:
    with pytest.raises(CaseValidationError, match="finite"):
        DCSeriesPort(
            "dc",
            "plasma",
            ballast_resistance_ohm=1.0e5,
            gap_m=4.0e-3,
            electrode_area_m2=5.0e-5,
            **overrides,
        )


@pytest.mark.parametrize("neutral_density", [1.0e24, 2.0e24])
def test_dc_series_solves_local_field_on_prepared_mobility_axis(
    neutral_density: float,
) -> None:
    port = DCSeriesPort(
        "dc",
        "plasma",
        ballast_resistance_ohm=1.0e5,
        gap_m=4.0e-3,
        electrode_area_m2=5.0e-5,
    )
    electron_density = 1.0e17
    kinetics = TabulatedElectronKinetics(
        source=Path("prepared.csv"),
        lookup="local_field",
        bounds="error",
        axis=np.array([1.0, 100.0, 300.0]),
        mean_energy_eV=np.array([1.0, 2.0, 3.0]),
        mobility_m2_V_s=np.array([1.0, 0.5, 0.2]),
        effective_field_Td=np.array([1.0, 100.0, 300.0]),
        rate_tables={},
        mobility_reference_neutral_density_m3=1.0e24,
    )
    field = port.solve_local_field(
        electron_density_m3=electron_density,
        neutral_density_m3=neutral_density,
        kinetics=kinetics,
        command=CompiledPowerCommand(kind="voltage", voltage_V=1_000.0),
    )
    mobility = kinetics.evaluate(
        reduced_field_Td=field,
        neutral_density_m3=neutral_density,
    ).mobility_m2_V_s
    result = port.evaluate(
        0.0,
        PowerState(
            electron_density_m3=electron_density,
            neutral_density_m3=neutral_density,
            electron_temperature_eV=3.0,
            electron_mobility_m2_V_s=mobility,
        ),
        CompiledPowerCommand(kind="voltage", voltage_V=1_000.0),
    )

    assert result.reduced_field_Td == pytest.approx(field, rel=1.0e-12)


def test_external_table_is_strict_and_interpolated(tmp_path: Path) -> None:
    path = tmp_path / "power.csv"
    path.write_text(
        "time_s,electron_power_W,gas_power_W,reduced_field_Td\n0,0,0,1\n1,10,2,3\n",
        encoding="utf-8",
    )
    port = _external_port(path, port_id="table", zone_id="plasma")
    result = port.evaluate(0.5, STATE, None)
    assert result.electron_power_W == 5.0
    assert result.gas_power_W == 1.0
    assert result.reduced_field_Td == 2.0
    with pytest.raises(ModelDomainError, match="outside"):
        port.evaluate(2.0, STATE, None)


@pytest.mark.parametrize(
    ("interpolation", "expected_total"), [("linear", 90.0), ("previous", 40.0)]
)
def test_external_vi_power_uses_scaled_interpolated_physical_inputs(
    tmp_path: Path, interpolation: str, expected_total: float
) -> None:
    path = tmp_path / "vi.csv"
    path.write_text(
        "time_s,voltage_V,current_A,gas_power_W\n0,2,2,1\n1,4,4,1\n", encoding="utf-8"
    )
    binding = replace(
        _external_port(path, port_id="table", zone_id="plasma").default,
        interpolation=interpolation,
        voltage_scale=2.0,
        current_scale=5.0,
        power_scale=3.0,
    )
    table = binding.bind_previous(0.5) if interpolation == "previous" else binding
    port = ExternalTablePowerPort("table", "plasma", binding)
    evaluated = port.evaluate(
        0.5, STATE, CompiledPowerCommand(kind="external_table", external_table=table)
    )
    assert evaluated.electron_power_W + evaluated.gas_power_W == pytest.approx(
        3.0 * expected_total
    )
    assert evaluated.gas_power_W == 3.0


def test_external_absorbed_power_is_total_including_gas_partition(
    tmp_path: Path,
) -> None:
    path = tmp_path / "total.csv"
    path.write_text(
        "time_s,absorbed_power_W,gas_power_W\n0,10,3\n1,20,5\n", encoding="utf-8"
    )
    evaluated = _external_port(path, port_id="table", zone_id="plasma").evaluate(
        0.5, STATE, None
    )
    assert evaluated.electron_power_W == 11.0
    assert evaluated.gas_power_W == 4.0


def test_external_binding_rejects_scaling_a_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "power.csv"
    path.write_text("time_s,electron_power_W\n0,1\n1,1\n", encoding="utf-8")
    port = _external_port(path, port_id="table", zone_id="plasma")
    with pytest.raises(CaseValidationError, match="corresponding CSV column"):
        replace(port.default, voltage_scale=2.0)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"interpolation": "cubic"}, "interpolation must"),
        ({"bounds": "clip"}, "bounds must"),
        ({"power_scale": np.nan}, "scales must be finite"),
        ({"time_offset_s": np.inf}, "offset and potential"),
        ({"gap_m": -1.0}, "finite and positive"),
        ({"gap_m": 0.1}, "gap_m requires voltage"),
        ({"total_density_m3": 1e20}, "total_density_m3 requires gap"),
    ],
)
def test_external_binding_rejects_unusable_controls(
    tmp_path: Path, update: dict, message: str
) -> None:
    path = tmp_path / "power.csv"
    path.write_text("time_s,electron_power_W\n0,1\n1,1\n", encoding="utf-8")
    port = _external_port(path, port_id="table", zone_id="plasma")
    with pytest.raises(CaseValidationError, match=message):
        replace(port.default, **update)


def test_dc_field_does_not_report_zero_for_an_undefined_density() -> None:
    port = DCSeriesPort(
        "dc",
        "plasma",
        ballast_resistance_ohm=1.0,
        gap_m=0.1,
        electrode_area_m2=0.01,
        configured_mobility_m2_V_s=0.5,
        default_voltage_V=10.0,
    )
    with pytest.raises(ModelDomainError, match="positive neutral density"):
        port.evaluate(0.0, replace(STATE, neutral_density_m3=0.0), None)


def test_vi_forcing_rejects_negative_power_between_valid_knots(tmp_path: Path) -> None:
    path = tmp_path / "vi.csv"
    path.write_text("time_s,voltage_V,current_A\n0,-2,-3\n1,4,5\n", encoding="utf-8")
    with pytest.raises(CaseValidationError, match="electron power cannot be negative"):
        _external_port(path, port_id="table", zone_id="plasma")


def test_external_table_rejects_unknown_columns(tmp_path: Path) -> None:
    path = tmp_path / "power.csv"
    path.write_text(
        "time_s,electron_power_W,power_source\n0,0,x\n1,1,x\n", encoding="utf-8"
    )
    with pytest.raises(CaseValidationError, match="unknown columns"):
        _external_port(path, port_id="table", zone_id="plasma")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (
            "time_s,electron_power_W,electron_power_W\n0,1,1\n1,2,2\n",
            "duplicate columns",
        ),
        (
            "time_s,electron_power_W\n0,1,unexpected\n1,2,unexpected\n",
            "extra CSV fields",
        ),
        ("time_s,electron_power_W\n0,\n1,2\n", "missing values"),
        ("time_s,electron_power_W\n0\n1,2\n", "missing values"),
    ],
)
def test_external_table_rejects_ambiguous_csv_shapes(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "invalid-power.csv"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(CaseValidationError, match=message):
        load_external_table(path)
