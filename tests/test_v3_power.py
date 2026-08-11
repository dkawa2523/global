from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plasma_global.errors import CaseValidationError, ModelDomainError
from plasma_global.models.external_table import (
    ExternalTableBinding,
    load_external_table,
)
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
    assert result.reduced_field_Td is not None and result.reduced_field_Td > 0.0


def test_dc_series_solves_local_field_on_prepared_mobility_axis() -> None:
    class Kinetics:
        axis = np.array([1.0, 100.0, 300.0])
        mobility_m2_V_s = np.array([1.0, 0.5, 0.2])

    port = DCSeriesPort(
        "dc",
        "plasma",
        ballast_resistance_ohm=1.0e5,
        gap_m=4.0e-3,
        electrode_area_m2=5.0e-5,
    )
    electron_density = 1.0e17
    neutral_density = 1.0e24
    field = port.solve_local_field(
        electron_density_m3=electron_density,
        neutral_density_m3=neutral_density,
        kinetics=Kinetics(),
        command=CompiledPowerCommand(kind="voltage", voltage_V=1_000.0),
    )
    mobility = float(np.interp(field, Kinetics.axis, Kinetics.mobility_m2_V_s))
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
