"""Pure calculations used by the public power-port classes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from plasma_global.errors import (
    CaseValidationError,
    CouplingConvergenceError,
    ModelDomainError,
)
from plasma_global.models.external_table import (
    ExternalTableBinding,
    ExternalTableSample,
)
from plasma_global.models.kinetics import TabulatedElectronKinetics

E_CHARGE = 1.602176634e-19
TD_TO_V_M2 = 1.0e-21

_CommandKind = Literal["off", "power", "voltage", "external_table"]


@dataclass(frozen=True, slots=True)
class _PortEvaluation:
    electron_power_W: float
    gas_power_W: float
    reduced_field_Td: float | None
    observables: Mapping[str, float]


def validate_prescribed_power_port(
    electron_fraction: float,
    gas_fraction: float,
    default_power_W: float | None,
) -> None:
    fractions = electron_fraction, gas_fraction
    if not np.isfinite(fractions).all() or min(fractions) < 0.0:
        raise CaseValidationError("power fractions must be finite and nonnegative")
    if abs(sum(fractions) - 1.0) > 1.0e-12:
        raise CaseValidationError("electron_fraction + gas_fraction must equal one")
    if default_power_W is not None and (
        not math.isfinite(default_power_W) or default_power_W < 0.0
    ):
        raise CaseValidationError("default_power_W must be finite and nonnegative")


def prescribed_power_evaluation(
    *,
    port_id: str,
    electron_fraction: float,
    gas_fraction: float,
    default_power_W: float | None,
    command_kind: _CommandKind | None,
    command_power_W: float | None,
) -> _PortEvaluation:
    if command_kind == "power" and command_power_W is not None:
        power_W = command_power_W
    elif command_kind is None and default_power_W is not None:
        power_W = default_power_W
    else:
        raise CaseValidationError(
            f"prescribed-power port {port_id!r} requires a power command"
        )
    return _PortEvaluation(
        electron_power_W=power_W * electron_fraction,
        gas_power_W=power_W * gas_fraction,
        reduced_field_Td=None,
        observables={"commanded_power_W": power_W},
    )


def _validate_optional_finite_parameter(
    name: str, value: float | None, *, positive: bool = False
) -> None:
    if value is None:
        return
    if not math.isfinite(value) or (positive and value <= 0.0):
        requirement = "finite and positive" if positive else "finite"
        raise CaseValidationError(f"{name} must be {requirement}")


def validate_dc_series_port(
    *,
    ballast_resistance_ohm: float,
    gap_m: float,
    electrode_area_m2: float,
    absorption_fraction: float,
    configured_mobility_m2_V_s: float | None,
    default_voltage_V: float | None,
) -> None:
    positive = {
        "ballast_resistance_ohm": ballast_resistance_ohm,
        "gap_m": gap_m,
        "electrode_area_m2": electrode_area_m2,
    }
    invalid = [
        name
        for name, value in positive.items()
        if not math.isfinite(value) or value <= 0.0
    ]
    if invalid:
        raise CaseValidationError(
            f"DC series parameters must be positive: {', '.join(invalid)}"
        )
    if not math.isfinite(absorption_fraction) or not (
        0.0 <= absorption_fraction <= 1.0
    ):
        raise CaseValidationError("absorption_fraction must be between zero and one")
    _validate_optional_finite_parameter(
        "configured mobility", configured_mobility_m2_V_s, positive=True
    )
    _validate_optional_finite_parameter("default voltage", default_voltage_V)


def _source_voltage(
    *,
    port_id: str,
    default_voltage_V: float | None,
    command_kind: _CommandKind | None,
    command_voltage_V: float | None,
) -> float:
    if command_kind == "voltage" and command_voltage_V is not None:
        return command_voltage_V
    if command_kind is None and default_voltage_V is not None:
        return default_voltage_V
    raise CaseValidationError(f"DC-series port {port_id!r} requires a voltage command")


def dc_series_evaluation(
    *,
    port_id: str,
    ballast_resistance_ohm: float,
    gap_m: float,
    electrode_area_m2: float,
    absorption_fraction: float,
    configured_mobility_m2_V_s: float | None,
    default_voltage_V: float | None,
    electron_density_m3: float,
    neutral_density_m3: float,
    electron_mobility_m2_V_s: float | None,
    command_kind: _CommandKind | None,
    command_voltage_V: float | None,
) -> _PortEvaluation:
    source_voltage = _source_voltage(
        port_id=port_id,
        default_voltage_V=default_voltage_V,
        command_kind=command_kind,
        command_voltage_V=command_voltage_V,
    )
    mobility = electron_mobility_m2_V_s or configured_mobility_m2_V_s
    if mobility is None:
        raise ModelDomainError(f"DC-series port {port_id!r} requires electron mobility")
    conductivity = E_CHARGE * electron_density_m3 * mobility
    if conductivity <= 0.0:
        plasma_resistance = math.inf
        current = 0.0
        plasma_voltage = source_voltage
        absorbed = 0.0
    else:
        plasma_resistance = gap_m / (conductivity * electrode_area_m2)
        current = source_voltage / (ballast_resistance_ohm + plasma_resistance)
        plasma_voltage = current * plasma_resistance
        absorbed = current * current * plasma_resistance * absorption_fraction
    field = abs(plasma_voltage) / gap_m
    reduced_field = (
        field / neutral_density_m3 / TD_TO_V_M2 if neutral_density_m3 > 0.0 else 0.0
    )
    return _PortEvaluation(
        electron_power_W=absorbed,
        gas_power_W=0.0,
        reduced_field_Td=reduced_field,
        observables={
            "source_voltage_V": source_voltage,
            "plasma_voltage_V": plasma_voltage,
            "current_A": current,
            "plasma_resistance_ohm": plasma_resistance,
        },
    )


def solve_dc_series_local_field(
    *,
    port_id: str,
    ballast_resistance_ohm: float,
    gap_m: float,
    electrode_area_m2: float,
    default_voltage_V: float | None,
    electron_density_m3: float,
    neutral_density_m3: float,
    kinetics: TabulatedElectronKinetics,
    command_kind: _CommandKind | None,
    command_voltage_V: float | None,
) -> float:
    """Solve the series-circuit E/N equation on a piecewise-linear table."""

    axis = np.asarray(kinetics.axis, dtype=float)
    mobility = np.asarray(kinetics.mobility_m2_V_s, dtype=float)
    if mobility.shape != axis.shape:
        raise CaseValidationError(
            "local-field kinetics mobility must share the E/N axis"
        )
    if neutral_density_m3 <= 0.0:
        raise ModelDomainError(
            f"DC-series port {port_id!r} requires positive neutral density"
        )
    source_voltage = _source_voltage(
        port_id=port_id,
        default_voltage_V=default_voltage_V,
        command_kind=command_kind,
        command_voltage_V=command_voltage_V,
    )
    vacuum_field_Td = abs(source_voltage) / gap_m / neutral_density_m3 / TD_TO_V_M2
    conductance_factor = (
        ballast_resistance_ohm
        * E_CHARGE
        * max(electron_density_m3, 0.0)
        * electrode_area_m2
        / gap_m
    )
    slopes = np.diff(mobility) / np.diff(axis)
    intercepts = mobility[:-1] - slopes * axis[:-1]
    quadratic = conductance_factor * slopes
    linear = 1.0 + conductance_factor * intercepts
    constant = -vacuum_field_Td

    machine_epsilon = np.finfo(float).eps
    linear_intervals = np.abs(quadratic) <= machine_epsilon * np.maximum(
        np.abs(linear), 1.0
    )
    discriminant = linear * linear - 4.0 * quadratic * constant
    real_quadratic = ~linear_intervals & (discriminant >= 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        root_scale = np.sqrt(np.maximum(discriminant, 0.0))
        root_sets = (
            np.where(linear_intervals, -constant / linear, np.nan),
            np.where(
                real_quadratic,
                (-linear - root_scale) / (2.0 * quadratic),
                np.nan,
            ),
            np.where(
                real_quadratic,
                (-linear + root_scale) / (2.0 * quadratic),
                np.nan,
            ),
        )
    left = axis[:-1]
    right = axis[1:]
    interval_tolerance = (
        32.0
        * machine_epsilon
        * np.maximum.reduce((np.abs(left), np.abs(right), np.ones_like(left)))
    )
    candidates: list[float] = []
    for roots in root_sets:
        in_interval = (
            np.isfinite(roots)
            & (roots >= left - interval_tolerance)
            & (roots <= right + interval_tolerance)
        )
        clipped = np.clip(roots, left, right)
        local_mobility = slopes * clipped + intercepts
        derivative = np.abs(
            -vacuum_field_Td
            * conductance_factor
            * slopes
            / (1.0 + conductance_factor * local_mobility) ** 2
        )
        candidates.extend(
            float(value) for value in clipped[in_interval & (derivative < 1.0)]
        )

    if not candidates:
        raise CouplingConvergenceError(
            f"DC-series port {port_id!r} has no stable E/N root in "
            f"[{axis[0]:g}, {axis[-1]:g}] Td"
        )

    selector = float(axis[len(axis) // 2])
    for _ in range(3):
        selector_mobility = float(np.interp(selector, axis, mobility))
        selector = vacuum_field_Td / (1.0 + conductance_factor * selector_mobility)
        selector = float(np.clip(selector, axis[0], axis[-1]))

    def distance_from_selector(value: float) -> float:
        return abs(value - selector)

    return min(candidates, key=distance_from_selector)


def _sample_external_table(
    table: ExternalTableBinding | ExternalTableSample,
    time_s: float,
) -> ExternalTableSample:
    if isinstance(table, ExternalTableSample):
        return table
    data = table.data
    return ExternalTableSample(
        electron_power_W=table.power_scale * table._at(data.electron_power_W, time_s),
        gas_power_W=table.power_scale * table._at(data.gas_power_W, time_s),
        voltage_V=(
            None
            if data.voltage_V is None
            else table.voltage_scale * table._at(data.voltage_V, time_s)
        ),
        current_A=(
            None
            if data.current_A is None
            else table.current_scale * table._at(data.current_A, time_s)
        ),
        reduced_field_Td=(
            None
            if data.reduced_field_Td is None
            else table._at(data.reduced_field_Td, time_s)
        ),
        gap_m=table.gap_m,
        total_density_m3=table.total_density_m3,
        plasma_potential_V=table.plasma_potential_V,
    )


def external_table_evaluation(
    *,
    table: ExternalTableBinding | ExternalTableSample,
    time_s: float,
    neutral_density_m3: float,
) -> _PortEvaluation:
    sample = _sample_external_table(table, time_s)
    reduced_field = sample.reduced_field_Td
    if (
        reduced_field is None
        and sample.voltage_V is not None
        and sample.gap_m is not None
    ):
        density = (
            neutral_density_m3
            if sample.total_density_m3 is None
            else sample.total_density_m3
        )
        reduced_field = (
            abs(sample.voltage_V) / sample.gap_m / density / TD_TO_V_M2
            if density > 0.0
            else 0.0
        )
    observables = {
        "absorbed_power_W": sample.electron_power_W + sample.gas_power_W,
        "plasma_potential_V": sample.plasma_potential_V,
    }
    if sample.voltage_V is not None:
        observables["source_voltage_V"] = sample.voltage_V
    if sample.current_A is not None:
        observables["current_A"] = sample.current_A
    return _PortEvaluation(
        electron_power_W=sample.electron_power_W,
        gas_power_W=sample.gas_power_W,
        reduced_field_Td=reduced_field,
        observables=observables,
    )
