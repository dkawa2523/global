"""Validation and numerical kernels for experimental RF power ports.

The public dataclasses live in :mod:`plasma_global.experimental.power`.  This
module owns their implementation so configuration, numerical iteration, and
result assembly can evolve independently without changing public type identity.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

from plasma_global.errors import ModelDomainError
from plasma_global.models.power import (
    CompiledPowerCommand,
    PowerPortResult,
    PowerState,
)

if TYPE_CHECKING:
    from plasma_global.experimental.power import (
        ICPPowerPort,
        RFEnvelopePort,
        SquarePulse,
    )

ELEMENTARY_CHARGE_C = 1.602176634e-19
BOLTZMANN_J_K = 1.380649e-23
VACUUM_PERMITTIVITY_F_M = 8.8541878128e-12
TD_TO_V_M2 = 1.0e-21
_ICP_FIELD_DENSITY_SCALE_M3 = 1.0e10


def _positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _fraction(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between zero and one")


def _validate_port_ids(port_id: str, zone_id: str) -> None:
    if not port_id or not zone_id:
        raise ValueError("port_id and zone_id must not be empty")


def _validate_default_commands(
    default_power_W: float | None,
    default_voltage_V: float | None,
    *,
    model_name: str,
) -> None:
    if default_power_W is not None and (
        not math.isfinite(default_power_W) or default_power_W < 0.0
    ):
        raise ValueError("default_power_W must be finite and nonnegative")
    if default_voltage_V is not None and not math.isfinite(default_voltage_V):
        raise ValueError("default_voltage_V must be finite")
    if default_power_W is not None and default_voltage_V is not None:
        raise ValueError(f"provide only one default {model_name} command")


def validate_square_pulse(pulse: SquarePulse) -> None:
    _positive("repetition_Hz", pulse.repetition_Hz)
    _fraction("duty_cycle", pulse.duty_cycle)
    if not math.isfinite(pulse.phase_fraction):
        raise ValueError("phase_fraction must be finite")


def _envelope_value(pulse: SquarePulse | None, time_s: float) -> float:
    return 1.0 if pulse is None else pulse(time_s)


def _delivered_power(absorbed_power_W: float, coupling_efficiency: float) -> float:
    """Recover source-side power from an explicitly absorbed-power command."""

    if absorbed_power_W == 0.0:
        return 0.0
    if coupling_efficiency == 0.0:
        raise ModelDomainError(
            "positive absorbed power requires a nonzero coupling_efficiency"
        )
    return absorbed_power_W / coupling_efficiency


def _command_value(
    command: CompiledPowerCommand | None,
    *,
    default_power_W: float | None,
    default_voltage_V: float | None,
) -> tuple[Literal["power", "voltage"], float]:
    if command is not None and command.kind == "power":
        if command.power_W is None:
            raise TypeError("compiled power command has no power value")
        return "power", command.power_W
    if command is not None and command.kind == "voltage":
        if command.voltage_V is None:
            raise TypeError("compiled voltage command has no voltage value")
        return "voltage", abs(command.voltage_V)
    if command is not None:
        raise TypeError("power ports accept compiled power or voltage commands")
    if default_power_W is not None:
        return "power", default_power_W
    if default_voltage_V is not None:
        return "voltage", abs(default_voltage_V)
    raise ValueError("the power port requires a recipe command or a default command")


def validate_rf_envelope_port(port: RFEnvelopePort) -> None:
    _validate_port_ids(port.port_id, port.zone_id)
    _positive("frequency_Hz", port.frequency_Hz)
    _positive("effective_impedance_ohm", port.effective_impedance_ohm)
    _fraction("coupling_efficiency", port.coupling_efficiency)
    if port.role not in {"source", "bias"}:
        raise ValueError("role must be 'source' or 'bias'")
    _validate_rf_coefficients(port)
    _validate_default_commands(
        port.default_power_W, port.default_voltage_V, model_name="RF"
    )


def _validate_rf_coefficients(port: RFEnvelopePort) -> None:
    nonnegative = {
        "base_reduced_field_Td": port.base_reduced_field_Td,
        "reduced_field_per_sqrt_W_Td": port.reduced_field_per_sqrt_W_Td,
        "self_bias_fraction": port.self_bias_fraction,
        "plasma_potential_offset_V": port.plasma_potential_offset_V,
        "plasma_potential_per_sqrt_W": port.plasma_potential_per_sqrt_W,
    }
    if any(not math.isfinite(value) or value < 0.0 for value in nonnegative.values()):
        raise ValueError(
            "RF envelope field and potential coefficients must be nonnegative"
        )


def evaluate_rf_envelope_port(
    port: RFEnvelopePort,
    time_s: float,
    command: CompiledPowerCommand | None,
) -> PowerPortResult:
    mode, value = _command_value(
        command,
        default_power_W=port.default_power_W,
        default_voltage_V=port.default_voltage_V,
    )
    envelope = _envelope_value(port.pulse, time_s)
    if mode == "power":
        absorbed = value * envelope
        delivered = _delivered_power(absorbed, port.coupling_efficiency)
        voltage_rms = math.sqrt(delivered * port.effective_impedance_ohm)
    else:
        voltage_rms = value * envelope
        delivered = voltage_rms**2 / port.effective_impedance_ohm
        absorbed = port.coupling_efficiency * delivered
    reduced_field = port.base_reduced_field_Td * envelope
    reduced_field += port.reduced_field_per_sqrt_W_Td * math.sqrt(absorbed)
    self_bias = (
        -port.self_bias_fraction * math.sqrt(2.0) * voltage_rms
        if port.role == "bias"
        else 0.0
    )
    plasma_potential = port.plasma_potential_offset_V * envelope
    plasma_potential += port.plasma_potential_per_sqrt_W * math.sqrt(absorbed)
    return PowerPortResult(
        port_id=port.port_id,
        zone_id=port.zone_id,
        electron_power_W=absorbed,
        reduced_field_Td=reduced_field,
        observables={
            "frequency_Hz": port.frequency_Hz,
            "delivered_power_W": delivered,
            "absorbed_power_W": absorbed,
            "voltage_rms_V": voltage_rms,
            "self_bias_V": self_bias,
            "plasma_potential_V": plasma_potential,
        },
    )


def validate_icp_power_port(port: ICPPowerPort) -> None:
    _validate_port_ids(port.port_id, port.zone_id)
    _positive("frequency_Hz", port.frequency_Hz)
    _positive("gas_temperature_K", port.gas_temperature_K)
    _fraction("downstream_fraction", port.downstream_fraction)
    if port.downstream_fraction > 0.0 and not port.downstream_zone_id:
        raise ValueError(
            "downstream_zone_id is required when power is split downstream"
        )
    _validate_default_commands(port.default_power_W, None, model_name="ICP")


def _icp_coupling(
    delivered_power_W: float,
    electron_density_m3: float,
    pressure_factor: float,
    frequency_factor: float,
) -> tuple[float, float, float]:
    """Return continuous coupling, mode index, and H-mode fraction."""

    density_factor = -math.expm1(-2.2 * electron_density_m3 / 1.0e17)
    mode_index = density_factor * math.sqrt(delivered_power_W / 150.0)
    transition = min(max((mode_index - 0.45) / 0.20, 0.0), 1.0)
    h_fraction = transition * transition * (3.0 - 2.0 * transition)
    plasma_factor = density_factor * pressure_factor * frequency_factor
    e_coupling = 0.18 + (0.55 - 0.18) * plasma_factor
    h_coupling = 0.40 + (0.88 - 0.40) * plasma_factor
    coupling = density_factor * (e_coupling + h_fraction * (h_coupling - e_coupling))
    return min(max(coupling, 0.0), 0.95), mode_index, h_fraction


def _icp_power_activity(delivered_power_W: float, electron_density_m3: float) -> float:
    """Smoothly suppress plasma-only observables at zero power or density."""

    density_activity = -math.expm1(-electron_density_m3 / _ICP_FIELD_DENSITY_SCALE_M3)
    return density_activity * -math.expm1(-delivered_power_W / 150.0)


def evaluate_icp_power_port(
    port: ICPPowerPort,
    time_s: float,
    state: PowerState,
    command: CompiledPowerCommand | None,
) -> PowerPortResult:
    mode, value = _command_value(
        command, default_power_W=port.default_power_W, default_voltage_V=None
    )
    if mode != "power":
        raise TypeError("ICP ports accept power commands only")
    delivered = value * _envelope_value(port.pulse, time_s)
    pressure_Pa = state.neutral_density_m3 * BOLTZMANN_J_K * port.gas_temperature_K
    pressure_scale = max(pressure_Pa / 10.0, 1.0e-3)
    frequency_scale = max(port.frequency_Hz / 13.56e6, 0.1)
    pressure_factor = 0.65 + 0.20 * math.tanh((pressure_scale - 0.3) / 0.6)
    frequency_factor = 0.85 + 0.08 * math.log10(1.0 + frequency_scale)
    coupling, mode_index, mode_H_fraction = _icp_coupling(
        delivered,
        state.electron_density_m3,
        pressure_factor,
        frequency_factor,
    )
    absorbed = coupling * delivered
    downstream_power = absorbed * port.downstream_fraction
    local_power = absorbed - downstream_power
    plasma_resistance = 2.5 + 12.0 / max(
        math.sqrt(state.electron_density_m3 / 1.0e16), 0.2
    )
    coil_current = math.sqrt(absorbed / plasma_resistance)
    coil_voltage = coil_current * plasma_resistance
    activity = _icp_power_activity(delivered, state.electron_density_m3)
    reduced_field = activity * (
        25.0
        + 30.0 * math.sqrt(max(state.electron_temperature_eV, 0.0))
        + 0.01 * absorbed
    )
    return PowerPortResult(
        port_id=port.port_id,
        zone_id=port.zone_id,
        electron_power_W=local_power,
        reduced_field_Td=reduced_field,
        observables={
            "frequency_Hz": port.frequency_Hz,
            "delivered_power_W": delivered,
            "absorbed_power_W": absorbed,
            "downstream_power_W": downstream_power,
            "reflected_power_W": delivered - absorbed,
            "coupling_efficiency": coupling,
            "plasma_resistance_ohm": plasma_resistance,
            "coil_voltage_rms_V": coil_voltage,
            "coil_current_rms_A": coil_current,
            "mode_index": mode_index,
            "mode_H": mode_H_fraction,
            "plasma_potential_V": activity * (5.0 + 0.015 * absorbed + 2.0 * coupling),
            "pressure_Pa": pressure_Pa,
        },
    )


__all__ = [
    "BOLTZMANN_J_K",
    "ELEMENTARY_CHARGE_C",
    "TD_TO_V_M2",
    "VACUUM_PERMITTIVITY_F_M",
    "evaluate_icp_power_port",
    "evaluate_rf_envelope_port",
    "validate_icp_power_port",
    "validate_rf_envelope_port",
    "validate_square_pulse",
]
