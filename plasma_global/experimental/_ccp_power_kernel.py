"""Equivalent-circuit kernel for experimental CCP bulk power and diagnostics."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from plasma_global.errors import ModelDomainError
from plasma_global.experimental._power_kernels import (
    BOLTZMANN_J_K,
    ELEMENTARY_CHARGE_C,
    TD_TO_V_M2,
    _command_value,
    _positive,
    _validate_constant_commands,
    _validate_port_ids,
)
from plasma_global.models.power import (
    CompiledPowerCommand,
    PowerContribution,
    PowerPortResult,
    PowerState,
)

if TYPE_CHECKING:
    from plasma_global.experimental.power import CCPPowerPort

CCP_OFF_OBSERVABLE_UNITS = {
    "frequency_Hz": "Hz",
    "delivered_power_W": "W",
    "apparent_power_VA": "VA",
    "absorbed_power_W": "W",
    "rf_voltage_rms_V": "V",
    "rf_current_rms_A": "A",
    "self_bias_V": "V",
    "plasma_potential_V": "V",
}
CCP_PLASMA_OBSERVABLE_UNITS = {
    "bulk_resistance_ohm": "ohm",
    "sheath_reactance_ohm": "ohm",
    "powered_sheath_thickness_m": "m",
    "grounded_sheath_thickness_m": "m",
    "powered_sheath_voltage_V": "V",
    "grounded_sheath_voltage_V": "V",
    "ion_flux_m2_s": "m^-2 s^-1",
    "estimated_mean_ion_energy_eV": "eV",
    "pressure_Pa": "Pa",
}
CCP_OBSERVABLE_UNITS = {**CCP_OFF_OBSERVABLE_UNITS, **CCP_PLASMA_OBSERVABLE_UNITS}


def validate_ccp_power_port(port: CCPPowerPort) -> None:
    _validate_port_ids(port.port_id, port.zone_id)
    positive = {
        "frequency_Hz": port.frequency_Hz,
        "zone_volume_m3": port.zone_volume_m3,
        "powered_area_m2": port.powered_area_m2,
        "grounded_area_m2": port.grounded_area_m2,
        "electrode_gap_m": port.electrode_gap_m,
        "dominant_ion_mass_kg": port.dominant_ion_mass_kg,
        "gas_temperature_K": port.gas_temperature_K,
    }
    for name, value in positive.items():
        _positive(name, value)
    _validate_constant_commands(
        port.absorbed_power_W, port.voltage_rms_V, model_name="CCP"
    )


def evaluate_ccp_power_port(
    port: CCPPowerPort,
    time_s: float,
    state: PowerState,
    command: CompiledPowerCommand | None,
) -> PowerPortResult:
    mode, value = _command_value(
        command,
        power_W=port.absorbed_power_W,
        voltage_V=port.voltage_rms_V,
    )
    del time_s
    absorbed_command = value
    if absorbed_command == 0.0:
        return PowerPortResult(
            port_id=port.port_id,
            zone_id=port.zone_id,
            reduced_field_Td=0.0,
            observables=port.off_observables,
        )
    if absorbed_command > 0.0 and state.electron_density_m3 == 0.0:
        raise ModelDomainError(
            "positive CCP power or voltage requires positive electron density"
        )
    bulk_resistance = port._bulk_resistance(state.electron_density_m3)
    if mode == "voltage":
        voltage_rms = absorbed_command
        current_rms = 0.0
    else:
        current_rms = math.sqrt(absorbed_command / bulk_resistance)
        voltage_rms = port._power_mode_voltage(
            state,
            current_rms,
            math.sqrt(absorbed_command * bulk_resistance),
        )
    powered_capacitance, powered_sheath_m = port._sheath_capacitance(
        port.powered_area_m2, state, voltage_rms
    )
    grounded_capacitance, grounded_sheath_m = port._sheath_capacitance(
        port.grounded_area_m2, state, voltage_rms
    )
    omega = 2.0 * math.pi * port.frequency_Hz
    sheath_reactance = 1.0 / (omega * powered_capacitance)
    sheath_reactance += 1.0 / (omega * grounded_capacitance)
    impedance = math.hypot(bulk_resistance, sheath_reactance)

    if mode == "voltage":
        current_rms = voltage_rms / impedance
        absorbed = current_rms**2 * bulk_resistance
    else:
        absorbed = absorbed_command
    # Lossless sheaths carry reactive power but consume no real power.  Source
    # real power is I^2 R; V*I is apparent power and never enters the ledger.
    delivered = absorbed
    apparent_power_VA = voltage_rms * current_rms

    area_ratio = max(port.grounded_area_m2 / port.powered_area_m2, 1.0)
    asymmetry = area_ratio**1.35
    bias_fraction = (asymmetry - 1.0) / (asymmetry + 1.0)
    peak_voltage = math.sqrt(2.0) * voltage_rms
    self_bias = -bias_fraction * peak_voltage
    powered_sheath_V = 0.5 * peak_voltage * (1.0 + bias_fraction)
    grounded_sheath_V = 0.5 * peak_voltage * (1.0 - bias_fraction)
    plasma_potential = max(3.0 * state.electron_temperature_eV, 8.0)
    plasma_potential += 0.25 * (powered_sheath_V + grounded_sheath_V)

    bohm_speed = math.sqrt(
        ELEMENTARY_CHARGE_C
        * max(state.electron_temperature_eV, 0.0)
        / port.dominant_ion_mass_kg
    )
    ion_flux = state.electron_density_m3 * bohm_speed
    collisional_opacity = state.neutral_density_m3 * 5.0e-20 * powered_sheath_m
    mean_ion_energy = powered_sheath_V / (1.0 + collisional_opacity)
    if state.neutral_density_m3 > 0.0:
        bulk_voltage_rms = current_rms * bulk_resistance
        reduced_field = (
            bulk_voltage_rms
            / port.electrode_gap_m
            / state.neutral_density_m3
            / TD_TO_V_M2
        )
    else:
        reduced_field = 0.0
    pressure_Pa = state.neutral_density_m3 * BOLTZMANN_J_K * port.gas_temperature_K
    # Sheath voltage and ion-energy estimates remain diagnostics.  The circuit
    # treats its sheaths as lossless, so feeding them into the wall ledger would
    # add an unaccounted real-power sink.
    return PowerPortResult(
        port_id=port.port_id,
        zone_id=port.zone_id,
        contributions=(PowerContribution(port.zone_id, absorbed),),
        reduced_field_Td=reduced_field,
        observables={
            "frequency_Hz": port.frequency_Hz,
            "delivered_power_W": delivered,
            "apparent_power_VA": apparent_power_VA,
            "absorbed_power_W": absorbed,
            "rf_voltage_rms_V": voltage_rms,
            "rf_current_rms_A": current_rms,
            "bulk_resistance_ohm": bulk_resistance,
            "sheath_reactance_ohm": sheath_reactance,
            "powered_sheath_thickness_m": powered_sheath_m,
            "grounded_sheath_thickness_m": grounded_sheath_m,
            "self_bias_V": self_bias,
            "plasma_potential_V": plasma_potential,
            "powered_sheath_voltage_V": powered_sheath_V,
            "grounded_sheath_voltage_V": grounded_sheath_V,
            "ion_flux_m2_s": ion_flux,
            "estimated_mean_ion_energy_eV": mean_ion_energy,
            "pressure_Pa": pressure_Pa,
        },
    )


__all__ = [
    "evaluate_ccp_power_port",
    "validate_ccp_power_port",
]
