"""Cycle-averaged RF power ports for opt-in exploratory models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from plasma_global.models.power import (
    CompiledPowerCommand,
    PowerPortResult,
    PowerState,
)

ELEMENTARY_CHARGE_C = 1.602176634e-19
BOLTZMANN_J_K = 1.380649e-23
VACUUM_PERMITTIVITY_F_M = 8.8541878128e-12
TD_TO_V_M2 = 1.0e-21


def _positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _fraction(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between zero and one")


@dataclass(frozen=True, slots=True)
class SquarePulse:
    """A dimensionless square envelope shared by experimental power ports."""

    repetition_Hz: float
    duty_cycle: float
    phase_fraction: float = 0.0

    def __post_init__(self) -> None:
        _positive("repetition_Hz", self.repetition_Hz)
        _fraction("duty_cycle", self.duty_cycle)
        if not math.isfinite(self.phase_fraction):
            raise ValueError("phase_fraction must be finite")

    def __call__(self, time_s: float) -> float:
        if not math.isfinite(time_s):
            raise ValueError("time_s must be finite")
        phase = (time_s * self.repetition_Hz + self.phase_fraction) % 1.0
        return 1.0 if phase < self.duty_cycle else 0.0


def _envelope_value(pulse: SquarePulse | None, time_s: float) -> float:
    return 1.0 if pulse is None else pulse(time_s)


def _command_value(
    command: CompiledPowerCommand | None,
    *,
    default_power_W: float | None,
    default_voltage_V: float | None,
) -> tuple[Literal["power", "voltage"], float]:
    if command is not None and command.kind == "power":
        assert command.power_W is not None
        return "power", command.power_W
    if command is not None and command.kind == "voltage":
        assert command.voltage_V is not None
        return "voltage", abs(command.voltage_V)
    if command is not None:
        raise TypeError("power ports accept compiled power or voltage commands")
    if default_power_W is not None:
        return "power", default_power_W
    if default_voltage_V is not None:
        return "voltage", abs(default_voltage_V)
    raise ValueError("the power port requires a recipe command or a default command")


@dataclass(frozen=True, slots=True)
class RFEnvelopePort:
    """HF/LF cycle-average envelope with explicit power coupling and bias."""

    port_id: str
    zone_id: str
    frequency_Hz: float
    role: Literal["source", "bias"] = "source"
    coupling_efficiency: float = 1.0
    effective_impedance_ohm: float = 50.0
    base_reduced_field_Td: float = 0.0
    reduced_field_per_sqrt_W_Td: float = 0.0
    self_bias_fraction: float = 0.35
    plasma_potential_offset_V: float = 0.0
    plasma_potential_per_sqrt_W: float = 0.0
    default_power_W: float | None = None
    default_voltage_V: float | None = None
    pulse: SquarePulse | None = None

    produces_reduced_field = True

    @property
    def time_dependent(self) -> bool:
        return self.pulse is not None

    def __post_init__(self) -> None:
        if not self.port_id or not self.zone_id:
            raise ValueError("port_id and zone_id must not be empty")
        _positive("frequency_Hz", self.frequency_Hz)
        _positive("effective_impedance_ohm", self.effective_impedance_ohm)
        _fraction("coupling_efficiency", self.coupling_efficiency)
        if self.role not in {"source", "bias"}:
            raise ValueError("role must be 'source' or 'bias'")
        nonnegative = {
            "base_reduced_field_Td": self.base_reduced_field_Td,
            "reduced_field_per_sqrt_W_Td": self.reduced_field_per_sqrt_W_Td,
            "self_bias_fraction": self.self_bias_fraction,
            "plasma_potential_offset_V": self.plasma_potential_offset_V,
            "plasma_potential_per_sqrt_W": self.plasma_potential_per_sqrt_W,
        }
        if any(
            not math.isfinite(value) or value < 0.0 for value in nonnegative.values()
        ):
            raise ValueError(
                "RF envelope field and potential coefficients must be nonnegative"
            )
        if self.default_power_W is not None and (
            not math.isfinite(self.default_power_W) or self.default_power_W < 0.0
        ):
            raise ValueError("default_power_W must be finite and nonnegative")
        if self.default_voltage_V is not None and not math.isfinite(
            self.default_voltage_V
        ):
            raise ValueError("default_voltage_V must be finite")
        if self.default_power_W is not None and self.default_voltage_V is not None:
            raise ValueError("provide only one default RF command")

    def evaluate(
        self, time_s: float, state: PowerState, command: CompiledPowerCommand | None
    ) -> PowerPortResult:
        del state
        mode, value = _command_value(
            command,
            default_power_W=self.default_power_W,
            default_voltage_V=self.default_voltage_V,
        )
        envelope = _envelope_value(self.pulse, time_s)
        if mode == "power":
            delivered = value * envelope
            voltage_rms = math.sqrt(delivered * self.effective_impedance_ohm)
        else:
            voltage_rms = value * envelope
            delivered = voltage_rms**2 / self.effective_impedance_ohm
        absorbed = self.coupling_efficiency * delivered
        reduced_field = self.base_reduced_field_Td * envelope
        reduced_field += self.reduced_field_per_sqrt_W_Td * math.sqrt(absorbed)
        self_bias = (
            -self.self_bias_fraction * math.sqrt(2.0) * voltage_rms
            if self.role == "bias"
            else 0.0
        )
        plasma_potential = self.plasma_potential_offset_V * envelope
        plasma_potential += self.plasma_potential_per_sqrt_W * math.sqrt(absorbed)
        return PowerPortResult(
            port_id=self.port_id,
            zone_id=self.zone_id,
            electron_power_W=absorbed,
            reduced_field_Td=reduced_field,
            observables={
                "frequency_Hz": self.frequency_Hz,
                "delivered_power_W": delivered,
                "absorbed_power_W": absorbed,
                "voltage_rms_V": voltage_rms,
                "self_bias_V": self_bias,
                "plasma_potential_V": plasma_potential,
            },
        )


@dataclass(frozen=True, slots=True)
class CCPPowerPort:
    """Lumped bulk-resistance/two-sheath CCP approximation."""

    port_id: str
    zone_id: str
    frequency_Hz: float
    zone_volume_m3: float
    powered_area_m2: float
    grounded_area_m2: float
    electrode_gap_m: float
    dominant_ion_mass_kg: float
    gas_temperature_K: float = 300.0
    default_power_W: float | None = None
    default_voltage_V: float | None = None
    pulse: SquarePulse | None = None

    produces_reduced_field = True

    @property
    def time_dependent(self) -> bool:
        return self.pulse is not None

    def __post_init__(self) -> None:
        if not self.port_id or not self.zone_id:
            raise ValueError("port_id and zone_id must not be empty")
        positive = {
            "frequency_Hz": self.frequency_Hz,
            "zone_volume_m3": self.zone_volume_m3,
            "powered_area_m2": self.powered_area_m2,
            "grounded_area_m2": self.grounded_area_m2,
            "electrode_gap_m": self.electrode_gap_m,
            "dominant_ion_mass_kg": self.dominant_ion_mass_kg,
            "gas_temperature_K": self.gas_temperature_K,
        }
        for name, value in positive.items():
            _positive(name, value)
        if self.default_power_W is not None and (
            not math.isfinite(self.default_power_W) or self.default_power_W < 0.0
        ):
            raise ValueError("default_power_W must be finite and nonnegative")
        if self.default_voltage_V is not None and not math.isfinite(
            self.default_voltage_V
        ):
            raise ValueError("default_voltage_V must be finite")
        if self.default_power_W is not None and self.default_voltage_V is not None:
            raise ValueError("provide only one default CCP command")

    def _bulk_resistance(self, electron_density_m3: float) -> float:
        density_scale = max(electron_density_m3 / 1.0e16, 1.0e-10)
        frequency_scale = self.frequency_Hz / 13.56e6
        length_scale = self.zone_volume_m3 / self.powered_area_m2
        return (
            5.0 * max(length_scale, 1.0e-6) / math.sqrt(density_scale * frequency_scale)
        )

    @staticmethod
    def _debye_length(
        electron_density_m3: float, electron_temperature_eV: float
    ) -> float:
        density = max(electron_density_m3, 1.0e6)
        temperature = max(electron_temperature_eV, 0.03)
        return math.sqrt(
            VACUUM_PERMITTIVITY_F_M * temperature / (density * ELEMENTARY_CHARGE_C)
        )

    def _sheath_capacitance(
        self, area_m2: float, state: PowerState, voltage_V: float
    ) -> tuple[float, float]:
        debye_length = self._debye_length(
            state.electron_density_m3, state.electron_temperature_eV
        )
        thickness = max(
            4.0
            * debye_length
            * math.sqrt(
                1.0 + max(voltage_V, 1.0) / max(state.electron_temperature_eV, 0.1)
            ),
            5.0e-5,
        )
        capacitance = VACUUM_PERMITTIVITY_F_M * area_m2 / thickness
        return capacitance, thickness

    def evaluate(
        self, time_s: float, state: PowerState, command: CompiledPowerCommand | None
    ) -> PowerPortResult:
        mode, value = _command_value(
            command,
            default_power_W=self.default_power_W,
            default_voltage_V=self.default_voltage_V,
        )
        envelope = _envelope_value(self.pulse, time_s)
        bulk_resistance = self._bulk_resistance(state.electron_density_m3)
        trial_voltage = (
            value * envelope
            if mode == "voltage"
            else math.sqrt(value * envelope * bulk_resistance)
        )
        powered_capacitance, powered_sheath_m = self._sheath_capacitance(
            self.powered_area_m2, state, trial_voltage
        )
        grounded_capacitance, grounded_sheath_m = self._sheath_capacitance(
            self.grounded_area_m2, state, trial_voltage
        )
        omega = 2.0 * math.pi * self.frequency_Hz
        sheath_reactance = 1.0 / (omega * powered_capacitance)
        sheath_reactance += 1.0 / (omega * grounded_capacitance)
        impedance = math.hypot(bulk_resistance, sheath_reactance)

        if mode == "voltage":
            voltage_rms = value * envelope
            current_rms = voltage_rms / impedance
            absorbed = current_rms**2 * bulk_resistance
            delivered = voltage_rms * current_rms
        else:
            delivered = value * envelope
            absorbed = delivered * (bulk_resistance / impedance) ** 2
            current_rms = math.sqrt(absorbed / bulk_resistance)
            voltage_rms = current_rms * impedance

        area_ratio = max(self.grounded_area_m2 / self.powered_area_m2, 1.0)
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
            / self.dominant_ion_mass_kg
        )
        ion_flux = state.electron_density_m3 * bohm_speed
        collisional_opacity = state.neutral_density_m3 * 5.0e-20 * powered_sheath_m
        mean_ion_energy = powered_sheath_V / (1.0 + collisional_opacity)
        if state.neutral_density_m3 > 0.0:
            reduced_field = (
                (powered_sheath_V + grounded_sheath_V)
                / self.electrode_gap_m
                / state.neutral_density_m3
                / TD_TO_V_M2
            )
        else:
            reduced_field = 0.0
        pressure_Pa = state.neutral_density_m3 * BOLTZMANN_J_K * self.gas_temperature_K
        return PowerPortResult(
            port_id=self.port_id,
            zone_id=self.zone_id,
            electron_power_W=absorbed,
            reduced_field_Td=reduced_field,
            observables={
                "frequency_Hz": self.frequency_Hz,
                "delivered_power_W": delivered,
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
                "mean_ion_energy_eV": mean_ion_energy,
                "pressure_Pa": pressure_Pa,
            },
        )


@dataclass(frozen=True, slots=True)
class ICPPowerPort:
    """Empirical ICP E/H-mode coupling with explicit downstream deposition."""

    port_id: str
    zone_id: str
    frequency_Hz: float = 13.56e6
    gas_temperature_K: float = 300.0
    downstream_zone_id: str | None = None
    downstream_fraction: float = 0.0
    default_power_W: float | None = None
    pulse: SquarePulse | None = None

    produces_reduced_field = True

    @property
    def time_dependent(self) -> bool:
        return self.pulse is not None

    def __post_init__(self) -> None:
        if not self.port_id or not self.zone_id:
            raise ValueError("port_id and zone_id must not be empty")
        _positive("frequency_Hz", self.frequency_Hz)
        _positive("gas_temperature_K", self.gas_temperature_K)
        _fraction("downstream_fraction", self.downstream_fraction)
        if self.downstream_fraction > 0.0 and not self.downstream_zone_id:
            raise ValueError(
                "downstream_zone_id is required when power is split downstream"
            )
        if self.default_power_W is not None and (
            not math.isfinite(self.default_power_W) or self.default_power_W < 0.0
        ):
            raise ValueError("default_power_W must be finite and nonnegative")

    def evaluate(
        self, time_s: float, state: PowerState, command: CompiledPowerCommand | None
    ) -> PowerPortResult:
        mode, value = _command_value(
            command, default_power_W=self.default_power_W, default_voltage_V=None
        )
        if mode != "power":
            raise TypeError("ICP ports accept power commands only")
        delivered = value * _envelope_value(self.pulse, time_s)
        if delivered == 0.0:
            return PowerPortResult(
                port_id=self.port_id,
                zone_id=self.zone_id,
                electron_power_W=0.0,
                reduced_field_Td=0.0,
                observables={
                    "delivered_power_W": 0.0,
                    "absorbed_power_W": 0.0,
                    "downstream_power_W": 0.0,
                    "coupling_efficiency": 0.0,
                    "mode_H": 0.0,
                },
            )

        density_scale = state.electron_density_m3 / 1.0e17
        pressure_Pa = state.neutral_density_m3 * BOLTZMANN_J_K * self.gas_temperature_K
        pressure_scale = max(pressure_Pa / 10.0, 1.0e-3)
        frequency_scale = max(self.frequency_Hz / 13.56e6, 0.1)
        density_factor = 1.0 - math.exp(-2.2 * density_scale)
        pressure_factor = 0.65 + 0.20 * math.tanh((pressure_scale - 0.3) / 0.6)
        frequency_factor = 0.85 + 0.08 * math.log10(1.0 + frequency_scale)
        mode_index = density_factor * math.sqrt(delivered / 150.0)
        mode_H = mode_index >= 0.55
        minimum_coupling, maximum_coupling = (0.40, 0.88) if mode_H else (0.18, 0.55)
        coupling = minimum_coupling + (
            (maximum_coupling - minimum_coupling)
            * density_factor
            * pressure_factor
            * frequency_factor
        )
        coupling = min(max(coupling, 0.05), 0.95)
        absorbed = coupling * delivered
        downstream_power = absorbed * self.downstream_fraction
        local_power = absorbed - downstream_power
        plasma_resistance = 2.5 + 12.0 / max(
            math.sqrt(state.electron_density_m3 / 1.0e16), 0.2
        )
        coil_current = math.sqrt(absorbed / plasma_resistance)
        coil_voltage = coil_current * plasma_resistance
        reduced_field = (
            25.0
            + 30.0 * math.sqrt(max(state.electron_temperature_eV, 0.0))
            + 0.01 * absorbed
        )
        return PowerPortResult(
            port_id=self.port_id,
            zone_id=self.zone_id,
            electron_power_W=local_power,
            reduced_field_Td=reduced_field,
            observables={
                "frequency_Hz": self.frequency_Hz,
                "delivered_power_W": delivered,
                "absorbed_power_W": absorbed,
                "downstream_power_W": downstream_power,
                "reflected_power_W": delivered - absorbed,
                "coupling_efficiency": coupling,
                "plasma_resistance_ohm": plasma_resistance,
                "coil_voltage_rms_V": coil_voltage,
                "coil_current_rms_A": coil_current,
                "mode_index": mode_index,
                "mode_H": float(mode_H),
                "plasma_potential_V": 5.0 + 0.015 * absorbed + 2.0 * coupling,
                "pressure_Pa": pressure_Pa,
            },
        )

    def power_by_zone(self, result: PowerPortResult) -> dict[str, float]:
        """Expand one ICP evaluation into the zone powers consumed by a core adapter."""

        values = {self.zone_id: result.electron_power_W}
        if self.downstream_zone_id is not None:
            downstream = float(result.observables.get("downstream_power_W", 0.0))
            values[self.downstream_zone_id] = (
                values.get(self.downstream_zone_id, 0.0) + downstream
            )
        return values


__all__ = ["CCPPowerPort", "ICPPowerPort", "RFEnvelopePort", "SquarePulse"]
