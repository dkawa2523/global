"""Cycle-averaged RF power ports for opt-in exploratory models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from plasma_global.errors import ModelDomainError
from plasma_global.experimental._ccp_power_kernel import (
    CCP_OBSERVABLE_UNITS,
    CCP_OFF_OBSERVABLE_UNITS,
    evaluate_ccp_power_port,
    validate_ccp_power_port,
)
from plasma_global.experimental._power_kernels import (
    BOLTZMANN_J_K as BOLTZMANN_J_K,
)
from plasma_global.experimental._power_kernels import (
    ELEMENTARY_CHARGE_C as ELEMENTARY_CHARGE_C,
)
from plasma_global.experimental._power_kernels import (
    ICP_OBSERVABLE_UNITS,
    ICP_OFF_OBSERVABLE_UNITS,
    RF_ENVELOPE_OBSERVABLE_UNITS,
    evaluate_icp_power_port,
    evaluate_rf_envelope_port,
    rf_off_observables,
    validate_icp_power_port,
    validate_rf_envelope_port,
)
from plasma_global.experimental._power_kernels import TD_TO_V_M2 as TD_TO_V_M2
from plasma_global.experimental._power_kernels import (
    VACUUM_PERMITTIVITY_F_M as VACUUM_PERMITTIVITY_F_M,
)
from plasma_global.models.power import (
    CompiledPowerCommand,
    PowerPortResult,
    PowerState,
    PowerTopology,
)


@dataclass(frozen=True, slots=True)
class RFEnvelopePort:
    """HF/LF envelope; power commands specify plasma-absorbed watts."""

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
    absorbed_power_W: float | None = None
    voltage_rms_V: float | None = None

    observable_units = RF_ENVELOPE_OBSERVABLE_UNITS

    @property
    def topology(self) -> PowerTopology:
        return PowerTopology(
            self.port_id, self.zone_id, (self.zone_id,), (), (self.zone_id,)
        )

    @property
    def off_observables(self) -> dict[str, float]:
        return rf_off_observables(self.frequency_Hz, self.observable_units)

    @property
    def time_dependent(self) -> bool:
        return False

    def __post_init__(self) -> None:
        validate_rf_envelope_port(self)

    def evaluate(
        self, time_s: float, state: PowerState, command: CompiledPowerCommand | None
    ) -> PowerPortResult:
        del state
        return evaluate_rf_envelope_port(self, time_s, command)


@dataclass(frozen=True, slots=True)
class CCPPowerPort:
    """Lumped CCP bulk-power model with diagnostic-only sheath estimates.

    Power commands specify absorbed bulk watts.  The
    ``estimated_mean_ion_energy_eV`` observable does not drive wall or surface
    kinetics because this lossless circuit has no real sheath-power partition.
    """

    port_id: str
    zone_id: str
    frequency_Hz: float
    zone_volume_m3: float
    powered_area_m2: float
    grounded_area_m2: float
    electrode_gap_m: float
    dominant_ion_mass_kg: float
    gas_temperature_K: float = 300.0
    absorbed_power_W: float | None = None
    voltage_rms_V: float | None = None

    observable_units = CCP_OBSERVABLE_UNITS

    @property
    def topology(self) -> PowerTopology:
        return PowerTopology(
            self.port_id, self.zone_id, (self.zone_id,), (), (self.zone_id,)
        )

    @property
    def off_observables(self) -> dict[str, float]:
        return rf_off_observables(self.frequency_Hz, CCP_OFF_OBSERVABLE_UNITS)

    @property
    def time_dependent(self) -> bool:
        return False

    def __post_init__(self) -> None:
        validate_ccp_power_port(self)

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

    def _power_mode_voltage(
        self, state: PowerState, current_rms_A: float, initial_voltage_V: float
    ) -> float:
        """Solve the voltage-dependent sheath impedance at fixed real power."""

        voltage = initial_voltage_V
        omega = 2.0 * math.pi * self.frequency_Hz
        bulk_resistance = self._bulk_resistance(state.electron_density_m3)
        for _ in range(48):
            powered, _ = self._sheath_capacitance(self.powered_area_m2, state, voltage)
            grounded, _ = self._sheath_capacitance(
                self.grounded_area_m2, state, voltage
            )
            reactance = 1.0 / (omega * powered) + 1.0 / (omega * grounded)
            updated = current_rms_A * math.hypot(bulk_resistance, reactance)
            if math.isclose(updated, voltage, rel_tol=1.0e-10, abs_tol=1.0e-12):
                return updated
            voltage = updated
        raise ModelDomainError("CCP sheath impedance did not converge")

    def evaluate(
        self, time_s: float, state: PowerState, command: CompiledPowerCommand | None
    ) -> PowerPortResult:
        return evaluate_ccp_power_port(self, time_s, state, command)


@dataclass(frozen=True, slots=True)
class ICPPowerPort:
    """Empirical ICP E/H-mode coupling with explicit downstream deposition."""

    port_id: str
    zone_id: str
    frequency_Hz: float = 13.56e6
    gas_temperature_K: float = 300.0
    downstream_zone_id: str | None = None
    downstream_fraction: float = 0.0
    delivered_power_W: float | None = None

    observable_units = ICP_OBSERVABLE_UNITS

    @property
    def topology(self) -> PowerTopology:
        targets = [self.zone_id]
        if self.downstream_fraction > 0.0 and self.downstream_zone_id is not None:
            targets.append(self.downstream_zone_id)
        return PowerTopology(
            self.port_id,
            self.zone_id,
            tuple(dict.fromkeys(targets)),
            (),
            (self.zone_id,),
        )

    @property
    def off_observables(self) -> dict[str, float]:
        return rf_off_observables(self.frequency_Hz, ICP_OFF_OBSERVABLE_UNITS)

    @property
    def time_dependent(self) -> bool:
        return False

    def __post_init__(self) -> None:
        validate_icp_power_port(self)

    def evaluate(
        self, time_s: float, state: PowerState, command: CompiledPowerCommand | None
    ) -> PowerPortResult:
        return evaluate_icp_power_port(self, time_s, state, command)


__all__ = ["CCPPowerPort", "ICPPowerPort", "RFEnvelopePort"]
