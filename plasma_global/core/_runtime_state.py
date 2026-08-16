"""Validate and prepare one compiled-model state for physical evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from plasma_global.core._runtime_assembly import (
    PreparedEvaluationState,
    PreparedZoneState,
)
from plasma_global.core.domain import RecipeSegment, Zone
from plasma_global.errors import ModelConfigurationError, StateDomainError
from plasma_global.models.electrons import ElectronState, resolved_electron_density

if TYPE_CHECKING:
    from plasma_global.core.compiled import CompiledGlobalModel


@dataclass(frozen=True, slots=True)
class RuntimeStatePreparer:
    """Turn a flat solver state into validated, zone-indexed physical values."""

    model: CompiledGlobalModel

    def electron_density(
        self,
        time_s: float,
        zone_id: str,
        net_heavy_charge_density_m3: float,
        segment: RecipeSegment,
    ) -> float:
        prescribed = segment.prescribed_electron_density_m3_by_zone.get(zone_id)
        if prescribed is None and self.model.electron_density_provider is not None:
            prescribed = self.model.electron_density_provider(time_s, zone_id)
        return resolved_electron_density(net_heavy_charge_density_m3, prescribed)

    def prepare(
        self,
        time_s: float,
        state: np.ndarray,
        segment: RecipeSegment,
        domain_atol: np.ndarray | None,
    ) -> PreparedEvaluationState:
        values = np.asarray(state, dtype=float)
        if values.shape != (self.model.layout.size,):
            raise StateDomainError(
                f"State has shape {values.shape}, expected {(self.model.layout.size,)}"
            )
        if not np.all(np.isfinite(values)):
            raise StateDomainError("State must contain only finite values")
        active_domain_atol = (
            np.asarray(self.model.domain_atol, dtype=float)
            if domain_atol is None
            else domain_atol
        )
        zone_count = len(self.model.zones)
        density_by_zone = np.empty(
            (zone_count, len(self.model.species_ids)), dtype=float
        )
        reaction_density_by_zone = np.empty_like(density_by_zone)
        electron_energy_by_zone = self._optional_zone_array(
            self.model.layout.evolves_electron_energy, zone_count
        )
        closure_electron_energy_by_zone = self._optional_zone_array(
            self.model.layout.evolves_electron_energy, zone_count
        )
        heavy_energy_by_zone = self._optional_zone_array(
            self.model.layout.evolves_heavy_energy, zone_count
        )
        net_charge_by_zone: dict[str, float] = {}
        electron_density_by_zone: dict[str, float] = {}
        charge_residual_by_zone: dict[str, float] = {}
        neutral_density_by_zone: dict[str, float] = {}
        mean_energy_for_power: dict[str, float | None] = {}
        electron_state_by_zone: dict[str, ElectronState | None] = {}
        for zone_index, zone in enumerate(self.model.zones):
            prepared = self._prepare_zone(
                time_s=time_s,
                values=values,
                domain_atol=active_domain_atol,
                segment=segment,
                zone=zone,
            )
            density_by_zone[zone_index] = prepared.density
            reaction_density_by_zone[zone_index] = prepared.reaction_density
            self._store_optional(
                electron_energy_by_zone, zone_index, prepared.electron_energy
            )
            self._store_optional(
                closure_electron_energy_by_zone,
                zone_index,
                prepared.closure_electron_energy,
            )
            self._store_optional(
                heavy_energy_by_zone, zone_index, prepared.heavy_energy
            )
            net_charge_by_zone[zone.zone_id] = prepared.net_charge
            electron_density_by_zone[zone.zone_id] = prepared.electron_density
            charge_residual_by_zone[zone.zone_id] = prepared.charge_residual
            neutral_density_by_zone[zone.zone_id] = prepared.neutral_density
            mean_energy_for_power[zone.zone_id] = prepared.mean_energy_for_power
            electron_state_by_zone[zone.zone_id] = prepared.electron_state
        gas_temperature_by_zone = self._gas_temperatures(
            density_by_zone, heavy_energy_by_zone
        )
        return PreparedEvaluationState(
            values=values,
            domain_atol=active_domain_atol,
            density_by_zone=density_by_zone,
            reaction_density_by_zone=reaction_density_by_zone,
            electron_energy_by_zone=electron_energy_by_zone,
            closure_electron_energy_by_zone=closure_electron_energy_by_zone,
            heavy_energy_by_zone=heavy_energy_by_zone,
            net_charge_by_zone=net_charge_by_zone,
            electron_density_by_zone=electron_density_by_zone,
            charge_residual_by_zone=charge_residual_by_zone,
            neutral_density_by_zone=neutral_density_by_zone,
            mean_energy_for_power=mean_energy_for_power,
            electron_state_by_zone=electron_state_by_zone,
            gas_temperature_by_zone=gas_temperature_by_zone,
        )

    def _prepare_zone(
        self,
        *,
        time_s: float,
        values: np.ndarray,
        domain_atol: np.ndarray,
        segment: RecipeSegment,
        zone: Zone,
    ) -> PreparedZoneState:
        density_slice = self.model.layout.density_slices[zone.zone_id]
        density = values[density_slice]
        density_atol = domain_atol[density_slice]
        materially_negative = density < (-10.0 * density_atol)
        if np.any(materially_negative):
            bad = {
                species_id: float(number_density)
                for species_id, number_density, failed in zip(
                    self.model.species_ids, density, materially_negative, strict=True
                )
                if failed
            }
            raise StateDomainError(
                "Negative density below -10*domain_atol in zone "
                f"{zone.zone_id!r}: {bad}"
            )
        reaction_density = np.where(density < 0.0, 0.0, density)
        raw_net_charge = float(self.model.charges @ density)
        charge_atol = float(np.abs(self.model.charges) @ density_atol)
        net_charge = (
            0.0 if -10.0 * charge_atol <= raw_net_charge < 0.0 else raw_net_charge
        )
        electron_density = self.electron_density(
            time_s, zone.zone_id, net_charge, segment
        )
        electron_energy, closure_energy, electron_state = self._electron_energy(
            values=values,
            domain_atol=domain_atol,
            segment=segment,
            zone=zone,
            net_charge=net_charge,
            electron_density=electron_density,
        )
        heavy_energy = self._heavy_energy(
            values=values, domain_atol=domain_atol, zone=zone
        )
        return PreparedZoneState(
            density=density,
            reaction_density=reaction_density,
            electron_energy=electron_energy,
            closure_electron_energy=closure_energy,
            heavy_energy=heavy_energy,
            net_charge=net_charge,
            electron_density=electron_density,
            charge_residual=raw_net_charge - electron_density,
            neutral_density=float(np.sum(reaction_density[self.model.charges == 0.0])),
            mean_energy_for_power=(
                None if electron_state is None else electron_state.mean_energy_eV
            ),
            electron_state=electron_state,
        )

    def _electron_energy(
        self,
        *,
        values: np.ndarray,
        domain_atol: np.ndarray,
        segment: RecipeSegment,
        zone: Zone,
        net_charge: float,
        electron_density: float,
    ) -> tuple[float | None, float | None, ElectronState | None]:
        if not self.model.layout.evolves_electron_energy:
            return None, None, None
        energy_index = self.model.layout.electron_energy_indices[zone.zone_id]
        energy = float(values[energy_index])
        energy_atol = float(domain_atol[energy_index])
        if energy < -10.0 * energy_atol:
            raise StateDomainError(
                f"Electron energy below -10*domain_atol in zone {zone.zone_id!r}"
            )
        closure_energy = max(energy, 0.0)
        if electron_density == 0.0 and closure_energy <= 10.0 * energy_atol:
            closure_energy = 0.0
        electron_state = self.model.electron_closure.evaluate(
            net_heavy_charge_density_m3=net_charge,
            energy_density_J_m3=closure_energy,
            reduced_field_Td=segment.reduced_field_Td_by_zone.get(zone.zone_id),
            electron_density_m3=electron_density,
        )
        return energy, closure_energy, electron_state

    def _heavy_energy(
        self, *, values: np.ndarray, domain_atol: np.ndarray, zone: Zone
    ) -> float | None:
        if not self.model.layout.evolves_heavy_energy:
            return None
        energy_index = self.model.layout.heavy_energy_indices[zone.zone_id]
        energy = float(values[energy_index])
        if energy < -10.0 * domain_atol[energy_index]:
            raise StateDomainError(
                f"Gas internal energy below -10*domain_atol in zone {zone.zone_id!r}"
            )
        return energy

    @staticmethod
    def _optional_zone_array(enabled: bool, zone_count: int) -> np.ndarray | None:
        return np.empty(zone_count, dtype=float) if enabled else None

    @staticmethod
    def _store_optional(
        target: np.ndarray | None, index: int, value: float | None
    ) -> None:
        if target is not None and value is not None:
            target[index] = value

    def _gas_temperatures(
        self, density_by_zone: np.ndarray, heavy_energy_by_zone: np.ndarray | None
    ) -> np.ndarray:
        if heavy_energy_by_zone is None:
            return np.array(
                [zone.gas_temperature_K for zone in self.model.zones], dtype=float
            )
        if self.model.heavy_energy_closure is None:
            raise ModelConfigurationError("Evolved gas energy needs its closure")
        closure_energy = np.where(heavy_energy_by_zone < 0.0, 0.0, heavy_energy_by_zone)
        temperatures = self.model.heavy_energy_closure.temperature_K(
            density_by_zone, closure_energy
        )
        if np.any(temperatures <= 0.0):
            raise StateDomainError("Evolved gas temperature must remain positive")
        return temperatures


__all__ = ["RuntimeStatePreparer"]
