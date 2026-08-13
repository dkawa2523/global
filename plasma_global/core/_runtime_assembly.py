"""Private evaluated-term payloads and deterministic RHS assembly."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from plasma_global.models.electrons import ElectronState
from plasma_global.models.gas_energy import BOLTZMANN_J_K
from plasma_global.models.kinetics import ElectronKineticsResult
from plasma_global.models.power import PowerCouplingResult
from plasma_global.models.surface import SurfaceEvaluation
from plasma_global.models.walls import WallFluxRecord

if TYPE_CHECKING:
    from plasma_global.core.compiled import CompiledGlobalModel


@dataclass(frozen=True, slots=True)
class PreparedEvaluationState:
    values: np.ndarray
    domain_atol: np.ndarray
    density_by_zone: np.ndarray
    reaction_density_by_zone: np.ndarray
    electron_energy_by_zone: np.ndarray | None
    closure_electron_energy_by_zone: np.ndarray | None
    heavy_energy_by_zone: np.ndarray | None
    net_charge_by_zone: dict[str, float]
    electron_density_by_zone: dict[str, float]
    charge_residual_by_zone: dict[str, float]
    neutral_density_by_zone: dict[str, float]
    mean_energy_for_power: dict[str, float | None]
    gas_temperature_by_zone: np.ndarray


@dataclass(frozen=True, slots=True)
class PreparedZoneState:
    density: np.ndarray
    reaction_density: np.ndarray
    electron_energy: float | None
    closure_electron_energy: float | None
    heavy_energy: float | None
    net_charge: float
    electron_density: float
    charge_residual: float
    neutral_density: float
    mean_energy_for_power: float | None


@dataclass(frozen=True, slots=True)
class PowerEvaluation:
    coupling: PowerCouplingResult | None
    kinetics_by_zone: dict[str, ElectronKineticsResult]
    electron_power_W_by_zone: Mapping[str, float]
    gas_power_W_by_zone: Mapping[str, float]
    reduced_field_Td_by_zone: dict[str, float]
    electron_states: dict[str, ElectronState]


@dataclass(frozen=True, slots=True)
class TransportEvaluation:
    density_rhs: np.ndarray
    electron_energy_rhs: np.ndarray | None
    heavy_energy_rhs: np.ndarray | None
    inlet_heavy_energy: np.ndarray


@dataclass(frozen=True, slots=True)
class ReactionWallEvaluation:
    reaction_rates_by_zone: np.ndarray
    electron_energy_transfer: np.ndarray
    gas_reaction_heating: np.ndarray
    wall_species_rhs: np.ndarray
    wall_energy_loss: np.ndarray
    wall_records_by_zone: dict[str, list[WallFluxRecord]]
    ion_flux_m2_s: dict[tuple[str, str], float]
    ion_energy_eV: dict[str, float]


@dataclass(frozen=True, slots=True)
class ZoneReactionWallEvaluation:
    rates: np.ndarray
    electron_energy_transfer: float
    gas_reaction_heating: float
    wall_species_rhs: np.ndarray
    wall_energy_loss: float
    wall_records: list[WallFluxRecord]
    ion_flux_m2_s: dict[tuple[str, str], float]
    ion_energy_eV: dict[str, float]


@dataclass(frozen=True, slots=True)
class SurfaceTerms:
    evaluation: SurfaceEvaluation | None
    rates_by_zone: dict[str, dict[str, float]]


@dataclass(frozen=True, slots=True)
class EnergyTerms:
    wall_heavy_exchange: np.ndarray
    elastic_heating: np.ndarray


@dataclass(frozen=True, slots=True)
class RuntimeZoneLedger:
    zone_id: str
    reaction_rates_m3_s: dict[str, float]
    absorbed_power_J_m3_s: float
    reaction_energy_loss_J_m3_s: float
    wall_energy_loss_J_m3_s: float
    gas_power_J_m3_s: float
    gas_reaction_heating_J_m3_s: float
    surface_reaction_heating_J_m3_s: float
    wall_heavy_energy_exchange_J_m3_s: float
    elastic_heating_J_m3_s: float
    transport_species_source_m3_s: dict[str, float]
    transport_electron_energy_J_m3_s: float
    transport_heavy_energy_J_m3_s: float
    inlet_heavy_energy_J_m3_s: float
    surface_rates_m2_s: Mapping[str, float]
    wall_fluxes: tuple[WallFluxRecord, ...]
    wall_species_energy_J_m3_s: float
    surface_species_energy_J_m3_s: float


@dataclass(frozen=True, slots=True)
class RuntimeEvaluation:
    derivative: np.ndarray
    electron_states: Mapping[str, ElectronState]
    kinetics_by_zone: Mapping[str, ElectronKineticsResult]
    gas_temperature_K_by_zone: Mapping[str, float]
    charge_residual_m3_by_zone: Mapping[str, float]
    ledger_by_zone: Mapping[str, RuntimeZoneLedger]
    power_coupling: PowerCouplingResult | None


ExtensionRHS = Callable[
    [PreparedEvaluationState, ReactionWallEvaluation, SurfaceTerms],
    np.ndarray | None,
]


@dataclass(frozen=True, slots=True)
class RuntimeAssembly:
    """Combine already evaluated physics terms into one state derivative."""

    model: CompiledGlobalModel
    extension_rhs: ExtensionRHS
    prepared: PreparedEvaluationState
    power: PowerEvaluation
    transport: TransportEvaluation
    reactions: ReactionWallEvaluation
    surface: SurfaceTerms
    energy: EnergyTerms
    collect_ledger: bool

    def assemble(self) -> tuple[np.ndarray, dict[str, RuntimeZoneLedger]]:
        derivative = np.zeros_like(self.prepared.values)
        surface_gas_rhs = self._surface_gas_rhs()
        surface_heating = self._surface_heating()
        wall_species_energy = self._species_internal_energy(
            self.reactions.wall_species_rhs
        )
        surface_species_energy = self._species_internal_energy(surface_gas_rhs)
        if self.surface.evaluation is not None:
            derivative[self.model.layout.surface_coverage_slice] = (
                self.surface.evaluation.coverage_derivative_s_inv
            )
        extension_rhs = self.extension_rhs(self.prepared, self.reactions, self.surface)
        if extension_rhs is not None:
            derivative[self.model.layout.extension_slice] = extension_rhs
        ledger_by_zone: dict[str, RuntimeZoneLedger] = {}
        for zone_index, zone in enumerate(self.model.zones):
            ledger = self._assemble_zone_derivative(
                derivative=derivative,
                zone_index=zone_index,
                surface_gas_rhs=surface_gas_rhs,
                surface_heating=surface_heating,
                wall_species_energy=wall_species_energy,
                surface_species_energy=surface_species_energy,
            )
            if ledger is not None:
                ledger_by_zone[zone.zone_id] = ledger
        return derivative, ledger_by_zone

    def _surface_gas_rhs(self) -> np.ndarray:
        if self.surface.evaluation is None:
            return np.zeros_like(self.prepared.density_by_zone)
        return self.surface.evaluation.gas_derivative_m3_s

    def _surface_heating(self) -> np.ndarray:
        if self.surface.evaluation is None:
            return np.zeros(len(self.model.zones))
        return self.surface.evaluation.gas_heating_J_m3_s

    def _species_internal_energy(self, species_rhs: np.ndarray) -> np.ndarray:
        closure = self.model.heavy_energy_closure
        if closure is None:
            return np.zeros(len(self.model.zones))
        return (
            BOLTZMANN_J_K
            * self.prepared.gas_temperature_by_zone
            * (species_rhs @ closure.cv_over_kb)
        )

    def _assemble_zone_derivative(
        self,
        *,
        derivative: np.ndarray,
        zone_index: int,
        surface_gas_rhs: np.ndarray,
        surface_heating: np.ndarray,
        wall_species_energy: np.ndarray,
        surface_species_energy: np.ndarray,
    ) -> RuntimeZoneLedger | None:
        zone = self.model.zones[zone_index]
        density_slice = self.model.layout.density_slices[zone.zone_id]
        rates = self.reactions.reaction_rates_by_zone[zone_index]
        derivative[density_slice] = (
            self.model.stoichiometry.T @ rates
            + self.transport.density_rhs[zone_index]
            + self.reactions.wall_species_rhs[zone_index]
            + surface_gas_rhs[zone_index]
        )
        power_density = (
            self.power.electron_power_W_by_zone.get(zone.zone_id, 0.0) / zone.volume_m3
        )
        gas_power_density = (
            self.power.gas_power_W_by_zone.get(zone.zone_id, 0.0) / zone.volume_m3
        )
        electron_transport = self._optional_value(
            self.transport.electron_energy_rhs, zone_index
        )
        heavy_transport = self._optional_value(
            self.transport.heavy_energy_rhs, zone_index
        )
        self._write_zone_energy_derivatives(
            derivative=derivative,
            zone_index=zone_index,
            power_density=power_density,
            gas_power_density=gas_power_density,
            electron_transport=electron_transport,
            heavy_transport=heavy_transport,
            surface_heating=surface_heating,
            wall_species_energy=wall_species_energy,
            surface_species_energy=surface_species_energy,
        )
        if not self.collect_ledger:
            return None
        return self._zone_ledger(
            zone_index=zone_index,
            rates=rates,
            power_density=power_density,
            gas_power_density=gas_power_density,
            electron_transport=electron_transport,
            heavy_transport=heavy_transport,
            surface_heating=surface_heating,
            wall_species_energy=wall_species_energy,
            surface_species_energy=surface_species_energy,
        )

    def _write_zone_energy_derivatives(
        self,
        *,
        derivative: np.ndarray,
        zone_index: int,
        power_density: float,
        gas_power_density: float,
        electron_transport: float,
        heavy_transport: float,
        surface_heating: np.ndarray,
        wall_species_energy: np.ndarray,
        surface_species_energy: np.ndarray,
    ) -> None:
        zone_id = self.model.zones[zone_index].zone_id
        if self.model.layout.evolves_electron_energy:
            derivative[self.model.layout.electron_energy_indices[zone_id]] = (
                power_density
                + self.reactions.electron_energy_transfer[zone_index]
                - self.reactions.wall_energy_loss[zone_index]
                - self.energy.elastic_heating[zone_index]
                + electron_transport
            )
        if self.model.layout.evolves_heavy_energy:
            derivative[self.model.layout.heavy_energy_indices[zone_id]] = (
                gas_power_density
                + self.reactions.gas_reaction_heating[zone_index]
                + surface_heating[zone_index]
                + self.energy.wall_heavy_exchange[zone_index]
                + self.energy.elastic_heating[zone_index]
                + wall_species_energy[zone_index]
                + surface_species_energy[zone_index]
                + heavy_transport
            )

    @staticmethod
    def _optional_value(values: np.ndarray | None, index: int) -> float:
        return 0.0 if values is None else float(values[index])

    def _zone_ledger(
        self,
        *,
        zone_index: int,
        rates: np.ndarray,
        power_density: float,
        gas_power_density: float,
        electron_transport: float,
        heavy_transport: float,
        surface_heating: np.ndarray,
        wall_species_energy: np.ndarray,
        surface_species_energy: np.ndarray,
    ) -> RuntimeZoneLedger:
        zone = self.model.zones[zone_index]
        return RuntimeZoneLedger(
            zone_id=zone.zone_id,
            reaction_rates_m3_s={
                reaction_id: float(rate)
                for reaction_id, rate in zip(
                    self.model.reaction_ids, rates, strict=True
                )
            },
            absorbed_power_J_m3_s=power_density,
            reaction_energy_loss_J_m3_s=float(
                -self.reactions.electron_energy_transfer[zone_index]
            ),
            wall_energy_loss_J_m3_s=float(self.reactions.wall_energy_loss[zone_index]),
            gas_power_J_m3_s=gas_power_density,
            gas_reaction_heating_J_m3_s=float(
                self.reactions.gas_reaction_heating[zone_index]
            ),
            surface_reaction_heating_J_m3_s=float(surface_heating[zone_index]),
            wall_heavy_energy_exchange_J_m3_s=float(
                self.energy.wall_heavy_exchange[zone_index]
            ),
            elastic_heating_J_m3_s=float(self.energy.elastic_heating[zone_index]),
            transport_species_source_m3_s={
                species_id: float(source)
                for species_id, source in zip(
                    self.model.species_ids,
                    self.transport.density_rhs[zone_index],
                    strict=True,
                )
            },
            transport_electron_energy_J_m3_s=electron_transport,
            transport_heavy_energy_J_m3_s=heavy_transport,
            inlet_heavy_energy_J_m3_s=float(
                self.transport.inlet_heavy_energy[zone_index]
            ),
            surface_rates_m2_s=self.surface.rates_by_zone[zone.zone_id],
            wall_fluxes=tuple(self.reactions.wall_records_by_zone[zone.zone_id]),
            wall_species_energy_J_m3_s=float(wall_species_energy[zone_index]),
            surface_species_energy_J_m3_s=float(surface_species_energy[zone_index]),
        )


__all__ = [
    "EnergyTerms",
    "PowerEvaluation",
    "PreparedEvaluationState",
    "PreparedZoneState",
    "ReactionWallEvaluation",
    "RuntimeAssembly",
    "RuntimeEvaluation",
    "RuntimeZoneLedger",
    "SurfaceTerms",
    "TransportEvaluation",
    "ZoneReactionWallEvaluation",
]
