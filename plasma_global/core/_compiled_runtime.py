"""Prepare, evaluate, and assemble the compiled physical runtime."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from plasma_global.core.domain import RecipeSegment, Zone
from plasma_global.core.exceptions import ModelConfigurationError, StateDomainError
from plasma_global.models.electrons import (
    ELEMENTARY_CHARGE_C,
    ElectronState,
    resolved_electron_density,
)
from plasma_global.models.kinetics import ElectronKineticsResult
from plasma_global.models.power import PowerCouplingResult
from plasma_global.models.rates import DensityView, RateContext, RateEvaluator
from plasma_global.models.surface import SurfaceEvaluation
from plasma_global.models.walls import (
    CompiledWallBoundary,
    WallFluxRecord,
    evaluate_compiled_wall_boundary,
)

if TYPE_CHECKING:
    from plasma_global.core.compiled import (
        CompiledGlobalModel,
        ModelEvaluation,
        ZoneTermLedger,
    )


@dataclass(frozen=True, slots=True)
class _PreparedEvaluationState:
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
class _PreparedZoneState:
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
class _PowerEvaluation:
    coupling: PowerCouplingResult | None
    kinetics_by_zone: dict[str, ElectronKineticsResult]
    electron_power_W_by_zone: Mapping[str, float]
    gas_power_W_by_zone: Mapping[str, float]
    reduced_field_Td_by_zone: dict[str, float]
    electron_states: dict[str, ElectronState]


@dataclass(frozen=True, slots=True)
class _TransportEvaluation:
    density_rhs: np.ndarray
    electron_energy_rhs: np.ndarray | None
    heavy_energy_rhs: np.ndarray | None
    inlet_heavy_energy: np.ndarray


@dataclass(frozen=True, slots=True)
class _ReactionWallEvaluation:
    reaction_rates_by_zone: np.ndarray
    reaction_energy_loss: np.ndarray
    gas_reaction_heating: np.ndarray
    wall_species_rhs: np.ndarray
    wall_energy_loss: np.ndarray
    wall_records_by_zone: dict[str, list[WallFluxRecord]]
    ion_flux_m2_s: dict[tuple[str, str], float]
    ion_energy_eV: dict[str, float]


@dataclass(frozen=True, slots=True)
class _ZoneReactionWallEvaluation:
    rates: np.ndarray
    reaction_energy_loss: float
    gas_reaction_heating: float
    wall_species_rhs: np.ndarray
    wall_energy_loss: float
    wall_records: list[WallFluxRecord]
    ion_flux_m2_s: dict[tuple[str, str], float]
    ion_energy_eV: dict[str, float]


@dataclass(frozen=True, slots=True)
class _SurfaceTerms:
    evaluation: SurfaceEvaluation | None
    rates_by_zone: dict[str, dict[str, float]]


@dataclass(frozen=True, slots=True)
class _EnergyTerms:
    wall_heavy_exchange: np.ndarray
    elastic_heating: np.ndarray


def _mass_action_rate(
    *,
    reaction_id: str,
    evaluator: RateEvaluator,
    context: RateContext,
    density: np.ndarray,
    reactant_orders: np.ndarray,
    electron_order: float,
) -> float:
    """Evaluate one active reaction without undefined zero-electron lookups."""

    if electron_order > 0.0 and context.electron_density_m3 == 0.0:
        return 0.0
    coefficient = float(cast(Any, evaluator(context)))
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise StateDomainError(
            f"Rate evaluator for {reaction_id!r} returned {coefficient!r}"
        )
    rate = coefficient
    if electron_order != 0.0:
        rate *= context.electron_density_m3**electron_order
    for species_index, order in enumerate(reactant_orders):
        if order != 0.0:
            rate *= density[species_index] ** order
    if not math.isfinite(rate) or rate < 0.0:
        raise StateDomainError(
            f"Mass-action rate for {reaction_id!r} is invalid: {rate!r}"
        )
    return rate


@dataclass(frozen=True, slots=True)
class RuntimeEvaluator:
    """Evaluate one compiled model state without expanding its public facade."""

    model: CompiledGlobalModel

    def _electron_density(
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

    def evaluate(
        self,
        time_s: float,
        state: np.ndarray,
        segment: RecipeSegment,
        *,
        collect_ledger: bool = True,
    ) -> ModelEvaluation:
        """Evaluate the diagnostic view used by output and audit paths."""

        from plasma_global.core.compiled import ModelEvaluation

        result = self._evaluate(
            time_s,
            state,
            segment,
            collect_ledger=collect_ledger,
            derivative_only=False,
        )
        if not isinstance(result, ModelEvaluation):
            raise RuntimeError("diagnostic evaluation returned only a derivative")
        return result

    def evaluate_derivative(
        self,
        time_s: float,
        state: np.ndarray,
        segment: RecipeSegment,
        *,
        domain_atol: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate only dy/dt without materializing diagnostic wrappers."""

        result = self._evaluate(
            time_s,
            state,
            segment,
            collect_ledger=False,
            derivative_only=True,
            domain_atol=domain_atol,
        )
        if not isinstance(result, np.ndarray):
            raise RuntimeError("derivative evaluation returned diagnostic data")
        return result

    def _prepare_zone_evaluation(
        self,
        *,
        time_s: float,
        values: np.ndarray,
        domain_atol: np.ndarray,
        segment: RecipeSegment,
        zone_index: int,
        zone: Zone,
    ) -> _PreparedZoneState:
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
        electron_density = self._electron_density(
            time_s, zone.zone_id, net_charge, segment
        )
        electron_energy, closure_energy, mean_energy = self._prepare_electron_energy(
            values=values,
            domain_atol=domain_atol,
            segment=segment,
            zone_index=zone_index,
            zone=zone,
            net_charge=net_charge,
            electron_density=electron_density,
        )
        heavy_energy = self._prepare_heavy_energy(
            values=values, domain_atol=domain_atol, zone=zone
        )
        return _PreparedZoneState(
            density=density,
            reaction_density=reaction_density,
            electron_energy=electron_energy,
            closure_electron_energy=closure_energy,
            heavy_energy=heavy_energy,
            net_charge=net_charge,
            electron_density=electron_density,
            charge_residual=raw_net_charge - electron_density,
            neutral_density=float(np.sum(reaction_density[self.model.charges == 0.0])),
            mean_energy_for_power=mean_energy,
        )

    def _prepare_electron_energy(
        self,
        *,
        values: np.ndarray,
        domain_atol: np.ndarray,
        segment: RecipeSegment,
        zone_index: int,
        zone: Zone,
        net_charge: float,
        electron_density: float,
    ) -> tuple[float | None, float | None, float | None]:
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
        initial_electrons = self.model.electron_closure.evaluate(
            net_heavy_charge_density_m3=net_charge,
            energy_density_J_m3=closure_energy,
            reduced_field_Td=segment.reduced_field_Td_by_zone.get(zone.zone_id),
            electron_density_m3=electron_density,
        )
        del zone_index
        return energy, closure_energy, initial_electrons.mean_energy_eV

    def _prepare_heavy_energy(
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

    def _prepare_evaluation_state(
        self,
        time_s: float,
        state: np.ndarray,
        segment: RecipeSegment,
        domain_atol: np.ndarray | None,
    ) -> _PreparedEvaluationState:
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
        for zone_index, zone in enumerate(self.model.zones):
            prepared = self._prepare_zone_evaluation(
                time_s=time_s,
                values=values,
                domain_atol=active_domain_atol,
                segment=segment,
                zone_index=zone_index,
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
        gas_temperature_by_zone = self._gas_temperatures(
            density_by_zone, heavy_energy_by_zone
        )
        return _PreparedEvaluationState(
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
            gas_temperature_by_zone=gas_temperature_by_zone,
        )

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

    def _evaluate_power(
        self,
        time_s: float,
        segment: RecipeSegment,
        prepared: _PreparedEvaluationState,
    ) -> _PowerEvaluation:
        if self.model.power_coordinator is None:
            coupling = None
            electron_power = segment.absorbed_power_W_by_zone
            gas_power: Mapping[str, float] = MappingProxyType({})
            reduced_field = dict(segment.reduced_field_Td_by_zone)
            kinetics = self._uncoordinated_kinetics(prepared, reduced_field)
        else:
            field_models: dict[str, Any] = {}
            field_model = self.model.electron_closure.mean_energy_from_field
            if callable(field_model):
                field_models = {zone.zone_id: field_model for zone in self.model.zones}
            coupling = self.model.power_coordinator.evaluate(
                time_s=time_s,
                commands=segment.port_commands,
                electron_density_m3_by_zone=prepared.electron_density_by_zone,
                neutral_density_m3_by_zone=prepared.neutral_density_by_zone,
                mean_energy_eV_by_zone=prepared.mean_energy_for_power,
                prescribed_electron_power_W_by_zone=segment.absorbed_power_W_by_zone,
                prescribed_reduced_field_Td_by_zone=segment.reduced_field_Td_by_zone,
                kinetics_by_zone=self.model.electron_kinetics_by_zone,
                mean_energy_from_field_by_zone=field_models,
            )
            electron_power = coupling.electron_power_W_by_zone
            gas_power = coupling.gas_power_W_by_zone
            reduced_field = dict(coupling.reduced_field_Td_by_zone)
            kinetics = dict(coupling.kinetics_by_zone)
        electron_states = self._evaluate_electron_states(
            prepared, reduced_field, kinetics
        )
        return _PowerEvaluation(
            coupling=coupling,
            kinetics_by_zone=kinetics,
            electron_power_W_by_zone=electron_power,
            gas_power_W_by_zone=gas_power,
            reduced_field_Td_by_zone=reduced_field,
            electron_states=electron_states,
        )

    def _uncoordinated_kinetics(
        self,
        prepared: _PreparedEvaluationState,
        reduced_field_by_zone: dict[str, float],
    ) -> dict[str, ElectronKineticsResult]:
        results: dict[str, ElectronKineticsResult] = {}
        for zone in self.model.zones:
            table = self.model.electron_kinetics_by_zone.get(zone.zone_id)
            if table is None:
                continue
            if table.lookup == "mean_energy":
                mean_energy = prepared.mean_energy_for_power[zone.zone_id]
                if mean_energy is None:
                    raise StateDomainError(
                        f"Zone {zone.zone_id!r} has no mean energy for table lookup"
                    )
                if (
                    prepared.electron_density_by_zone[zone.zone_id] == 0.0
                    and mean_energy == 0.0
                ):
                    # No electron property or electron-driven rate is defined or
                    # needed in this state.  In particular, do not force zero onto
                    # a positive, tail-safe mean-energy lookup axis.
                    continue
                result = table.evaluate(mean_energy_eV=mean_energy)
                reduced_field_by_zone.setdefault(
                    zone.zone_id, result.effective_field_Td
                )
            else:
                field_value = reduced_field_by_zone.get(zone.zone_id)
                if field_value is None:
                    raise StateDomainError(
                        f"Zone {zone.zone_id!r} has no E/N for local-field table lookup"
                    )
                result = table.evaluate(reduced_field_Td=field_value)
            results[zone.zone_id] = result
        return results

    def _evaluate_electron_states(
        self,
        prepared: _PreparedEvaluationState,
        reduced_field_by_zone: Mapping[str, float],
        kinetics_by_zone: Mapping[str, ElectronKineticsResult],
    ) -> dict[str, ElectronState]:
        states: dict[str, ElectronState] = {}
        for zone_index, zone in enumerate(self.model.zones):
            field_value = reduced_field_by_zone.get(zone.zone_id)
            kinetics = kinetics_by_zone.get(zone.zone_id)
            electron_density = prepared.electron_density_by_zone[zone.zone_id]
            if (
                self.model.electron_closure.mode == "local_field"
                and kinetics is not None
            ):
                states[zone.zone_id] = self._local_field_electron_state(
                    electron_density, field_value, kinetics
                )
                continue
            energy_density = (
                None
                if prepared.closure_electron_energy_by_zone is None
                else float(prepared.closure_electron_energy_by_zone[zone_index])
            )
            states[zone.zone_id] = self.model.electron_closure.evaluate(
                net_heavy_charge_density_m3=prepared.net_charge_by_zone[zone.zone_id],
                energy_density_J_m3=energy_density,
                reduced_field_Td=field_value,
                electron_density_m3=electron_density,
            )
        return states

    @staticmethod
    def _local_field_electron_state(
        electron_density: float,
        field_value: float | None,
        kinetics: ElectronKineticsResult,
    ) -> ElectronState:
        mean_energy = kinetics.mean_energy_eV
        return ElectronState(
            density_m3=electron_density,
            mean_energy_eV=mean_energy,
            temperature_eV=kinetics.electron_temperature_eV,
            energy_density_J_m3=(electron_density * ELEMENTARY_CHARGE_C * mean_energy),
            reduced_field_Td=(
                field_value if field_value is not None else kinetics.effective_field_Td
            ),
        )

    def _evaluate_transport(
        self, segment: RecipeSegment, prepared: _PreparedEvaluationState
    ) -> _TransportEvaluation:
        zone_count = len(self.model.zones)
        density_rhs = np.zeros_like(prepared.density_by_zone)
        electron_rhs = self._optional_zero_zone_array(
            self.model.layout.evolves_electron_energy, zone_count
        )
        heavy_rhs = self._optional_zero_zone_array(
            self.model.layout.evolves_heavy_energy, zone_count
        )
        inlet_heavy_energy = np.zeros(zone_count, dtype=float)
        if self.model.transport is not None:
            forcing = self.model._transport_by_segment[segment.segment_id]
            density_rhs, electron_rhs, heavy_rhs = self.model.transport.evaluate(
                prepared.density_by_zone,
                forcing,
                electron_energy_J_m3=prepared.electron_energy_by_zone,
                heavy_energy_J_m3=prepared.heavy_energy_by_zone,
                heavy_flow_densities_m3=prepared.reaction_density_by_zone,
            )
            inlet_heavy_energy = forcing.inlet_heavy_energy_J_m3_s
        return _TransportEvaluation(
            density_rhs=density_rhs,
            electron_energy_rhs=electron_rhs,
            heavy_energy_rhs=heavy_rhs,
            inlet_heavy_energy=inlet_heavy_energy,
        )

    @staticmethod
    def _optional_zero_zone_array(enabled: bool, zone_count: int) -> np.ndarray | None:
        return np.zeros(zone_count, dtype=float) if enabled else None

    def _evaluate_reactions_and_walls(
        self,
        time_s: float,
        prepared: _PreparedEvaluationState,
        power: _PowerEvaluation,
        *,
        collect_ledger: bool,
    ) -> _ReactionWallEvaluation:
        zone_count = len(self.model.zones)
        reaction_rates = np.zeros((zone_count, len(self.model.reaction_ids)))
        reaction_energy_loss = np.zeros(zone_count)
        gas_reaction_heating = np.zeros(zone_count)
        wall_species_rhs = np.zeros_like(prepared.density_by_zone)
        wall_energy_loss = np.zeros(zone_count)
        wall_records_by_zone: dict[str, list[WallFluxRecord]] = {}
        ion_flux_m2_s: dict[tuple[str, str], float] = {}
        ion_energy_eV: dict[str, float] = {}
        for zone_index, zone in enumerate(self.model.zones):
            result = self._evaluate_zone_reactions_and_walls(
                time_s=time_s,
                zone_index=zone_index,
                zone=zone,
                prepared=prepared,
                power=power,
                collect_ledger=collect_ledger,
            )
            reaction_rates[zone_index] = result.rates
            reaction_energy_loss[zone_index] = result.reaction_energy_loss
            gas_reaction_heating[zone_index] = result.gas_reaction_heating
            wall_species_rhs[zone_index] = result.wall_species_rhs
            wall_energy_loss[zone_index] = result.wall_energy_loss
            wall_records_by_zone[zone.zone_id] = result.wall_records
            for key, flux in result.ion_flux_m2_s.items():
                ion_flux_m2_s[key] = ion_flux_m2_s.get(key, 0.0) + flux
            ion_energy_eV.update(result.ion_energy_eV)
        return _ReactionWallEvaluation(
            reaction_rates_by_zone=reaction_rates,
            reaction_energy_loss=reaction_energy_loss,
            gas_reaction_heating=gas_reaction_heating,
            wall_species_rhs=wall_species_rhs,
            wall_energy_loss=wall_energy_loss,
            wall_records_by_zone=wall_records_by_zone,
            ion_flux_m2_s=ion_flux_m2_s,
            ion_energy_eV=ion_energy_eV,
        )

    def _evaluate_zone_reactions_and_walls(
        self,
        *,
        time_s: float,
        zone_index: int,
        zone: Zone,
        prepared: _PreparedEvaluationState,
        power: _PowerEvaluation,
        collect_ledger: bool,
    ) -> _ZoneReactionWallEvaluation:
        reaction_density = prepared.reaction_density_by_zone[zone_index]
        electrons = power.electron_states[zone.zone_id]
        kinetics = power.kinetics_by_zone.get(zone.zone_id)
        context = RateContext(
            time_s=time_s,
            zone_id=zone.zone_id,
            gas_temperature_K=float(prepared.gas_temperature_by_zone[zone_index]),
            pressure_Pa=(
                float(np.sum(reaction_density))
                * 1.380649e-23
                * float(prepared.gas_temperature_by_zone[zone_index])
            ),
            electron_density_m3=electrons.density_m3,
            mean_energy_eV=electrons.mean_energy_eV,
            electron_temperature_eV=electrons.temperature_eV,
            reduced_field_Td=electrons.reduced_field_Td,
            electron_mobility_m2_V_s=(
                None if kinetics is None else kinetics.mobility_m2_V_s
            ),
            rate_coefficients=(
                MappingProxyType({}) if kinetics is None else kinetics.rate_coefficients
            ),
            densities_m3=DensityView(
                self.model.species_ids, self.model._species_index, reaction_density
            ),
        )
        rates = self._reaction_rates(
            context,
            reaction_density,
            self.model._active_reactions_by_zone[zone.zone_id],
        )
        wall_rhs = np.zeros(len(self.model.species_ids), dtype=float)
        wall_loss = 0.0
        wall_records: list[WallFluxRecord] = []
        ion_flux: dict[tuple[str, str], float] = {}
        ion_energy: dict[str, float] = {}
        for compiled_wall in self.model._walls_by_zone[zone.zone_id]:
            wall = evaluate_compiled_wall_boundary(
                compiled=compiled_wall,
                volume_m3=zone.volume_m3,
                species_ids=self.model.species_ids,
                densities_m3=reaction_density,
                electrons=electrons,
                collect_records=collect_ledger,
            )
            wall_rhs += wall.species_derivative_m3_s
            wall_loss += wall.electron_energy_loss_J_m3_s
            wall_records.extend(wall.records)
            self._record_wall_ion_flux(compiled_wall, wall, ion_flux, ion_energy)
        return _ZoneReactionWallEvaluation(
            rates=rates,
            reaction_energy_loss=float(self.model.energy_loss_eV @ rates)
            * ELEMENTARY_CHARGE_C,
            gas_reaction_heating=float(self.model.gas_heating_eV @ rates)
            * ELEMENTARY_CHARGE_C,
            wall_species_rhs=wall_rhs,
            wall_energy_loss=wall_loss,
            wall_records=wall_records,
            ion_flux_m2_s=ion_flux,
            ion_energy_eV=ion_energy,
        )

    def _record_wall_ion_flux(
        self,
        compiled_wall: CompiledWallBoundary,
        wall: Any,
        ion_flux: dict[tuple[str, str], float],
        ion_energy: dict[str, float],
    ) -> None:
        surface_id = compiled_wall.boundary.surface_id
        if surface_id is None:
            return
        ion_energy[surface_id] = wall.sheath_energy_eV
        for species_index in compiled_wall.ion_indices:
            flux = float(wall.incident_flux_m2_s[int(species_index)])
            if flux != 0.0:
                key = (surface_id, self.model.species_ids[int(species_index)])
                ion_flux[key] = ion_flux.get(key, 0.0) + flux

    def _evaluate_surface_terms(
        self,
        segment: RecipeSegment,
        prepared: _PreparedEvaluationState,
        reactions: _ReactionWallEvaluation,
        *,
        collect_ledger: bool,
    ) -> _SurfaceTerms:
        rates_by_zone: dict[str, dict[str, float]] = (
            {zone.zone_id: {} for zone in self.model.zones} if collect_ledger else {}
        )
        if self.model.surface_model is None:
            return _SurfaceTerms(None, rates_by_zone)
        evaluation = self.model.surface_model.evaluate(
            prepared.values[self.model.layout.surface_coverage_slice],
            prepared.reaction_density_by_zone,
            prepared.gas_temperature_by_zone,
            ion_flux_m2_s=reactions.ion_flux_m2_s,
            ion_energy_eV=reactions.ion_energy_eV,
            surface_temperature_K=segment.surface_temperature_K_by_surface,
            domain_atol=prepared.domain_atol[self.model.layout.surface_coverage_slice],
            collect_rates=(
                collect_ledger or self.model.extension_accumulator is not None
            ),
        )
        if collect_ledger:
            surface_zone = {
                surface.surface_id: surface.zone_id
                for surface in self.model.surface_model.surfaces
            }
            for rate_id, rate in evaluation.rates_m2_s.items():
                surface_id = rate_id.rsplit("@", 1)[-1]
                rates_by_zone[surface_zone[surface_id]][rate_id] = rate
        return _SurfaceTerms(evaluation, rates_by_zone)

    def _evaluate_energy_terms(
        self,
        segment: RecipeSegment,
        prepared: _PreparedEvaluationState,
        power: _PowerEvaluation,
    ) -> _EnergyTerms:
        return _EnergyTerms(
            wall_heavy_exchange=self._wall_heavy_exchange(segment, prepared),
            elastic_heating=self._elastic_heating(prepared, power),
        )

    def _wall_heavy_exchange(
        self, segment: RecipeSegment, prepared: _PreparedEvaluationState
    ) -> np.ndarray:
        result = np.zeros(len(self.model.zones))
        if prepared.heavy_energy_by_zone is None:
            return result
        if self.model.heavy_energy_closure is None:
            raise ModelConfigurationError("Evolved gas energy needs its closure")
        wall_temperature = np.asarray(
            [
                segment.wall_temperature_K_by_zone.get(
                    zone.zone_id,
                    self.model.heavy_energy_closure.wall_temperature_K[zone_index],
                )
                for zone_index, zone in enumerate(self.model.zones)
            ],
            dtype=float,
        )
        equilibrium = self.model.heavy_energy_closure.energy_J_m3(
            prepared.density_by_zone, wall_temperature
        )
        return -self.model.heavy_energy_closure.wall_relaxation_s_inv * (
            prepared.heavy_energy_by_zone - equilibrium
        )

    def _elastic_heating(
        self, prepared: _PreparedEvaluationState, power: _PowerEvaluation
    ) -> np.ndarray:
        result = np.zeros(len(self.model.zones))
        if self.model.elastic_heating_evaluator is None:
            return result
        for zone_index, zone in enumerate(self.model.zones):
            value = float(
                cast(
                    Any,
                    self.model.elastic_heating_evaluator(
                        zone.zone_id,
                        power.electron_states[zone.zone_id],
                        prepared.reaction_density_by_zone[zone_index],
                        float(prepared.gas_temperature_by_zone[zone_index]),
                        power.kinetics_by_zone.get(zone.zone_id),
                    ),
                )
            )
            if not math.isfinite(value):
                raise StateDomainError(
                    f"Elastic heating evaluator returned {value!r} in zone "
                    f"{zone.zone_id!r}"
                )
            result[zone_index] = value
        return result

    def _extension_rhs(
        self,
        prepared: _PreparedEvaluationState,
        reactions: _ReactionWallEvaluation,
        surface: _SurfaceTerms,
    ) -> np.ndarray | None:
        if self.model.extension_accumulator is None:
            return None
        ion_flux_by_surface: dict[str, float] = {}
        for (surface_id, _species_id), flux in reactions.ion_flux_m2_s.items():
            ion_flux_by_surface[surface_id] = (
                ion_flux_by_surface.get(surface_id, 0.0) + flux
            )
        try:
            extension_rhs = np.asarray(
                self.model.extension_accumulator.rhs(
                    prepared.values[self.model.layout.extension_slice],
                    drivers={"ion_flux_m2_s": ion_flux_by_surface},
                    surface_rates_m2_s=(
                        MappingProxyType({})
                        if surface.evaluation is None
                        else surface.evaluation.rates_m2_s
                    ),
                ),
                dtype=float,
            )
        except (FloatingPointError, KeyError, ValueError) as exc:
            raise StateDomainError(
                f"Experimental accumulator state/domain failure: {exc}"
            ) from exc
        extension_slice = self.model.layout.extension_slice
        expected = extension_slice.stop - extension_slice.start
        if extension_rhs.shape != (expected,) or not np.all(np.isfinite(extension_rhs)):
            raise StateDomainError(
                "Experimental accumulator returned an invalid RHS vector"
            )
        return extension_rhs

    def _evaluate(
        self,
        time_s: float,
        state: np.ndarray,
        segment: RecipeSegment,
        *,
        collect_ledger: bool,
        derivative_only: bool,
        domain_atol: np.ndarray | None = None,
    ) -> ModelEvaluation | np.ndarray:
        prepared_state = self._prepare_evaluation_state(
            time_s, state, segment, domain_atol
        )

        power = self._evaluate_power(time_s, segment, prepared_state)
        transport = self._evaluate_transport(segment, prepared_state)
        reactions = self._evaluate_reactions_and_walls(
            time_s, prepared_state, power, collect_ledger=collect_ledger
        )
        surface = self._evaluate_surface_terms(
            segment, prepared_state, reactions, collect_ledger=collect_ledger
        )
        energy = self._evaluate_energy_terms(segment, prepared_state, power)
        derivative, ledger_by_zone = RuntimeAssembly(
            runtime=self,
            prepared=prepared_state,
            power=power,
            transport=transport,
            reactions=reactions,
            surface=surface,
            energy=energy,
            collect_ledger=collect_ledger,
        ).assemble()

        if derivative_only:
            derivative.setflags(write=False)
            return derivative
        from plasma_global.core.compiled import ModelEvaluation

        return ModelEvaluation(
            derivative=derivative,
            electron_states=power.electron_states,
            kinetics_by_zone=power.kinetics_by_zone,
            gas_temperature_K_by_zone={
                zone.zone_id: float(prepared_state.gas_temperature_by_zone[index])
                for index, zone in enumerate(self.model.zones)
            },
            charge_residual_m3_by_zone=prepared_state.charge_residual_by_zone,
            ledger_by_zone=ledger_by_zone,
            power_coupling=power.coupling,
        )

    def _reaction_rates(
        self,
        context: RateContext,
        density: np.ndarray,
        active_reactions: np.ndarray,
    ) -> np.ndarray:
        rates = np.empty(len(self.model.reaction_ids), dtype=float)
        for reaction_index, evaluator in enumerate(self.model.rate_evaluators):
            if not active_reactions[reaction_index]:
                rates[reaction_index] = 0.0
                continue
            rates[reaction_index] = _mass_action_rate(
                reaction_id=self.model.reaction_ids[reaction_index],
                evaluator=evaluator,
                context=context,
                density=density,
                reactant_orders=self.model.reactant_orders[reaction_index],
                electron_order=float(self.model.electron_orders[reaction_index]),
            )
        return rates


@dataclass(frozen=True, slots=True)
class RuntimeAssembly:
    """Combine already evaluated physics terms into one state derivative."""

    runtime: RuntimeEvaluator
    prepared: _PreparedEvaluationState
    power: _PowerEvaluation
    transport: _TransportEvaluation
    reactions: _ReactionWallEvaluation
    surface: _SurfaceTerms
    energy: _EnergyTerms
    collect_ledger: bool

    def assemble(self) -> tuple[np.ndarray, dict[str, ZoneTermLedger]]:
        model = self.runtime.model
        derivative = np.zeros_like(self.prepared.values)
        surface_gas_rhs = self._surface_gas_rhs()
        surface_heating = self._surface_heating()
        if self.surface.evaluation is not None:
            derivative[model.layout.surface_coverage_slice] = (
                self.surface.evaluation.coverage_derivative_s_inv
            )
        extension_rhs = self.runtime._extension_rhs(
            self.prepared, self.reactions, self.surface
        )
        if extension_rhs is not None:
            derivative[model.layout.extension_slice] = extension_rhs
        ledger_by_zone: dict[str, ZoneTermLedger] = {}
        for zone_index, zone in enumerate(model.zones):
            ledger = self._assemble_zone_derivative(
                derivative=derivative,
                zone_index=zone_index,
                surface_gas_rhs=surface_gas_rhs,
                surface_heating=surface_heating,
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
            return np.zeros(len(self.runtime.model.zones))
        return self.surface.evaluation.gas_heating_J_m3_s

    def _assemble_zone_derivative(
        self,
        *,
        derivative: np.ndarray,
        zone_index: int,
        surface_gas_rhs: np.ndarray,
        surface_heating: np.ndarray,
    ) -> ZoneTermLedger | None:
        model = self.runtime.model
        zone = model.zones[zone_index]
        density_slice = model.layout.density_slices[zone.zone_id]
        rates = self.reactions.reaction_rates_by_zone[zone_index]
        derivative[density_slice] = (
            model.stoichiometry.T @ rates
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
        if model.layout.evolves_electron_energy:
            derivative[model.layout.electron_energy_indices[zone.zone_id]] = (
                power_density
                - self.reactions.reaction_energy_loss[zone_index]
                - self.reactions.wall_energy_loss[zone_index]
                - self.energy.elastic_heating[zone_index]
                + electron_transport
            )
        if model.layout.evolves_heavy_energy:
            derivative[model.layout.heavy_energy_indices[zone.zone_id]] = (
                gas_power_density
                + self.reactions.gas_reaction_heating[zone_index]
                + surface_heating[zone_index]
                + self.energy.wall_heavy_exchange[zone_index]
                + self.energy.elastic_heating[zone_index]
                + heavy_transport
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
    ) -> ZoneTermLedger:
        # Local import avoids a module cycle while preserving the public class's
        # identity and pickle path in ``plasma_global.core.compiled``.
        from plasma_global.core.compiled import ZoneTermLedger

        model = self.runtime.model
        zone = model.zones[zone_index]
        return ZoneTermLedger(
            zone_id=zone.zone_id,
            reaction_rates_m3_s={
                reaction_id: float(rate)
                for reaction_id, rate in zip(model.reaction_ids, rates, strict=True)
            },
            absorbed_power_J_m3_s=power_density,
            reaction_energy_loss_J_m3_s=float(
                self.reactions.reaction_energy_loss[zone_index]
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
                    model.species_ids,
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
        )


__all__ = ["RuntimeEvaluator"]
