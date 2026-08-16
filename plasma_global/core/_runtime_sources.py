"""Evaluate transport, chemistry, boundary, and energy source terms."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from plasma_global.core._runtime_assembly import (
    EnergyTerms,
    PowerEvaluation,
    PreparedEvaluationState,
    ReactionWallEvaluation,
    SurfaceTerms,
    TransportEvaluation,
    ZoneReactionWallEvaluation,
)
from plasma_global.core.domain import RecipeSegment, Zone
from plasma_global.errors import ModelConfigurationError, StateDomainError
from plasma_global.models.electrons import ELEMENTARY_CHARGE_C
from plasma_global.models.rates import DensityView, RateContext, RateEvaluator
from plasma_global.models.walls import (
    CompiledWallBoundary,
    WallEvaluation,
    WallFluxRecord,
    evaluate_compiled_wall_boundary,
)

if TYPE_CHECKING:
    from plasma_global.core.compiled import CompiledGlobalModel


def _mass_action_rate(
    *,
    reaction_id: str,
    evaluator: RateEvaluator,
    context: RateContext,
    density: np.ndarray,
    reactant_powers: tuple[tuple[int, float], ...],
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
    for species_index, order in reactant_powers:
        rate *= density[species_index] ** order
    if not math.isfinite(rate) or rate < 0.0:
        raise StateDomainError(
            f"Mass-action rate for {reaction_id!r} is invalid: {rate!r}"
        )
    return rate


@dataclass(frozen=True, slots=True)
class RuntimeSourceEvaluator:
    """Evaluate source terms after state and electron conditions are resolved."""

    model: CompiledGlobalModel

    def evaluate_transport(
        self, segment: RecipeSegment, prepared: PreparedEvaluationState
    ) -> TransportEvaluation:
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
        return TransportEvaluation(
            density_rhs=density_rhs,
            electron_energy_rhs=electron_rhs,
            heavy_energy_rhs=heavy_rhs,
            inlet_heavy_energy=inlet_heavy_energy,
        )

    def evaluate_reactions_and_walls(
        self,
        time_s: float,
        prepared: PreparedEvaluationState,
        power: PowerEvaluation,
        *,
        collect_ledger: bool,
    ) -> ReactionWallEvaluation:
        zone_count = len(self.model.zones)
        reaction_rates = np.zeros((zone_count, len(self.model.reaction_ids)))
        electron_energy_transfer = np.zeros(zone_count)
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
            electron_energy_transfer[zone_index] = result.electron_energy_transfer
            gas_reaction_heating[zone_index] = result.gas_reaction_heating
            wall_species_rhs[zone_index] = result.wall_species_rhs
            wall_energy_loss[zone_index] = result.wall_energy_loss
            wall_records_by_zone[zone.zone_id] = result.wall_records
            for key, flux in result.ion_flux_m2_s.items():
                ion_flux_m2_s[key] = ion_flux_m2_s.get(key, 0.0) + flux
            ion_energy_eV.update(result.ion_energy_eV)
        return ReactionWallEvaluation(
            reaction_rates_by_zone=reaction_rates,
            electron_energy_transfer=electron_energy_transfer,
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
        prepared: PreparedEvaluationState,
        power: PowerEvaluation,
        collect_ledger: bool,
    ) -> ZoneReactionWallEvaluation:
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
            self.model._active_reaction_indices_by_zone[zone.zone_id],
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
        return ZoneReactionWallEvaluation(
            rates=rates,
            electron_energy_transfer=float(
                self.model.electron_energy_transfer_eV @ rates
            )
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
        wall: WallEvaluation,
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

    def evaluate_surface_terms(
        self,
        segment: RecipeSegment,
        prepared: PreparedEvaluationState,
        reactions: ReactionWallEvaluation,
        *,
        collect_ledger: bool,
    ) -> SurfaceTerms:
        rates_by_zone: dict[str, dict[str, float]] = (
            {zone.zone_id: {} for zone in self.model.zones} if collect_ledger else {}
        )
        if self.model.surface_model is None:
            return SurfaceTerms(None, rates_by_zone)
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
        return SurfaceTerms(evaluation, rates_by_zone)

    def evaluate_energy_terms(
        self,
        segment: RecipeSegment,
        prepared: PreparedEvaluationState,
        power: PowerEvaluation,
    ) -> EnergyTerms:
        return EnergyTerms(
            wall_heavy_exchange=self._wall_heavy_exchange(segment, prepared),
            elastic_heating=self._elastic_heating(prepared, power),
        )

    def _wall_heavy_exchange(
        self, segment: RecipeSegment, prepared: PreparedEvaluationState
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
        self, prepared: PreparedEvaluationState, power: PowerEvaluation
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

    def extension_rhs(
        self,
        prepared: PreparedEvaluationState,
        reactions: ReactionWallEvaluation,
        surface: SurfaceTerms,
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

    @staticmethod
    def _optional_zero_zone_array(enabled: bool, zone_count: int) -> np.ndarray | None:
        return np.zeros(zone_count, dtype=float) if enabled else None

    def _reaction_rates(
        self,
        context: RateContext,
        density: np.ndarray,
        active_reaction_indices: tuple[int, ...],
    ) -> np.ndarray:
        rates = np.zeros(len(self.model.reaction_ids), dtype=float)
        for reaction_index in active_reaction_indices:
            evaluator = self.model.rate_evaluators[reaction_index]
            rates[reaction_index] = _mass_action_rate(
                reaction_id=self.model.reaction_ids[reaction_index],
                evaluator=evaluator,
                context=context,
                density=density,
                reactant_powers=(
                    self.model._chemistry_data.reactant_powers_by_reaction[
                        reaction_index
                    ]
                ),
                electron_order=float(self.model.electron_orders[reaction_index]),
            )
        return rates


__all__ = ["RuntimeSourceEvaluator"]
