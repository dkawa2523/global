"""Cross-component validation for a fully compiled global model."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from plasma_global.core.domain import RecipeSegment
from plasma_global.core.exceptions import ModelConfigurationError
from plasma_global.models.walls import BoundaryReaction, WallBoundary

if TYPE_CHECKING:
    from plasma_global.core.compiled import CompiledGlobalModel


@dataclass(frozen=True, slots=True)
class DomainValidator:
    """Validate references and shapes spanning otherwise independent models."""

    model: CompiledGlobalModel

    def validate(self) -> None:
        zone_ids, known_zones = self._identifiers()
        self._optional_models()
        known_surfaces = self._surfaces(zone_ids, known_zones)
        self._transport_and_power(zone_ids)
        self._kinetics_and_reactions(known_zones)
        self._segments(known_zones, known_surfaces)
        self._electron_closure()
        self._wall_boundaries(known_zones)

    def _identifiers(self) -> tuple[tuple[str, ...], set[str]]:
        model = self.model
        if not model.zones:
            raise ModelConfigurationError("At least one zone is required")
        zone_ids = tuple(zone.zone_id for zone in model.zones)
        if len(set(zone_ids)) != len(zone_ids):
            raise ModelConfigurationError("Zone IDs must be unique")
        if not model.segments:
            raise ModelConfigurationError("At least one recipe segment is required")
        segment_ids = tuple(segment.segment_id for segment in model.segments)
        if len(set(segment_ids)) != len(segment_ids):
            raise ModelConfigurationError("Recipe segment IDs must be unique")
        return zone_ids, set(zone_ids)

    def _optional_models(self) -> None:
        self._heavy_energy_model()
        model = self.model
        callbacks = (
            ("elastic_heating_evaluator", model.elastic_heating_evaluator),
            ("electron_density_provider", model.electron_density_provider),
        )
        for name, callback in callbacks:
            if callback is not None and not callable(callback):
                raise ModelConfigurationError(f"{name} must be callable")

    def _heavy_energy_model(self) -> None:
        model = self.model
        closure = model.heavy_energy_closure
        if closure is None:
            return
        if closure.cv_over_kb.shape != (len(model.species_ids),):
            raise ModelConfigurationError(
                "HeavyEnergyClosure heat capacities must match compiled species order"
            )
        if model.transport is not None:
            transport_cv = model.transport.heavy_cv_over_kb
            if transport_cv is None or not np.array_equal(
                transport_cv, closure.cv_over_kb
            ):
                raise ModelConfigurationError(
                    "Evolved gas energy requires transport heat capacities to "
                    "match HeavyEnergyClosure"
                )
        expected_zone_shape = (len(model.zones),)
        if (
            closure.wall_temperature_K.shape != expected_zone_shape
            or closure.wall_relaxation_s_inv.shape != expected_zone_shape
        ):
            raise ModelConfigurationError(
                "HeavyEnergyClosure wall arrays must match compiled zone order"
            )

    def _surfaces(self, zone_ids: tuple[str, ...], known_zones: set[str]) -> set[str]:
        model = self.model
        surface_model = model.surface_model
        if surface_model is None:
            return set()
        if surface_model.gas_species_ids != model.species_ids:
            raise ModelConfigurationError(
                "CompiledSurfaceModel gas species/order must match chemistry"
            )
        if surface_model.zone_ids != zone_ids:
            raise ModelConfigurationError(
                "CompiledSurfaceModel zone IDs/order must match model zones"
            )
        model_volumes = np.array([zone.volume_m3 for zone in model.zones])
        if not np.array_equal(surface_model.zone_volumes_m3, model_volumes):
            raise ModelConfigurationError(
                "CompiledSurfaceModel volumes/order must match model zones"
            )
        surface_ids = tuple(surface.surface_id for surface in surface_model.surfaces)
        if len(set(surface_ids)) != len(surface_ids):
            raise ModelConfigurationError("Surface IDs must be unique")
        unknown_surface_zones = {
            surface.zone_id for surface in surface_model.surfaces
        } - known_zones
        if unknown_surface_zones:
            raise ModelConfigurationError(
                f"Surfaces reference unknown zones {sorted(unknown_surface_zones)}"
            )
        known_surfaces = set(surface_ids)
        self._surface_wall_bindings(known_surfaces)
        return known_surfaces

    def _surface_wall_bindings(self, surface_ids: set[str]) -> None:
        boundary_surface_ids = self._boundary_surface_ids()
        unknown_wall_surfaces = set(boundary_surface_ids) - surface_ids
        if unknown_wall_surfaces:
            raise ModelConfigurationError(
                "Wall boundaries reference unknown surfaces "
                f"{sorted(unknown_wall_surfaces)}"
            )
        self._validate_surface_wall_zones()

    def _boundary_surface_ids(self) -> tuple[str, ...]:
        boundary_surface_ids = tuple(
            boundary.surface_id
            for boundary in self.model.wall_boundaries
            if boundary.surface_id is not None
        )
        if len(set(boundary_surface_ids)) != len(boundary_surface_ids):
            raise ModelConfigurationError(
                "At most one wall transport boundary may own each surface"
            )
        return boundary_surface_ids

    def _validate_surface_wall_zones(self) -> None:
        model = self.model
        if model.surface_model is None:
            return
        surface_zones = {
            surface.surface_id: surface.zone_id
            for surface in model.surface_model.surfaces
        }
        mismatched_wall_zones = [
            boundary.surface_id
            for boundary in model.wall_boundaries
            if boundary.surface_id is not None
            and surface_zones[boundary.surface_id] != boundary.zone_id
        ]
        if mismatched_wall_zones:
            raise ModelConfigurationError(
                "Wall boundary zones disagree with their surfaces: "
                f"{sorted(mismatched_wall_zones)}"
            )

    def _transport_and_power(self, zone_ids: tuple[str, ...]) -> None:
        model = self.model
        if model.transport is not None:
            if model.transport.n_zones != len(model.zones):
                raise ModelConfigurationError(
                    "Transport zone count does not match model zones"
                )
            if model.transport.n_species != len(model.species_ids):
                raise ModelConfigurationError(
                    "Transport species count does not match compiled chemistry"
                )
            model_volumes = np.array([zone.volume_m3 for zone in model.zones])
            if not np.array_equal(model.transport.volumes_m3, model_volumes):
                raise ModelConfigurationError(
                    "Transport volumes/order must exactly match "
                    "CompiledGlobalModel.zones"
                )
        elif any(segment.transport is not None for segment in model.segments):
            raise ModelConfigurationError(
                "Recipe segment transport forcing requires "
                "CompiledGlobalModel.transport"
            )
        if (
            model.power_coordinator is not None
            and tuple(model.power_coordinator.zone_ids) != zone_ids
        ):
            raise ModelConfigurationError(
                "PowerCoordinator zone_ids/order must match CompiledGlobalModel.zones"
            )

    def _kinetics_and_reactions(self, known_zones: set[str]) -> None:
        model = self.model
        unknown_kinetics = set(model.electron_kinetics_by_zone) - known_zones
        if unknown_kinetics:
            raise ModelConfigurationError(
                f"Electron kinetics reference unknown zones {sorted(unknown_kinetics)}"
            )
        required_lookup = (
            "mean_energy"
            if model.electron_closure.mode == "electron_energy"
            else "local_field"
        )
        for zone_id, kinetics in model.electron_kinetics_by_zone.items():
            if kinetics.lookup != required_lookup:
                raise ModelConfigurationError(
                    f"Zone {zone_id!r} electron closure "
                    f"{model.electron_closure.mode!r} requires a "
                    f"{required_lookup!r} kinetics table, got {kinetics.lookup!r}"
                )
        for reaction_id, selected_zones in zip(
            model.reaction_ids, model.reaction_zones, strict=True
        ):
            unknown = set(selected_zones) - known_zones
            if unknown:
                raise ModelConfigurationError(
                    f"Reaction {reaction_id!r} selects unknown zones {sorted(unknown)}"
                )

    def _segments(self, known_zones: set[str], known_surfaces: set[str]) -> None:
        model = self.model
        previous_end: float | None = None
        requires_electron_density = any(
            segment.prescribed_electron_density_m3_by_zone for segment in model.segments
        )
        for segment in model.segments:
            if previous_end is not None and segment.start_s != previous_end:
                raise ModelConfigurationError(
                    "Recipe segments must be exactly contiguous: "
                    f"{previous_end} != {segment.start_s}"
                )
            previous_end = segment.end_s
            self._segment_references(
                segment,
                known_zones=known_zones,
                known_surfaces=known_surfaces,
                require_electron_density=requires_electron_density,
            )

    def _segment_references(
        self,
        segment: RecipeSegment,
        *,
        known_zones: set[str],
        known_surfaces: set[str],
        require_electron_density: bool,
    ) -> None:
        zone_references = (
            set(segment.absorbed_power_W_by_zone)
            | set(segment.reduced_field_Td_by_zone)
            | set(segment.wall_temperature_K_by_zone)
            | set(segment.prescribed_electron_density_m3_by_zone)
        )
        unknown_zones = zone_references - known_zones
        if unknown_zones:
            raise ModelConfigurationError(
                f"Recipe segment {segment.segment_id!r} references unknown zones "
                f"{sorted(unknown_zones)}"
            )
        electron_density_zones = set(segment.prescribed_electron_density_m3_by_zone)
        if require_electron_density and electron_density_zones != known_zones:
            raise ModelConfigurationError(
                f"Recipe segment {segment.segment_id!r} must prescribe electron "
                f"density for every zone {sorted(known_zones)}"
            )
        unknown_surfaces = (
            set(segment.surface_temperature_K_by_surface) - known_surfaces
        )
        if unknown_surfaces:
            raise ModelConfigurationError(
                f"Recipe segment {segment.segment_id!r} references unknown surfaces "
                f"{sorted(unknown_surfaces)}"
            )
        self._segment_electron_controls(segment, known_zones)

    def _segment_electron_controls(
        self, segment: RecipeSegment, known_zones: set[str]
    ) -> None:
        model = self.model
        if (
            model.electron_closure.mode == "local_field"
            and model.power_coordinator is None
        ):
            missing = known_zones - set(segment.reduced_field_Td_by_zone)
            if missing:
                raise ModelConfigurationError(
                    f"local_field segment {segment.segment_id!r} lacks E/N for zones "
                    f"{sorted(missing)}"
                )
        if model.power_coordinator is None and segment.port_commands:
            raise ModelConfigurationError(
                f"Segment {segment.segment_id!r} has port commands but no "
                "PowerCoordinator"
            )
        if model.power_coordinator is not None:
            model.power_coordinator.validate_commands(segment.port_commands)

    def _electron_closure(self) -> None:
        closure = self.model.electron_closure
        if closure.mode not in {"electron_energy", "local_field"}:
            raise ModelConfigurationError(
                f"Unknown electron closure mode {closure.mode!r}"
            )
        if closure.evolves_energy != (closure.mode == "electron_energy"):
            raise ModelConfigurationError(
                "Electron closure mode and evolves_energy flag are inconsistent"
            )

    def _wall_boundaries(self, known_zones: set[str]) -> None:
        model = self.model
        species_index = {
            species_id: index for index, species_id in enumerate(model.species_ids)
        }
        for boundary in model.wall_boundaries:
            self._wall_boundary(boundary, known_zones, species_index)

    def _wall_boundary(
        self,
        boundary: WallBoundary,
        known_zones: set[str],
        species_index: Mapping[str, int],
    ) -> None:
        if boundary.zone_id not in known_zones:
            raise ModelConfigurationError(
                f"Wall boundary references unknown zone {boundary.zone_id!r}"
            )
        for reaction in boundary.reactions:
            self._boundary_reaction(reaction, species_index)
        if boundary.transport_kind != "off":
            self._boundary_probabilities(boundary)

    def _boundary_reaction(
        self, reaction: BoundaryReaction, species_index: Mapping[str, int]
    ) -> None:
        model = self.model
        incident_index = species_index.get(reaction.incident_species)
        if incident_index is None:
            raise ModelConfigurationError(
                f"Boundary reaction {reaction.reaction_id!r} has unknown "
                "incident species"
            )
        if model.charges[incident_index] <= 0.0:
            raise ModelConfigurationError(
                f"Boundary incident species {reaction.incident_species!r} must "
                "be a positive ion"
            )
        unknown_products = set(reaction.products) - set(model.species_ids)
        if unknown_products:
            raise ModelConfigurationError(
                f"Boundary reaction {reaction.reaction_id!r} has unknown products "
                f"{sorted(unknown_products)}"
            )

    def _boundary_probabilities(self, boundary: WallBoundary) -> None:
        model = self.model
        branch_probability = {
            species_id: sum(
                reaction.probability
                for reaction in boundary.reactions
                if reaction.incident_species == species_id
            )
            for species_id, charge in zip(model.species_ids, model.charges, strict=True)
            if charge > 0.0
        }
        incomplete = {
            species_id: probability
            for species_id, probability in branch_probability.items()
            if not math.isclose(probability, 1.0, rel_tol=0.0, abs_tol=1.0e-12)
        }
        if incomplete:
            raise ModelConfigurationError(
                f"Wall {boundary.surface_id or boundary.zone_id!r} must define "
                "unit-probability boundary products for every positive ion; "
                f"got {incomplete}"
            )


__all__ = ["DomainValidator"]
