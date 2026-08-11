"""Compile opt-in schema and chemistry declarations into runtime accumulators."""

from __future__ import annotations

from plasma_global.chemistry.data import ChemistryData
from plasma_global.errors import CaseValidationError
from plasma_global.experimental.accumulators import SurfaceEventRate
from plasma_global.experimental.runtime import (
    ExperimentalRuntimeAccumulator,
    compile_generic_state_accumulator,
)
from plasma_global.input.schema import CaseSpec


def compile_experimental_accumulator(
    case: CaseSpec,
    chemistry_data: ChemistryData,
    surface_model: object | None,
) -> tuple[ExperimentalRuntimeAccumulator | None, tuple[str, ...]]:
    """Compile the selected experimental state block, if any."""

    selected = case.experimental
    if selected is None:
        return None, ()

    surfaces = tuple(case.reactor.surfaces)
    surface_ids = tuple(surface.surface_id for surface in surfaces)
    generic = None
    features: list[str] = []
    if selected.extensions is not None:
        try:
            generic = compile_generic_state_accumulator(
                chemistry_data.experimental,
                zone_ids=tuple(zone.zone_id for zone in case.reactor.zones),
                surface_ids=surface_ids,
            )
        except (TypeError, ValueError) as exc:
            raise CaseValidationError(
                f"Cannot compile experimental chemistry extensions: {exc}"
            ) from exc
        if generic is None:
            raise CaseValidationError(
                "experimental.extensions requires chemistry.experimental.state_variables"
            )
        features.append("extensions")

    film_config = selected.film
    film_enabled = film_config is not None
    film_species = {
        species.id
        for species in chemistry_data.species
        if species.phase == "surface" and "film_fragment" in species.state_tags
    }
    film_delta_by_reaction = {
        reaction.id: sum(
            float(reaction.products.get(species_id, 0.0))
            - float(reaction.reactants.get(species_id, 0.0))
            for species_id in film_species
        )
        for reaction in chemistry_data.surface_reactions
    }
    if film_enabled:
        if surface_model is None:
            raise CaseValidationError(
                "experimental.film requires models.surface_kinetics"
            )
        if not any(value != 0.0 for value in film_delta_by_reaction.values()):
            raise CaseValidationError(
                "experimental.film requires a surface reaction with a "
                "film_fragment stoichiometric change"
            )
        features.append("film")

    inventory_config = selected.wall_inventory
    inventory = (
        {}
        if inventory_config is None
        else {
            surface_id: dict(entries)
            for surface_id, entries in inventory_config.initial_by_surface.items()
        }
    )
    if inventory_config is not None:
        if not any(inventory.values()):
            raise CaseValidationError(
                "experimental.wall_inventory requires at least one initial inventory key"
            )
        features.append("wall_inventory")
    empty_inventory_events: dict[tuple[str, str], dict[str, float]] = {}
    inventory_events: dict[tuple[str, str], dict[str, float]] = (
        empty_inventory_events
        if inventory_config is None
        else {
            (event.reaction_id, event.surface_id): dict(
                event.inventory_particles_per_event
            )
            for event in inventory_config.events
        }
    )
    if inventory_events and surface_model is None:
        raise CaseValidationError(
            "experimental wall-inventory events require models.surface_kinetics"
        )

    templates: list[SurfaceEventRate] = []
    seen_inventory_events: set[tuple[str, str]] = set()
    if surface_model is not None and (film_enabled or inventory_events):
        surface_by_id = {surface.surface_id: surface for surface in surfaces}
        for reaction in chemistry_data.surface_reactions:
            film_factor = film_delta_by_reaction[reaction.id] if film_enabled else 0.0
            target_surfaces = reaction.surfaces or surface_ids
            for surface_id in target_surfaces:
                surface = surface_by_id.get(surface_id)
                if surface is None or (
                    reaction.zones and surface.zone_id not in reaction.zones
                ):
                    continue
                event_key = reaction.id, surface_id
                inventory_particles = inventory_events.get(event_key, {})
                if inventory_particles:
                    seen_inventory_events.add(event_key)
                if film_factor == 0.0 and not inventory_particles:
                    continue
                if film_factor != 0.0 and surface.site_density_m2 <= 0.0:
                    raise CaseValidationError(
                        f"experimental film event {reaction.id!r} on {surface_id!r} "
                        "requires positive site_density_m2"
                    )
                templates.append(
                    SurfaceEventRate(
                        event_id=f"{reaction.id}@{surface_id}",
                        zone_id=surface.zone_id,
                        surface_id=surface_id,
                        rate_m2_s=0.0,
                        area_m2=surface.area_m2,
                        site_density_m2=max(float(surface.site_density_m2), 1.0),
                        inventory_particles_per_event=inventory_particles,
                        film_layers_per_event=film_factor,
                    )
                )
    unbound_inventory_events = set(inventory_events) - seen_inventory_events
    if unbound_inventory_events:
        raise CaseValidationError(
            "experimental wall-inventory events reference unknown or inapplicable "
            f"reaction/surface pairs: {sorted(unbound_inventory_events)}"
        )

    if not features:
        return None, ()
    try:
        accumulator = ExperimentalRuntimeAccumulator(
            generic=generic,
            film_surfaces=surface_ids if film_enabled else (),
            initial_inventory_by_surface=inventory,
            surface_event_templates=templates,
            monolayer_thickness_m=(
                3.0e-10
                if film_config is None
                else float(film_config.monolayer_thickness_m)
            ),
        )
    except (TypeError, ValueError) as exc:
        raise CaseValidationError(
            f"Cannot compile experimental runtime accumulator: {exc}"
        ) from exc
    return accumulator, tuple(features)


__all__ = ["compile_experimental_accumulator"]
