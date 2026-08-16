"""Compile opt-in schema and chemistry declarations into runtime accumulators."""

from __future__ import annotations

from collections.abc import Mapping

from plasma_global.chemistry.data import ChemistryData, ReactionData
from plasma_global.errors import CaseValidationError
from plasma_global.experimental.accumulators import (
    GenericStateAccumulator,
    SurfaceEventRate,
)
from plasma_global.experimental.runtime import (
    ExperimentalRuntimeAccumulator,
    compile_generic_state_accumulator,
)
from plasma_global.input.schema import (
    CaseSpec,
    ExperimentalConfig,
    ExperimentalWallInventoryConfig,
    SurfaceConfig,
)


def _compile_generic_extension(
    selected: ExperimentalConfig,
    chemistry_data: ChemistryData,
    *,
    zone_ids: tuple[str, ...],
    surface_ids: tuple[str, ...],
) -> GenericStateAccumulator | None:
    if selected.extensions is None:
        return None
    try:
        generic = compile_generic_state_accumulator(
            chemistry_data.experimental,
            zone_ids=zone_ids,
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
    return generic


def _compile_film_changes(
    selected: ExperimentalConfig,
    chemistry_data: ChemistryData,
    surface_model: object | None,
) -> dict[str, float]:
    if selected.film is None:
        return {}
    if surface_model is None:
        raise CaseValidationError("experimental.film requires models.surface_kinetics")
    film_species = {
        species.id
        for species in chemistry_data.species
        if species.phase == "surface" and "film_fragment" in species.state_tags
    }
    changes = {
        reaction.id: sum(
            float(reaction.products.get(species_id, 0.0))
            - float(reaction.reactants.get(species_id, 0.0))
            for species_id in film_species
        )
        for reaction in chemistry_data.surface_reactions
    }
    if not any(value != 0.0 for value in changes.values()):
        raise CaseValidationError(
            "experimental.film requires a surface reaction with a "
            "film_fragment stoichiometric change"
        )
    return changes


def _compile_wall_inventory(
    config: ExperimentalWallInventoryConfig | None,
    surface_model: object | None,
) -> tuple[
    Mapping[str, Mapping[str, float]],
    Mapping[tuple[str, str], Mapping[str, float]],
]:
    if config is None:
        return {}, {}
    inventory = config.initial_by_surface
    if not any(inventory.values()):
        raise CaseValidationError(
            "experimental.wall_inventory requires at least one initial inventory key"
        )
    events = {
        (event.reaction_id, event.surface_id): event.inventory_particles_per_event
        for event in config.events
    }
    if events and surface_model is None:
        raise CaseValidationError(
            "experimental wall-inventory events require models.surface_kinetics"
        )
    return inventory, events


def _applicable_surface_pairs(
    reaction: ReactionData,
    surface_ids: tuple[str, ...],
    surface_by_id: Mapping[str, SurfaceConfig],
) -> tuple[tuple[str, SurfaceConfig], ...]:
    pairs: list[tuple[str, SurfaceConfig]] = []
    for surface_id in reaction.surfaces or surface_ids:
        surface = surface_by_id.get(surface_id)
        if surface is None or (
            reaction.zones and surface.zone_id not in reaction.zones
        ):
            continue
        pairs.append((surface_id, surface))
    return tuple(pairs)


def _surface_event_template(
    reaction_id: str,
    surface_id: str,
    surface: SurfaceConfig,
    film_factor: float,
    inventory_particles: Mapping[str, float],
) -> SurfaceEventRate | None:
    if film_factor == 0.0 and not inventory_particles:
        return None
    if film_factor != 0.0 and surface.site_density_m2 <= 0.0:
        raise CaseValidationError(
            f"experimental film event {reaction_id!r} on {surface_id!r} "
            "requires positive site_density_m2"
        )
    return SurfaceEventRate(
        event_id=f"{reaction_id}@{surface_id}",
        zone_id=surface.zone_id,
        surface_id=surface_id,
        rate_m2_s=0.0,
        area_m2=surface.area_m2,
        site_density_m2=max(surface.site_density_m2, 1.0),
        inventory_particles_per_event=inventory_particles,
        film_layers_per_event=film_factor,
    )


def _compile_surface_event_templates(
    chemistry_data: ChemistryData,
    surfaces: tuple[SurfaceConfig, ...],
    film_changes: Mapping[str, float],
    inventory_events: Mapping[tuple[str, str], Mapping[str, float]],
) -> tuple[SurfaceEventRate, ...]:
    surface_ids = tuple(surface.surface_id for surface in surfaces)
    surface_by_id = {surface.surface_id: surface for surface in surfaces}
    templates: list[SurfaceEventRate] = []
    seen_inventory_events: set[tuple[str, str]] = set()
    for reaction in chemistry_data.surface_reactions:
        film_factor = film_changes.get(reaction.id, 0.0)
        for surface_id, surface in _applicable_surface_pairs(
            reaction, surface_ids, surface_by_id
        ):
            event_key = reaction.id, surface_id
            inventory_particles = inventory_events.get(event_key, {})
            if inventory_particles:
                seen_inventory_events.add(event_key)
            template = _surface_event_template(
                reaction.id,
                surface_id,
                surface,
                film_factor,
                inventory_particles,
            )
            if template is not None:
                templates.append(template)
    unbound_inventory_events = set(inventory_events) - seen_inventory_events
    if unbound_inventory_events:
        raise CaseValidationError(
            "experimental wall-inventory events reference unknown or inapplicable "
            f"reaction/surface pairs: {sorted(unbound_inventory_events)}"
        )
    return tuple(templates)


def _build_runtime_accumulator(
    selected: ExperimentalConfig,
    *,
    generic: GenericStateAccumulator | None,
    surface_ids: tuple[str, ...],
    inventory: Mapping[str, Mapping[str, float]],
    templates: tuple[SurfaceEventRate, ...],
) -> ExperimentalRuntimeAccumulator:
    film_config = selected.film
    try:
        return ExperimentalRuntimeAccumulator(
            generic=generic,
            film_surfaces=surface_ids if film_config is not None else (),
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
    generic = _compile_generic_extension(
        selected,
        chemistry_data,
        zone_ids=tuple(zone.zone_id for zone in case.reactor.zones),
        surface_ids=surface_ids,
    )
    film_changes = _compile_film_changes(selected, chemistry_data, surface_model)
    inventory, inventory_events = _compile_wall_inventory(
        selected.wall_inventory, surface_model
    )
    features = tuple(
        name
        for name, config in (
            ("extensions", selected.extensions),
            ("film", selected.film),
            ("wall_inventory", selected.wall_inventory),
        )
        if config is not None
    )
    if not features:
        return None, ()
    templates = (
        _compile_surface_event_templates(
            chemistry_data, surfaces, film_changes, inventory_events
        )
        if film_changes or inventory_events
        else ()
    )
    accumulator = _build_runtime_accumulator(
        selected,
        generic=generic,
        surface_ids=surface_ids,
        inventory=inventory,
        templates=templates,
    )
    return accumulator, features


__all__ = ["compile_experimental_accumulator"]
