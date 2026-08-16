"""Translate schema-v2 experimental stopping and surface state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from plasma_global.errors import MigrationError
from plasma_global.input._migrate_v2_common import record_extra_keys


def _experimental_quasi_steady(run: Any, unused: set[str]) -> dict[str, float] | None:
    events = dict(run.numerics.events or {})
    steady = dict(events.get("steady_state", {}) or {})
    record_extra_keys(events, {"steady_state"}, "numerics.events", unused)
    record_extra_keys(
        steady,
        {"enabled", "relative_rhs_norm_s_inv", "min_step_time_s"},
        "numerics.events.steady_state",
        unused,
    )
    if not bool(steady.get("enabled", False)):
        if "steady_state" in events:
            unused.add("numerics.events.steady_state")
            unused.update(f"numerics.events.steady_state.{key}" for key in steady)
        return None
    return {
        "relative_rhs_norm_s_inv": float(steady.get("relative_rhs_norm_s_inv", 1.0e-3)),
        "min_time_s": float(steady.get("min_step_time_s", 0.0)),
    }


def _legacy_initial_inventory(chamber: Any) -> dict[str, dict[str, float]]:
    return {
        str(surface.surface_id): {
            str(key): float(value) for key, value in surface.initial_inventory.items()
        }
        for surface in chamber.surfaces
        if surface.initial_inventory
    }


def _inventory_reaction(
    *,
    loaded: Any,
    surface_id: str,
    inventory_id: str,
    gas_species_ids: set[str],
) -> Any:
    matching = [
        reaction
        for reaction in loaded.mechanism.surface_reactions
        if _reaction_matches_inventory(
            reaction,
            surface_id=surface_id,
            inventory_id=inventory_id,
            gas_species_ids=gas_species_ids,
        )
    ]
    if len(matching) != 1:
        raise MigrationError(
            "cannot migrate wall inventory without one unambiguous surface event: "
            f"{surface_id}.{inventory_id} matched "
            f"{[item.reaction_id for item in matching]}"
        )
    return matching[0]


def _reaction_matches_inventory(
    reaction: Any,
    *,
    surface_id: str,
    inventory_id: str,
    gas_species_ids: set[str],
) -> bool:
    if not reaction.enabled or (
        reaction.surface_filter and surface_id not in reaction.surface_filter
    ):
        return False
    gas_reactants = [
        species_id
        for species_id in reaction.gas_reactants
        if species_id in gas_species_ids
    ]
    return bool(gas_reactants) and f"{gas_reactants[0]}_reservoir" == inventory_id


def _migrate_wall_inventory(
    loaded: Any,
    initial_inventory: Mapping[str, Mapping[str, float]],
    warnings: list[str],
) -> dict[str, Any]:
    gas_species_ids = {
        str(species.canonical_id)
        for species in loaded.mechanism.gas_state_species
        if str(species.canonical_id) != "e"
    }
    event_yields: dict[tuple[str, str], dict[str, float]] = {}
    for surface_id, inventory_entries in initial_inventory.items():
        for inventory_id in inventory_entries:
            reaction = _inventory_reaction(
                loaded=loaded,
                surface_id=surface_id,
                inventory_id=inventory_id,
                gas_species_ids=gas_species_ids,
            )
            key = (str(reaction.reaction_id), surface_id)
            if key not in event_yields:
                event_yields[key] = {}
            event_yields[key][inventory_id] = 1.0
    warnings.append(
        "wall inventory event yields were made explicit from the unique v2 "
        "unit-per-event mapping; review each migrated yield"
    )
    return {
        "initial_by_surface": initial_inventory,
        "events": [
            {
                "reaction_id": reaction_id,
                "surface_id": surface_id,
                "inventory_particles_per_event": yields,
            }
            for (reaction_id, surface_id), yields in event_yields.items()
        ],
    }


def migrate_experimental(
    *,
    loaded: Any,
    run: Any,
    chamber: Any,
    unused: set[str],
    warnings: list[str],
) -> dict[str, Any]:
    experimental: dict[str, Any] = {}
    quasi_steady = _experimental_quasi_steady(run, unused)
    if quasi_steady is not None:
        experimental["stop_when_quasi_steady"] = quasi_steady
    has_film_state = any(
        "film_fragment" in set(getattr(species, "state_tags", ()) or ())
        for species in loaded.mechanism.surface_species
    )
    if run.physics.enable_surface_coverages and has_film_state:
        experimental["film"] = dict[str, Any]()
    initial_inventory = _legacy_initial_inventory(chamber)
    if run.physics.enable_wall_inventory and initial_inventory:
        experimental["wall_inventory"] = _migrate_wall_inventory(
            loaded, initial_inventory, warnings
        )
    if loaded.mechanism.state_variables or loaded.mechanism.processes:
        experimental["extensions"] = dict[str, Any]()
    return experimental
