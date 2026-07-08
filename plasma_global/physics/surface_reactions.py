from __future__ import annotations

from typing import Any

from plasma_global.physics.types import CompiledSurfaceReaction


def compile_surface_reactions(system: Any) -> list[CompiledSurfaceReaction]:
    compiled: list[CompiledSurfaceReaction] = []
    for rxn in system.mechanism.surface_reactions:
        if not rxn.enabled:
            continue
        for surface_id in rxn.surface_filter:
            compiled_rxn = _compile_surface_reaction(system, rxn, surface_id)
            if compiled_rxn is not None:
                compiled.append(compiled_rxn)
    return compiled


def _compile_surface_reaction(system: Any, rxn: Any, surface_id: str) -> CompiledSurfaceReaction | None:
    surface = system.chamber.surface_by_id[surface_id]
    zone_id = surface.zone_id
    if rxn.zone_filter and zone_id not in rxn.zone_filter:
        return None
    gas_reactants = _gas_side(system, rxn.reactants)
    surface_delta = _surface_delta(system, rxn, surface_id)
    return CompiledSurfaceReaction(
        reaction_id=rxn.reaction_id,
        zone_id=zone_id,
        surface_id=surface_id,
        gas_reactants=gas_reactants,
        surface_reactants=_surface_side(system, rxn.reactants, surface_id),
        delta_gas=_gas_delta(system, rxn),
        delta_surface=surface_delta,
        area_over_volume=surface.area_m2 / max(system.chamber.zone_by_id[zone_id].volume_m3, 1.0e-30),
        area_m2=surface.area_m2,
        site_density_m2=max(surface.site_density_m2, 1.0),
        rate_model=system.mechanism.model(rxn.rate_model_key),
        inventory_idx=_inventory_index(system, surface_id, gas_reactants),
        film_factor=_film_factor(system, surface_delta),
    )


def _gas_side(system: Any, side: dict[str, float]) -> list[tuple[int, float, str]]:
    return [
        (system.gas_species_index[sp_id], float(nu), sp_id)
        for sp_id, nu in side.items()
        if sp_id in system.gas_species_index
    ]


def _surface_side(system: Any, side: dict[str, float], surface_id: str) -> list[tuple[int, float, str]]:
    mapping = system.state_layout.surface_index.get(surface_id, {})
    return [(mapping[sp_id], float(nu), sp_id) for sp_id, nu in side.items() if sp_id in mapping]


def _surface_delta(system: Any, rxn: Any, surface_id: str) -> list[tuple[int, float, str]]:
    return _delta_side(rxn, system.state_layout.surface_index.get(surface_id, {}))


def _gas_delta(system: Any, rxn: Any) -> list[tuple[int, float, str]]:
    return _delta_side(rxn, system.gas_species_index)


def _delta_side(rxn: Any, mapping: dict[str, int]) -> list[tuple[int, float, str]]:
    return [
        (idx, float(rxn.products.get(sp_id, 0.0) - rxn.reactants.get(sp_id, 0.0)), sp_id)
        for sp_id, idx in mapping.items()
        if abs(rxn.products.get(sp_id, 0.0) - rxn.reactants.get(sp_id, 0.0)) > 0.0
    ]


def _inventory_index(system: Any, surface_id: str, gas_reactants: list[tuple[int, float, str]]) -> int | None:
    if not gas_reactants:
        return None
    candidate = f'{gas_reactants[0][2]}_reservoir'
    return system.state_layout.inventory_index.get(surface_id, {}).get(candidate)


def _film_factor(system: Any, surface_delta: list[tuple[int, float, str]]) -> float:
    return sum(
        nu
        for _idx, nu, species_id in surface_delta
        if 'film_fragment' in system.surface_species_tags.get(species_id, set())
    )
