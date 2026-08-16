"""Pure surface-species classification shared by input and model compilers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from plasma_global.chemistry.data import SpeciesData


@dataclass(frozen=True, slots=True)
class SurfaceSpeciesRoles:
    """Applicable surface species grouped by their state-layout responsibility."""

    applicable: tuple[SpeciesData, ...]
    free_site_ids: tuple[str, ...]
    independent_ids: tuple[str, ...]
    occupancy_by_species: MappingProxyType[str, float]


def _is_applicable_surface_species(
    species: SpeciesData,
    surface_id: str | None,
) -> bool:
    return species.phase == "surface" and (
        surface_id is None or not species.surfaces or surface_id in species.surfaces
    )


def surface_species_roles(
    species: Sequence[SpeciesData],
    surface_id: str | None,
) -> SurfaceSpeciesRoles:
    """Classify species for one surface, preserving chemistry declaration order."""

    applicable = tuple(
        item for item in species if _is_applicable_surface_species(item, surface_id)
    )
    free_site_ids = tuple(item.id for item in applicable if "site" in item.state_tags)
    free_site_set = set(free_site_ids)
    independent_ids = tuple(
        item.id
        for item in applicable
        if item.id not in free_site_set and "film_fragment" not in item.state_tags
    )
    occupancy = MappingProxyType(
        {item.id: item.elements.get("site", 1.0) for item in applicable}
    )
    return SurfaceSpeciesRoles(
        applicable=applicable,
        free_site_ids=free_site_ids,
        independent_ids=independent_ids,
        occupancy_by_species=occupancy,
    )


def _surface_initial_state_error(
    surface_id: str,
    initial_coverages: Mapping[str, float],
    roles: SurfaceSpeciesRoles,
) -> str | None:
    """Return the shared independent-coverage validation error, if any."""

    supplied = set(initial_coverages)
    if explicit_free := supplied & set(roles.free_site_ids):
        return (
            f"surface {surface_id!r} must not initialize algebraic "
            f"free site(s) {sorted(explicit_free)}"
        )
    if unknown := supplied - set(roles.independent_ids):
        return (
            f"surface {surface_id!r} initializes unknown or dependent "
            f"coverage species {sorted(unknown)}"
        )
    occupied = sum(
        roles.occupancy_by_species[species_id] * coverage
        for species_id, coverage in initial_coverages.items()
    )
    if occupied > 1.0 + 1.0e-12:
        return f"surface {surface_id!r} initial site occupancy exceeds one"
    return None


__all__ = [
    "SurfaceSpeciesRoles",
    "surface_species_roles",
]
