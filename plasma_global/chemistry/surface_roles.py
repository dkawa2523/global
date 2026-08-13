"""Pure surface-species classification shared by input and model compilers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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


__all__ = ["SurfaceSpeciesRoles", "surface_species_roles"]
