from __future__ import annotations

from plasma_global.chemistry.data import SpeciesData
from plasma_global.chemistry.surface_roles import surface_species_roles


def test_surface_species_roles_are_scoped_ordered_and_occupancy_aware() -> None:
    species = (
        SpeciesData("A", "gas", 0, 40.0, {"A": 1.0}),
        SpeciesData(
            "site",
            "surface",
            0,
            0.0,
            {"site": 1.0},
            state_tags=frozenset({"site"}),
        ),
        SpeciesData("pair", "surface", 0, 80.0, {"A": 2.0, "site": 2.0}),
        SpeciesData(
            "film",
            "surface",
            0,
            40.0,
            {"A": 1.0, "site": 1.0},
            state_tags=frozenset({"film_fragment"}),
        ),
        SpeciesData(
            "other",
            "surface",
            0,
            40.0,
            {"A": 1.0, "site": 1.0},
            surfaces=("other_wall",),
        ),
    )

    roles = surface_species_roles(species, "wall")

    assert tuple(item.id for item in roles.applicable) == ("site", "pair", "film")
    assert roles.free_site_ids == ("site",)
    assert roles.independent_ids == ("pair",)
    assert roles.occupancy_by_species == {"site": 1.0, "pair": 2.0, "film": 1.0}
