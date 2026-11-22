"""Static surface-event declarations consumed by the runtime accumulator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from plasma_global.experimental._surface_inventory import validate_surface_event


@dataclass(frozen=True, slots=True)
class SurfaceEventRate:
    """Geometry and stoichiometry for one surface bookkeeping event.

    The surface chemistry model supplies the event rate to the runtime RHS.
    Gas-particle changes remain owned by that chemistry model.
    Event identity is the pair (reaction_id, surface_id), without ID encoding.
    """

    reaction_id: str
    zone_id: str
    surface_id: str
    area_m2: float
    site_density_m2: float
    inventory_particles_per_event: Mapping[str, float] = field(default_factory=dict)
    film_layers_per_event: float = 0.0

    @property
    def event_key(self) -> tuple[str, str]:
        return self.reaction_id, self.surface_id

    def __post_init__(self) -> None:
        inventory = validate_surface_event(
            reaction_id=self.reaction_id,
            zone_id=self.zone_id,
            surface_id=self.surface_id,
            area_m2=self.area_m2,
            site_density_m2=self.site_density_m2,
            inventory_particles_per_event=self.inventory_particles_per_event,
            film_layers_per_event=self.film_layers_per_event,
        )
        object.__setattr__(self, "inventory_particles_per_event", inventory)


__all__ = ["SurfaceEventRate"]
