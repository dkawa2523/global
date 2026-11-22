"""One immutable definition for evaluated physical balance terms."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from plasma_global.models.walls import WallFluxRecord


@dataclass(frozen=True, slots=True)
class ZoneTermLedger:
    zone_id: str
    reaction_rates_m3_s: Mapping[str, float]
    absorbed_power_J_m3_s: float
    reaction_energy_loss_J_m3_s: float
    wall_energy_loss_J_m3_s: float
    gas_power_J_m3_s: float
    gas_reaction_heating_J_m3_s: float
    surface_reaction_heating_J_m3_s: float
    wall_heavy_energy_exchange_J_m3_s: float
    elastic_heating_J_m3_s: float
    transport_species_source_m3_s: Mapping[str, float]
    transport_electron_energy_J_m3_s: float
    transport_heavy_energy_J_m3_s: float
    inlet_heavy_energy_J_m3_s: float
    surface_rates_m2_s: Mapping[tuple[str, str], float]
    wall_fluxes: tuple[WallFluxRecord, ...]
    wall_species_energy_J_m3_s: float = 0.0
    surface_species_energy_J_m3_s: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "reaction_rates_m3_s",
            "transport_species_source_m3_s",
            "surface_rates_m2_s",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


__all__ = ["ZoneTermLedger"]
