from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class StateSlice:
    name: str
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start


@dataclass
class StateLayout:
    slices: dict[str, StateSlice]
    size: int
    gas_index: dict[str, dict[str, int]] = field(default_factory=dict)
    electron_energy_index: dict[str, int] = field(default_factory=dict)
    gas_temperature_index: dict[str, int] = field(default_factory=dict)
    surface_index: dict[str, dict[str, int]] = field(default_factory=dict)
    inventory_index: dict[str, dict[str, int]] = field(default_factory=dict)
    film_index: dict[str, int] = field(default_factory=dict)
    gas_species_ids: list[str] = field(default_factory=list)
    zone_ids: list[str] = field(default_factory=list)

    def make_state(self) -> np.ndarray:
        return np.zeros(self.size, dtype=float)

    def slice(self, name: str) -> slice:
        s = self.slices[name]
        return slice(s.start, s.stop)


def mechanism_has_film_state(mechanism: Any) -> bool:
    film_species = {
        sp.canonical_id
        for sp in getattr(mechanism, 'surface_species', []) or []
        if 'film_fragment' in set(getattr(sp, 'state_tags', []) or [])
    }
    if not film_species:
        return False
    for rxn in getattr(mechanism, 'surface_reactions', []) or []:
        if not getattr(rxn, 'enabled', True):
            continue
        for sp_id in film_species:
            delta = float((getattr(rxn, 'products', {}) or {}).get(sp_id, 0.0)) - float((getattr(rxn, 'reactants', {}) or {}).get(sp_id, 0.0))
            if abs(delta) > 0.0:
                return True
    return False


def _add_gas_density_state(
    *,
    start: int,
    slices: dict[str, StateSlice],
    gas_index: dict[str, dict[str, int]],
    gas_species: list[Any],
    zone_ids: list[str],
) -> int:
    slices['gas_densities'] = StateSlice('gas_densities', start, start + len(gas_species) * len(zone_ids))
    for z_idx, zone_id in enumerate(zone_ids):
        gas_index[zone_id] = {}
        for s_idx, sp in enumerate(gas_species):
            gas_index[zone_id][sp.canonical_id] = start + z_idx * len(gas_species) + s_idx
    return slices['gas_densities'].stop


def _add_electron_energy_state(
    *,
    start: int,
    slices: dict[str, StateSlice],
    electron_energy_index: dict[str, int],
    zone_ids: list[str],
) -> int:
    slices['electron_energy'] = StateSlice('electron_energy', start, start + len(zone_ids))
    for i, zone_id in enumerate(zone_ids):
        electron_energy_index[zone_id] = start + i
    return slices['electron_energy'].stop


def _add_gas_temperature_state(
    *,
    start: int,
    slices: dict[str, StateSlice],
    gas_temperature_index: dict[str, int],
    zone_ids: list[str],
) -> int:
    slices['gas_temperature'] = StateSlice('gas_temperature', start, start + len(zone_ids))
    for i, zone_id in enumerate(zone_ids):
        gas_temperature_index[zone_id] = start + i
    return slices['gas_temperature'].stop


def _add_surface_coverage_state(
    *,
    start: int,
    slices: dict[str, StateSlice],
    surface_index: dict[str, dict[str, int]],
    chamber: Any,
    mechanism: Any,
) -> int:
    surf_start = start
    for surface in chamber.surfaces:
        mapping: dict[str, int] = {}
        for sp in mechanism.surface_species:
            if surface.surface_id in sp.surfaces:
                mapping[sp.canonical_id] = start
                start += 1
        if mapping:
            surface_index[surface.surface_id] = mapping
    if start > surf_start:
        slices['surface_coverages'] = StateSlice('surface_coverages', surf_start, start)
    return start


def _add_wall_inventory_state(
    *,
    start: int,
    slices: dict[str, StateSlice],
    inventory_index: dict[str, dict[str, int]],
    chamber: Any,
) -> int:
    inv_start = start
    for surface in chamber.surfaces:
        mapping: dict[str, int] = {}
        for key in surface.initial_inventory.keys():
            mapping[key] = start
            start += 1
        if mapping:
            inventory_index[surface.surface_id] = mapping
    if start > inv_start:
        slices['wall_inventory'] = StateSlice('wall_inventory', inv_start, start)
    return start


def _add_film_thickness_state(
    *,
    start: int,
    slices: dict[str, StateSlice],
    film_index: dict[str, int],
    chamber: Any,
) -> int:
    film_start = start
    for surface in chamber.surfaces:
        film_index[surface.surface_id] = start
        start += 1
    if start > film_start:
        slices['film_thickness'] = StateSlice('film_thickness', film_start, start)
    return start


def build_state_layout(mechanism: Any, chamber: Any, run_config: Any) -> StateLayout:
    gas_species = [s for s in mechanism.gas_state_species]
    zone_ids = [z.zone_id for z in chamber.zones]
    start = 0
    slices: dict[str, StateSlice] = {}
    gas_index: dict[str, dict[str, int]] = {}
    surface_index: dict[str, dict[str, int]] = {}
    inventory_index: dict[str, dict[str, int]] = {}
    film_index: dict[str, int] = {}
    electron_energy_index: dict[str, int] = {}
    gas_temperature_index: dict[str, int] = {}

    start = _add_gas_density_state(
        start=start,
        slices=slices,
        gas_index=gas_index,
        gas_species=gas_species,
        zone_ids=zone_ids,
    )
    start = _add_electron_energy_state(
        start=start,
        slices=slices,
        electron_energy_index=electron_energy_index,
        zone_ids=zone_ids,
    )
    if run_config.physics.enable_gas_temperature:
        start = _add_gas_temperature_state(
            start=start,
            slices=slices,
            gas_temperature_index=gas_temperature_index,
            zone_ids=zone_ids,
        )
    if run_config.physics.enable_surface_coverages:
        start = _add_surface_coverage_state(
            start=start,
            slices=slices,
            surface_index=surface_index,
            chamber=chamber,
            mechanism=mechanism,
        )
    if run_config.physics.enable_wall_inventory:
        start = _add_wall_inventory_state(
            start=start,
            slices=slices,
            inventory_index=inventory_index,
            chamber=chamber,
        )
    if run_config.physics.enable_surface_coverages and mechanism_has_film_state(mechanism):
        start = _add_film_thickness_state(
            start=start,
            slices=slices,
            film_index=film_index,
            chamber=chamber,
        )

    return StateLayout(
        slices=slices,
        size=start,
        gas_index=gas_index,
        electron_energy_index=electron_energy_index,
        gas_temperature_index=gas_temperature_index,
        surface_index=surface_index,
        inventory_index=inventory_index,
        film_index=film_index,
        gas_species_ids=[sp.canonical_id for sp in gas_species],
        zone_ids=zone_ids,
    )
