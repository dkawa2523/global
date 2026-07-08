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
    extra_state_index: dict[str, dict[str, int]] = field(default_factory=dict)
    gas_species_ids: list[str] = field(default_factory=list)
    zone_ids: list[str] = field(default_factory=list)

    def make_state(self) -> np.ndarray:
        return np.zeros(self.size, dtype=float)

    def slice(self, name: str) -> slice:
        s = self.slices[name]
        return slice(s.start, s.stop)


@dataclass
class _LayoutBuilder:
    slices: dict[str, StateSlice] = field(default_factory=dict)
    next_index: int = 0

    def add_block(self, name: str, size: int) -> StateSlice:
        block = StateSlice(name, self.next_index, self.next_index + size)
        self.slices[name] = block
        self.next_index = block.stop
        return block

    def allocate(self) -> int:
        index = self.next_index
        self.next_index += 1
        return index

    def add_optional_block(self, name: str, start: int) -> None:
        if self.next_index > start:
            self.slices[name] = StateSlice(name, start, self.next_index)


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
    builder: _LayoutBuilder,
    gas_index: dict[str, dict[str, int]],
    gas_species: list[Any],
    zone_ids: list[str],
) -> None:
    block = builder.add_block('gas_densities', len(gas_species) * len(zone_ids))
    for z_idx, zone_id in enumerate(zone_ids):
        gas_index[zone_id] = {}
        for s_idx, sp in enumerate(gas_species):
            gas_index[zone_id][sp.canonical_id] = block.start + z_idx * len(gas_species) + s_idx


def _add_electron_energy_state(
    *,
    builder: _LayoutBuilder,
    electron_energy_index: dict[str, int],
    zone_ids: list[str],
) -> None:
    block = builder.add_block('electron_energy', len(zone_ids))
    for i, zone_id in enumerate(zone_ids):
        electron_energy_index[zone_id] = block.start + i


def _add_gas_temperature_state(
    *,
    builder: _LayoutBuilder,
    gas_temperature_index: dict[str, int],
    zone_ids: list[str],
) -> None:
    block = builder.add_block('gas_temperature', len(zone_ids))
    for i, zone_id in enumerate(zone_ids):
        gas_temperature_index[zone_id] = block.start + i


def _add_surface_coverage_state(
    *,
    builder: _LayoutBuilder,
    surface_index: dict[str, dict[str, int]],
    chamber: Any,
    mechanism: Any,
) -> None:
    start = builder.next_index
    for surface in chamber.surfaces:
        mapping: dict[str, int] = {}
        for sp in mechanism.surface_species:
            if surface.surface_id in sp.surfaces:
                mapping[sp.canonical_id] = builder.allocate()
        if mapping:
            surface_index[surface.surface_id] = mapping
    builder.add_optional_block('surface_coverages', start)


def _add_wall_inventory_state(
    *,
    builder: _LayoutBuilder,
    inventory_index: dict[str, dict[str, int]],
    chamber: Any,
) -> None:
    start = builder.next_index
    for surface in chamber.surfaces:
        mapping: dict[str, int] = {}
        for key in surface.initial_inventory.keys():
            mapping[key] = builder.allocate()
        if mapping:
            inventory_index[surface.surface_id] = mapping
    builder.add_optional_block('wall_inventory', start)


def _add_film_thickness_state(
    *,
    builder: _LayoutBuilder,
    film_index: dict[str, int],
    chamber: Any,
) -> None:
    start = builder.next_index
    for surface in chamber.surfaces:
        film_index[surface.surface_id] = builder.allocate()
    builder.add_optional_block('film_thickness', start)


def extra_state_owner_ids(spec: Any, chamber: Any) -> list[str]:
    if spec.scope == 'zone':
        return list(spec.zones) if spec.zones else [z.zone_id for z in chamber.zones]
    if spec.scope == 'surface':
        return list(spec.surfaces) if spec.surfaces else [s.surface_id for s in chamber.surfaces]
    return []


def _add_extra_states(
    *,
    builder: _LayoutBuilder,
    extra_state_index: dict[str, dict[str, int]],
    chamber: Any,
    mechanism: Any,
) -> None:
    start = builder.next_index
    for spec in getattr(mechanism, 'state_variables', []) or []:
        mapping: dict[str, int] = {}
        for owner_id in extra_state_owner_ids(spec, chamber):
            mapping[owner_id] = builder.allocate()
        if mapping:
            extra_state_index[spec.state_id] = mapping
    builder.add_optional_block('extra_states', start)


def build_state_layout(mechanism: Any, chamber: Any, run_config: Any) -> StateLayout:
    gas_species = [s for s in mechanism.gas_state_species]
    zone_ids = [z.zone_id for z in chamber.zones]
    builder = _LayoutBuilder()
    gas_index: dict[str, dict[str, int]] = {}
    surface_index: dict[str, dict[str, int]] = {}
    inventory_index: dict[str, dict[str, int]] = {}
    film_index: dict[str, int] = {}
    extra_state_index: dict[str, dict[str, int]] = {}
    electron_energy_index: dict[str, int] = {}
    gas_temperature_index: dict[str, int] = {}

    _add_gas_density_state(
        builder=builder,
        gas_index=gas_index,
        gas_species=gas_species,
        zone_ids=zone_ids,
    )
    _add_electron_energy_state(
        builder=builder,
        electron_energy_index=electron_energy_index,
        zone_ids=zone_ids,
    )
    if run_config.physics.enable_gas_temperature:
        _add_gas_temperature_state(
            builder=builder,
            gas_temperature_index=gas_temperature_index,
            zone_ids=zone_ids,
        )
    if run_config.physics.enable_surface_coverages:
        _add_surface_coverage_state(
            builder=builder,
            surface_index=surface_index,
            chamber=chamber,
            mechanism=mechanism,
        )
    if run_config.physics.enable_wall_inventory:
        _add_wall_inventory_state(
            builder=builder,
            inventory_index=inventory_index,
            chamber=chamber,
        )
    if run_config.physics.enable_surface_coverages and mechanism_has_film_state(mechanism):
        _add_film_thickness_state(
            builder=builder,
            film_index=film_index,
            chamber=chamber,
        )
    _add_extra_states(
        builder=builder,
        extra_state_index=extra_state_index,
        chamber=chamber,
        mechanism=mechanism,
    )

    return StateLayout(
        slices=builder.slices,
        size=builder.next_index,
        gas_index=gas_index,
        electron_energy_index=electron_energy_index,
        gas_temperature_index=gas_temperature_index,
        surface_index=surface_index,
        inventory_index=inventory_index,
        film_index=film_index,
        extra_state_index=extra_state_index,
        gas_species_ids=[sp.canonical_id for sp in gas_species],
        zone_ids=zone_ids,
    )
