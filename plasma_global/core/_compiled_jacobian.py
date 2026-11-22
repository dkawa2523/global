"""Compile physical state dependencies before materializing a Jacobian pattern."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from plasma_global.core.compiled import CompiledGlobalModel


@dataclass(frozen=True, slots=True)
class StateDependency:
    rows: tuple[int, ...]
    columns: tuple[int, ...]


@dataclass(slots=True)
class DependencyCompiler:
    model: CompiledGlobalModel
    dependencies: list[StateDependency] = field(default_factory=list)

    def add(self, rows: tuple[int, ...], columns: tuple[int, ...]) -> None:
        if rows and columns:
            self.dependencies.append(StateDependency(rows, columns))

    def zone_indices(self, zone_id: str) -> tuple[int, ...]:
        layout = self.model.layout
        density = layout.density_slices[zone_id]
        indices = tuple(range(density.start, density.stop))
        for mapping in (layout.electron_energy_indices, layout.heavy_energy_indices):
            if zone_id in mapping:
                indices += (mapping[zone_id],)
        return indices

    def compile(self) -> tuple[StateDependency, ...]:
        self._zones()
        self._transport()
        self._surfaces()
        self._power()
        # Existing film/inventory states consume events without feeding them back.
        extension = self.model.layout.extension_slice
        self.add(
            tuple(range(extension.start, extension.stop)), tuple(range(extension.start))
        )
        return tuple(dict.fromkeys(self.dependencies))

    def _zones(self) -> None:
        model, layout = self.model, self.model.layout
        reactive = np.any(model.stoichiometry != 0.0, axis=0)
        for zone_id in layout.zone_ids:
            zone = self.zone_indices(zone_id)
            density = layout.density_slices[zone_id]
            species = tuple(range(density.start, density.stop))
            for row, mask in enumerate(model.chemistry_jacobian_species_pattern):
                self.add(
                    (species[row],),
                    tuple(species[index] for index in np.flatnonzero(mask)),
                )
            reactive_rows = tuple(species[index] for index in np.flatnonzero(reactive))
            for mapping in (
                layout.electron_energy_indices,
                layout.heavy_energy_indices,
            ):
                if zone_id in mapping:
                    self.add(reactive_rows, (mapping[zone_id],))
                    self.add((mapping[zone_id],), zone)
            if model._walls_by_zone.get(zone_id):
                self.add(zone, zone)

    def _transport(self) -> None:
        transport = self.model.transport
        if transport is None:
            return
        for zone_id in self.model.layout.zone_ids:
            for index in self.zone_indices(zone_id):
                self.add((index,), (index,))
        for source, target in zip(transport.edge_from, transport.edge_to, strict=True):
            self.add(
                self.zone_indices(self.model.layout.zone_ids[int(target)]),
                self.zone_indices(self.model.layout.zone_ids[int(source)]),
            )

    def _surfaces(self) -> None:
        surface_model = self.model.surface_model
        if surface_model is None:
            return
        for surface in surface_model.surfaces:
            zone = self.zone_indices(surface.zone_id)
            coverage = tuple(
                index
                for key, index in self.model.layout.surface_coverage_indices.items()
                if key[0] == surface.surface_id
            )
            self.add(zone, zone)
            self.add(zone, coverage)
            self.add(coverage, zone)
            self.add(coverage, coverage)

    def _power(self) -> None:
        coordinator = self.model.power_coordinator
        if coordinator is None:
            return
        for segment in self.model.segments:
            for topology in coordinator.topology(segment.port_commands):
                source = self.zone_indices(topology.source_zone_id)
                for zone_id in topology.field_zone_ids:
                    zone = self.zone_indices(zone_id)
                    self.add(zone, zone)
                self._deposition(
                    topology.electron_deposition_zone_ids,
                    self.model.layout.electron_energy_indices,
                    source,
                )
                self._deposition(
                    topology.gas_deposition_zone_ids,
                    self.model.layout.heavy_energy_indices,
                    source,
                )

    def _deposition(
        self,
        targets: tuple[str, ...],
        energy_indices: Mapping[str, int],
        source: tuple[int, ...],
    ) -> None:
        self.add(
            tuple(
                energy_indices[target] for target in targets if target in energy_indices
            ),
            source,
        )


def compile_dependencies(model: CompiledGlobalModel) -> tuple[StateDependency, ...]:
    return DependencyCompiler(model).compile()
