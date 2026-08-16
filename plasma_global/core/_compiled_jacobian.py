"""Build the conservative Jacobian sparsity pattern for a compiled model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from plasma_global.core.compiled import CompiledGlobalModel
    from plasma_global.models.power import PowerCoordinator


@dataclass(frozen=True, slots=True)
class JacobianBuilder:
    """Assemble cross-component dependency blocks without evaluating the RHS."""

    model: CompiledGlobalModel

    def build(self) -> csr_matrix:
        model = self.model
        pattern = np.zeros((model.layout.size, model.layout.size), dtype=bool)
        self._add_zone_blocks(pattern)
        self._add_transport_blocks(pattern)
        self._add_surface_blocks(pattern)
        self._add_power_blocks(pattern)
        if model.extension_accumulator is not None:
            # Extensions consume physical drivers; the physical core never
            # depends on extension state.
            pattern[model.layout.extension_slice, :] = True
        return csr_matrix(pattern)

    def _zone_state_bounds(self, zone_id: str) -> tuple[int, int]:
        layout = self.model.layout
        density_slice = layout.density_slices[zone_id]
        stop = density_slice.stop
        if layout.evolves_electron_energy:
            stop = layout.electron_energy_indices[zone_id] + 1
        if layout.evolves_heavy_energy:
            stop = layout.heavy_energy_indices[zone_id] + 1
        return density_slice.start, stop

    def _add_zone_blocks(self, pattern: np.ndarray) -> None:
        model = self.model
        layout = model.layout
        reactive_species = np.any(model.stoichiometry != 0.0, axis=0)
        for zone_id in layout.zone_ids:
            start, stop = self._zone_state_bounds(zone_id)
            density_slice = layout.density_slices[zone_id]
            pattern[density_slice, density_slice] = (
                model.chemistry_jacobian_species_pattern
            )
            if layout.evolves_electron_energy:
                energy_index = layout.electron_energy_indices[zone_id]
                pattern[density_slice, energy_index] = reactive_species
                pattern[energy_index, start:stop] = True
            if layout.evolves_heavy_energy:
                heavy_index = layout.heavy_energy_indices[zone_id]
                pattern[density_slice, heavy_index] = reactive_species
                pattern[heavy_index, start:stop] = True
            if model._walls_by_zone.get(zone_id):
                pattern[start:stop, start:stop] = True

    def _add_transport_blocks(self, pattern: np.ndarray) -> None:
        model = self.model
        transport = model.transport
        if transport is None:
            return
        for zone_id in model.layout.zone_ids:
            start, stop = self._zone_state_bounds(zone_id)
            pattern[start:stop, start:stop] |= np.eye(stop - start, dtype=bool)
        for source_index, target_index in zip(
            transport.edge_from, transport.edge_to, strict=True
        ):
            source_zone = model.layout.zone_ids[int(source_index)]
            target_zone = model.layout.zone_ids[int(target_index)]
            source_start, source_stop = self._zone_state_bounds(source_zone)
            target_start, target_stop = self._zone_state_bounds(target_zone)
            pattern[target_start:target_stop, source_start:source_stop] = True

    def _add_surface_blocks(self, pattern: np.ndarray) -> None:
        model = self.model
        surface_model = model.surface_model
        if surface_model is None:
            return
        for surface in surface_model.surfaces:
            zone_start, zone_stop = self._zone_state_bounds(surface.zone_id)
            pattern[zone_start:zone_stop, zone_start:zone_stop] = True
            coverage_indices = [
                index
                for key, index in model.layout.surface_coverage_indices.items()
                if key[0] == surface.surface_id
            ]
            if coverage_indices:
                pattern[zone_start:zone_stop, coverage_indices] = True
                pattern[coverage_indices, zone_start:zone_stop] = True
                pattern[np.ix_(coverage_indices, coverage_indices)] = True

    def _add_power_blocks(self, pattern: np.ndarray) -> None:
        coordinator = self.model.power_coordinator
        if coordinator is None:
            return
        self._add_field_power_blocks(pattern, coordinator)
        self._add_distributed_power_blocks(pattern, coordinator)

    def _add_field_power_blocks(
        self, pattern: np.ndarray, coordinator: PowerCoordinator
    ) -> None:
        for zone_id in self.model.layout.zone_ids:
            if not any(
                coordinator.has_reduced_field_source(zone_id, segment.port_commands)
                for segment in self.model.segments
            ):
                continue
            start, stop = self._zone_state_bounds(zone_id)
            pattern[start:stop, start:stop] = True

    def _add_distributed_power_blocks(
        self, pattern: np.ndarray, coordinator: PowerCoordinator
    ) -> None:
        if not self.model.layout.evolves_electron_energy:
            return
        energy_indices = self.model.layout.electron_energy_indices
        all_energy_rows = list(energy_indices.values())
        for distributor in coordinator._zone_power_distributors:
            target_ids = distributor.static_target_zone_ids
            target_rows = (
                all_energy_rows
                if target_ids is None
                else [
                    energy_indices[target]
                    for target in target_ids
                    if target in energy_indices
                ]
            )
            if not target_rows:
                continue
            source_start, source_stop = self._zone_state_bounds(
                distributor.source_zone_id
            )
            pattern[target_rows, source_start:source_stop] = True


__all__ = ["JacobianBuilder"]
