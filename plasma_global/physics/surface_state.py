from __future__ import annotations

from typing import Any

import numpy as np


def surface_state_enabled(system: Any) -> bool:
    slices = system.state_layout.slices
    return any(name in slices for name in ('surface_coverages', 'wall_inventory', 'film_thickness'))


def build_surface_free_site_species_map(system: Any) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for surface_id, mapping in system.state_layout.surface_index.items():
        candidates = [sp_id for sp_id in mapping if 'site' in system.surface_species_tags.get(sp_id, set())]
        out[surface_id] = candidates[0] if candidates else None
    return out


def build_surface_site_occupancy_map(system: Any) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for surface_id, mapping in system.state_layout.surface_index.items():
        out[surface_id] = {}
        for sp_id in mapping:
            spec = system.mechanism.species_by_id[sp_id]
            out[surface_id][sp_id] = max(float(spec.elements.get('site', 1.0)), 1.0e-12)
    return out


def initialize_surface_state(system: Any, y0: np.ndarray) -> None:
    if 'surface_coverages' in system.state_layout.slices:
        for surface in system.chamber.surfaces:
            for sp_id, idx in system.state_layout.surface_index.get(surface.surface_id, {}).items():
                y0[idx] = float(surface.initial_coverages.get(sp_id, 0.0))
    if 'wall_inventory' in system.state_layout.slices:
        for surface in system.chamber.surfaces:
            for key, idx in system.state_layout.inventory_index.get(surface.surface_id, {}).items():
                y0[idx] = float(surface.initial_inventory.get(key, 0.0))
    if 'film_thickness' in system.state_layout.slices:
        for idx in system.state_layout.film_index.values():
            y0[idx] = 0.0


def project_surface_state(
    system: Any,
    y: np.ndarray,
    free_site_species: dict[str, str | None],
    site_occupancy: dict[str, dict[str, float]],
) -> np.ndarray:
    if 'surface_coverages' in system.state_layout.slices:
        _project_surface_coverages(system, y, free_site_species, site_occupancy)
    if 'wall_inventory' in system.state_layout.slices:
        _project_wall_inventory(system, y)
    return y


def clip_negative_surface_rhs(system: Any, y: np.ndarray, dydt: np.ndarray) -> None:
    if 'surface_coverages' not in system.state_layout.slices:
        return
    surf_slice = system.state_layout.slice('surface_coverages')
    surf_rhs = dydt[surf_slice]
    surf_state = y[surf_slice]
    surf_rhs[(surf_state <= 0.0) & (surf_rhs < 0.0)] = 0.0


def _project_surface_coverages(
    system: Any,
    y: np.ndarray,
    free_site_species: dict[str, str | None],
    site_occupancy: dict[str, dict[str, float]],
) -> None:
    for surface_id, mapping in system.state_layout.surface_index.items():
        for idx in mapping.values():
            y[idx] = np.clip(y[idx], 0.0, 1.0)
        if not mapping:
            continue
        free_site = free_site_species.get(surface_id)
        if free_site is not None and free_site in mapping:
            _project_coverages_with_free_site(y, surface_id, mapping, free_site, site_occupancy)
        else:
            _project_coverages_without_free_site(y, surface_id, mapping, site_occupancy)


def _project_coverages_with_free_site(
    y: np.ndarray,
    surface_id: str,
    mapping: dict[str, int],
    free_site: str,
    site_occupancy: dict[str, dict[str, float]],
) -> None:
    occ_map = site_occupancy.get(surface_id, {})
    occ_other = sum(occ_map.get(sp_id, 1.0) * y[idx] for sp_id, idx in mapping.items() if sp_id != free_site)
    if occ_other > 1.0:
        scale = 1.0 / occ_other
        for sp_id, idx in mapping.items():
            if sp_id != free_site:
                y[idx] *= scale
        occ_other = 1.0
    y[mapping[free_site]] = max(1.0 - occ_other, 0.0) / max(occ_map.get(free_site, 1.0), 1.0e-12)


def _project_coverages_without_free_site(
    y: np.ndarray,
    surface_id: str,
    mapping: dict[str, int],
    site_occupancy: dict[str, dict[str, float]],
) -> None:
    occ_map = site_occupancy.get(surface_id, {})
    total = sum(occ_map.get(sp_id, 1.0) * y[idx] for sp_id, idx in mapping.items())
    if total > 1.0:
        scale = 1.0 / total
        for idx in mapping.values():
            y[idx] *= scale


def _project_wall_inventory(system: Any, y: np.ndarray) -> None:
    inv_slice = system.state_layout.slice('wall_inventory')
    y[inv_slice] = np.clip(y[inv_slice], 0.0, None)
