from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from plasma_global.physics.surface_reactions import compile_surface_reactions
from plasma_global.physics.surface_rates import evaluate_surface_rate
from plasma_global.physics.surface_state import (
    build_surface_free_site_species_map,
    build_surface_site_occupancy_map,
    clip_negative_surface_rhs,
    initialize_surface_state,
    project_surface_state,
    surface_state_enabled,
)
from plasma_global.physics.types import CompiledSurfaceReaction, CoupledPlasmaEvaluation, SurfaceRateContext
from plasma_global.reactor.surface_models import bohm_ion_flux_m2_s

MONOLAYER_THICKNESS_M = 3.0e-10


@dataclass
class _SurfaceRhsViews:
    gas_rhs: np.ndarray
    surface_rhs: np.ndarray | None
    inventory_rhs: np.ndarray | None
    film_rhs: np.ndarray | None
    surface_offset: int
    inventory_offset: int
    film_offset: int


@dataclass
class SurfaceCore:
    system: Any
    enabled: bool = field(init=False)
    surface_reactions: list[CompiledSurfaceReaction] = field(init=False)
    surface_free_site_species: dict[str, str | None] = field(init=False)
    surface_site_occupancy: dict[str, dict[str, float]] = field(init=False)

    def __post_init__(self) -> None:
        self.enabled = surface_state_enabled(self.system)
        if not self.enabled:
            self.surface_free_site_species = {}
            self.surface_site_occupancy = {}
            self.surface_reactions = []
            return
        self.surface_free_site_species = build_surface_free_site_species_map(self.system)
        self.surface_site_occupancy = build_surface_site_occupancy_map(self.system)
        self.surface_reactions = compile_surface_reactions(self.system)

    def initialize_state(self, y0: np.ndarray) -> None:
        if not self.enabled:
            return
        initialize_surface_state(self.system, y0)

    def project_state(self, y: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return y
        return project_surface_state(self.system, y, self.surface_free_site_species, self.surface_site_occupancy)

    def clip_negative_rhs(self, y: np.ndarray, dydt: np.ndarray) -> None:
        if not self.enabled:
            return
        clip_negative_surface_rhs(self.system, y, dydt)

    def surface_temperature(self, surface_id: str, step: Any) -> float:
        sys = self.system
        base = float(sys.chamber.surface_by_id[surface_id].temperature_K)
        override = (step.surface_overrides.get(surface_id, {}) or {}).get('temperature_K')
        return float(override) if override is not None else base

    def _bohm_proxy_ion_flux_m2_s(self, zone_id: str, coupled: CoupledPlasmaEvaluation) -> float:
        return bohm_ion_flux_m2_s(
            coupled.ne_by_zone[zone_id],
            coupled.mean_e_by_zone[zone_id],
            coupled.ion_mass_by_zone[zone_id],
        )

    def surface_ion_flux_m2_s(
        self,
        surface_id: str,
        zone_id: str,
        gas_row: np.ndarray,
        coupled: CoupledPlasmaEvaluation,
    ) -> float:
        ied = coupled.power.surface_ied.get(surface_id)
        if ied is not None:
            return max(float(ied.ion_flux_m2_s), 0.0)

        wall_flux = self.system.gas_core.ion_wall_loss_flux_m2_s(
            zone_id,
            gas_row,
            coupled.mean_e_by_zone[zone_id],
        )
        if wall_flux > 0.0:
            return wall_flux
        return max(self._bohm_proxy_ion_flux_m2_s(zone_id, coupled), 0.0)

    def surface_rate_context(
        self,
        rxn: CompiledSurfaceReaction,
        y: np.ndarray,
        step: Any,
        coupled: CoupledPlasmaEvaluation,
    ) -> SurfaceRateContext:
        z = self.system.zone_index[rxn.zone_id]
        gas_row = coupled.gas[z]
        area_surface_ied = coupled.power.surface_ied.get(rxn.surface_id)
        ion_energy_eV = (
            float(area_surface_ied.mean_ion_energy_eV)
            if area_surface_ied is not None
            else max(coupled.power.plasma_potential_V - coupled.power.self_bias_V, 0.0)
        )
        return SurfaceRateContext(
            reaction=rxn,
            gas_row=gas_row,
            gas_temperature_K=float(coupled.gas_temperature[z]),
            state=y,
            surface_temperature_K=self.surface_temperature(rxn.surface_id, step),
            ion_energy_eV=ion_energy_eV,
            positive_ion_density_m3=coupled.pos_by_zone[rxn.zone_id],
            ion_flux_m2_s=self.surface_ion_flux_m2_s(rxn.surface_id, rxn.zone_id, gas_row, coupled),
        )

    def _rhs_views(self, dydt: np.ndarray) -> _SurfaceRhsViews:
        sys = self.system
        slices = sys.state_layout.slices
        return _SurfaceRhsViews(
            gas_rhs=dydt[sys.state_layout.slice('gas_densities')].reshape(sys.n_zones, sys.n_gas_species),
            surface_rhs=dydt[sys.state_layout.slice('surface_coverages')] if 'surface_coverages' in slices else None,
            inventory_rhs=dydt[sys.state_layout.slice('wall_inventory')] if 'wall_inventory' in slices else None,
            film_rhs=dydt[sys.state_layout.slice('film_thickness')] if 'film_thickness' in slices else None,
            surface_offset=slices['surface_coverages'].start if 'surface_coverages' in slices else 0,
            inventory_offset=slices['wall_inventory'].start if 'wall_inventory' in slices else 0,
            film_offset=slices['film_thickness'].start if 'film_thickness' in slices else 0,
        )

    def apply_rhs(self, _time_s: float, y: np.ndarray, step: Any, coupled: CoupledPlasmaEvaluation, dydt: np.ndarray) -> None:
        if not self.enabled or not self.surface_reactions:
            return
        sys = self.system
        view = self._rhs_views(dydt)

        for rxn in self.surface_reactions:
            z = sys.zone_index[rxn.zone_id]
            rate = evaluate_surface_rate(self, self.surface_rate_context(rxn, y, step, coupled))
            if rate == 0.0:
                continue
            for idx, nu, _sp in rxn.delta_gas:
                view.gas_rhs[z, idx] += rxn.area_over_volume * nu * rate
            if view.surface_rhs is not None:
                for idx, nu, _sp in rxn.delta_surface:
                    view.surface_rhs[idx - view.surface_offset] += nu * rate / rxn.site_density_m2
            if view.inventory_rhs is not None and rxn.inventory_idx is not None:
                view.inventory_rhs[rxn.inventory_idx - view.inventory_offset] += rxn.area_m2 * rate
            if view.film_rhs is not None and abs(rxn.film_factor) > 0.0 and rxn.surface_id in sys.state_layout.film_index:
                fidx = sys.state_layout.film_index[rxn.surface_id] - view.film_offset
                view.film_rhs[fidx] += MONOLAYER_THICKNESS_M * rxn.film_factor * rate / rxn.site_density_m2
