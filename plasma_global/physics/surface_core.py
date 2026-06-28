from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from plasma_global.chemistry.models import E_CHARGE
from plasma_global.physics.surface_rates import evaluate_surface_rate
from plasma_global.physics.types import CompiledSurfaceReaction, CoupledPlasmaEvaluation, SurfaceRateContext

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
        self.enabled = self._surface_rhs_enabled()
        if not self.enabled:
            self.surface_free_site_species = {}
            self.surface_site_occupancy = {}
            self.surface_reactions = []
            return
        self.surface_free_site_species = self._build_surface_free_site_species_map()
        self.surface_site_occupancy = self._build_surface_site_occupancy_map()
        self.surface_reactions = self._compile_surface_reactions()

    def _surface_rhs_enabled(self) -> bool:
        slices = self.system.state_layout.slices
        return any(name in slices for name in ('surface_coverages', 'wall_inventory', 'film_thickness'))

    def _build_surface_free_site_species_map(self) -> dict[str, str | None]:
        sys = self.system
        out: dict[str, str | None] = {}
        for surface_id, mapping in sys.state_layout.surface_index.items():
            candidates = [sp_id for sp_id in mapping if 'site' in sys.surface_species_tags.get(sp_id, set())]
            out[surface_id] = candidates[0] if candidates else None
        return out

    def _build_surface_site_occupancy_map(self) -> dict[str, dict[str, float]]:
        sys = self.system
        out: dict[str, dict[str, float]] = {}
        for surface_id, mapping in sys.state_layout.surface_index.items():
            out[surface_id] = {}
            for sp_id in mapping:
                spec = sys.mechanism.species_by_id[sp_id]
                out[surface_id][sp_id] = max(float(spec.elements.get('site', 1.0)), 1.0e-12)
        return out

    def _compile_surface_reactions(self) -> list[CompiledSurfaceReaction]:
        compiled: list[CompiledSurfaceReaction] = []
        for rxn in self.system.mechanism.surface_reactions:
            if not rxn.enabled:
                continue
            for surface_id in rxn.surface_filter:
                compiled_rxn = self._compile_surface_reaction(rxn, surface_id)
                if compiled_rxn is not None:
                    compiled.append(compiled_rxn)
        return compiled

    def _compile_surface_reaction(self, rxn: Any, surface_id: str) -> CompiledSurfaceReaction | None:
        sys = self.system
        surface = sys.chamber.surface_by_id[surface_id]
        zone_id = surface.zone_id
        if rxn.zone_filter and zone_id not in rxn.zone_filter:
            return None
        gas_reactants = self._gas_side(rxn.reactants)
        surface_delta = self._surface_delta(rxn, surface_id)
        return CompiledSurfaceReaction(
            reaction_id=rxn.reaction_id,
            zone_id=zone_id,
            surface_id=surface_id,
            gas_reactants=gas_reactants,
            surface_reactants=self._surface_side(rxn.reactants, surface_id),
            delta_gas=self._gas_delta(rxn),
            delta_surface=surface_delta,
            area_over_volume=surface.area_m2 / max(sys.chamber.zone_by_id[zone_id].volume_m3, 1.0e-30),
            area_m2=surface.area_m2,
            site_density_m2=max(surface.site_density_m2, 1.0),
            rate_model=sys.mechanism.model(rxn.rate_model_key),
            inventory_idx=self._inventory_index(surface_id, gas_reactants),
            film_factor=self._film_factor(surface_delta),
        )

    def _gas_side(self, side: dict[str, float]) -> list[tuple[int, float, str]]:
        return [
            (self.system.gas_species_index[sp_id], float(nu), sp_id)
            for sp_id, nu in side.items()
            if sp_id in self.system.gas_species_index
        ]

    def _surface_side(self, side: dict[str, float], surface_id: str) -> list[tuple[int, float, str]]:
        mapping = self.system.state_layout.surface_index.get(surface_id, {})
        return [(mapping[sp_id], float(nu), sp_id) for sp_id, nu in side.items() if sp_id in mapping]

    def _surface_delta(self, rxn: Any, surface_id: str) -> list[tuple[int, float, str]]:
        return self._delta_side(rxn, self.system.state_layout.surface_index.get(surface_id, {}))

    def _gas_delta(self, rxn: Any) -> list[tuple[int, float, str]]:
        return self._delta_side(rxn, self.system.gas_species_index)

    @staticmethod
    def _delta_side(rxn: Any, mapping: dict[str, int]) -> list[tuple[int, float, str]]:
        return [
            (idx, float(rxn.products.get(sp_id, 0.0) - rxn.reactants.get(sp_id, 0.0)), sp_id)
            for sp_id, idx in mapping.items()
            if abs(rxn.products.get(sp_id, 0.0) - rxn.reactants.get(sp_id, 0.0)) > 0.0
        ]

    def _inventory_index(self, surface_id: str, gas_reactants: list[tuple[int, float, str]]) -> int | None:
        if not gas_reactants:
            return None
        candidate = f'{gas_reactants[0][2]}_reservoir'
        return self.system.state_layout.inventory_index.get(surface_id, {}).get(candidate)

    def _film_factor(self, surface_delta: list[tuple[int, float, str]]) -> float:
        return sum(
            nu
            for _idx, nu, species_id in surface_delta
            if 'film_fragment' in self.system.surface_species_tags.get(species_id, set())
        )

    def initialize_state(self, y0: np.ndarray) -> None:
        if not self.enabled:
            return
        sys = self.system
        if 'surface_coverages' in sys.state_layout.slices:
            for surface in sys.chamber.surfaces:
                for sp_id, idx in sys.state_layout.surface_index.get(surface.surface_id, {}).items():
                    y0[idx] = float(surface.initial_coverages.get(sp_id, 0.0))
        if 'wall_inventory' in sys.state_layout.slices:
            for surface in sys.chamber.surfaces:
                for key, idx in sys.state_layout.inventory_index.get(surface.surface_id, {}).items():
                    y0[idx] = float(surface.initial_inventory.get(key, 0.0))
        if 'film_thickness' in sys.state_layout.slices:
            for surface_id, idx in sys.state_layout.film_index.items():
                y0[idx] = 0.0

    def project_state(self, y: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return y
        sys = self.system
        if 'surface_coverages' in sys.state_layout.slices:
            self._project_surface_coverages(y)
        if 'wall_inventory' in sys.state_layout.slices:
            self._project_wall_inventory(y)
        return y

    def _project_surface_coverages(self, y: np.ndarray) -> None:
        for surface_id, mapping in self.system.state_layout.surface_index.items():
            for idx in mapping.values():
                y[idx] = np.clip(y[idx], 0.0, 1.0)
            if not mapping:
                continue
            free_site = self.surface_free_site_species.get(surface_id)
            if free_site is not None and free_site in mapping:
                self._project_coverages_with_free_site(y, surface_id, mapping, free_site)
            else:
                self._project_coverages_without_free_site(y, surface_id, mapping)

    def _project_coverages_with_free_site(self, y: np.ndarray, surface_id: str, mapping: dict[str, int], free_site: str) -> None:
        occ_map = self.surface_site_occupancy.get(surface_id, {})
        occ_other = sum(
            occ_map.get(sp_id, 1.0) * y[idx]
            for sp_id, idx in mapping.items()
            if sp_id != free_site
        )
        if occ_other > 1.0:
            scale = 1.0 / occ_other
            for sp_id, idx in mapping.items():
                if sp_id != free_site:
                    y[idx] *= scale
            occ_other = 1.0
        y[mapping[free_site]] = max(1.0 - occ_other, 0.0) / max(occ_map.get(free_site, 1.0), 1.0e-12)

    def _project_coverages_without_free_site(self, y: np.ndarray, surface_id: str, mapping: dict[str, int]) -> None:
        occ_map = self.surface_site_occupancy.get(surface_id, {})
        total = sum(occ_map.get(sp_id, 1.0) * y[idx] for sp_id, idx in mapping.items())
        if total > 1.0:
            scale = 1.0 / total
            for idx in mapping.values():
                y[idx] *= scale

    def _project_wall_inventory(self, y: np.ndarray) -> None:
        inv_slice = self.system.state_layout.slice('wall_inventory')
        y[inv_slice] = np.clip(y[inv_slice], 0.0, None)

    def clip_negative_rhs(self, y: np.ndarray, dydt: np.ndarray) -> None:
        if not self.enabled:
            return
        sys = self.system
        if 'surface_coverages' in sys.state_layout.slices:
            surf_rhs = dydt[sys.state_layout.slice('surface_coverages')]
            surf_state = y[sys.state_layout.slice('surface_coverages')]
            surf_rhs[(surf_state <= 0.0) & (surf_rhs < 0.0)] = 0.0

    def surface_temperature(self, surface_id: str, step: Any) -> float:
        sys = self.system
        base = float(sys.chamber.surface_by_id[surface_id].temperature_K)
        override = (step.surface_overrides.get(surface_id, {}) or {}).get('temperature_K')
        return float(override) if override is not None else base

    def _bohm_proxy_ion_flux_m2_s(self, zone_id: str, coupled: CoupledPlasmaEvaluation) -> float:
        ne_zone = coupled.ne_by_zone[zone_id]
        mean_e_zone = coupled.mean_e_by_zone[zone_id]
        ion_mass_zone = coupled.ion_mass_by_zone[zone_id]
        return float(
            0.61
            * ne_zone
            * np.sqrt(max(mean_e_zone, 0.05) * E_CHARGE / max(ion_mass_zone, 1.0e-30))
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
            step=step,
            coupled=coupled,
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
            eval_ = evaluate_surface_rate(self, self.surface_rate_context(rxn, y, step, coupled))
            rate = eval_.rate_m2_s
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
