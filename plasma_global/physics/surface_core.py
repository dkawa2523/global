from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from plasma_global.chemistry.models import E_CHARGE, K_B
from plasma_global.physics.types import CompiledSurfaceReaction, CoupledPlasmaEvaluation, SurfaceRateContext, SurfaceRateEvaluation

MONOLAYER_THICKNESS_M = 3.0e-10


@dataclass
class _SurfaceRateAccumulator:
    core: Any
    context: SurfaceRateContext
    rate: float = 1.0
    dlog_gas: dict[int, float] = field(default_factory=dict)
    dlog_surface: dict[int, float] = field(default_factory=dict)
    dlog_Tg: float = 0.0
    used_surface_species: set[str] = field(default_factory=set)

    def add_gas_power(self, idx: int, power_exp: float) -> None:
        n = max(float(self.context.gas_row[idx]), self.core.system.floor_density)
        self.rate *= n ** power_exp
        self.dlog_gas[idx] = self.dlog_gas.get(idx, 0.0) + power_exp / n

    def add_surface_power(self, state_idx: int, power_exp: float) -> None:
        theta = max(float(np.clip(self.context.state[state_idx], 0.0, 1.0)), 1.0e-12)
        self.rate *= theta ** power_exp
        self.dlog_surface[state_idx] = self.dlog_surface.get(state_idx, 0.0) + power_exp / theta

    def add_surface_reactants(self, reactants: list[tuple[int, float, str]]) -> None:
        for state_idx, nu, species_id in reactants:
            if species_id not in self.used_surface_species:
                self.add_surface_power(state_idx, nu)

    def result(self) -> SurfaceRateEvaluation:
        return SurfaceRateEvaluation(
            rate_m2_s=float(self.rate),
            d_gas={idx: self.rate * val for idx, val in self.dlog_gas.items()},
            d_surface={idx: self.rate * val for idx, val in self.dlog_surface.items()},
            dTg=float(self.rate * self.dlog_Tg),
        )


@dataclass
class SurfaceCore:
    system: Any
    surface_reactions: list[CompiledSurfaceReaction] = field(init=False)
    surface_free_site_species: dict[str, str | None] = field(init=False)
    surface_site_occupancy: dict[str, dict[str, float]] = field(init=False)

    def __post_init__(self) -> None:
        self.surface_free_site_species = self._build_surface_free_site_species_map()
        self.surface_site_occupancy = self._build_surface_site_occupancy_map()
        self.surface_reactions = self._compile_surface_reactions()

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

    def surface_coverage_factor(self, cfg: dict[str, Any] | None, surface_id: str, y: np.ndarray) -> tuple[float, dict[int, float], set[str]]:
        sys = self.system
        if not cfg:
            return 1.0, {}, set()
        kind = str(cfg.get('kind', 'constant')).lower()
        if kind == 'constant':
            return 1.0, {}, set()
        if kind in {'site_blocking', 'species_power'}:
            site_species = str(cfg.get('site_species') or cfg.get('species'))
            exponent = float(cfg.get('exponent', 1.0))
            idx = sys.state_layout.surface_index[surface_id][site_species]
            theta = float(np.clip(y[idx], 0.0, 1.0))
            theta_eff = max(theta, 1.0e-12)
            factor = theta_eff ** exponent
            deriv = {idx: factor * exponent / theta_eff}
            return factor, deriv, {site_species}
        if kind == 'logistic_switch':
            species = str(cfg['species'])
            midpoint = float(cfg.get('midpoint', 0.5))
            sharpness = float(cfg.get('sharpness', 12.0))
            low = float(cfg.get('low', 0.0))
            high = float(cfg.get('high', 1.0))
            idx = sys.state_layout.surface_index[surface_id][species]
            theta = float(np.clip(y[idx], 0.0, 1.0))
            z = sharpness * (theta - midpoint)
            logistic = 1.0 / (1.0 + np.exp(-z))
            factor = low + (high - low) * logistic
            deriv = {idx: (high - low) * logistic * (1.0 - logistic) * sharpness}
            return float(factor), {idx: float(v) for idx, v in deriv.items()}, {species}
        raise NotImplementedError(f'Unsupported coverage factor kind: {kind}')

    def surface_thermal_prefactor(self, model: dict[str, Any], Ts: float) -> tuple[float, float]:
        beta = float(model.get('beta', 0.0))
        Ea_eV = float(model.get('Ea_eV', model.get('activation_eV', 0.0)))
        factor = (max(Ts, 1.0) / 300.0) ** beta * np.exp(-Ea_eV * E_CHARGE / (K_B * max(Ts, 1.0)))
        dlog_dTs = beta / max(Ts, 1.0) + Ea_eV * E_CHARGE / (K_B * max(Ts, 1.0) ** 2)
        return float(factor), float(dlog_dTs)

    def surface_energy_factor(self, model: dict[str, Any], ion_energy_eV: float) -> float:
        threshold = float(model.get('threshold_eV', 0.0))
        exponent = float(model.get('energy_exponent', 1.0))
        ref = float(model.get('reference_energy_eV', max(threshold + 1.0, 100.0)))
        e = max(float(ion_energy_eV), 0.0)
        if e <= threshold:
            return 0.0
        return ((e - threshold) / max(ref - threshold, 1.0e-6)) ** exponent

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

    def evaluate_surface_rate(self, context: SurfaceRateContext) -> SurfaceRateEvaluation:
        acc = _SurfaceRateAccumulator(self, context)
        self._apply_surface_common_factors(acc)
        backend = str(context.reaction.rate_model.get('backend', '')).lower()
        if backend in {'sticking', 'eley_rideal'}:
            return self._sticking_surface_rate(acc)
        if backend == 'ion_assisted':
            return self._ion_assisted_surface_rate(acc)
        if backend == 'desorption':
            return self._desorption_surface_rate(acc)
        if backend == 'langmuir_hinshelwood':
            return self._langmuir_hinshelwood_surface_rate(acc)
        raise NotImplementedError(f'Unsupported surface rate backend: {backend}')

    def _apply_surface_common_factors(self, acc: _SurfaceRateAccumulator) -> None:
        rxn = acc.context.reaction
        model = rxn.rate_model
        thermal_factor, _ = self.surface_thermal_prefactor(model, acc.context.surface_temperature_K)
        cov_factor, cov_derivs, cov_used = self.surface_coverage_factor(model.get('coverage_factor'), rxn.surface_id, acc.context.state)
        acc.rate *= thermal_factor * cov_factor
        acc.used_surface_species |= cov_used
        for idx, deriv in cov_derivs.items():
            acc.dlog_surface[idx] = acc.dlog_surface.get(idx, 0.0) + deriv / max(cov_factor, 1.0e-30)

    def _sticking_surface_rate(self, acc: _SurfaceRateAccumulator) -> SurfaceRateEvaluation:
        rxn = acc.context.reaction
        primary = rxn.gas_reactants[0] if rxn.gas_reactants else None
        if primary is None:
            return SurfaceRateEvaluation(0.0)
        idx, _nu, _sp_id = primary
        if self.system.gas_species[idx].charge > 0:
            self._apply_ion_sticking_primary(acc, idx)
        else:
            self._apply_neutral_sticking_primary(acc, idx)
        for idx2, nu, _sp in rxn.gas_reactants[1:]:
            acc.add_gas_power(idx2, nu)
        acc.add_surface_reactants(rxn.surface_reactants)
        return acc.result()

    def _apply_ion_sticking_primary(self, acc: _SurfaceRateAccumulator, idx: int) -> None:
        model = acc.context.reaction.rate_model
        n_i = max(float(acc.context.gas_row[idx]), self.system.floor_density)
        frac_i = min(max(n_i / max(acc.context.positive_ion_density_m3, self.system.floor_density), 0.0), 1.0)
        acc.add_gas_power(idx, 1.0)
        acc.rate *= acc.context.ion_flux_m2_s * frac_i / n_i
        acc.rate *= self.surface_energy_factor(model, acc.context.ion_energy_eV)
        acc.rate *= float(model.get('sticking_value', model.get('yield_value', 1.0)))

    def _apply_neutral_sticking_primary(self, acc: _SurfaceRateAccumulator, idx: int) -> None:
        model = acc.context.reaction.rate_model
        Tg = acc.context.gas_temperature_K
        mass = self.system.gas_masses[idx]
        thermal_speed = np.sqrt(8.0 * K_B * max(Tg, 1.0) / (np.pi * max(mass, 1.0e-30)))
        acc.rate *= 0.25 * thermal_speed * float(model.get('sticking_value', model.get('yield_value', 0.0)))
        acc.add_gas_power(idx, 1.0)
        acc.dlog_Tg += 0.5 / max(Tg, 1.0)

    def _ion_assisted_surface_rate(self, acc: _SurfaceRateAccumulator) -> SurfaceRateEvaluation:
        rxn = acc.context.reaction
        primary = self._primary_ion_reactant(rxn) or (rxn.gas_reactants[0] if rxn.gas_reactants else None)
        if primary is None:
            return SurfaceRateEvaluation(0.0)
        idx, _nu, _sp_id = primary
        energy_factor = self.surface_energy_factor(rxn.rate_model, acc.context.ion_energy_eV)
        if energy_factor <= 0.0:
            return SurfaceRateEvaluation(0.0)
        n_i = max(float(acc.context.gas_row[idx]), self.system.floor_density)
        frac_i = min(max(n_i / max(acc.context.positive_ion_density_m3, self.system.floor_density), 0.0), 1.0)
        yield_base = float(rxn.rate_model.get('yield_value', rxn.rate_model.get('yield_at_ref', rxn.rate_model.get('sticking_value', 1.0))))
        acc.rate *= acc.context.ion_flux_m2_s * frac_i * yield_base * energy_factor
        acc.add_gas_power(idx, 1.0)
        acc.rate /= n_i
        acc.add_surface_reactants(rxn.surface_reactants)
        for idx2, nu, _sp in rxn.gas_reactants[1:]:
            acc.add_gas_power(idx2, nu)
        return acc.result()

    def _primary_ion_reactant(self, rxn: CompiledSurfaceReaction) -> tuple[int, float, str] | None:
        for item in rxn.gas_reactants:
            if self.system.gas_species[item[0]].charge > 0:
                return item
        return None

    def _desorption_surface_rate(self, acc: _SurfaceRateAccumulator) -> SurfaceRateEvaluation:
        rxn = acc.context.reaction
        acc.rate *= float(rxn.rate_model.get('nu0_s_inv', rxn.rate_model.get('prefactor_s_inv', rxn.rate_model.get('A', 0.0)))) * rxn.site_density_m2
        acc.add_surface_reactants(rxn.surface_reactants)
        return acc.result()

    def _langmuir_hinshelwood_surface_rate(self, acc: _SurfaceRateAccumulator) -> SurfaceRateEvaluation:
        rxn = acc.context.reaction
        acc.rate *= float(rxn.rate_model.get('A_m2_s_inv', rxn.rate_model.get('A', 0.0))) * rxn.site_density_m2
        acc.add_surface_reactants(rxn.surface_reactants)
        for idx, nu, _sp in rxn.gas_reactants:
            acc.add_gas_power(idx, nu)
        return acc.result()

    def apply_rhs(self, _time_s: float, y: np.ndarray, step: Any, coupled: CoupledPlasmaEvaluation, dydt: np.ndarray) -> None:
        sys = self.system
        gas_rhs = dydt[sys.state_layout.slice('gas_densities')].reshape(sys.n_zones, sys.n_gas_species)
        surf_rhs = dydt[sys.state_layout.slice('surface_coverages')] if 'surface_coverages' in sys.state_layout.slices else None
        inv_rhs = dydt[sys.state_layout.slice('wall_inventory')] if 'wall_inventory' in sys.state_layout.slices else None
        film_rhs = dydt[sys.state_layout.slice('film_thickness')] if 'film_thickness' in sys.state_layout.slices else None
        surf_offset = sys.state_layout.slices['surface_coverages'].start if 'surface_coverages' in sys.state_layout.slices else 0
        inv_offset = sys.state_layout.slices['wall_inventory'].start if 'wall_inventory' in sys.state_layout.slices else 0
        film_offset = sys.state_layout.slices['film_thickness'].start if 'film_thickness' in sys.state_layout.slices else 0

        for rxn in self.surface_reactions:
            z = sys.zone_index[rxn.zone_id]
            eval_ = self.evaluate_surface_rate(self.surface_rate_context(rxn, y, step, coupled))
            rate = eval_.rate_m2_s
            if rate == 0.0:
                continue
            for idx, nu, _sp in rxn.delta_gas:
                gas_rhs[z, idx] += rxn.area_over_volume * nu * rate
            if surf_rhs is not None:
                for idx, nu, _sp in rxn.delta_surface:
                    surf_rhs[idx - surf_offset] += nu * rate / rxn.site_density_m2
            if inv_rhs is not None and rxn.inventory_idx is not None:
                inv_rhs[rxn.inventory_idx - inv_offset] += rxn.area_m2 * rate
            if film_rhs is not None and abs(rxn.film_factor) > 0.0 and rxn.surface_id in sys.state_layout.film_index:
                fidx = sys.state_layout.film_index[rxn.surface_id] - film_offset
                film_rhs[fidx] += MONOLAYER_THICKNESS_M * rxn.film_factor * rate / rxn.site_density_m2
