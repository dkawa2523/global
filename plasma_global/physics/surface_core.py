from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from plasma_global.chemistry.models import E_CHARGE, K_B
from plasma_global.physics.types import CompiledSurfaceReaction, CoupledPlasmaEvaluation, SurfaceRateEvaluation

MONOLAYER_THICKNESS_M = 3.0e-10


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
        sys = self.system
        compiled: list[CompiledSurfaceReaction] = []
        for rxn in sys.mechanism.surface_reactions:
            if not rxn.enabled:
                continue
            for surface_id in rxn.surface_filter:
                surface = sys.chamber.surface_by_id[surface_id]
                zone_id = surface.zone_id
                if rxn.zone_filter and zone_id not in rxn.zone_filter:
                    continue
                gas_reactants: list[tuple[int, float, str]] = []
                for sp_id, nu in rxn.reactants.items():
                    if sp_id in sys.gas_species_index:
                        gas_reactants.append((sys.gas_species_index[sp_id], float(nu), sp_id))
                surf_reactants: list[tuple[int, float, str]] = []
                surf_delta: list[tuple[int, float, str]] = []
                for sp_id, nu in rxn.reactants.items():
                    if sp_id in sys.state_layout.surface_index.get(surface_id, {}):
                        surf_reactants.append((sys.state_layout.surface_index[surface_id][sp_id], float(nu), sp_id))
                for sp_id, idx in sys.state_layout.surface_index.get(surface_id, {}).items():
                    nu = rxn.products.get(sp_id, 0.0) - rxn.reactants.get(sp_id, 0.0)
                    if abs(nu) > 0.0:
                        surf_delta.append((idx, float(nu), sp_id))
                gas_delta = []
                for sp_id, idx in sys.gas_species_index.items():
                    nu = rxn.products.get(sp_id, 0.0) - rxn.reactants.get(sp_id, 0.0)
                    if abs(nu) > 0.0:
                        gas_delta.append((idx, float(nu), sp_id))
                inventory_idx = None
                if gas_reactants:
                    primary_species_id = gas_reactants[0][2]
                    candidate = f'{primary_species_id}_reservoir'
                    if candidate in sys.state_layout.inventory_index.get(surface_id, {}):
                        inventory_idx = sys.state_layout.inventory_index[surface_id][candidate]
                film_factor = 0.0
                for _, nu, sp_id in surf_delta:
                    if 'film_fragment' in sys.surface_species_tags.get(sp_id, set()):
                        film_factor += nu
                compiled.append(
                    CompiledSurfaceReaction(
                        reaction_id=rxn.reaction_id,
                        zone_id=zone_id,
                        surface_id=surface_id,
                        gas_reactants=gas_reactants,
                        surface_reactants=surf_reactants,
                        delta_gas=gas_delta,
                        delta_surface=surf_delta,
                        area_over_volume=surface.area_m2 / max(sys.chamber.zone_by_id[zone_id].volume_m3, 1.0e-30),
                        area_m2=surface.area_m2,
                        site_density_m2=max(surface.site_density_m2, 1.0),
                        rate_model=sys.mechanism.model(rxn.rate_model_key),
                        inventory_idx=inventory_idx,
                        film_factor=film_factor,
                    )
                )
        return compiled

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
            for surface_id, mapping in sys.state_layout.surface_index.items():
                free_site = self.surface_free_site_species.get(surface_id)
                occ_map = self.surface_site_occupancy.get(surface_id, {})
                for idx in mapping.values():
                    y[idx] = np.clip(y[idx], 0.0, 1.0)
                if not mapping:
                    continue
                if free_site is not None and free_site in mapping:
                    occ_other = 0.0
                    for sp_id, idx in mapping.items():
                        if sp_id == free_site:
                            continue
                        occ_other += occ_map.get(sp_id, 1.0) * y[idx]
                    if occ_other > 1.0:
                        scale = 1.0 / occ_other
                        for sp_id, idx in mapping.items():
                            if sp_id != free_site:
                                y[idx] *= scale
                        occ_other = 1.0
                    free_occ = max(1.0 - occ_other, 0.0)
                    y[mapping[free_site]] = free_occ / max(occ_map.get(free_site, 1.0), 1.0e-12)
                else:
                    total = sum(occ_map.get(sp_id, 1.0) * y[idx] for sp_id, idx in mapping.items())
                    if total > 1.0:
                        scale = 1.0 / total
                        for idx in mapping.values():
                            y[idx] *= scale
        if 'wall_inventory' in sys.state_layout.slices:
            inv_slice = sys.state_layout.slice('wall_inventory')
            y[inv_slice] = np.clip(y[inv_slice], 0.0, None)
        return y

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

    def surface_site_metrics(self, surface_id: str, y: np.ndarray) -> dict[str, float]:
        sys = self.system
        mapping = sys.state_layout.surface_index.get(surface_id, {})
        occ_map = self.surface_site_occupancy.get(surface_id, {})
        free_site = self.surface_free_site_species.get(surface_id)
        total = 0.0
        occupied = 0.0
        free = 0.0
        for sp_id, idx in mapping.items():
            occ = occ_map.get(sp_id, 1.0) * max(float(np.clip(y[idx], 0.0, 1.0)), 0.0)
            total += occ
            if sp_id == free_site:
                free += occ
            else:
                occupied += occ
        return {'occupied_fraction': occupied, 'free_fraction': free, 'total_fraction': total}

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

    def surface_rate(self, rxn: CompiledSurfaceReaction, gas_row: np.ndarray, Tg: float, y: np.ndarray, step: Any, coupled: CoupledPlasmaEvaluation) -> SurfaceRateEvaluation:
        sys = self.system
        model = rxn.rate_model
        backend = str(model.get('backend', '')).lower()
        Ts = self.surface_temperature(rxn.surface_id, step)
        area_surface_ied = (coupled.power.metadata or {}).get('surface_ied', {}).get(rxn.surface_id, {})
        ion_energy_eV = float(area_surface_ied.get('mean_ion_energy_eV', max(coupled.power.plasma_potential_V - coupled.power.self_bias_V, 0.0)))
        ne_zone = coupled.ne_by_zone[rxn.zone_id]
        pos_zone = coupled.pos_by_zone[rxn.zone_id]
        mean_e_zone = coupled.mean_e_by_zone[rxn.zone_id]
        ion_mass_zone = coupled.ion_mass_by_zone[rxn.zone_id]
        ion_flux_total = float(area_surface_ied.get('ion_flux_m2_s', 0.61 * ne_zone * np.sqrt(max(mean_e_zone, 0.05) * E_CHARGE / max(ion_mass_zone, 1.0e-30))))
        rate = 1.0
        dlog_gas: dict[int, float] = {}
        dlog_surface: dict[int, float] = {}
        dlog_Tg = 0.0
        used_surface_species: set[str] = set()

        def add_gas_power(idx: int, power_exp: float) -> None:
            nonlocal rate
            n = max(float(gas_row[idx]), sys.floor_density)
            rate *= n ** power_exp
            dlog_gas[idx] = dlog_gas.get(idx, 0.0) + power_exp / n

        def add_surface_power(state_idx: int, power_exp: float) -> None:
            nonlocal rate
            theta = max(float(np.clip(y[state_idx], 0.0, 1.0)), 1.0e-12)
            rate *= theta ** power_exp
            dlog_surface[state_idx] = dlog_surface.get(state_idx, 0.0) + power_exp / theta

        thermal_factor, _ = self.surface_thermal_prefactor(model, Ts)
        cov_factor, cov_derivs, cov_used = self.surface_coverage_factor(model.get('coverage_factor'), rxn.surface_id, y)
        rate *= thermal_factor * cov_factor
        used_surface_species |= cov_used
        for idx, deriv in cov_derivs.items():
            dlog_surface[idx] = dlog_surface.get(idx, 0.0) + deriv / max(cov_factor, 1.0e-30)

        gas_reactants = rxn.gas_reactants
        ion_reactants = [g for g in gas_reactants if sys.gas_species[g[0]].charge > 0]

        if backend in {'sticking', 'eley_rideal'}:
            primary = gas_reactants[0] if gas_reactants else None
            if primary is None:
                return SurfaceRateEvaluation(0.0)
            idx, _nu, _sp_id = primary
            if sys.gas_species[idx].charge > 0:
                n_i = max(float(gas_row[idx]), sys.floor_density)
                frac_i = min(max(n_i / max(pos_zone, sys.floor_density), 0.0), 1.0)
                incident_flux = ion_flux_total * frac_i
                add_gas_power(idx, 1.0)
                rate *= incident_flux / n_i
                energy_factor = self.surface_energy_factor(model, ion_energy_eV)
                rate *= energy_factor
            else:
                m = sys.gas_masses[idx]
                vth = np.sqrt(8.0 * K_B * max(Tg, 1.0) / (np.pi * max(m, 1.0e-30)))
                rate *= 0.25 * vth * float(model.get('sticking_value', model.get('yield_value', 0.0)))
                add_gas_power(idx, 1.0)
                dlog_Tg += 0.5 / max(Tg, 1.0)
            if sys.gas_species[idx].charge > 0:
                rate *= float(model.get('sticking_value', model.get('yield_value', 1.0)))
            for idx2, nu, _sp in gas_reactants[1:]:
                add_gas_power(idx2, nu)
            for sidx, nu, sp_id in rxn.surface_reactants:
                if sp_id in used_surface_species:
                    continue
                add_surface_power(sidx, nu)
        elif backend in {'ion_assisted', 'sputter_yield'}:
            primary = ion_reactants[0] if ion_reactants else (gas_reactants[0] if gas_reactants else None)
            if primary is None:
                return SurfaceRateEvaluation(0.0)
            idx, _nu, _sp_id = primary
            n_i = max(float(gas_row[idx]), sys.floor_density)
            frac_i = min(max(n_i / max(pos_zone, sys.floor_density), 0.0), 1.0)
            energy_factor = self.surface_energy_factor(model, ion_energy_eV)
            if energy_factor <= 0.0:
                return SurfaceRateEvaluation(0.0)
            yield_base = float(model.get('yield_value', model.get('yield_at_ref', model.get('sticking_value', 1.0))))
            rate *= ion_flux_total * frac_i * yield_base * energy_factor
            add_gas_power(idx, 1.0)
            rate /= n_i
            for sidx, nu, sp_id in rxn.surface_reactants:
                if sp_id in used_surface_species:
                    continue
                add_surface_power(sidx, nu)
            for idx2, nu, _sp in gas_reactants[1:]:
                add_gas_power(idx2, nu)
        elif backend == 'desorption':
            pref = float(model.get('nu0_s_inv', model.get('prefactor_s_inv', model.get('A', 0.0))))
            rate *= pref * rxn.site_density_m2
            for sidx, nu, _sp in rxn.surface_reactants:
                add_surface_power(sidx, nu)
        elif backend == 'langmuir_hinshelwood':
            pref = float(model.get('A_m2_s_inv', model.get('A', 0.0)))
            rate *= pref * rxn.site_density_m2
            for sidx, nu, _sp in rxn.surface_reactants:
                add_surface_power(sidx, nu)
            for idx2, nu, _sp in gas_reactants:
                add_gas_power(idx2, nu)
        elif backend == 'arrhenius':
            pref = float(model.get('A', 0.0))
            rate *= pref * rxn.site_density_m2
            for sidx, nu, _sp in rxn.surface_reactants:
                add_surface_power(sidx, nu)
            for idx2, nu, _sp in gas_reactants:
                add_gas_power(idx2, nu)
        else:
            raise NotImplementedError(f'Unsupported surface rate backend: {backend}')

        d_gas = {idx: rate * val for idx, val in dlog_gas.items()}
        d_surface = {idx: rate * val for idx, val in dlog_surface.items()}
        dTg = rate * dlog_Tg
        return SurfaceRateEvaluation(
            rate_m2_s=float(rate),
            d_gas=d_gas,
            d_surface=d_surface,
            dTg=float(dTg),
            diagnostics={'ion_energy_eV': ion_energy_eV, 'ion_flux_total': ion_flux_total},
        )

    def apply_rhs(self, time_s: float, y: np.ndarray, step: Any, coupled: CoupledPlasmaEvaluation, dydt: np.ndarray) -> None:
        sys = self.system
        gas = coupled.gas
        Tg = coupled.gas_temperature
        gas_rhs = dydt[sys.state_layout.slice('gas_densities')].reshape(sys.n_zones, sys.n_gas_species)
        surf_rhs = dydt[sys.state_layout.slice('surface_coverages')] if 'surface_coverages' in sys.state_layout.slices else None
        inv_rhs = dydt[sys.state_layout.slice('wall_inventory')] if 'wall_inventory' in sys.state_layout.slices else None
        film_rhs = dydt[sys.state_layout.slice('film_thickness')] if 'film_thickness' in sys.state_layout.slices else None
        surf_offset = sys.state_layout.slices['surface_coverages'].start if 'surface_coverages' in sys.state_layout.slices else 0
        inv_offset = sys.state_layout.slices['wall_inventory'].start if 'wall_inventory' in sys.state_layout.slices else 0
        film_offset = sys.state_layout.slices['film_thickness'].start if 'film_thickness' in sys.state_layout.slices else 0

        for rxn in self.surface_reactions:
            z = sys.zone_index[rxn.zone_id]
            eval_ = self.surface_rate(rxn, gas[z], float(Tg[z]), y, step, coupled)
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

    def apply_jacobian(self, time_s: float, y: np.ndarray, step: Any, coupled: CoupledPlasmaEvaluation, J) -> None:
        sys = self.system
        gas = coupled.gas
        Tg = coupled.gas_temperature
        for rxn in self.surface_reactions:
            z = sys.zone_index[rxn.zone_id]
            eval_ = self.surface_rate(rxn, gas[z], float(Tg[z]), y, step, coupled)
            rate = eval_.rate_m2_s
            if rate == 0.0:
                continue
            for s, d_rate_dn in eval_.d_gas.items():
                col = sys.gas_core.gas_state_idx(z, s)
                for idx, nu, _sp in rxn.delta_gas:
                    J[sys.gas_core.gas_state_idx(z, idx), col] += rxn.area_over_volume * nu * d_rate_dn
                if rxn.inventory_idx is not None:
                    J[rxn.inventory_idx, col] += rxn.area_m2 * d_rate_dn
                if abs(rxn.film_factor) > 0.0 and rxn.surface_id in sys.state_layout.film_index:
                    J[sys.state_layout.film_index[rxn.surface_id], col] += MONOLAYER_THICKNESS_M * rxn.film_factor * d_rate_dn / rxn.site_density_m2
            for sidx, d_rate_dtheta in eval_.d_surface.items():
                for idx, nu, _sp in rxn.delta_gas:
                    J[sys.gas_core.gas_state_idx(z, idx), sidx] += rxn.area_over_volume * nu * d_rate_dtheta
                for idx, nu, _sp in rxn.delta_surface:
                    J[idx, sidx] += nu * d_rate_dtheta / rxn.site_density_m2
                if rxn.inventory_idx is not None:
                    J[rxn.inventory_idx, sidx] += rxn.area_m2 * d_rate_dtheta
                if abs(rxn.film_factor) > 0.0 and rxn.surface_id in sys.state_layout.film_index:
                    J[sys.state_layout.film_index[rxn.surface_id], sidx] += MONOLAYER_THICKNESS_M * rxn.film_factor * d_rate_dtheta / rxn.site_density_m2
            if 'gas_temperature' in sys.state_layout.slices and eval_.dTg != 0.0:
                col = sys.state_layout.gas_temperature_index[rxn.zone_id]
                for idx, nu, _sp in rxn.delta_gas:
                    J[sys.gas_core.gas_state_idx(z, idx), col] += rxn.area_over_volume * nu * eval_.dTg
                for idx, nu, _sp in rxn.delta_surface:
                    J[idx, col] += nu * eval_.dTg / rxn.site_density_m2
                if rxn.inventory_idx is not None:
                    J[rxn.inventory_idx, col] += rxn.area_m2 * eval_.dTg
                if abs(rxn.film_factor) > 0.0 and rxn.surface_id in sys.state_layout.film_index:
                    J[sys.state_layout.film_index[rxn.surface_id], col] += MONOLAYER_THICKNESS_M * rxn.film_factor * eval_.dTg / rxn.site_density_m2

    def surface_net_gas_fluxes(self, state: np.ndarray, step: Any, coupled: CoupledPlasmaEvaluation) -> dict[str, dict[str, float]]:
        sys = self.system
        gas = coupled.gas
        fluxes: dict[str, dict[str, float]] = {surface_id: {} for surface_id in sys.surface_ids}
        for rxn in self.surface_reactions:
            z = sys.zone_index[rxn.zone_id]
            eval_ = self.surface_rate(rxn, gas[z], float(coupled.gas_temperature[z]), state, step, coupled)
            rate = eval_.rate_m2_s
            if rate == 0.0:
                continue
            bucket = fluxes.setdefault(rxn.surface_id, {})
            for idx, nu, sp_id in rxn.delta_gas:
                bucket[sp_id] = bucket.get(sp_id, 0.0) - nu * rate
        return fluxes
