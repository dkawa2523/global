from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from plasma_global.chemistry.models import E_CHARGE, K_B
from plasma_global.physics.types import CompiledGasReaction, CoupledPlasmaEvaluation

SCCM_TO_PARTICLES_PER_S = 4.477962e17
BOHM_FLUX_COEFF = 0.61
EV_TO_K = E_CHARGE / K_B


@dataclass
class GasPhaseCore:
    system: Any
    gas_reactions: list[CompiledGasReaction] = field(init=False)
    zone_residence_time_s: dict[str, float] = field(init=False)
    zone_wall_temperature: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self.zone_wall_temperature = self._compute_zone_wall_temperatures()
        self.zone_residence_time_s = self._compute_zone_residence_times()
        self.gas_reactions = self._compile_gas_reactions()

    def _compute_zone_wall_temperatures(self) -> dict[str, float]:
        sys = self.system
        out: dict[str, float] = {}
        for zone_id in sys.zone_ids:
            surfaces = sys.chamber.surfaces_by_zone.get(zone_id, [])
            if not surfaces:
                out[zone_id] = sys.chamber.zone_by_id[zone_id].gas_temperature_K
                continue
            num = sum(s.area_m2 * s.temperature_K for s in surfaces)
            den = sum(s.area_m2 for s in surfaces)
            out[zone_id] = num / max(den, 1.0)
        return out

    def _compute_zone_residence_times(self) -> dict[str, float]:
        sys = self.system
        out: dict[str, float] = {}
        for zone in sys.chamber.zones:
            q_out = sum(p.speed_m3_s for p in sys.chamber.pumps if p.zone_id == zone.zone_id)
            q_out += sum(e.conductance_m3_s for e in sys.chamber.edges if e.from_zone == zone.zone_id)
            if q_out <= 0.0:
                out[zone.zone_id] = float('inf')
            else:
                out[zone.zone_id] = zone.volume_m3 / q_out
        return out

    def _compile_gas_reactions(self) -> list[CompiledGasReaction]:
        sys = self.system
        compiled: list[CompiledGasReaction] = []
        for rxn in sys.mechanism.gas_reactions:
            if not rxn.enabled:
                continue
            reactant_gas: list[tuple[int, float, str]] = []
            electron_sto = 0.0
            for sp_id, nu in rxn.reactants.items():
                if sp_id == sys.mechanism.electron_species_id:
                    electron_sto += nu
                elif sp_id in sys.gas_species_index:
                    reactant_gas.append((sys.gas_species_index[sp_id], float(nu), sp_id))
            delta_gas = []
            for sp_id, idx in sys.gas_species_index.items():
                nu = rxn.products.get(sp_id, 0.0) - rxn.reactants.get(sp_id, 0.0)
                if abs(nu) > 0.0:
                    delta_gas.append((idx, float(nu), sp_id))
            compiled.append(
                CompiledGasReaction(
                    reaction_id=rxn.reaction_id,
                    zones=rxn.zone_filter or sys.zone_ids,
                    reactant_gas=reactant_gas,
                    electron_reactant_stoich=electron_sto,
                    delta_gas=delta_gas,
                    rate_model=sys.mechanism.model(rxn.rate_model_key),
                    energy_model=sys.mechanism.model(rxn.energy_model_key) if rxn.energy_model_key else None,
                )
            )
        return compiled

    def initialize_state(self, y0: np.ndarray) -> None:
        sys = self.system
        first_step = sys.recipe.steps[0]
        global_mix = self.step_global_mix(first_step)
        for zone_id in sys.zone_ids:
            zone = sys.chamber.zone_by_id[zone_id]
            T = zone.gas_temperature_K
            n_total = zone.pressure_Pa / (K_B * max(T, 1.0))
            zone_mix = self.step_zone_mix(first_step, zone_id)
            mix = zone_mix if zone_mix else global_mix
            for sp_id, frac in mix.items():
                if sp_id in sys.gas_species_index:
                    y0[sys.state_layout.gas_index[zone_id][sp_id]] = max(frac, 0.0) * n_total
            for sp in sys.gas_species:
                if sp.charge > 0:
                    idx = sys.state_layout.gas_index[zone_id][sp.canonical_id]
                    y0[idx] = max(y0[idx], 1.0e12)
                elif sp.charge < 0:
                    idx = sys.state_layout.gas_index[zone_id][sp.canonical_id]
                    y0[idx] = max(min(y0[idx], 1.0e10), 0.0)
            for sp_id, density in (zone.initial_densities_m3 or {}).items():
                if sp_id in sys.gas_species_index:
                    y0[sys.state_layout.gas_index[zone_id][sp_id]] = max(float(density), 0.0)
            prescribed_ne = sys.prescribed_electron_density_by_zone(sys.recipe.steps[0].t_start_s)
            ne = (
                max(float(prescribed_ne[zone_id]), sys.floor_density)
                if prescribed_ne is not None and zone_id in prescribed_ne
                else self.electron_density_from_state_row(self.gas_row(y0, zone_id))
            )
            y0[sys.state_layout.electron_energy_index[zone_id]] = 3.0 * ne * E_CHARGE
            if 'gas_temperature' in sys.state_layout.slices:
                y0[sys.state_layout.gas_temperature_index[zone_id]] = T

    def project_state(self, y: np.ndarray) -> np.ndarray:
        sys = self.system
        gas_slice = sys.state_layout.slice('gas_densities')
        y[gas_slice] = np.clip(y[gas_slice], 0.0, None)
        if 'electron_energy' in sys.state_layout.slices:
            e_slice = sys.state_layout.slice('electron_energy')
            y[e_slice] = np.clip(y[e_slice], sys.floor_energy, None)
        if 'gas_temperature' in sys.state_layout.slices:
            t_slice = sys.state_layout.slice('gas_temperature')
            y[t_slice] = np.clip(y[t_slice], 50.0, None)
        return y

    def clip_negative_rhs(self, y: np.ndarray, dydt: np.ndarray) -> None:
        sys = self.system
        gas_rhs = dydt[sys.state_layout.slice('gas_densities')].reshape(sys.n_zones, sys.n_gas_species)
        gas_state = y[sys.state_layout.slice('gas_densities')].reshape(sys.n_zones, sys.n_gas_species)
        gas_floor = sys.floor_density
        low_mask = (gas_state <= gas_floor) & (gas_rhs < 0.0)
        gas_rhs[low_mask] = 0.0
        if 'electron_energy' in sys.state_layout.slices:
            We_rhs = dydt[sys.state_layout.slice('electron_energy')]
            low = y[sys.state_layout.slice('electron_energy')] <= sys.floor_energy
            We_rhs[low & (We_rhs < 0.0)] = 0.0

    def step_zone_mix(self, step: Any, zone_id: str) -> dict[str, float]:
        sys = self.system
        mix: dict[str, float] = {}
        total = 0.0
        for inlet_id, flows in step.gas_inlets.items():
            inlet = sys.chamber.inlet_by_id.get(inlet_id)
            if inlet is None or inlet.zone_id != zone_id:
                continue
            for sp_id, value in flows.items():
                total += float(value)
                mix[sp_id] = mix.get(sp_id, 0.0) + float(value)
        if total <= 0.0:
            return {}
        return {sp: val / total for sp, val in mix.items()}

    def step_global_mix(self, step: Any) -> dict[str, float]:
        mix: dict[str, float] = {}
        total = 0.0
        for flows in step.gas_inlets.values():
            for sp_id, value in flows.items():
                total += float(value)
                mix[sp_id] = mix.get(sp_id, 0.0) + float(value)
        if total <= 0.0:
            return {}
        return {sp: val / total for sp, val in mix.items()}

    def gas_row(self, state: np.ndarray, zone_id: str) -> np.ndarray:
        sys = self.system
        base = state[sys.state_layout.slice('gas_densities')].reshape(sys.n_zones, sys.n_gas_species)
        return base[sys.zone_index[zone_id]]

    def gas_state_idx(self, zone_idx: int, species_idx: int) -> int:
        sys = self.system
        return sys.state_layout.gas_index[sys.zone_ids[zone_idx]][sys.gas_species_ids[species_idx]]

    def negative_ion_density_from_state_row(self, gas_row: np.ndarray) -> float:
        sys = self.system
        total = 0.0
        for i, sp in enumerate(sys.gas_species):
            if sp.charge < 0:
                total += abs(sp.charge) * max(float(gas_row[i]), 0.0)
        return max(total, 0.0)

    def ion_species_payload_from_state_row(self, gas_row: np.ndarray) -> dict[str, dict[str, float]]:
        sys = self.system
        payload: dict[str, dict[str, float]] = {}
        for i in sys.positive_ion_local_indices:
            density = max(float(gas_row[i]), 0.0)
            if density <= 0.0:
                continue
            sp = sys.gas_species[i]
            payload[sp.canonical_id] = {
                'density_m3': density,
                'charge': float(sp.charge),
                'mass_kg': float(sys.gas_masses[i]),
            }
        return payload

    def electron_density_from_state_row(self, gas_row: np.ndarray) -> float:
        sys = self.system
        ne = float(np.dot(sys.gas_charges, gas_row))
        return max(ne, sys.floor_density)

    def electron_density_deriv_active(self, gas_row: np.ndarray) -> bool:
        return float(np.dot(self.system.gas_charges, gas_row)) > self.system.floor_density

    def positive_ion_density_from_state_row(self, gas_row: np.ndarray) -> float:
        sys = self.system
        total = 0.0
        for i, sp in enumerate(sys.gas_species):
            if sp.charge > 0:
                total += sp.charge * max(float(gas_row[i]), 0.0)
        return max(total, sys.floor_density)

    def dominant_ion_mass_kg_from_state_row(self, gas_row: np.ndarray) -> float:
        sys = self.system
        best_density = -1.0
        best_mass = None
        for i in sys.positive_ion_local_indices:
            n_i = max(float(gas_row[i]), 0.0)
            if n_i > best_density:
                best_density = n_i
                best_mass = sys.gas_masses[i]
        return float(best_mass if best_mass is not None else 6.63e-26)

    def zone_pressure_from_state_row(self, gas_row: np.ndarray, gas_temperature_K: float) -> float:
        sys = self.system
        n_total = max(float(np.sum(np.clip(gas_row, 0.0, None))), sys.floor_density)
        return n_total * K_B * max(float(gas_temperature_K), 1.0)

    def mean_energy_eV(self, We_J_m3: float, ne_m3: float) -> float:
        sys = self.system
        return max(float(We_J_m3), sys.floor_energy) / max(ne_m3, sys.floor_density) / E_CHARGE

    def zone_state_meta(self, gas: np.ndarray, We: np.ndarray) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
        sys = self.system
        ne_by_zone: dict[str, float] = {}
        mean_e_by_zone: dict[str, float] = {}
        pos_by_zone: dict[str, float] = {}
        ion_mass_by_zone: dict[str, float] = {}
        for z_idx, zone_id in enumerate(sys.zone_ids):
            ne = self.electron_density_from_state_row(gas[z_idx])
            pos = self.positive_ion_density_from_state_row(gas[z_idx])
            mean_e = self.mean_energy_eV(We[z_idx], ne)
            ion_mass = self.dominant_ion_mass_kg_from_state_row(gas[z_idx])
            ne_by_zone[zone_id] = ne
            mean_e_by_zone[zone_id] = mean_e
            pos_by_zone[zone_id] = pos
            ion_mass_by_zone[zone_id] = ion_mass
        return ne_by_zone, mean_e_by_zone, pos_by_zone, ion_mass_by_zone

    def incident_neutral_flux_m2_s(self, gas_row: np.ndarray, Tg: float, local_idx: int) -> float:
        sys = self.system
        n = max(float(gas_row[local_idx]), 0.0)
        m = max(float(sys.gas_masses[local_idx]), 1.0e-30)
        vth = np.sqrt(8.0 * K_B * max(float(Tg), 1.0) / (np.pi * m))
        return 0.25 * n * vth

    def reaction_mass_action(self, rxn: CompiledGasReaction, gas_row: np.ndarray, ne: float) -> float:
        mass_action = 1.0
        for idx, nu, _sp in rxn.reactant_gas:
            mass_action *= max(gas_row[idx], self.system.floor_density) ** nu
        if rxn.electron_reactant_stoich:
            mass_action *= max(ne, self.system.floor_density) ** rxn.electron_reactant_stoich
        return float(mass_action)

    def reaction_rate_coefficient(self, model: dict[str, Any], Tg: float, eedf) -> tuple[float, float, float]:
        backend = str(model.get('backend', '')).lower()
        if backend == 'electron_impact_xsec':
            cs_id = model['cross_section_id']
            branch = float(model.get('branching_yield', 1.0))
            k = branch * float(eedf.rate_coefficients.get(cs_id, 0.0))
            dk_de = branch * float(eedf.d_rate_d_mean_energy_eV.get(cs_id, 0.0))
            dk_dT = 0.0
        elif backend == 'arrhenius':
            A = float(model.get('A', 0.0))
            beta = float(model.get('beta', 0.0))
            Ea_eV = float(model.get('Ea_eV', 0.0))
            kT_eV = K_B * max(Tg, 1.0) / E_CHARGE
            k = A * (max(Tg, 1.0) / 300.0) ** beta * np.exp(-Ea_eV / max(kT_eV, 1.0e-12))
            dk_dT = k * (beta / max(Tg, 1.0) + Ea_eV * E_CHARGE / (K_B * max(Tg, 1.0) ** 2))
            dk_de = 0.0
        elif backend == 'constant':
            k = float(model.get('value', 0.0))
            dk_de = 0.0
            dk_dT = 0.0
        elif backend == 'first_order_loss':
            k = float(model.get('rate_s_inv', model.get('value', 0.0)))
            dk_de = 0.0
            dk_dT = 0.0
        elif backend in {'te_power_law', 'electron_temperature_power_law'}:
            A = float(model.get('A', model.get('value', 0.0)))
            alpha = float(model.get('alpha', model.get('exponent', 0.0)))
            Tref_K = max(float(model.get('Tref_K', model.get('reference_temperature_K', 1.0))), 1.0e-30)
            mean_e_raw = float((getattr(eedf, 'transport', {}) or {}).get('mean_energy_eV', 0.0))
            mean_floor = float(model.get('mean_energy_floor_eV', 1.0e-6))
            mean_e = max(mean_e_raw, mean_floor)
            temperature_factor = float(model.get('electron_temperature_factor', 2.0 / 3.0))
            Te_floor_K = float(model.get('electron_temperature_floor_K', 1.0))
            Te_K = max(temperature_factor * mean_e * EV_TO_K, Te_floor_K)
            k = A * (Te_K / Tref_K) ** alpha
            dk_de = k * alpha / mean_e if mean_e_raw >= mean_floor and Te_K > Te_floor_K else 0.0
            dk_dT = 0.0
        else:
            raise NotImplementedError(f'Unsupported gas rate backend: {backend}')
        return float(k), float(dk_de), float(dk_dT)

    def reaction_rate(self, rxn: CompiledGasReaction, gas_row: np.ndarray, Tg: float, eedf, ne: float) -> tuple[float, float, float]:
        k, dk_de, dk_dT = self.reaction_rate_coefficient(rxn.rate_model, Tg, eedf)
        R = k * self.reaction_mass_action(rxn, gas_row, ne)
        return float(R), float(dk_de), float(dk_dT)

    def _ion_wall_loss_items(self, zone_id: str, gas_row: np.ndarray, mean_e: float) -> list[tuple[str, float]]:
        sys = self.system
        family = sys.zone_ion_loss_family.get(zone_id, 'disabled')
        if family == 'disabled':
            return []
        volume = sys.chamber.zone_by_id[zone_id].volume_m3
        if family == 'bohm':
            area = sys.zone_ion_loss_area.get(zone_id, 0.0)
            h_factor = sys.zone_ion_loss_h_factor.get(zone_id, 1.0)
            pref_common = area / max(volume, 1.0e-30) * h_factor * BOHM_FLUX_COEFF
        elif family == 'ambipolar_diffusion':
            pref_common = sys.zone_ambipolar_loss_rate_s.get(zone_id, 0.0)
        else:
            return []
        if pref_common <= 0.0:
            return []
        items: list[tuple[str, float]] = []
        for idx in sys.positive_ion_local_indices:
            n_i = max(float(gas_row[idx]), 0.0)
            if n_i <= 0.0:
                continue
            if family == 'bohm':
                mass = max(float(sys.gas_masses[idx]), 1.0e-30)
                sound_speed = np.sqrt(max(mean_e, 0.05) * E_CHARGE / mass)
                loss = pref_common * n_i * sound_speed
            else:
                loss = pref_common * n_i
            items.append((sys.gas_species_ids[idx], float(loss)))
        return items

    def reaction_budget(
        self,
        time_s: float,
        state: np.ndarray,
        step: Any,
        coupled: CoupledPlasmaEvaluation,
        *,
        species_filter: list[str] | None = None,
        max_reactions_per_species: int = 5,
    ) -> dict[str, Any]:
        sys = self.system
        wanted = set(species_filter or sys.gas_species_ids)
        out: dict[str, Any] = {
            'time_s': float(time_s),
            'step_id': step.step_id,
            'included_processes': ['gas_reactions', 'ion_wall_loss'],
            'zones': {},
        }
        gas = coupled.gas
        Tg = coupled.gas_temperature
        max_items = max(int(max_reactions_per_species), 0)
        for zone_id in sys.zone_ids:
            z = sys.zone_index[zone_id]
            zone_species = {
                sp_id: {
                    'density_m3': float(gas[z, sys.gas_species_index[sp_id]]),
                    'production_m3_s': 0.0,
                    'loss_m3_s': 0.0,
                    'net_m3_s': 0.0,
                    '_production_terms': [],
                    '_loss_terms': [],
                }
                for sp_id in sys.gas_species_ids
                if sp_id in wanted
            }
            zone_reactions: dict[str, dict[str, Any]] = {}
            for rxn in self.gas_reactions:
                if zone_id not in rxn.zones:
                    continue
                ne = coupled.ne_by_zone[zone_id]
                R, _, _ = self.reaction_rate(rxn, gas[z], float(Tg[z]), coupled.eedf_by_zone[zone_id], ne)
                if R == 0.0:
                    continue
                net_species: dict[str, float] = {}
                for _idx, nu, sp_id in rxn.delta_gas:
                    val = float(nu * R)
                    net_species[sp_id] = val
                    if sp_id not in zone_species:
                        continue
                    payload = zone_species[sp_id]
                    payload['net_m3_s'] += val
                    term = {'process_id': rxn.reaction_id, 'contribution_m3_s': abs(val), 'signed_m3_s': val}
                    if val >= 0.0:
                        payload['production_m3_s'] += val
                        payload['_production_terms'].append(term)
                    else:
                        payload['loss_m3_s'] += -val
                        payload['_loss_terms'].append(term)
                if not species_filter or any(sp_id in wanted for sp_id in net_species):
                    zone_reactions[rxn.reaction_id] = {
                        'rate_m3_s': float(R),
                        'net_species_m3_s': net_species,
                    }
            for sp_id, loss in self._ion_wall_loss_items(zone_id, gas[z], coupled.mean_e_by_zone[zone_id]):
                if sp_id not in zone_species:
                    continue
                payload = zone_species[sp_id]
                payload['loss_m3_s'] += loss
                payload['net_m3_s'] -= loss
                payload['_loss_terms'].append(
                    {
                        'process_id': f'ion_wall_loss:{sys.zone_ion_loss_family.get(zone_id, "unknown")}',
                        'contribution_m3_s': loss,
                        'signed_m3_s': -loss,
                    }
                )
            for payload in zone_species.values():
                prod_terms = sorted(payload.pop('_production_terms'), key=lambda x: x['contribution_m3_s'], reverse=True)
                loss_terms = sorted(payload.pop('_loss_terms'), key=lambda x: x['contribution_m3_s'], reverse=True)
                payload['top_production'] = prod_terms[:max_items]
                payload['top_loss'] = loss_terms[:max_items]
            out['zones'][zone_id] = {
                'species': zone_species,
                'reactions': zone_reactions,
            }
        return out

    def _reaction_energy_category(self, rxn: CompiledGasReaction) -> str:
        sys = self.system
        model = rxn.rate_model
        backend = str(model.get('backend', 'gas_reaction')).lower()
        if backend == 'electron_impact_xsec':
            cs = sys.mechanism.cross_sections.get(str(model.get('cross_section_id', '')))
            if cs is not None:
                return str(cs.kind or 'electron_impact').lower()
            return 'electron_impact'
        return backend

    def electron_energy_budget(
        self,
        time_s: float,
        state: np.ndarray,
        step: Any,
        coupled: CoupledPlasmaEvaluation,
        *,
        max_reactions: int = 8,
    ) -> dict[str, Any]:
        sys = self.system
        out: dict[str, Any] = {
            'time_s': float(time_s),
            'step_id': step.step_id,
            'sign_convention': 'positive terms add electron energy density; negative terms remove it',
            'zones': {},
        }
        gas = coupled.gas
        Tg = coupled.gas_temperature
        max_items = max(int(max_reactions), 0)

        for zone_id in sys.zone_ids:
            z = sys.zone_index[zone_id]
            volume = sys.chamber.zone_by_id[zone_id].volume_m3
            pabs = float(coupled.power.absorbed_power_W_by_zone.get(zone_id, 0.0))
            terms: dict[str, float] = {
                'absorbed_power_W_m3': pabs / max(volume, 1.0e-30),
                'electron_impact_loss_W_m3': 0.0,
                'ion_wall_loss_W_m3': 0.0,
                'pump_loss_W_m3': 0.0,
                'edge_transport_W_m3': 0.0,
            }
            category_losses: dict[str, float] = {}
            losses: list[dict[str, Any]] = []

            for rxn in self.gas_reactions:
                if zone_id not in rxn.zones:
                    continue
                if not rxn.energy_model or str(rxn.energy_model.get('backend', '')).lower() != 'constant_event_loss':
                    continue
                energy_loss_eV = float(rxn.energy_model.get('energy_loss_eV', 0.0))
                if energy_loss_eV == 0.0:
                    continue
                ne = coupled.ne_by_zone[zone_id]
                R, _, _ = self.reaction_rate(rxn, gas[z], float(Tg[z]), coupled.eedf_by_zone[zone_id], ne)
                if R == 0.0:
                    continue
                loss_W_m3 = float(R * energy_loss_eV * E_CHARGE)
                category = self._reaction_energy_category(rxn)
                category_losses[category] = category_losses.get(category, 0.0) + loss_W_m3
                terms['electron_impact_loss_W_m3'] -= loss_W_m3
                losses.append(
                    {
                        'process_id': rxn.reaction_id,
                        'category': category,
                        'energy_loss_eV': energy_loss_eV,
                        'rate_m3_s': float(R),
                        'loss_W_m3': loss_W_m3,
                    }
                )

            mean_e = coupled.mean_e_by_zone[zone_id]
            for sp_id, particle_loss in self._ion_wall_loss_items(zone_id, gas[z], mean_e):
                sp = sys.mechanism.species_by_id[sp_id]
                wall_loss_W_m3 = max(float(sp.charge), 1.0) * mean_e * E_CHARGE * particle_loss
                terms['ion_wall_loss_W_m3'] -= wall_loss_W_m3
                losses.append(
                    {
                        'process_id': f'ion_wall_loss:{sys.zone_ion_loss_family.get(zone_id, "unknown")}:{sp_id}',
                        'category': 'ion_wall_loss',
                        'energy_loss_eV': max(float(sp.charge), 1.0) * mean_e,
                        'rate_m3_s': particle_loss,
                        'loss_W_m3': wall_loss_W_m3,
                    }
                )

            for pump in sys.chamber.pumps:
                if pump.zone_id != zone_id:
                    continue
                lam = pump.speed_m3_s / max(volume, 1.0e-30)
                terms['pump_loss_W_m3'] -= lam * float(coupled.electron_energy[z])

            for edge in sys.chamber.edges:
                if edge.from_zone == zone_id:
                    terms['edge_transport_W_m3'] -= edge.conductance_m3_s / max(volume, 1.0e-30) * float(coupled.electron_energy[z])
                if edge.to_zone == zone_id:
                    zi = sys.zone_index[edge.from_zone]
                    terms['edge_transport_W_m3'] += edge.conductance_m3_s / max(volume, 1.0e-30) * float(coupled.electron_energy[zi])

            notes = [
                'Gas-temperature heating is currently a separate reduced heavy-particle equation and is not subtracted from electron energy.',
            ]
            tau = self.field_table_energy_relaxation_tau_s()
            if tau is not None and str(coupled.eedf_by_zone[zone_id].transport.get('lookup_mode', '')).lower() == 'field':
                target_mean_e = max(float(coupled.eedf_by_zone[zone_id].transport.get('mean_energy_eV', mean_e)), 1.0e-6)
                target_We = max(coupled.ne_by_zone[zone_id], sys.floor_density) * target_mean_e * E_CHARGE
                relaxation = (target_We - float(coupled.electron_energy[z])) / tau
                terms['field_table_energy_relaxation_override_W_m3'] = relaxation - float(sum(terms.values()))
                notes.append('Field-rate-table mode overrides the electron-energy RHS with relaxation to the table mean energy.')

            net = float(sum(terms.values()))
            out['zones'][zone_id] = {
                'electron_energy_density_J_m3': float(coupled.electron_energy[z]),
                'electron_density_m3': float(coupled.ne_by_zone[zone_id]),
                'mean_energy_eV': float(mean_e),
                'terms_W_m3': terms,
                'loss_by_category_W_m3': dict(sorted(category_losses.items())),
                'top_losses': sorted(losses, key=lambda item: item['loss_W_m3'], reverse=True)[:max_items],
                'net_W_m3': net,
                'notes': notes,
            }
        return out

    def eedf_rates_depend_on_mean_energy(self, eedf) -> bool:
        lookup_mode = str((getattr(eedf, 'transport', {}) or {}).get('lookup_mode', 'mean_energy')).lower()
        return lookup_mode not in {'field', 'local_field'}

    def apply_ion_wall_losses_rhs(self, gas: np.ndarray, coupled: CoupledPlasmaEvaluation, gas_rhs: np.ndarray, We_rhs: np.ndarray) -> None:
        sys = self.system
        for zone_id in sys.zone_ids:
            family = sys.zone_ion_loss_family.get(zone_id, 'disabled')
            if family == 'disabled':
                continue
            z = sys.zone_index[zone_id]
            mean_e = coupled.mean_e_by_zone[zone_id]
            volume = sys.chamber.zone_by_id[zone_id].volume_m3
            if family == 'bohm':
                area = sys.zone_ion_loss_area.get(zone_id, 0.0)
                h_factor = sys.zone_ion_loss_h_factor.get(zone_id, 1.0)
                pref_common = area / max(volume, 1.0e-30) * h_factor * BOHM_FLUX_COEFF
            elif family == 'ambipolar_diffusion':
                pref_common = sys.zone_ambipolar_loss_rate_s.get(zone_id, 0.0)
            else:
                continue
            if pref_common <= 0.0:
                continue
            for idx in sys.positive_ion_local_indices:
                n_i = max(float(gas[z, idx]), 0.0)
                if n_i <= 0.0:
                    continue
                if family == 'bohm':
                    mass = max(float(sys.gas_masses[idx]), 1.0e-30)
                    sound_speed = np.sqrt(max(mean_e, 0.05) * E_CHARGE / mass)
                    loss = pref_common * n_i * sound_speed
                else:
                    loss = pref_common * n_i
                charge = max(float(sys.gas_charges[idx]), 1.0)
                gas_rhs[z, idx] -= loss
                We_rhs[z] -= charge * mean_e * E_CHARGE * loss

    def apply_ion_wall_loss_jacobian(self, gas: np.ndarray, coupled: CoupledPlasmaEvaluation, J) -> None:
        sys = self.system
        for zone_id in sys.zone_ids:
            family = sys.zone_ion_loss_family.get(zone_id, 'disabled')
            if family == 'disabled':
                continue
            z = sys.zone_index[zone_id]
            volume = sys.chamber.zone_by_id[zone_id].volume_m3
            if family == 'bohm':
                area = sys.zone_ion_loss_area.get(zone_id, 0.0)
                h_factor = sys.zone_ion_loss_h_factor.get(zone_id, 1.0)
                pref_common = area / max(volume, 1.0e-30) * h_factor * BOHM_FLUX_COEFF
            elif family == 'ambipolar_diffusion':
                pref_common = sys.zone_ambipolar_loss_rate_s.get(zone_id, 0.0)
            else:
                continue
            if pref_common <= 0.0:
                continue
            ne = coupled.ne_by_zone[zone_id]
            mean_e = coupled.mean_e_by_zone[zone_id]
            active_ne = self.electron_density_deriv_active(gas[z])
            We_idx = sys.state_layout.electron_energy_index[zone_id]
            for ion_idx in sys.positive_ion_local_indices:
                n_i_raw = float(gas[z, ion_idx])
                if n_i_raw <= 0.0:
                    continue
                if family == 'bohm':
                    mass = max(float(sys.gas_masses[ion_idx]), 1.0e-30)
                    sound_speed = np.sqrt(max(mean_e, 0.05) * E_CHARGE / mass)
                    pref = pref_common * sound_speed
                else:
                    pref = pref_common
                loss = pref * n_i_raw
                charge = max(float(sys.gas_charges[ion_idx]), 1.0)
                row = self.gas_state_idx(z, ion_idx)
                for s in range(sys.n_gas_species):
                    dloss = pref if s == ion_idx else 0.0
                    dmean_dn = 0.0
                    if active_ne:
                        q = sys.gas_charges[s]
                        if q != 0.0:
                            dmean_dn = -mean_e * q / max(ne, sys.floor_density)
                            if family == 'bohm' and mean_e > 0.05:
                                dloss += loss * 0.5 * dmean_dn / max(mean_e, 1.0e-30)
                    col = self.gas_state_idx(z, s)
                    if dloss != 0.0:
                        J[row, col] += -dloss
                    d_we_loss = charge * E_CHARGE * (dmean_dn * loss + mean_e * dloss)
                    if d_we_loss != 0.0:
                        J[We_idx, col] += -d_we_loss
                if family == 'bohm' and mean_e > 0.05:
                    dmean_dWe = 1.0 / max(ne, sys.floor_density) / E_CHARGE
                    dloss_dWe = loss * 0.5 * dmean_dWe / max(mean_e, 1.0e-30)
                    J[row, We_idx] += -dloss_dWe
                    J[We_idx, We_idx] += -charge * E_CHARGE * (dmean_dWe * loss + mean_e * dloss_dWe)
                elif family == 'ambipolar_diffusion':
                    dmean_dWe = 1.0 / max(ne, sys.floor_density) / E_CHARGE
                    J[We_idx, We_idx] += -charge * E_CHARGE * dmean_dWe * loss

    def apply_rhs(self, time_s: float, y: np.ndarray, step: Any, coupled: CoupledPlasmaEvaluation, dydt: np.ndarray) -> None:
        sys = self.system
        gas = coupled.gas
        We = coupled.electron_energy
        Tg = coupled.gas_temperature
        gas_rhs = dydt[sys.state_layout.slice('gas_densities')].reshape(sys.n_zones, sys.n_gas_species)
        We_rhs = dydt[sys.state_layout.slice('electron_energy')]
        Tg_rhs = dydt[sys.state_layout.slice('gas_temperature')] if 'gas_temperature' in sys.state_layout.slices else None

        for rxn in self.gas_reactions:
            for zone_id in rxn.zones:
                z = sys.zone_index[zone_id]
                ne = coupled.ne_by_zone[zone_id]
                R, _, _ = self.reaction_rate(rxn, gas[z], float(Tg[z]), coupled.eedf_by_zone[zone_id], ne)
                if R == 0.0:
                    continue
                for idx, nu, _sp in rxn.delta_gas:
                    gas_rhs[z, idx] += nu * R
                if rxn.energy_model and str(rxn.energy_model.get('backend', '')).lower() == 'constant_event_loss':
                    loss = float(rxn.energy_model.get('energy_loss_eV', 0.0)) * E_CHARGE
                    We_rhs[z] -= loss * R

        self.apply_ion_wall_losses_rhs(gas, coupled, gas_rhs, We_rhs)

        for inlet_id, default_inlet in sys.chamber.inlet_by_id.items():
            flows = step.gas_inlets.get(inlet_id, default_inlet.flow_sccm)
            z = sys.zone_index[default_inlet.zone_id]
            V = sys.chamber.zone_by_id[default_inlet.zone_id].volume_m3
            for sp_id, flow_sccm in flows.items():
                if sp_id not in sys.gas_species_index:
                    continue
                gas_rhs[z, sys.gas_species_index[sp_id]] += flow_sccm * SCCM_TO_PARTICLES_PER_S / max(V, 1.0e-30)
            if Tg_rhs is not None:
                n_tot = max(float(np.sum(gas[z])), sys.floor_density)
                particle_source = sum(float(v) for v in flows.values()) * SCCM_TO_PARTICLES_PER_S / max(V, 1.0e-30)
                Tg_rhs[z] += particle_source / n_tot * (default_inlet.temperature_K - Tg[z])

        for pump in sys.chamber.pumps:
            z = sys.zone_index[pump.zone_id]
            lam = pump.speed_m3_s / max(sys.chamber.zone_by_id[pump.zone_id].volume_m3, 1.0e-30)
            gas_rhs[z, :] -= lam * gas[z, :]
            We_rhs[z] -= lam * We[z]

        for edge in sys.chamber.edges:
            zi = sys.zone_index[edge.from_zone]
            zj = sys.zone_index[edge.to_zone]
            Vi = sys.chamber.zone_by_id[edge.from_zone].volume_m3
            Vj = sys.chamber.zone_by_id[edge.to_zone].volume_m3
            Ci = edge.conductance_m3_s
            gas_rhs[zi, :] -= Ci / max(Vi, 1.0e-30) * gas[zi, :]
            gas_rhs[zj, :] += Ci / max(Vj, 1.0e-30) * gas[zi, :]
            We_rhs[zi] -= Ci / max(Vi, 1.0e-30) * We[zi]
            We_rhs[zj] += Ci / max(Vj, 1.0e-30) * We[zi]
            if Tg_rhs is not None:
                n_i = max(float(np.sum(gas[zi])), sys.floor_density)
                n_j = max(float(np.sum(gas[zj])), sys.floor_density)
                Tg_rhs[zj] += Ci / max(Vj, 1.0e-30) * n_i / n_j * (Tg[zi] - Tg[zj])

        for zone_id, pabs in coupled.power.absorbed_power_W_by_zone.items():
            z = sys.zone_index[zone_id]
            V = sys.chamber.zone_by_id[zone_id].volume_m3
            We_rhs[z] += pabs / max(V, 1.0e-30)

        self.apply_field_table_energy_relaxation(coupled, We_rhs)

        if Tg_rhs is not None:
            gas_heating_fraction = float(sys.run_config.physics.gas_heating_fraction)
            wall_relax = float(sys.run_config.physics.wall_relaxation_s_inv)
            for zone_id in sys.zone_ids:
                z = sys.zone_index[zone_id]
                n_tot = max(float(np.sum(gas[z])), sys.floor_density)
                V = sys.chamber.zone_by_id[zone_id].volume_m3
                pabs = coupled.power.absorbed_power_W_by_zone.get(zone_id, 0.0)
                Tg_rhs[z] += gas_heating_fraction * pabs / max(V, 1.0e-30) / (1.5 * K_B * n_tot)
                Tg_rhs[z] -= wall_relax * (Tg[z] - self.zone_wall_temperature[zone_id])

    def field_table_energy_relaxation_tau_s(self) -> float | None:
        table_cfg = getattr(getattr(self.system.run_config, 'swarm', None), 'table', None)
        mode = str(getattr(table_cfg, 'electron_energy_mode', '') or '').lower()
        if mode not in {'table_relaxation', 'from_table', 'prescribed_from_table'}:
            return None
        return max(float(getattr(table_cfg, 'energy_relaxation_time_s', 1.0e-6) or 1.0e-6), 1.0e-12)

    def apply_field_table_energy_relaxation(self, coupled: CoupledPlasmaEvaluation, We_rhs: np.ndarray) -> None:
        tau = self.field_table_energy_relaxation_tau_s()
        if tau is None:
            return
        sys = self.system
        for zone_id in sys.zone_ids:
            eedf = coupled.eedf_by_zone[zone_id]
            if str(eedf.transport.get('lookup_mode', '')).lower() != 'field':
                continue
            z = sys.zone_index[zone_id]
            target_mean_e = max(float(eedf.transport.get('mean_energy_eV', coupled.mean_e_by_zone[zone_id])), 1.0e-6)
            target_We = max(coupled.ne_by_zone[zone_id], sys.floor_density) * target_mean_e * E_CHARGE
            We_rhs[z] = (target_We - float(coupled.electron_energy[z])) / tau

    def apply_jacobian(self, time_s: float, y: np.ndarray, step: Any, coupled: CoupledPlasmaEvaluation, J) -> None:
        sys = self.system
        gas = coupled.gas
        We = coupled.electron_energy
        Tg = coupled.gas_temperature

        for rxn in self.gas_reactions:
            for zone_id in rxn.zones:
                z = sys.zone_index[zone_id]
                ne = coupled.ne_by_zone[zone_id]
                mean_e = coupled.mean_e_by_zone[zone_id]
                eedf = coupled.eedf_by_zone[zone_id]
                eedf_mean_energy_dependent = self.eedf_rates_depend_on_mean_energy(eedf)
                R, dk_de, dk_dT = self.reaction_rate(rxn, gas[z], float(Tg[z]), eedf, ne)
                if R == 0.0:
                    continue
                active_ne = self.electron_density_deriv_active(gas[z])
                mass_action = self.reaction_mass_action(rxn, gas[z], ne)
                for s in range(sys.n_gas_species):
                    dR = 0.0
                    for ridx, nu, _sp in rxn.reactant_gas:
                        if ridx == s:
                            dR += R * nu / max(gas[z, s], sys.floor_density)
                    if active_ne and (rxn.electron_reactant_stoich or (eedf_mean_energy_dependent and dk_de != 0.0)):
                        q = sys.gas_charges[s]
                        if q != 0.0:
                            if rxn.electron_reactant_stoich:
                                dR += R * rxn.electron_reactant_stoich * q / max(ne, sys.floor_density)
                            if eedf_mean_energy_dependent and dk_de != 0.0:
                                dmean_dn = -mean_e * q / max(ne, sys.floor_density)
                                dR += dk_de * mass_action * dmean_dn
                    if dR == 0.0:
                        continue
                    col = self.gas_state_idx(z, s)
                    for idx, nu, _sp in rxn.delta_gas:
                        row = self.gas_state_idx(z, idx)
                        J[row, col] += nu * dR
                    if rxn.energy_model and str(rxn.energy_model.get('backend', '')).lower() == 'constant_event_loss':
                        loss = float(rxn.energy_model.get('energy_loss_eV', 0.0)) * E_CHARGE
                        J[sys.state_layout.electron_energy_index[zone_id], col] += -loss * dR
                dmean_dWe = 1.0 / max(ne, sys.floor_density) / E_CHARGE
                if eedf_mean_energy_dependent and dk_de != 0.0:
                    dR_dWe = dk_de * mass_action * dmean_dWe
                    col = sys.state_layout.electron_energy_index[zone_id]
                    for idx, nu, _sp in rxn.delta_gas:
                        row = self.gas_state_idx(z, idx)
                        J[row, col] += nu * dR_dWe
                    if rxn.energy_model and str(rxn.energy_model.get('backend', '')).lower() == 'constant_event_loss':
                        loss = float(rxn.energy_model.get('energy_loss_eV', 0.0)) * E_CHARGE
                        J[sys.state_layout.electron_energy_index[zone_id], col] += -loss * dR_dWe
                if dk_dT != 0.0 and 'gas_temperature' in sys.state_layout.slices:
                    col = sys.state_layout.gas_temperature_index[zone_id]
                    mass_action = 1.0
                    for idx, nu, _sp in rxn.reactant_gas:
                        mass_action *= max(gas[z, idx], sys.floor_density) ** nu
                    if rxn.electron_reactant_stoich:
                        mass_action *= max(ne, sys.floor_density) ** rxn.electron_reactant_stoich
                    dR_dT = dk_dT * mass_action
                    for idx, nu, _sp in rxn.delta_gas:
                        row = self.gas_state_idx(z, idx)
                        J[row, col] += nu * dR_dT
                    if rxn.energy_model and str(rxn.energy_model.get('backend', '')).lower() == 'constant_event_loss':
                        loss = float(rxn.energy_model.get('energy_loss_eV', 0.0)) * E_CHARGE
                        J[sys.state_layout.electron_energy_index[zone_id], col] += -loss * dR_dT

        self.apply_ion_wall_loss_jacobian(gas, coupled, J)

        for inlet_id, default_inlet in sys.chamber.inlet_by_id.items():
            flows = step.gas_inlets.get(inlet_id, default_inlet.flow_sccm)
            z = sys.zone_index[default_inlet.zone_id]
            V = sys.chamber.zone_by_id[default_inlet.zone_id].volume_m3
            if 'gas_temperature' in sys.state_layout.slices:
                col = sys.state_layout.gas_temperature_index[default_inlet.zone_id]
                n_tot = max(float(np.sum(gas[z])), sys.floor_density)
                particle_source = sum(float(v) for v in flows.values()) * SCCM_TO_PARTICLES_PER_S / max(V, 1.0e-30)
                temp_delta = default_inlet.temperature_K - Tg[z]
                J[col, col] += -particle_source / n_tot
                for s in range(sys.n_gas_species):
                    gas_col = self.gas_state_idx(z, s)
                    J[col, gas_col] += -particle_source * temp_delta / (n_tot ** 2)

        for pump in sys.chamber.pumps:
            z = sys.zone_index[pump.zone_id]
            lam = pump.speed_m3_s / max(sys.chamber.zone_by_id[pump.zone_id].volume_m3, 1.0e-30)
            for s in range(sys.n_gas_species):
                idx = self.gas_state_idx(z, s)
                J[idx, idx] += -lam
            J[sys.state_layout.electron_energy_index[pump.zone_id], sys.state_layout.electron_energy_index[pump.zone_id]] += -lam

        for edge in sys.chamber.edges:
            zi = sys.zone_index[edge.from_zone]
            zj = sys.zone_index[edge.to_zone]
            Vi = sys.chamber.zone_by_id[edge.from_zone].volume_m3
            Vj = sys.chamber.zone_by_id[edge.to_zone].volume_m3
            C = edge.conductance_m3_s
            for s in range(sys.n_gas_species):
                idx_i = self.gas_state_idx(zi, s)
                idx_j = self.gas_state_idx(zj, s)
                J[idx_i, idx_i] += -C / max(Vi, 1.0e-30)
                J[idx_j, idx_i] += C / max(Vj, 1.0e-30)
            Wi = sys.state_layout.electron_energy_index[edge.from_zone]
            Wj = sys.state_layout.electron_energy_index[edge.to_zone]
            J[Wi, Wi] += -C / max(Vi, 1.0e-30)
            J[Wj, Wi] += C / max(Vj, 1.0e-30)
            if 'gas_temperature' in sys.state_layout.slices:
                Ti = sys.state_layout.gas_temperature_index[edge.from_zone]
                Tj = sys.state_layout.gas_temperature_index[edge.to_zone]
                n_i = max(float(np.sum(gas[zi])), sys.floor_density)
                n_j = max(float(np.sum(gas[zj])), sys.floor_density)
                coeff = C / max(Vj, 1.0e-30)
                temp_delta = Tg[zi] - Tg[zj]
                J[Tj, Ti] += coeff * n_i / n_j
                J[Tj, Tj] += -coeff * n_i / n_j
                for s in range(sys.n_gas_species):
                    J[Tj, self.gas_state_idx(zi, s)] += coeff * temp_delta / n_j
                    J[Tj, self.gas_state_idx(zj, s)] += -coeff * n_i * temp_delta / (n_j ** 2)

        if 'gas_temperature' in sys.state_layout.slices:
            gas_heating_fraction = float(sys.run_config.physics.gas_heating_fraction)
            wall_relax = float(sys.run_config.physics.wall_relaxation_s_inv)
            for zone_id in sys.zone_ids:
                z = sys.zone_index[zone_id]
                T_idx = sys.state_layout.gas_temperature_index[zone_id]
                n_tot = max(float(np.sum(gas[z])), sys.floor_density)
                V = sys.chamber.zone_by_id[zone_id].volume_m3
                pabs = coupled.power.absorbed_power_W_by_zone.get(zone_id, 0.0)
                coeff = gas_heating_fraction * pabs / max(V, 1.0e-30) / (1.5 * K_B)
                for s in range(sys.n_gas_species):
                    idx = self.gas_state_idx(z, s)
                    J[T_idx, idx] += -coeff / (n_tot ** 2)
                J[T_idx, T_idx] += -wall_relax

        self.apply_field_table_energy_relaxation_jacobian(coupled, J)

    def apply_field_table_energy_relaxation_jacobian(self, coupled: CoupledPlasmaEvaluation, J) -> None:
        tau = self.field_table_energy_relaxation_tau_s()
        if tau is None:
            return
        sys = self.system
        for zone_id in sys.zone_ids:
            eedf = coupled.eedf_by_zone[zone_id]
            if str(eedf.transport.get('lookup_mode', '')).lower() != 'field':
                continue
            row = sys.state_layout.electron_energy_index[zone_id]
            J.rows[row] = []
            J.data[row] = []
            target_mean_e = max(float(eedf.transport.get('mean_energy_eV', coupled.mean_e_by_zone[zone_id])), 1.0e-6)
            z = sys.zone_index[zone_id]
            for s, charge in enumerate(sys.gas_charges):
                if charge == 0.0:
                    continue
                col = self.gas_state_idx(z, s)
                J[row, col] += charge * target_mean_e * E_CHARGE / tau
            J[row, row] += -1.0 / tau
