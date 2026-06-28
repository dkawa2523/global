from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from plasma_global.chemistry.models import E_CHARGE, K_B
from plasma_global.diagnostics.budgets import ReactionBudget
from plasma_global.physics.types import CompiledGasReaction, CoupledPlasmaEvaluation, GasReactionTerm, IonWallLossTerm
from plasma_global.reactor.surface_models import bohm_ion_loss_frequency_s, ion_loss_uses_effective_frequency

SCCM_TO_PARTICLES_PER_S = 4.477962e17
EV_TO_K = E_CHARGE / K_B


@dataclass
class _GasRhsViews:
    gas: np.ndarray
    electron_energy: np.ndarray
    gas_temperature: np.ndarray
    gas_rhs: np.ndarray
    electron_energy_rhs: np.ndarray
    gas_temperature_rhs: np.ndarray | None


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
            out[zone_id] = num / max(den, 1.0e-30)
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

    def electron_density_from_state_row(self, gas_row: np.ndarray) -> float:
        sys = self.system
        ne = float(np.dot(sys.gas_charges, gas_row))
        return max(ne, sys.floor_density)

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

    def ion_wall_loss_frequency_s(self, zone_id: str, ion_local_idx: int, mean_energy_eV: float) -> float:
        sys = self.system
        family = sys.zone_ion_loss_family.get(zone_id, 'disabled')
        if family == 'bohm':
            zone = sys.chamber.zone_by_id[zone_id]
            return bohm_ion_loss_frequency_s(
                area_m2=sys.zone_ion_loss_area.get(zone_id, 0.0),
                volume_m3=zone.volume_m3,
                h_factor=sys.zone_ion_loss_h_factor.get(zone_id, 0.0),
                mean_energy_eV=mean_energy_eV,
                ion_mass_kg=sys.gas_masses[ion_local_idx],
            )
        if ion_loss_uses_effective_frequency(family):
            return max(float(sys.zone_effective_ion_loss_frequency_s.get(zone_id, 0.0)), 0.0)
        return 0.0

    def ion_wall_loss_flux_m2_s(self, zone_id: str, gas_row: np.ndarray, mean_energy_eV: float) -> float:
        sys = self.system
        total_loss_source = 0.0
        for idx in sys.positive_ion_local_indices:
            frequency = self.ion_wall_loss_frequency_s(zone_id, idx, mean_energy_eV)
            n_i = max(float(gas_row[idx]), 0.0)
            total_loss_source += frequency * n_i
        volume = sys.chamber.zone_by_id[zone_id].volume_m3
        area = sys.zone_ion_loss_area.get(zone_id, 0.0)
        flux = total_loss_source * volume / max(area, 1.0e-30) if area > 0.0 else 0.0
        return max(float(flux), 0.0)

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
            mean_e_raw = float(eedf.transport.mean_energy_eV)
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

    def gas_reaction_terms(self, coupled: CoupledPlasmaEvaluation) -> list[GasReactionTerm]:
        sys = self.system
        terms: list[GasReactionTerm] = []
        for rxn in self.gas_reactions:
            for zone_id in rxn.zones:
                z = sys.zone_index[zone_id]
                ne = coupled.ne_by_zone[zone_id]
                R, _, _ = self.reaction_rate(rxn, coupled.gas[z], float(coupled.gas_temperature[z]), coupled.eedf_by_zone[zone_id], ne)
                if R == 0.0:
                    continue
                energy_loss = self.reaction_energy_loss_J_m3_s(rxn, R)
                changes = [(idx, float(nu) * R, sp_id) for idx, nu, sp_id in rxn.delta_gas]
                terms.append(GasReactionTerm(zone_id, rxn.reaction_id, R, changes, energy_loss))
        return terms

    def ion_wall_loss_terms(self, coupled: CoupledPlasmaEvaluation) -> list[IonWallLossTerm]:
        sys = self.system
        terms: list[IonWallLossTerm] = []
        for zone_id in sys.zone_ids:
            family = sys.zone_ion_loss_family.get(zone_id, 'disabled')
            if family == 'disabled':
                continue
            z = sys.zone_index[zone_id]
            mean_e = coupled.mean_e_by_zone[zone_id]
            for idx in sys.positive_ion_local_indices:
                n_i = max(float(coupled.gas[z, idx]), 0.0)
                if n_i <= 0.0:
                    continue
                frequency = self.ion_wall_loss_frequency_s(zone_id, idx, mean_e)
                if frequency <= 0.0:
                    continue
                loss = frequency * n_i
                charge = max(float(sys.gas_charges[idx]), 1.0)
                terms.append(IonWallLossTerm(zone_id, idx, sys.gas_species_ids[idx], loss, charge * mean_e * E_CHARGE * loss))
        return terms

    def reaction_energy_loss_J_m3_s(self, rxn: CompiledGasReaction, rate_m3_s: float) -> float:
        if rxn.energy_model and str(rxn.energy_model.get('backend', '')).lower() == 'constant_event_loss':
            return float(rxn.energy_model.get('energy_loss_eV', 0.0)) * E_CHARGE * rate_m3_s
        return 0.0

    def reaction_source_loss_budget(self, coupled: CoupledPlasmaEvaluation) -> ReactionBudget:
        budget = ReactionBudget()
        for term in self.gas_reaction_terms(coupled):
            budget.add_reaction_rate(term.zone_id, term.reaction_id, term.rate_m3_s)
            for _idx, change, sp_id in term.species_changes:
                budget.add_species_change(term.zone_id, sp_id, change)
            budget.add_electron_energy_loss(term.zone_id, term.reaction_id, term.electron_energy_loss_J_m3_s)
        wall_energy_loss_by_zone: dict[str, float] = {}
        for term in self.ion_wall_loss_terms(coupled):
            budget.add_wall_loss(term.zone_id, term.species_id, term.loss_m3_s)
            wall_energy_loss_by_zone[term.zone_id] = wall_energy_loss_by_zone.get(term.zone_id, 0.0) + term.electron_energy_loss_J_m3_s
        for zone_id, loss in wall_energy_loss_by_zone.items():
            budget.add_electron_energy_loss(zone_id, 'ion_wall', loss)
        return budget

    def apply_reaction_terms(self, terms: list[GasReactionTerm], gas_rhs: np.ndarray, We_rhs: np.ndarray) -> None:
        sys = self.system
        for term in terms:
            z = sys.zone_index[term.zone_id]
            for idx, change, _sp in term.species_changes:
                gas_rhs[z, idx] += change
            We_rhs[z] -= term.electron_energy_loss_J_m3_s

    def apply_ion_wall_loss_terms(self, terms: list[IonWallLossTerm], gas_rhs: np.ndarray, We_rhs: np.ndarray) -> None:
        sys = self.system
        for term in terms:
            z = sys.zone_index[term.zone_id]
            gas_rhs[z, term.species_index] -= term.loss_m3_s
            We_rhs[z] -= term.electron_energy_loss_J_m3_s

    def apply_inlet_terms(self, step: Any, gas: np.ndarray, Tg: np.ndarray, gas_rhs: np.ndarray, Tg_rhs: np.ndarray | None) -> None:
        sys = self.system
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

    def apply_pump_terms(self, gas: np.ndarray, We: np.ndarray, gas_rhs: np.ndarray, We_rhs: np.ndarray) -> None:
        sys = self.system
        for pump in sys.chamber.pumps:
            z = sys.zone_index[pump.zone_id]
            lam = pump.speed_m3_s / max(sys.chamber.zone_by_id[pump.zone_id].volume_m3, 1.0e-30)
            gas_rhs[z, :] -= lam * gas[z, :]
            We_rhs[z] -= lam * We[z]

    def apply_interzone_terms(self, view: _GasRhsViews) -> None:
        sys = self.system
        for edge in sys.chamber.edges:
            zi = sys.zone_index[edge.from_zone]
            zj = sys.zone_index[edge.to_zone]
            Vi = sys.chamber.zone_by_id[edge.from_zone].volume_m3
            Vj = sys.chamber.zone_by_id[edge.to_zone].volume_m3
            Ci = edge.conductance_m3_s
            view.gas_rhs[zi, :] -= Ci / max(Vi, 1.0e-30) * view.gas[zi, :]
            view.gas_rhs[zj, :] += Ci / max(Vj, 1.0e-30) * view.gas[zi, :]
            view.electron_energy_rhs[zi] -= Ci / max(Vi, 1.0e-30) * view.electron_energy[zi]
            view.electron_energy_rhs[zj] += Ci / max(Vj, 1.0e-30) * view.electron_energy[zi]
            if view.gas_temperature_rhs is not None:
                n_i = max(float(np.sum(view.gas[zi])), sys.floor_density)
                n_j = max(float(np.sum(view.gas[zj])), sys.floor_density)
                view.gas_temperature_rhs[zj] += Ci / max(Vj, 1.0e-30) * n_i / n_j * (view.gas_temperature[zi] - view.gas_temperature[zj])

    def apply_power_terms(self, coupled: CoupledPlasmaEvaluation, We_rhs: np.ndarray) -> None:
        sys = self.system
        for zone_id, pabs in coupled.power.absorbed_power_W_by_zone.items():
            z = sys.zone_index[zone_id]
            V = sys.chamber.zone_by_id[zone_id].volume_m3
            We_rhs[z] += pabs / max(V, 1.0e-30)

    def apply_gas_temperature_terms(self, coupled: CoupledPlasmaEvaluation, gas: np.ndarray, Tg: np.ndarray, Tg_rhs: np.ndarray | None) -> None:
        if Tg_rhs is None:
            return
        sys = self.system
        gas_heating_fraction = float(sys.run_config.physics.gas_heating_fraction)
        wall_relax = float(sys.run_config.physics.wall_relaxation_s_inv)
        for zone_id in sys.zone_ids:
            z = sys.zone_index[zone_id]
            n_tot = max(float(np.sum(gas[z])), sys.floor_density)
            V = sys.chamber.zone_by_id[zone_id].volume_m3
            pabs = coupled.power.absorbed_power_W_by_zone.get(zone_id, 0.0)
            Tg_rhs[z] += gas_heating_fraction * pabs / max(V, 1.0e-30) / (1.5 * K_B * n_tot)
            Tg_rhs[z] -= wall_relax * (Tg[z] - self.zone_wall_temperature[zone_id])

    def apply_rhs(self, _time_s: float, _y: np.ndarray, step: Any, coupled: CoupledPlasmaEvaluation, dydt: np.ndarray) -> None:
        sys = self.system
        gas = coupled.gas
        We = coupled.electron_energy
        Tg = coupled.gas_temperature
        view = _GasRhsViews(
            gas=gas,
            electron_energy=We,
            gas_temperature=Tg,
            gas_rhs=dydt[sys.state_layout.slice('gas_densities')].reshape(sys.n_zones, sys.n_gas_species),
            electron_energy_rhs=dydt[sys.state_layout.slice('electron_energy')],
            gas_temperature_rhs=dydt[sys.state_layout.slice('gas_temperature')] if 'gas_temperature' in sys.state_layout.slices else None,
        )

        self.apply_reaction_terms(self.gas_reaction_terms(coupled), view.gas_rhs, view.electron_energy_rhs)
        self.apply_ion_wall_loss_terms(self.ion_wall_loss_terms(coupled), view.gas_rhs, view.electron_energy_rhs)
        self.apply_inlet_terms(step, gas, Tg, view.gas_rhs, view.gas_temperature_rhs)
        self.apply_pump_terms(gas, We, view.gas_rhs, view.electron_energy_rhs)
        self.apply_interzone_terms(view)
        self.apply_power_terms(coupled, view.electron_energy_rhs)
        self.apply_field_table_energy_relaxation(coupled, view.electron_energy_rhs)
        self.apply_gas_temperature_terms(coupled, gas, Tg, view.gas_temperature_rhs)

    def field_table_energy_relaxation_tau_s(self) -> float | None:
        table_cfg = self.system.run_config.swarm.table
        mode = str(table_cfg.electron_energy_mode or '').lower()
        if mode not in {'table_relaxation', 'from_table', 'prescribed_from_table'}:
            return None
        return max(float(table_cfg.energy_relaxation_time_s or 1.0e-6), 1.0e-12)

    def apply_field_table_energy_relaxation(self, coupled: CoupledPlasmaEvaluation, We_rhs: np.ndarray) -> None:
        tau = self.field_table_energy_relaxation_tau_s()
        if tau is None:
            return
        sys = self.system
        for zone_id in sys.zone_ids:
            eedf = coupled.eedf_by_zone[zone_id]
            if eedf.transport.lookup_mode != 'field':
                continue
            z = sys.zone_index[zone_id]
            target_mean_e = max(float(eedf.transport.mean_energy_eV), 1.0e-6)
            target_We = max(coupled.ne_by_zone[zone_id], sys.floor_density) * target_mean_e * E_CHARGE
            We_rhs[z] = (target_We - float(coupled.electron_energy[z])) / tau
