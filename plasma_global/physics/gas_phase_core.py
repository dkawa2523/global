from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from plasma_global.chemistry.models import E_CHARGE, K_B
from plasma_global.physics.gas_closure import electron_density_from_state_row
from plasma_global.physics.gas_flow import apply_inlet_terms, apply_interzone_terms, apply_pump_terms
from plasma_global.physics.gas_rates import gas_rate_coefficient
from plasma_global.physics.types import CompiledGasReaction, CoupledPlasmaEvaluation, GasReactionTerm, IonWallLossTerm
from plasma_global.reactor.surface_models import bohm_ion_loss_frequency_s, ion_loss_uses_effective_frequency


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
    zone_wall_temperature: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self.zone_wall_temperature = self._compute_zone_wall_temperatures()
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
                else electron_density_from_state_row(sys, self.gas_row(y0, zone_id))
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

    def reaction_mass_action(self, rxn: CompiledGasReaction, gas_row: np.ndarray, ne: float) -> float:
        mass_action = 1.0
        for idx, nu, _sp in rxn.reactant_gas:
            mass_action *= max(gas_row[idx], self.system.floor_density) ** nu
        if rxn.electron_reactant_stoich:
            mass_action *= max(ne, self.system.floor_density) ** rxn.electron_reactant_stoich
        return float(mass_action)

    def reaction_rate(self, rxn: CompiledGasReaction, gas_row: np.ndarray, Tg: float, eedf, ne: float) -> tuple[float, float, float]:
        k, dk_de, dk_dT = gas_rate_coefficient(rxn.rate_model, Tg, eedf)
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

    def apply_reaction_terms(self, terms: list[GasReactionTerm], gas_rhs: np.ndarray, We_rhs: np.ndarray) -> None:
        sys = self.system
        for term in terms:
            z = sys.zone_index[term.zone_id]
            for idx, change, _sp in term.species_changes:
                gas_rhs[z, idx] += change
            We_rhs[z] -= term.electron_energy_loss_J_m3_s

    def apply_wall_loss_terms(self, terms: list[IonWallLossTerm], gas_rhs: np.ndarray, We_rhs: np.ndarray) -> None:
        sys = self.system
        for term in terms:
            z = sys.zone_index[term.zone_id]
            gas_rhs[z, term.species_index] -= term.loss_m3_s
            We_rhs[z] -= term.electron_energy_loss_J_m3_s

    def apply_flow_terms(self, step: Any, view: _GasRhsViews) -> None:
        sys = self.system
        apply_inlet_terms(sys, step, view.gas, view.gas_temperature, view.gas_rhs, view.gas_temperature_rhs)
        apply_pump_terms(sys, view.gas, view.electron_energy, view.gas_rhs, view.electron_energy_rhs)
        apply_interzone_terms(
            sys,
            view.gas,
            view.electron_energy,
            view.gas_temperature,
            view.gas_rhs,
            view.electron_energy_rhs,
            view.gas_temperature_rhs,
        )

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

    def _rhs_views(self, coupled: CoupledPlasmaEvaluation, dydt: np.ndarray) -> _GasRhsViews:
        sys = self.system
        gas = coupled.gas
        We = coupled.electron_energy
        Tg = coupled.gas_temperature
        return _GasRhsViews(
            gas=gas,
            electron_energy=We,
            gas_temperature=Tg,
            gas_rhs=dydt[sys.state_layout.slice('gas_densities')].reshape(sys.n_zones, sys.n_gas_species),
            electron_energy_rhs=dydt[sys.state_layout.slice('electron_energy')],
            gas_temperature_rhs=dydt[sys.state_layout.slice('gas_temperature')] if 'gas_temperature' in sys.state_layout.slices else None,
        )

    def apply_rhs(self, _time_s: float, _y: np.ndarray, step: Any, coupled: CoupledPlasmaEvaluation, dydt: np.ndarray) -> None:
        view = self._rhs_views(coupled, dydt)

        self.apply_reaction_terms(self.gas_reaction_terms(coupled), view.gas_rhs, view.electron_energy_rhs)
        self.apply_wall_loss_terms(self.ion_wall_loss_terms(coupled), view.gas_rhs, view.electron_energy_rhs)
        self.apply_flow_terms(step, view)
        self.apply_power_terms(coupled, view.electron_energy_rhs)
        self.apply_field_table_energy_relaxation(coupled, view.electron_energy_rhs)
        self.apply_gas_temperature_terms(coupled, view.gas, view.gas_temperature, view.gas_temperature_rhs)

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
