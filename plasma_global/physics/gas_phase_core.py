from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from plasma_global.chemistry.models import K_B
from plasma_global.physics.electron_energy_relaxation import (
    apply_field_table_energy_relaxation as apply_field_table_energy_relaxation_term,
    field_table_energy_relaxation_zones,
    field_table_energy_relaxation_tau_s as field_table_energy_relaxation_tau_s_for,
)
from plasma_global.physics.gas_flow import apply_inlet_terms, apply_interzone_terms, apply_pump_terms
from plasma_global.physics.gas_reactions import compile_gas_reactions, gas_reaction_terms
from plasma_global.physics.gas_state import (
    clip_negative_gas_rhs,
    compute_zone_wall_temperatures,
    gas_row as gas_state_row,
    initialize_gas_state,
    project_gas_state,
)
from plasma_global.physics.types import CompiledGasReaction, CoupledPlasmaEvaluation, GasReactionTerm, IonWallLossTerm
from plasma_global.physics.wall_loss_terms import (
    ion_wall_loss_flux_m2_s as calculate_ion_wall_loss_flux_m2_s,
    ion_wall_loss_frequency_s as calculate_ion_wall_loss_frequency_s,
    ion_wall_loss_terms as calculate_ion_wall_loss_terms,
)


@dataclass
class _GasRhsViews:
    gas: np.ndarray
    electron_energy: np.ndarray
    gas_temperature: np.ndarray
    gas_rhs: np.ndarray
    electron_energy_rhs: np.ndarray
    gas_temperature_rhs: np.ndarray | None


def electron_absorbed_power_fraction(system: Any) -> float:
    if 'gas_temperature' not in system.state_layout.slices:
        return 1.0
    gas_fraction = float(system.run_config.physics.gas_heating_fraction)
    return min(max(1.0 - gas_fraction, 0.0), 1.0)


@dataclass
class GasPhaseCore:
    system: Any
    gas_reactions: list[CompiledGasReaction] = field(init=False)
    zone_wall_temperature: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self.zone_wall_temperature = compute_zone_wall_temperatures(self.system)
        self.gas_reactions = compile_gas_reactions(self.system)

    def initialize_state(self, y0: np.ndarray) -> None:
        initialize_gas_state(self.system, y0)

    def project_state(self, y: np.ndarray) -> np.ndarray:
        return project_gas_state(self.system, y)

    def clip_negative_rhs(self, y: np.ndarray, dydt: np.ndarray) -> None:
        clip_negative_gas_rhs(self.system, y, dydt)

    def gas_row(self, state: np.ndarray, zone_id: str) -> np.ndarray:
        return gas_state_row(self.system, state, zone_id)

    def ion_wall_loss_frequency_s(self, zone_id: str, ion_local_idx: int, mean_energy_eV: float) -> float:
        return calculate_ion_wall_loss_frequency_s(self.system, zone_id, ion_local_idx, mean_energy_eV)

    def ion_wall_loss_flux_m2_s(self, zone_id: str, gas_row: np.ndarray, mean_energy_eV: float) -> float:
        return calculate_ion_wall_loss_flux_m2_s(self.system, zone_id, gas_row, mean_energy_eV)

    def ion_wall_loss_terms(self, coupled: CoupledPlasmaEvaluation) -> list[IonWallLossTerm]:
        return calculate_ion_wall_loss_terms(self.system, coupled)

    def apply_reaction_terms(
        self,
        terms: list[GasReactionTerm],
        gas_rhs: np.ndarray,
        We_rhs: np.ndarray,
        electron_energy_relaxed_zones: set[str] | None = None,
    ) -> None:
        sys = self.system
        relaxed_zones = electron_energy_relaxed_zones or set()
        for term in terms:
            z = sys.zone_index[term.zone_id]
            for idx, change, _sp in term.species_changes:
                gas_rhs[z, idx] += change
            if term.zone_id not in relaxed_zones:
                We_rhs[z] -= term.electron_energy_loss_J_m3_s

    def apply_wall_loss_terms(
        self,
        terms: list[IonWallLossTerm],
        gas_rhs: np.ndarray,
        We_rhs: np.ndarray,
        electron_energy_relaxed_zones: set[str] | None = None,
    ) -> None:
        sys = self.system
        relaxed_zones = electron_energy_relaxed_zones or set()
        for term in terms:
            z = sys.zone_index[term.zone_id]
            gas_rhs[z, term.species_index] -= term.loss_m3_s
            if term.zone_id not in relaxed_zones:
                We_rhs[z] -= term.electron_energy_loss_J_m3_s

    def apply_flow_terms(self, step: Any, view: _GasRhsViews, electron_energy_relaxed_zones: set[str]) -> None:
        sys = self.system
        apply_inlet_terms(sys, step, view.gas, view.gas_temperature, view.gas_rhs, view.gas_temperature_rhs)
        apply_pump_terms(
            sys,
            view.gas,
            view.electron_energy,
            view.gas_rhs,
            view.electron_energy_rhs,
            electron_energy_relaxed_zones,
        )
        apply_interzone_terms(
            sys,
            view.gas,
            view.electron_energy,
            view.gas_temperature,
            view.gas_rhs,
            view.electron_energy_rhs,
            view.gas_temperature_rhs,
            electron_energy_relaxed_zones,
        )

    def apply_power_terms(
        self,
        coupled: CoupledPlasmaEvaluation,
        We_rhs: np.ndarray,
        electron_energy_relaxed_zones: set[str] | None = None,
    ) -> None:
        sys = self.system
        relaxed_zones = electron_energy_relaxed_zones or set()
        electron_fraction = electron_absorbed_power_fraction(sys)
        for zone_id, pabs in coupled.power.absorbed_power_W_by_zone.items():
            if zone_id in relaxed_zones:
                continue
            z = sys.zone_index[zone_id]
            V = sys.chamber.zone_by_id[zone_id].volume_m3
            We_rhs[z] += electron_fraction * pabs / max(V, 1.0e-30)

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
        relaxed_zones = self.field_table_energy_relaxation_zones(coupled)

        self.apply_reaction_terms(
            gas_reaction_terms(self.system, self.gas_reactions, coupled),
            view.gas_rhs,
            view.electron_energy_rhs,
            relaxed_zones,
        )
        self.apply_wall_loss_terms(self.ion_wall_loss_terms(coupled), view.gas_rhs, view.electron_energy_rhs, relaxed_zones)
        self.apply_flow_terms(step, view, relaxed_zones)
        self.apply_power_terms(coupled, view.electron_energy_rhs, relaxed_zones)
        self.apply_field_table_energy_relaxation(coupled, view.electron_energy_rhs)
        self.apply_gas_temperature_terms(coupled, view.gas, view.gas_temperature, view.gas_temperature_rhs)

    def field_table_energy_relaxation_tau_s(self) -> float | None:
        return field_table_energy_relaxation_tau_s_for(self.system)

    def apply_field_table_energy_relaxation(self, coupled: CoupledPlasmaEvaluation, We_rhs: np.ndarray) -> None:
        apply_field_table_energy_relaxation_term(self.system, coupled, We_rhs)

    def field_table_energy_relaxation_zones(self, coupled: CoupledPlasmaEvaluation) -> set[str]:
        return field_table_energy_relaxation_zones(self.system, coupled)
