"""Derived per-zone state used by RHS coupling.

The ODE state stores densities and energies. Electrical and EEDF backends need
derived zone metadata such as electron density, mean energy, pressure, and
dominant ion mass. This module owns that projection so the coupling evaluator
does not need to know state-vector slice details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from plasma_global.electrical.base import ZoneElectricalState
from plasma_global.physics.gas_closure import mean_energy_eV, zone_pressure_from_state_row, zone_state_meta


@dataclass
class CoupledStateView:
    gas: np.ndarray
    electron_energy: np.ndarray
    gas_temperature: np.ndarray
    zone_state: dict[str, ZoneElectricalState]
    ne_by_zone: dict[str, float]
    mean_e_by_zone: dict[str, float]
    pos_by_zone: dict[str, float]
    ion_mass_by_zone: dict[str, float]
    pressure_by_zone: dict[str, float]
    total_density_by_zone: dict[str, float]


def build_coupled_state_view(system: Any, time_s: float, y: np.ndarray) -> CoupledStateView:
    gas = y[system.state_layout.slice('gas_densities')].reshape(system.n_zones, system.n_gas_species)
    electron_energy = y[system.state_layout.slice('electron_energy')]
    gas_temperature = _gas_temperature(system, y)
    ne_by_zone, mean_e_by_zone, pos_by_zone, ion_mass_by_zone = zone_state_meta(system, gas, electron_energy)
    _apply_prescribed_electrons(system, time_s, electron_energy, ne_by_zone, mean_e_by_zone)
    pressure_by_zone, total_density_by_zone = _zone_pressure_and_density(system, gas, gas_temperature)
    return CoupledStateView(
        gas=gas,
        electron_energy=electron_energy,
        gas_temperature=gas_temperature,
        zone_state={
            zone_id: ZoneElectricalState(
                electron_density_m3=ne_by_zone[zone_id],
                mean_energy_eV=mean_e_by_zone[zone_id],
                positive_ion_density_m3=pos_by_zone[zone_id],
                dominant_ion_mass_kg=ion_mass_by_zone[zone_id],
                pressure_Pa=pressure_by_zone[zone_id],
                gas_temperature_K=float(gas_temperature[system.zone_index[zone_id]]),
                total_density_m3=total_density_by_zone[zone_id],
            )
            for zone_id in system.zone_ids
        },
        ne_by_zone=ne_by_zone,
        mean_e_by_zone=mean_e_by_zone,
        pos_by_zone=pos_by_zone,
        ion_mass_by_zone=ion_mass_by_zone,
        pressure_by_zone=pressure_by_zone,
        total_density_by_zone=total_density_by_zone,
    )


def _gas_temperature(system: Any, y: np.ndarray) -> np.ndarray:
    if 'gas_temperature' in system.state_layout.slices:
        return y[system.state_layout.slice('gas_temperature')]
    return np.array([system.chamber.zone_by_id[z].gas_temperature_K for z in system.zone_ids], dtype=float)


def _apply_prescribed_electrons(
    system: Any,
    time_s: float,
    electron_energy: np.ndarray,
    ne_by_zone: dict[str, float],
    mean_e_by_zone: dict[str, float],
) -> None:
    prescribed_ne = system.prescribed_electron_density_by_zone(time_s)
    if prescribed_ne is None:
        return
    for zone_id, ne in prescribed_ne.items():
        z = system.zone_index[zone_id]
        ne_by_zone[zone_id] = max(float(ne), system.floor_density)
        mean_e_by_zone[zone_id] = mean_energy_eV(system, float(electron_energy[z]), ne_by_zone[zone_id])


def _zone_pressure_and_density(
    system: Any,
    gas: np.ndarray,
    gas_temperature: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    pressure_by_zone: dict[str, float] = {}
    total_density_by_zone: dict[str, float] = {}
    for z_idx, zone_id in enumerate(system.zone_ids):
        pressure_by_zone[zone_id] = zone_pressure_from_state_row(system, gas[z_idx], float(gas_temperature[z_idx]))
        total_density_by_zone[zone_id] = max(float(np.sum(np.clip(gas[z_idx], 0.0, None))), system.floor_density)
    return pressure_by_zone, total_density_by_zone


__all__ = ['CoupledStateView', 'build_coupled_state_view']
