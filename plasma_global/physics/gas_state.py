from __future__ import annotations

from typing import Any

import numpy as np

from plasma_global.chemistry.models import E_CHARGE, K_B
from plasma_global.physics.gas_closure import electron_density_from_state_row


def compute_zone_wall_temperatures(system: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    for zone_id in system.zone_ids:
        surfaces = system.chamber.surfaces_by_zone.get(zone_id, [])
        if not surfaces:
            out[zone_id] = system.chamber.zone_by_id[zone_id].gas_temperature_K
            continue
        area = sum(surface.area_m2 for surface in surfaces)
        weighted_temperature = sum(surface.area_m2 * surface.temperature_K for surface in surfaces)
        out[zone_id] = weighted_temperature / max(area, 1.0e-30)
    return out


def initialize_gas_state(system: Any, y0: np.ndarray) -> None:
    first_step = system.recipe.steps[0]
    global_mix = step_global_mix(first_step)
    for zone_id in system.zone_ids:
        zone = system.chamber.zone_by_id[zone_id]
        n_total = zone.pressure_Pa / (K_B * max(zone.gas_temperature_K, 1.0))
        mix = step_zone_mix(system, first_step, zone_id) or global_mix
        for sp_id, frac in mix.items():
            if sp_id in system.gas_species_index:
                y0[system.state_layout.gas_index[zone_id][sp_id]] = max(frac, 0.0) * n_total
        _apply_species_density_floors(system, y0, zone_id)
        _apply_initial_densities(system, y0, zone_id)
        ne = _initial_electron_density(system, y0, zone_id, first_step.t_start_s)
        y0[system.state_layout.electron_energy_index[zone_id]] = 3.0 * ne * E_CHARGE
        if 'gas_temperature' in system.state_layout.slices:
            y0[system.state_layout.gas_temperature_index[zone_id]] = zone.gas_temperature_K


def _apply_species_density_floors(system: Any, y0: np.ndarray, zone_id: str) -> None:
    for sp in system.gas_species:
        idx = system.state_layout.gas_index[zone_id][sp.canonical_id]
        if sp.charge > 0:
            y0[idx] = max(y0[idx], 1.0e12)
        elif sp.charge < 0:
            y0[idx] = max(min(y0[idx], 1.0e10), 0.0)


def _apply_initial_densities(system: Any, y0: np.ndarray, zone_id: str) -> None:
    zone = system.chamber.zone_by_id[zone_id]
    for sp_id, density in (zone.initial_densities_m3 or {}).items():
        if sp_id in system.gas_species_index:
            y0[system.state_layout.gas_index[zone_id][sp_id]] = max(float(density), 0.0)


def _initial_electron_density(system: Any, y0: np.ndarray, zone_id: str, time_s: float) -> float:
    prescribed_ne = system.prescribed_electron_density_by_zone(time_s)
    if prescribed_ne is not None and zone_id in prescribed_ne:
        return max(float(prescribed_ne[zone_id]), system.floor_density)
    return electron_density_from_state_row(system, gas_row(system, y0, zone_id))


def project_gas_state(system: Any, y: np.ndarray) -> np.ndarray:
    y[system.state_layout.slice('gas_densities')] = np.clip(
        y[system.state_layout.slice('gas_densities')],
        0.0,
        None,
    )
    if 'electron_energy' in system.state_layout.slices:
        e_slice = system.state_layout.slice('electron_energy')
        y[e_slice] = np.clip(y[e_slice], system.floor_energy, None)
    if 'gas_temperature' in system.state_layout.slices:
        t_slice = system.state_layout.slice('gas_temperature')
        y[t_slice] = np.clip(y[t_slice], 50.0, None)
    return y


def clip_negative_gas_rhs(system: Any, y: np.ndarray, dydt: np.ndarray) -> None:
    gas_rhs = dydt[system.state_layout.slice('gas_densities')].reshape(system.n_zones, system.n_gas_species)
    gas_state = y[system.state_layout.slice('gas_densities')].reshape(system.n_zones, system.n_gas_species)
    gas_rhs[(gas_state <= system.floor_density) & (gas_rhs < 0.0)] = 0.0
    if 'electron_energy' in system.state_layout.slices:
        energy_slice = system.state_layout.slice('electron_energy')
        energy_rhs = dydt[energy_slice]
        low_energy = y[energy_slice] <= system.floor_energy
        energy_rhs[low_energy & (energy_rhs < 0.0)] = 0.0


def step_zone_mix(system: Any, step: Any, zone_id: str) -> dict[str, float]:
    mix: dict[str, float] = {}
    total = 0.0
    for inlet_id, flows in step.gas_inlets.items():
        inlet = system.chamber.inlet_by_id.get(inlet_id)
        if inlet is None or inlet.zone_id != zone_id:
            continue
        for sp_id, value in flows.items():
            total += float(value)
            mix[sp_id] = mix.get(sp_id, 0.0) + float(value)
    if total <= 0.0:
        return {}
    return {sp: val / total for sp, val in mix.items()}


def step_global_mix(step: Any) -> dict[str, float]:
    mix: dict[str, float] = {}
    total = 0.0
    for flows in step.gas_inlets.values():
        for sp_id, value in flows.items():
            total += float(value)
            mix[sp_id] = mix.get(sp_id, 0.0) + float(value)
    if total <= 0.0:
        return {}
    return {sp: val / total for sp, val in mix.items()}


def gas_row(system: Any, state: np.ndarray, zone_id: str) -> np.ndarray:
    gas = state[system.state_layout.slice('gas_densities')].reshape(system.n_zones, system.n_gas_species)
    return gas[system.zone_index[zone_id]]
