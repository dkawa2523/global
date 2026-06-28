from __future__ import annotations

from typing import Any

import numpy as np

from plasma_global.chemistry.models import E_CHARGE, K_B


def electron_density_from_state_row(system: Any, gas_row: np.ndarray) -> float:
    ne = float(np.dot(system.gas_charges, gas_row))
    return max(ne, system.floor_density)


def positive_ion_density_from_state_row(system: Any, gas_row: np.ndarray) -> float:
    total = 0.0
    for i, sp in enumerate(system.gas_species):
        if sp.charge > 0:
            total += sp.charge * max(float(gas_row[i]), 0.0)
    return max(total, system.floor_density)


def dominant_ion_mass_kg_from_state_row(system: Any, gas_row: np.ndarray) -> float:
    best_density = -1.0
    best_mass = None
    for i in system.positive_ion_local_indices:
        n_i = max(float(gas_row[i]), 0.0)
        if n_i > best_density:
            best_density = n_i
            best_mass = system.gas_masses[i]
    return float(best_mass if best_mass is not None else 6.63e-26)


def zone_pressure_from_state_row(system: Any, gas_row: np.ndarray, gas_temperature_K: float) -> float:
    n_total = max(float(np.sum(np.clip(gas_row, 0.0, None))), system.floor_density)
    return n_total * K_B * max(float(gas_temperature_K), 1.0)


def mean_energy_eV(system: Any, We_J_m3: float, ne_m3: float) -> float:
    return max(float(We_J_m3), system.floor_energy) / max(ne_m3, system.floor_density) / E_CHARGE


def zone_state_meta(system: Any, gas: np.ndarray, We: np.ndarray) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    ne_by_zone: dict[str, float] = {}
    mean_e_by_zone: dict[str, float] = {}
    pos_by_zone: dict[str, float] = {}
    ion_mass_by_zone: dict[str, float] = {}
    for z_idx, zone_id in enumerate(system.zone_ids):
        gas_row = gas[z_idx]
        ne = electron_density_from_state_row(system, gas_row)
        pos = positive_ion_density_from_state_row(system, gas_row)
        mean_e = mean_energy_eV(system, We[z_idx], ne)
        ion_mass = dominant_ion_mass_kg_from_state_row(system, gas_row)
        ne_by_zone[zone_id] = ne
        mean_e_by_zone[zone_id] = mean_e
        pos_by_zone[zone_id] = pos
        ion_mass_by_zone[zone_id] = ion_mass
    return ne_by_zone, mean_e_by_zone, pos_by_zone, ion_mass_by_zone
