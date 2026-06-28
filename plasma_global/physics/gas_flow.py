from __future__ import annotations

from typing import Any

import numpy as np

SCCM_TO_PARTICLES_PER_S = 4.477962e17


def apply_inlet_terms(
    system: Any,
    step: Any,
    gas: np.ndarray,
    gas_temperature: np.ndarray,
    gas_rhs: np.ndarray,
    gas_temperature_rhs: np.ndarray | None,
) -> None:
    for inlet_id, default_inlet in system.chamber.inlet_by_id.items():
        flows = step.gas_inlets.get(inlet_id, default_inlet.flow_sccm)
        z = system.zone_index[default_inlet.zone_id]
        volume = system.chamber.zone_by_id[default_inlet.zone_id].volume_m3
        for sp_id, flow_sccm in flows.items():
            if sp_id in system.gas_species_index:
                gas_rhs[z, system.gas_species_index[sp_id]] += flow_sccm * SCCM_TO_PARTICLES_PER_S / max(volume, 1.0e-30)
        if gas_temperature_rhs is not None:
            n_tot = max(float(np.sum(gas[z])), system.floor_density)
            particle_source = sum(float(v) for v in flows.values()) * SCCM_TO_PARTICLES_PER_S / max(volume, 1.0e-30)
            gas_temperature_rhs[z] += particle_source / n_tot * (default_inlet.temperature_K - gas_temperature[z])


def apply_pump_terms(
    system: Any,
    gas: np.ndarray,
    electron_energy: np.ndarray,
    gas_rhs: np.ndarray,
    electron_energy_rhs: np.ndarray,
) -> None:
    for pump in system.chamber.pumps:
        z = system.zone_index[pump.zone_id]
        volume = system.chamber.zone_by_id[pump.zone_id].volume_m3
        loss_frequency = pump.speed_m3_s / max(volume, 1.0e-30)
        gas_rhs[z, :] -= loss_frequency * gas[z, :]
        electron_energy_rhs[z] -= loss_frequency * electron_energy[z]


def apply_interzone_terms(
    system: Any,
    gas: np.ndarray,
    electron_energy: np.ndarray,
    gas_temperature: np.ndarray,
    gas_rhs: np.ndarray,
    electron_energy_rhs: np.ndarray,
    gas_temperature_rhs: np.ndarray | None,
) -> None:
    for edge in system.chamber.edges:
        zi = system.zone_index[edge.from_zone]
        zj = system.zone_index[edge.to_zone]
        Vi = system.chamber.zone_by_id[edge.from_zone].volume_m3
        Vj = system.chamber.zone_by_id[edge.to_zone].volume_m3
        Ci = edge.conductance_m3_s
        gas_rhs[zi, :] -= Ci / max(Vi, 1.0e-30) * gas[zi, :]
        gas_rhs[zj, :] += Ci / max(Vj, 1.0e-30) * gas[zi, :]
        electron_energy_rhs[zi] -= Ci / max(Vi, 1.0e-30) * electron_energy[zi]
        electron_energy_rhs[zj] += Ci / max(Vj, 1.0e-30) * electron_energy[zi]
        if gas_temperature_rhs is not None:
            n_i = max(float(np.sum(gas[zi])), system.floor_density)
            n_j = max(float(np.sum(gas[zj])), system.floor_density)
            gas_temperature_rhs[zj] += Ci / max(Vj, 1.0e-30) * n_i / n_j * (gas_temperature[zi] - gas_temperature[zj])
