from __future__ import annotations

from typing import Any

import numpy as np

from plasma_global.chemistry.models import E_CHARGE
from plasma_global.physics.types import CoupledPlasmaEvaluation

FIELD_TABLE_RELAXATION_MODES = {'table_relaxation', 'from_table', 'prescribed_from_table'}


def field_table_energy_relaxation_tau_s(system: Any) -> float | None:
    table_cfg = system.run_config.swarm.table
    mode = str(table_cfg.electron_energy_mode or '').lower()
    if mode not in FIELD_TABLE_RELAXATION_MODES:
        return None
    return max(float(table_cfg.energy_relaxation_time_s or 1.0e-6), 1.0e-12)


def field_table_energy_relaxation_zones(system: Any, coupled: CoupledPlasmaEvaluation) -> set[str]:
    if field_table_energy_relaxation_tau_s(system) is None:
        return set()
    return {
        zone_id
        for zone_id in system.zone_ids
        if coupled.eedf_by_zone[zone_id].transport.lookup_mode == 'field'
    }


def apply_field_table_energy_relaxation(
    system: Any,
    coupled: CoupledPlasmaEvaluation,
    electron_energy_rhs: np.ndarray,
) -> None:
    tau = field_table_energy_relaxation_tau_s(system)
    if tau is None:
        return
    for zone_id in field_table_energy_relaxation_zones(system, coupled):
        eedf = coupled.eedf_by_zone[zone_id]
        z = system.zone_index[zone_id]
        target_mean_e = max(float(eedf.transport.mean_energy_eV), 1.0e-6)
        target_energy = max(coupled.ne_by_zone[zone_id], system.floor_density) * target_mean_e * E_CHARGE
        electron_energy_rhs[z] += (target_energy - float(coupled.electron_energy[z])) / tau
