from __future__ import annotations

from typing import Any

import numpy as np

from plasma_global.chemistry.models import E_CHARGE
from plasma_global.physics.types import CoupledPlasmaEvaluation, IonWallLossTerm
from plasma_global.reactor.surface_models import bohm_ion_loss_frequency_s, ion_loss_uses_effective_frequency


def ion_wall_loss_frequency_s(system: Any, zone_id: str, ion_local_idx: int, mean_energy_eV: float) -> float:
    family = system.zone_ion_loss_family.get(zone_id, 'disabled')
    if family == 'bohm':
        zone = system.chamber.zone_by_id[zone_id]
        return bohm_ion_loss_frequency_s(
            area_m2=system.zone_ion_loss_area.get(zone_id, 0.0),
            volume_m3=zone.volume_m3,
            h_factor=system.zone_ion_loss_h_factor.get(zone_id, 0.0),
            mean_energy_eV=mean_energy_eV,
            ion_mass_kg=system.gas_masses[ion_local_idx],
        )
    if ion_loss_uses_effective_frequency(family):
        return max(float(system.zone_effective_ion_loss_frequency_s.get(zone_id, 0.0)), 0.0)
    return 0.0


def ion_wall_loss_flux_m2_s(system: Any, zone_id: str, gas_row: np.ndarray, mean_energy_eV: float) -> float:
    total_loss_source = 0.0
    for idx in system.positive_ion_local_indices:
        frequency = ion_wall_loss_frequency_s(system, zone_id, idx, mean_energy_eV)
        total_loss_source += frequency * max(float(gas_row[idx]), 0.0)
    volume = system.chamber.zone_by_id[zone_id].volume_m3
    area = system.zone_ion_loss_area.get(zone_id, 0.0)
    flux = total_loss_source * volume / max(area, 1.0e-30) if area > 0.0 else 0.0
    return max(float(flux), 0.0)


def ion_wall_loss_terms(system: Any, coupled: CoupledPlasmaEvaluation) -> list[IonWallLossTerm]:
    terms: list[IonWallLossTerm] = []
    for zone_id in system.zone_ids:
        if system.zone_ion_loss_family.get(zone_id, 'disabled') == 'disabled':
            continue
        z = system.zone_index[zone_id]
        mean_e = coupled.mean_e_by_zone[zone_id]
        for idx in system.positive_ion_local_indices:
            n_i = max(float(coupled.gas[z, idx]), 0.0)
            if n_i <= 0.0:
                continue
            frequency = ion_wall_loss_frequency_s(system, zone_id, idx, mean_e)
            if frequency <= 0.0:
                continue
            loss = frequency * n_i
            charge = max(float(system.gas_charges[idx]), 1.0)
            terms.append(
                IonWallLossTerm(
                    zone_id=zone_id,
                    species_index=idx,
                    species_id=system.gas_species_ids[idx],
                    loss_m3_s=loss,
                    electron_energy_loss_J_m3_s=charge * mean_e * E_CHARGE * loss,
                )
            )
    return terms
