from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plasma_global.eedf.base import EEDFResult, EEDFTransport


@dataclass
class BoltzmannSwarmTable:
    key: tuple[object, ...]
    field_grid_Td: np.ndarray
    mean_energy_by_field_eV: np.ndarray
    k_by_field: dict[str, np.ndarray]
    mobility_by_field: np.ndarray
    diffusion_by_field: np.ndarray
    mean_energy_grid_eV: np.ndarray
    field_by_mean_Td: np.ndarray
    k_by_mean: dict[str, np.ndarray]
    mobility_by_mean: np.ndarray
    diffusion_by_mean: np.ndarray


def interp_by_field(field_Td: float, table: BoltzmannSwarmTable) -> EEDFResult:
    field = float(np.clip(field_Td, table.field_grid_Td[0], table.field_grid_Td[-1]))
    k_map = {cs_id: float(np.interp(field, table.field_grid_Td, arr)) for cs_id, arr in table.k_by_field.items()}
    mean_e = float(np.interp(field, table.field_grid_Td, table.mean_energy_by_field_eV))
    return EEDFResult(
        rate_coefficients=k_map,
        transport=EEDFTransport(
            mean_energy_eV=mean_e,
            mobility_m2_V_s=float(np.interp(field, table.field_grid_Td, table.mobility_by_field)),
            diffusion_m2_s=float(np.interp(field, table.field_grid_Td, table.diffusion_by_field)),
            effective_field_Td=field,
            lookup_mode='field',
        ),
    )


def interp_by_mean_energy(mean_energy_eV: float, table: BoltzmannSwarmTable) -> EEDFResult:
    eps = float(np.clip(mean_energy_eV, table.mean_energy_grid_eV[0], table.mean_energy_grid_eV[-1]))
    k_map = {cs_id: float(np.interp(eps, table.mean_energy_grid_eV, arr)) for cs_id, arr in table.k_by_mean.items()}
    return EEDFResult(
        rate_coefficients=k_map,
        transport=EEDFTransport(
            mean_energy_eV=eps,
            mobility_m2_V_s=float(np.interp(eps, table.mean_energy_grid_eV, table.mobility_by_mean)),
            diffusion_m2_s=float(np.interp(eps, table.mean_energy_grid_eV, table.diffusion_by_mean)),
            effective_field_Td=float(np.interp(eps, table.mean_energy_grid_eV, table.field_by_mean_Td)),
            lookup_mode='mean_energy',
        ),
    )
