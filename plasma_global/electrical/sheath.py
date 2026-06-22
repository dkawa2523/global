from __future__ import annotations

import math

from plasma_global.chemistry.models import E_CHARGE, K_B
from plasma_global.electrical.base import SurfaceIED

EPS0 = 8.8541878128e-12


def debye_length_m(ne_m3: float, te_eV: float) -> float:
    ne = max(float(ne_m3), 1.0)
    te_J = max(float(te_eV), 1.0e-3) * E_CHARGE
    return math.sqrt(EPS0 * te_J / (ne * E_CHARGE * E_CHARGE))


def sheath_thickness_m(ne_m3: float, te_eV: float, sheath_voltage_V: float, area_m2: float | None = None) -> float:
    ld = debye_length_m(ne_m3, te_eV)
    v_ratio = max(float(sheath_voltage_V), 0.1) / max(float(te_eV), 0.1)
    geom = 1.0
    if area_m2 is not None and area_m2 > 0.0:
        geom = min(max(math.sqrt(area_m2 / 0.03), 0.5), 2.0)
    return max(3.0 * ld * math.sqrt(1.0 + v_ratio) * geom, 1.0e-5)


def charge_exchange_mfp_m(pressure_Pa: float, gas_temperature_K: float = 300.0, sigma_cx_m2: float = 3.0e-19) -> float:
    n_n = max(float(pressure_Pa), 1.0e-6) / (K_B * max(float(gas_temperature_K), 50.0))
    return 1.0 / max(n_n * sigma_cx_m2, 1.0e-30)


def ion_sound_speed_m_s(te_eV: float, ion_mass_kg: float) -> float:
    return math.sqrt(max(float(te_eV), 0.05) * E_CHARGE / max(float(ion_mass_kg), 1.0e-30))


def bohm_ion_flux_m2_s(ne_m3: float, te_eV: float, ion_mass_kg: float) -> float:
    return 0.61 * max(float(ne_m3), 0.0) * ion_sound_speed_m_s(te_eV, ion_mass_kg)


def mean_ion_energy_eV(sheath_voltage_V: float, collision_ratio: float, waveform_factor: float = 1.0) -> float:
    vs = max(float(sheath_voltage_V), 0.0)
    xi = max(float(collision_ratio), 0.0)
    collisional_loss = 1.0 / (1.0 + 0.35 * xi + 0.02 * xi * xi)
    return max(vs * collisional_loss * max(float(waveform_factor), 0.05), 0.0)


def build_surface_ied(
    *,
    ne_m3: float,
    te_eV: float,
    ion_mass_kg: float,
    sheath_voltage_V: float,
    pressure_Pa: float,
    gas_temperature_K: float,
    area_m2: float | None = None,
) -> SurfaceIED:
    thickness = sheath_thickness_m(ne_m3, te_eV, sheath_voltage_V, area_m2=area_m2)
    mfp = charge_exchange_mfp_m(pressure_Pa, gas_temperature_K=gas_temperature_K)
    collision_ratio = max(float(thickness), 0.0) / max(float(mfp), 1.0e-30)
    return SurfaceIED(
        ion_flux_m2_s=bohm_ion_flux_m2_s(ne_m3, te_eV, ion_mass_kg),
        mean_ion_energy_eV=mean_ion_energy_eV(sheath_voltage_V, collision_ratio),
    )
