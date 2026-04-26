from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from plasma_global.chemistry.models import E_CHARGE, K_B

EPS0 = 8.8541878128e-12
M_E = 9.1093837015e-31


@dataclass
class IEDProxy:
    ion_flux_m2_s: float
    mean_energy_eV: float
    width_eV: float
    collisionality: float
    angle_spread_deg: float
    sheath_voltage_V: float
    sheath_thickness_m: float
    plasma_potential_V: float
    transit_time_s: float
    iedf_energy_eV: list[float]
    iedf_probability: list[float]
    metadata: dict[str, Any]


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


def collisionality_index(sheath_thickness_m: float, mean_free_path_m: float) -> float:
    return max(float(sheath_thickness_m), 0.0) / max(float(mean_free_path_m), 1.0e-30)


def ion_sound_speed_m_s(te_eV: float, ion_mass_kg: float) -> float:
    return math.sqrt(max(float(te_eV), 0.05) * E_CHARGE / max(float(ion_mass_kg), 1.0e-30))


def bohm_ion_flux_m2_s(ne_m3: float, te_eV: float, ion_mass_kg: float, ion_fraction: float = 1.0) -> float:
    return 0.61 * max(float(ne_m3), 0.0) * max(float(ion_fraction), 0.0) * ion_sound_speed_m_s(te_eV, ion_mass_kg)


def ion_transit_time_s(sheath_thickness_m: float, mean_energy_eV: float, ion_mass_kg: float) -> float:
    energy_J = max(float(mean_energy_eV), 1.0e-3) * E_CHARGE
    v = math.sqrt(2.0 * energy_J / max(float(ion_mass_kg), 1.0e-30))
    return 2.0 * max(float(sheath_thickness_m), 1.0e-6) / max(v, 1.0e-9)


def mean_ion_energy_eV(sheath_voltage_V: float, collisionality: float, waveform_factor: float = 1.0) -> float:
    vs = max(float(sheath_voltage_V), 0.0)
    xi = max(float(collisionality), 0.0)
    collisional_loss = 1.0 / (1.0 + 0.35 * xi + 0.02 * xi * xi)
    return max(vs * collisional_loss * max(float(waveform_factor), 0.05), 0.0)


def ied_width_eV(mean_energy_eV_value: float, rf_angular_frequency_Hz: float, transit_time_s_value: float, collisionality: float, pulsed: bool = False) -> float:
    omega_tau = max(float(rf_angular_frequency_Hz), 0.0) * max(float(transit_time_s_value), 0.0)
    rf_factor = 0.15 + 0.55 / (1.0 + omega_tau ** 2)
    coll_factor = 1.0 + 0.4 * max(float(collisionality), 0.0)
    pulse_factor = 0.8 if pulsed else 1.0
    return max(float(mean_energy_eV_value), 0.0) * rf_factor * coll_factor * pulse_factor


def angle_spread_deg(collisionality: float) -> float:
    xi = max(float(collisionality), 0.0)
    return min(45.0, 5.0 + 28.0 * (1.0 - math.exp(-0.9 * xi)))


def build_bimodal_iedf(mean_eV: float, width_eV: float, n_points: int = 48) -> tuple[list[float], list[float]]:
    mean_eV = max(float(mean_eV), 0.0)
    width_eV = max(float(width_eV), 1.0e-6)
    left = max(mean_eV - 0.5 * width_eV, 0.0)
    right = mean_eV + 0.5 * width_eV
    grid = np.linspace(max(left - width_eV, 0.0), right + width_eV, n_points)
    sigma = max(width_eV / 5.0, 1.0e-3)
    pdf = np.exp(-0.5 * ((grid - left) / sigma) ** 2) + np.exp(-0.5 * ((grid - right) / sigma) ** 2)
    if pdf.sum() <= 0.0:
        pdf[:] = 1.0
    pdf /= max(float(np.sum(pdf)), 1.0e-30)
    return grid.tolist(), pdf.tolist()


def build_ied_proxy(
    *,
    ne_m3: float,
    te_eV: float,
    ion_mass_kg: float,
    sheath_voltage_V: float,
    rf_frequency_Hz: float,
    pressure_Pa: float,
    gas_temperature_K: float,
    ion_fraction: float = 1.0,
    area_m2: float | None = None,
    pulsed: bool = False,
    plasma_potential_V: float = 0.0,
) -> IEDProxy:
    thickness = sheath_thickness_m(ne_m3, te_eV, sheath_voltage_V, area_m2=area_m2)
    mfp = charge_exchange_mfp_m(pressure_Pa, gas_temperature_K=gas_temperature_K)
    xi = collisionality_index(thickness, mfp)
    gamma_i = bohm_ion_flux_m2_s(ne_m3, te_eV, ion_mass_kg, ion_fraction=ion_fraction)
    e_mean = mean_ion_energy_eV(sheath_voltage_V, xi, waveform_factor=1.0)
    transit = ion_transit_time_s(thickness, e_mean, ion_mass_kg)
    width = ied_width_eV(e_mean, 2.0 * math.pi * max(rf_frequency_Hz, 0.0), transit, xi, pulsed=pulsed)
    angle = angle_spread_deg(xi)
    energy_grid, prob = build_bimodal_iedf(e_mean, width)
    return IEDProxy(
        ion_flux_m2_s=gamma_i,
        mean_energy_eV=e_mean,
        width_eV=width,
        collisionality=xi,
        angle_spread_deg=angle,
        sheath_voltage_V=max(float(sheath_voltage_V), 0.0),
        sheath_thickness_m=thickness,
        plasma_potential_V=float(plasma_potential_V),
        transit_time_s=transit,
        iedf_energy_eV=energy_grid,
        iedf_probability=prob,
        metadata={'mean_free_path_m': mfp},
    )


def build_species_resolved_ied(
    *,
    ion_species: dict[str, dict[str, float]] | None,
    ne_m3: float,
    te_eV: float,
    sheath_voltage_V: float,
    rf_frequency_Hz: float,
    pressure_Pa: float,
    gas_temperature_K: float,
    area_m2: float | None = None,
    pulsed: bool = False,
    plasma_potential_V: float = 0.0,
    fallback_ion_mass_kg: float = 6.63e-26,
) -> dict[str, Any]:
    species = ion_species or {}
    total_charge_density = 0.0
    for payload in species.values():
        total_charge_density += abs(float(payload.get('charge', 1.0))) * max(float(payload.get('density_m3', 0.0)), 0.0)

    species_payload: dict[str, dict[str, float | list[float]]] = {}
    flux_sum = 0.0
    mean_energy_flux_weighted = 0.0
    width_flux_weighted = 0.0
    coll_flux_weighted = 0.0
    angle_flux_weighted = 0.0
    chosen_proxy: IEDProxy | None = None

    for sp_id, payload in species.items():
        density = max(float(payload.get('density_m3', 0.0)), 0.0)
        charge = abs(float(payload.get('charge', 1.0)))
        if density <= 0.0 or charge <= 0.0:
            continue
        frac = charge * density / max(total_charge_density, 1.0e-30)
        proxy = build_ied_proxy(
            ne_m3=ne_m3,
            te_eV=te_eV,
            ion_mass_kg=max(float(payload.get('mass_kg', fallback_ion_mass_kg)), 1.0e-30),
            sheath_voltage_V=sheath_voltage_V,
            rf_frequency_Hz=rf_frequency_Hz,
            pressure_Pa=pressure_Pa,
            gas_temperature_K=gas_temperature_K,
            ion_fraction=frac,
            area_m2=area_m2,
            pulsed=pulsed,
            plasma_potential_V=plasma_potential_V,
        )
        if chosen_proxy is None:
            chosen_proxy = proxy
        species_payload[sp_id] = {
            'density_m3': density,
            'charge_state': charge,
            'charge_fraction': frac,
            'ion_flux_m2_s': proxy.ion_flux_m2_s,
            'mean_ion_energy_eV': proxy.mean_energy_eV,
            'width_eV': proxy.width_eV,
            'collisionality': proxy.collisionality,
            'angle_spread_deg': proxy.angle_spread_deg,
            'sheath_voltage_V': proxy.sheath_voltage_V,
            'plasma_potential_V': proxy.plasma_potential_V,
            'transit_time_s': proxy.transit_time_s,
            'iedf_energy_eV': proxy.iedf_energy_eV,
            'iedf_probability': proxy.iedf_probability,
            'mass_kg': float(payload.get('mass_kg', fallback_ion_mass_kg)),
        }
        flux_sum += proxy.ion_flux_m2_s
        mean_energy_flux_weighted += proxy.ion_flux_m2_s * proxy.mean_energy_eV
        width_flux_weighted += proxy.ion_flux_m2_s * proxy.width_eV
        coll_flux_weighted += proxy.ion_flux_m2_s * proxy.collisionality
        angle_flux_weighted += proxy.ion_flux_m2_s * proxy.angle_spread_deg

    if chosen_proxy is None:
        chosen_proxy = build_ied_proxy(
            ne_m3=ne_m3,
            te_eV=te_eV,
            ion_mass_kg=fallback_ion_mass_kg,
            sheath_voltage_V=sheath_voltage_V,
            rf_frequency_Hz=rf_frequency_Hz,
            pressure_Pa=pressure_Pa,
            gas_temperature_K=gas_temperature_K,
            ion_fraction=1.0,
            area_m2=area_m2,
            pulsed=pulsed,
            plasma_potential_V=plasma_potential_V,
        )
        flux_sum = chosen_proxy.ion_flux_m2_s
        mean_energy_flux_weighted = flux_sum * chosen_proxy.mean_energy_eV
        width_flux_weighted = flux_sum * chosen_proxy.width_eV
        coll_flux_weighted = flux_sum * chosen_proxy.collisionality
        angle_flux_weighted = flux_sum * chosen_proxy.angle_spread_deg

    flux_ref = max(flux_sum, 1.0e-30)
    return {
        'ion_flux_m2_s': flux_sum,
        'mean_ion_energy_eV': mean_energy_flux_weighted / flux_ref,
        'width_eV': width_flux_weighted / flux_ref,
        'collisionality': coll_flux_weighted / flux_ref,
        'angle_spread_deg': angle_flux_weighted / flux_ref,
        'sheath_voltage_V': chosen_proxy.sheath_voltage_V,
        'plasma_potential_V': chosen_proxy.plasma_potential_V,
        'sheath_thickness_m': chosen_proxy.sheath_thickness_m,
        'transit_time_s': chosen_proxy.transit_time_s,
        'iedf_energy_eV': chosen_proxy.iedf_energy_eV,
        'iedf_probability': chosen_proxy.iedf_probability,
        'ion_species': species_payload,
    }
