from __future__ import annotations

from typing import Any

import numpy as np

from plasma_global.chemistry.models import E_CHARGE, K_B

EV_TO_K = E_CHARGE / K_B


def gas_rate_coefficient(model: dict[str, Any], gas_temperature_K: float, eedf: Any) -> tuple[float, float, float]:
    backend = str(model.get('backend', '')).lower()
    if backend == 'electron_impact_xsec':
        cs_id = model['cross_section_id']
        branch = float(model.get('branching_yield', 1.0))
        k = branch * float(eedf.rate_coefficients.get(cs_id, 0.0))
        dk_de = branch * float(eedf.d_rate_d_mean_energy_eV.get(cs_id, 0.0))
        dk_dT = 0.0
    elif backend == 'arrhenius':
        A = float(model.get('A', 0.0))
        beta = float(model.get('beta', 0.0))
        Ea_eV = float(model.get('Ea_eV', 0.0))
        Tg = max(float(gas_temperature_K), 1.0)
        kT_eV = K_B * Tg / E_CHARGE
        k = A * (Tg / 300.0) ** beta * np.exp(-Ea_eV / max(kT_eV, 1.0e-12))
        dk_dT = k * (beta / Tg + Ea_eV * E_CHARGE / (K_B * Tg ** 2))
        dk_de = 0.0
    elif backend == 'constant':
        k = float(model.get('value', 0.0))
        dk_de = 0.0
        dk_dT = 0.0
    elif backend == 'first_order_loss':
        k = float(model.get('rate_s_inv', model.get('value', 0.0)))
        dk_de = 0.0
        dk_dT = 0.0
    elif backend in {'te_power_law', 'electron_temperature_power_law'}:
        A = float(model.get('A', model.get('value', 0.0)))
        alpha = float(model.get('alpha', model.get('exponent', 0.0)))
        Tref_K = max(float(model.get('Tref_K', model.get('reference_temperature_K', 1.0))), 1.0e-30)
        mean_e_raw = float(eedf.transport.mean_energy_eV)
        mean_floor = float(model.get('mean_energy_floor_eV', 1.0e-6))
        mean_e = max(mean_e_raw, mean_floor)
        temperature_factor = float(model.get('electron_temperature_factor', 2.0 / 3.0))
        Te_floor_K = float(model.get('electron_temperature_floor_K', 1.0))
        Te_K = max(temperature_factor * mean_e * EV_TO_K, Te_floor_K)
        k = A * (Te_K / Tref_K) ** alpha
        dk_de = k * alpha / mean_e if mean_e_raw >= mean_floor and Te_K > Te_floor_K else 0.0
        dk_dT = 0.0
    else:
        raise NotImplementedError(f'Unsupported gas rate backend: {backend}')
    return float(k), float(dk_de), float(dk_dT)
