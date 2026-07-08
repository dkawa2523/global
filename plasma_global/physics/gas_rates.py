from __future__ import annotations

from typing import Any

import numpy as np

from plasma_global.chemistry.models import E_CHARGE, K_B
from plasma_global.chemistry.rate_tables import lookup_table_1d
from plasma_global.chemistry.rate_model_schema import GAS_RATE_BACKEND_NAMES

EV_TO_K = E_CHARGE / K_B


def _electron_impact_xsec_rate(model: dict[str, Any], _gas_temperature_K: float, eedf: Any, _pressure_Pa: float | None) -> float:
    cs_id = model['cross_section_id']
    branch = float(model.get('branching_yield', 1.0))
    return branch * float(eedf.rate_coefficients.get(cs_id, 0.0))


def _arrhenius_rate(model: dict[str, Any], gas_temperature_K: float, _eedf: Any, _pressure_Pa: float | None) -> float:
    A = float(model.get('A', 0.0))
    beta = float(model.get('beta', 0.0))
    Ea_eV = float(model.get('Ea_eV', 0.0))
    Tg = max(float(gas_temperature_K), 1.0)
    kT_eV = K_B * Tg / E_CHARGE
    return A * (Tg / 300.0) ** beta * np.exp(-Ea_eV / max(kT_eV, 1.0e-12))


def _constant_rate(model: dict[str, Any], _gas_temperature_K: float, _eedf: Any, _pressure_Pa: float | None) -> float:
    return float(model.get('value', 0.0))


def _first_order_loss_rate(model: dict[str, Any], _gas_temperature_K: float, _eedf: Any, _pressure_Pa: float | None) -> float:
    return float(model.get('rate_s_inv', model.get('value', 0.0)))


def _electron_temperature_power_law_rate(model: dict[str, Any], _gas_temperature_K: float, eedf: Any, _pressure_Pa: float | None) -> float:
    A = float(model.get('A', model.get('value', 0.0)))
    alpha = float(model.get('alpha', model.get('exponent', 0.0)))
    Tref_K = max(float(model.get('Tref_K', model.get('reference_temperature_K', 1.0))), 1.0e-30)
    mean_floor = float(model.get('mean_energy_floor_eV', 1.0e-6))
    mean_e = max(float(eedf.transport.mean_energy_eV), mean_floor)
    temperature_factor = float(model.get('electron_temperature_factor', 2.0 / 3.0))
    Te_floor_K = float(model.get('electron_temperature_floor_K', 1.0))
    Te_K = max(temperature_factor * mean_e * EV_TO_K, Te_floor_K)
    return A * (Te_K / Tref_K) ** alpha


def _tabulated_1d_rate(model: dict[str, Any], gas_temperature_K: float, eedf: Any, pressure_Pa: float | None) -> float:
    axis_name = str(model.get('x', '')).strip()
    if axis_name == 'gas_temperature_K':
        x_value = float(gas_temperature_K)
    elif axis_name == 'mean_energy_eV':
        x_value = float(eedf.transport.mean_energy_eV)
    elif axis_name == 'reduced_field_Td':
        x_value = float(eedf.transport.effective_field_Td)
    elif axis_name == 'pressure_Pa':
        if pressure_Pa is None:
            raise ValueError('tabulated_1d pressure_Pa lookup requires pressure_Pa')
        x_value = float(pressure_Pa)
    else:
        raise ValueError(f'Unsupported tabulated_1d axis: {axis_name!r}')
    return lookup_table_1d(model, x_value)


GAS_RATE_BACKENDS = {
    'electron_impact_xsec': _electron_impact_xsec_rate,
    'arrhenius': _arrhenius_rate,
    'constant': _constant_rate,
    'first_order_loss': _first_order_loss_rate,
    'te_power_law': _electron_temperature_power_law_rate,
    'electron_temperature_power_law': _electron_temperature_power_law_rate,
    'tabulated_1d': _tabulated_1d_rate,
}

if set(GAS_RATE_BACKENDS) != GAS_RATE_BACKEND_NAMES:
    raise RuntimeError('gas rate backend registry is out of sync with chemistry rate model schema')


def gas_rate_coefficient(model: dict[str, Any], gas_temperature_K: float, eedf: Any, pressure_Pa: float | None = None) -> float:
    backend = str(model.get('backend', '')).lower()
    evaluator = GAS_RATE_BACKENDS.get(backend)
    if evaluator is None:
        raise NotImplementedError(f'Unsupported gas rate backend: {backend}')
    return float(evaluator(model, gas_temperature_K, eedf, pressure_Pa))
