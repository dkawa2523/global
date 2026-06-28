from __future__ import annotations

from collections.abc import Mapping
from typing import Any

K_B = 1.380649e-23
E_CHARGE = 1.602176634e-19
BOHM_FLUX_COEFF = 0.61

BOHM_ION_LOSS_MODES = {'bohm'}
PRESCRIBED_ION_LOSS_MODES = {'prescribed_loss_frequency'}
AMBIPOLAR_ION_LOSS_MODES = {'ambipolar_diffusion'}
EFFECTIVE_FREQUENCY_ION_LOSS_FAMILIES = {'prescribed_loss_frequency', 'ambipolar_diffusion'}
ION_LOSS_ACTIVE_MODES = BOHM_ION_LOSS_MODES | PRESCRIBED_ION_LOSS_MODES | AMBIPOLAR_ION_LOSS_MODES
ION_LOSS_DISABLED_MODES = {'off'}
ION_LOSS_KNOWN_MODES = ION_LOSS_ACTIVE_MODES | ION_LOSS_DISABLED_MODES


def normalized_model_value(value: Any) -> str:
    return str(value).strip().lower().replace('-', '_')


def configured_ion_loss_mode(models: Mapping[str, Any]) -> str:
    if 'ion_loss' not in models:
        return 'bohm'
    return normalized_model_value(models.get('ion_loss'))


def ion_loss_family(models: Mapping[str, Any]) -> str:
    mode = configured_ion_loss_mode(models)
    if mode in ION_LOSS_DISABLED_MODES:
        return 'disabled'
    if mode in BOHM_ION_LOSS_MODES:
        return 'bohm'
    if mode in PRESCRIBED_ION_LOSS_MODES:
        return 'prescribed_loss_frequency'
    if mode in AMBIPOLAR_ION_LOSS_MODES:
        return 'ambipolar_diffusion'
    raise ValueError(f'Unknown ion_loss mode {models.get("ion_loss")!r}')


def ion_loss_enabled(models: Mapping[str, Any]) -> bool:
    return ion_loss_family(models) != 'disabled'


def ion_loss_uses_effective_frequency(family: str) -> bool:
    return normalized_model_value(family) in EFFECTIVE_FREQUENCY_ION_LOSS_FAMILIES


def _model_float(models: Mapping[str, Any], keys: tuple[str, ...], default: float | None = None) -> float | None:
    for key in keys:
        if key in models and models[key] is not None:
            return float(models[key])
    return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def bohm_h_factor(
    models: Mapping[str, Any],
    *,
    pressure_Pa: float,
    gas_temperature_K: float,
    characteristic_length_m: float,
) -> float:
    value = models.get('h_factor')
    if value is None:
        value = 1.0
    min_h = _model_float(models, ('min_h_factor',), 0.02) or 0.02
    max_h = _model_float(models, ('max_h_factor',), 1.0) or 1.0
    if normalized_model_value(value) == 'auto':
        sigma = _model_float(models, ('ion_neutral_cross_section_m2',), 1.0e-18) or 1.0e-18
        n_gas = max(float(pressure_Pa), 0.0) / (K_B * max(float(gas_temperature_K), 1.0))
        mean_free_path = 1.0 / max(n_gas * max(sigma, 1.0e-30), 1.0e-30)
        h = 1.0 / (1.0 + max(characteristic_length_m, 0.0) / mean_free_path) ** 0.5
        return _clamp(h, min_h, max_h)
    return _clamp(float(value), min_h, max_h)


def effective_ion_loss_frequency_s(models: Mapping[str, Any], *, volume_m3: float, area_m2: float) -> float:
    family = ion_loss_family(models)
    if family == 'prescribed_loss_frequency':
        explicit = _model_float(models, ('frequency_s',))
        if explicit is None:
            raise ValueError('prescribed_loss_frequency requires frequency_s')
        return max(explicit, 0.0)
    if family != 'ambipolar_diffusion':
        return 0.0
    D = _model_float(models, ('diffusion_coefficient_m2_s',))
    if D is None:
        raise ValueError('ambipolar_diffusion requires diffusion_coefficient_m2_s')
    diffusion_length = _model_float(models, ('diffusion_length_m',))
    if diffusion_length is None:
        diffusion_length = volume_m3 / max(area_m2, 1.0e-30)
    return max(D, 0.0) / max(diffusion_length, 1.0e-30) ** 2


def bohm_ion_loss_frequency_s(
    *,
    area_m2: float,
    volume_m3: float,
    h_factor: float,
    mean_energy_eV: float,
    ion_mass_kg: float,
) -> float:
    if area_m2 <= 0.0 or volume_m3 <= 0.0 or h_factor <= 0.0:
        return 0.0
    mass = max(float(ion_mass_kg), 1.0e-30)
    sound_speed = (max(float(mean_energy_eV), 0.05) * E_CHARGE / mass) ** 0.5
    return max(float(area_m2), 0.0) / max(float(volume_m3), 1.0e-30) * max(float(h_factor), 0.0) * BOHM_FLUX_COEFF * sound_speed
