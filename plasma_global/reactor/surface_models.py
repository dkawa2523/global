from __future__ import annotations

from collections.abc import Mapping
from typing import Any

K_B = 1.380649e-23

BOHM_ION_LOSS_MODES = {
    'active',
    'bohm',
    'bohm_edge_loss',
    'bohm_global_loss',
    'bohm_like',
    'default',
    'on',
    'sheath_flux',
    'true',
    'yes',
    '1',
}
AMBIPOLAR_ION_LOSS_MODES = {
    'ambipolar',
    'ambipolar_diffusion',
    'ambipolar_diffusion_loss',
    'diffusion',
    'diffusion_loss',
}
ION_LOSS_ACTIVE_MODES = BOHM_ION_LOSS_MODES | AMBIPOLAR_ION_LOSS_MODES
ION_LOSS_DISABLED_MODES = {'disabled', 'false', 'no', 'none', 'off', '0'}
ION_LOSS_KNOWN_MODES = ION_LOSS_ACTIVE_MODES | ION_LOSS_DISABLED_MODES
ION_LOSS_CONSUMED_MODEL_KEYS = {
    'ion_loss',
    'h_factor',
    'edge_to_center_factor',
    'ion_neutral_cross_section_m2',
    'characteristic_length_m',
    'min_h_factor',
    'max_h_factor',
    'ambipolar_loss_rate_s',
    'loss_rate_s',
    'ambipolar_diffusion_coefficient_m2_s',
    'diffusion_coefficient_m2_s',
    'diffusion_length_m',
}


def normalized_model_value(value: Any) -> str:
    return str(value).strip().lower().replace('-', '_')


def configured_ion_loss_mode(models: Mapping[str, Any]) -> str:
    if 'ion_loss' not in models:
        return 'bohm_edge_loss'
    return normalized_model_value(models.get('ion_loss'))


def ion_loss_family(models: Mapping[str, Any]) -> str:
    mode = configured_ion_loss_mode(models)
    if mode in ION_LOSS_DISABLED_MODES:
        return 'disabled'
    if mode in AMBIPOLAR_ION_LOSS_MODES:
        return 'ambipolar_diffusion'
    return 'bohm'


def ion_loss_enabled(models: Mapping[str, Any]) -> bool:
    return ion_loss_family(models) != 'disabled'


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
    value = models.get('h_factor', models.get('edge_to_center_factor'))
    if value is None:
        value = 'auto' if configured_ion_loss_mode(models) == 'bohm_global_loss' else 1.0
    min_h = _model_float(models, ('min_h_factor',), 0.02) or 0.02
    max_h = _model_float(models, ('max_h_factor',), 1.0) or 1.0
    if normalized_model_value(value) == 'auto':
        sigma = _model_float(models, ('ion_neutral_cross_section_m2',), 1.0e-18) or 1.0e-18
        n_gas = max(float(pressure_Pa), 0.0) / (K_B * max(float(gas_temperature_K), 1.0))
        mean_free_path = 1.0 / max(n_gas * max(sigma, 1.0e-30), 1.0e-30)
        h = 1.0 / (1.0 + max(characteristic_length_m, 0.0) / mean_free_path) ** 0.5
        return _clamp(h, min_h, max_h)
    return _clamp(float(value), min_h, max_h)


def ambipolar_loss_rate_s(models: Mapping[str, Any], *, volume_m3: float, area_m2: float) -> float:
    explicit = _model_float(models, ('ambipolar_loss_rate_s', 'loss_rate_s'))
    if explicit is not None:
        return max(explicit, 0.0)
    D = _model_float(models, ('ambipolar_diffusion_coefficient_m2_s', 'diffusion_coefficient_m2_s'))
    if D is None:
        raise ValueError('ambipolar_diffusion ion_loss requires ambipolar_loss_rate_s or diffusion_coefficient_m2_s')
    diffusion_length = _model_float(models, ('diffusion_length_m',))
    if diffusion_length is None:
        diffusion_length = volume_m3 / max(area_m2, 1.0e-30)
    return max(D, 0.0) / max(diffusion_length, 1.0e-30) ** 2
