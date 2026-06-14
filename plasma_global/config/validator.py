from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from plasma_global.electrical.circuit_models import DCSeriesCircuitConfig, dc_series_config_mapping_at_time
from plasma_global.electrical.external_table import (
    read_circuit_table,
    resolve_circuit_table_path,
    validate_circuit_table_columns,
)
from plasma_global.electrical.rf_envelope import rf_envelope_calibration_warnings, validate_rf_envelope_port
from plasma_global.config.models import ResolvedPaths, RunConfig
from plasma_global.physics.prescribed_electrons import build_prescribed_electron_profile
from plasma_global.reactor.surface_models import (
    ION_LOSS_CONSUMED_MODEL_KEYS,
    ION_LOSS_KNOWN_MODES,
    bohm_h_factor,
    configured_ion_loss_mode,
    effective_ion_loss_frequency_s,
    ion_loss_enabled,
    ion_loss_family,
    ion_loss_uses_effective_frequency,
)


@dataclass
class ConfigMessage:
    level: str
    code: str
    message: str
    entity_id: str = ''


@dataclass
class ConfigValidationReport:
    messages: list[ConfigMessage]

    @property
    def has_errors(self) -> bool:
        return any(m.level == 'ERROR' for m in self.messages)


def _surface_model_warnings(chamber_path: Path) -> list[ConfigMessage]:
    if not chamber_path.is_file():
        return []
    with chamber_path.open('r', encoding='utf-8') as fh:
        raw = yaml.safe_load(fh) or {}
    messages: list[ConfigMessage] = []
    families_by_zone: dict[str, set[str]] = {}
    effective_frequency_count_by_zone: dict[str, int] = {}
    zones = {str(z.get('zone_id')): z for z in raw.get('zones', []) or []}
    for surface in raw.get('surfaces', []) or []:
        models = surface.get('models', {}) or {}
        if models:
            surface_id = str(surface.get('surface_id', 'unknown_surface'))
            metadata_only_keys = sorted(str(k) for k in models if str(k) not in ION_LOSS_CONSUMED_MODEL_KEYS)
            if metadata_only_keys:
                keys = ', '.join(metadata_only_keys)
                messages.append(
                    ConfigMessage(
                        'WARNING',
                        'SURFACE_MODELS_METADATA_ONLY',
                        f'surface.models for {surface_id} includes metadata not directly consumed by the current solver: {keys}',
                        surface_id,
                    )
                )
            if 'ion_loss' in models:
                mode = configured_ion_loss_mode(models)
                if mode not in ION_LOSS_KNOWN_MODES:
                    messages.append(
                        ConfigMessage(
                            'WARNING',
                            'ION_LOSS_MODEL_UNRECOGNIZED',
                            f'surface.models.ion_loss for {surface_id} uses unrecognized mode {models.get("ion_loss")!r}; treating this surface as active for the Bohm-like ion wall loss.',
                            surface_id,
                        )
                    )
            if ion_loss_enabled(models):
                zone_id = str(surface.get('zone_id', ''))
                family = ion_loss_family(models)
                families_by_zone.setdefault(zone_id, set()).add(family)
                if ion_loss_uses_effective_frequency(family):
                    effective_frequency_count_by_zone[zone_id] = effective_frequency_count_by_zone.get(zone_id, 0) + 1
                    try:
                        zone = zones.get(zone_id, {})
                        effective_ion_loss_frequency_s(
                            models,
                            volume_m3=float(zone.get('volume_m3', 0.0) or 0.0),
                            area_m2=float(surface.get('area_m2', 0.0) or 0.0),
                        )
                    except Exception as exc:
                        code = 'AMBIPOLAR_LOSS_CONFIG_INVALID' if family == 'ambipolar_diffusion' else 'ION_LOSS_FREQUENCY_CONFIG_INVALID'
                        messages.append(
                            ConfigMessage(
                                'ERROR',
                                code,
                                f'effective-frequency ion loss for {surface_id} needs loss_rate_s, ion_loss_frequency_s, ambipolar_loss_rate_s, or diffusion_coefficient_m2_s: {exc}',
                                surface_id,
                            )
                        )
                else:
                    try:
                        zone = zones.get(zone_id, {})
                        area = float(surface.get('area_m2', 0.0) or 0.0)
                        volume = float(zone.get('volume_m3', 0.0) or 0.0)
                        char_len = float(models.get('characteristic_length_m') or volume / max(area, 1.0e-30))
                        bohm_h_factor(
                            models,
                            pressure_Pa=float(zone.get('pressure_Pa', 0.0) or 0.0),
                            gas_temperature_K=float(zone.get('gas_temperature_K', 300.0) or 300.0),
                            characteristic_length_m=char_len,
                        )
                    except Exception as exc:
                        messages.append(
                            ConfigMessage(
                                'ERROR',
                                'BOHM_H_FACTOR_INVALID',
                                f'Bohm ion loss h_factor for {surface_id} is invalid: {exc}',
                                surface_id,
                            )
                        )
    for zone_id, families in families_by_zone.items():
        if len(families) > 1:
            messages.append(
                ConfigMessage(
                    'ERROR',
                    'ION_LOSS_MODE_MIXED_IN_ZONE',
                    f'Zone {zone_id} mixes ion wall-loss mode families {sorted(families)}; choose one family to avoid double counting.',
                    zone_id,
                )
            )
        if effective_frequency_count_by_zone.get(zone_id, 0) > 1:
            messages.append(
                ConfigMessage(
                    'WARNING',
                    'ION_LOSS_FREQUENCY_MULTIPLE_SURFACES',
                    f'Zone {zone_id} has multiple effective-frequency ion wall-loss surfaces; the solver uses an area-weighted rate, not an additive sum.',
                    zone_id,
                )
            )
    return messages


def validate_run_config(run_config: RunConfig, resolved_paths: ResolvedPaths) -> ConfigValidationReport:
    messages: list[ConfigMessage] = []
    required_files = {
        'chamber': resolved_paths.chamber_file,
        'recipe': resolved_paths.recipe_file,
    }
    if resolved_paths.chemistry_manifest is not None:
        required_files['chemistry_manifest'] = resolved_paths.chemistry_manifest
    elif resolved_paths.chemistry_dir is not None:
        required_files['chemistry_dir'] = resolved_paths.chemistry_dir
    else:
        messages.append(ConfigMessage('ERROR', 'CHEMISTRY_PATH_MISSING', 'Either files.chemistry.manifest or files.chemistry.directory must be set.'))

    for name, p_str in required_files.items():
        p = Path(p_str)
        if not p.exists():
            messages.append(ConfigMessage('ERROR', 'FILE_NOT_FOUND', f'Required {name} path does not exist: {p}', name))

    chamber_path = Path(resolved_paths.chamber_file)
    messages.extend(_surface_model_warnings(chamber_path))

    if run_config.physics.integrator != 'scipy_bdf':
        messages.append(ConfigMessage('WARNING', 'INTEGRATOR_UNVERIFIED', f'Integrator {run_config.physics.integrator} is not part of the smoke-tested default set.', 'integrator'))

    if not 0.0 <= run_config.physics.gas_heating_fraction <= 1.0:
        messages.append(ConfigMessage('ERROR', 'GAS_HEATING_FRACTION_RANGE', 'physics.gas_heating_fraction must be between 0 and 1.', 'physics.gas_heating_fraction'))

    if run_config.physics.wall_relaxation_s_inv < 0.0:
        messages.append(ConfigMessage('ERROR', 'WALL_RELAXATION_RANGE', 'physics.wall_relaxation_s_inv must be non-negative.', 'physics.wall_relaxation_s_inv'))

    electron_closure = str(getattr(run_config.physics, 'electron_density_closure', 'quasi_neutral') or 'quasi_neutral').lower()
    allowed_electron_closures = {'quasi_neutral', 'algebraic', 'prescribed_profile', 'external_profile', 'profile'}
    if electron_closure not in allowed_electron_closures:
        messages.append(
            ConfigMessage(
                'ERROR',
                'ELECTRON_DENSITY_CLOSURE_UNRECOGNIZED',
                f'physics.electron_density_closure must be one of {sorted(allowed_electron_closures)}, got {electron_closure!r}.',
                'physics.electron_density_closure',
            )
        )
    elif electron_closure in {'prescribed_profile', 'external_profile', 'profile'}:
        try:
            profile_cfg = getattr(getattr(run_config, 'swarm', None), 'prescribed_electron_profile', None)
            zone_columns = getattr(profile_cfg, 'zone_columns', None)
            if isinstance(zone_columns, dict):
                validation_zones = [str(zone_id) for zone_id in zone_columns]
            elif zone_columns is not None:
                validation_zones = [str(zone_id) for zone_id in vars(zone_columns)]
            else:
                validation_zones = ['*']
            build_prescribed_electron_profile(run_config, validation_zones)
        except Exception as exc:
            messages.append(
                ConfigMessage(
                    'ERROR',
                    'PRESCRIBED_ELECTRON_PROFILE_INVALID',
                    f'prescribed electron density profile is missing or invalid: {exc}',
                    'swarm.prescribed_electron_profile',
                )
            )

    closure = str(getattr(run_config.swarm, 'closure', 'auto') or 'auto').lower()
    allowed_closures = {'auto', 'mean_energy', 'local_field'}
    if closure not in allowed_closures:
        messages.append(
            ConfigMessage(
                'ERROR',
                'SWARM_CLOSURE_UNRECOGNIZED',
                f'swarm.closure must be one of {sorted(allowed_closures)}, got {closure!r}.',
                'swarm.closure',
            )
        )
    elif closure == 'local_field':
        messages.append(
            ConfigMessage(
                'WARNING',
                'LOCAL_FIELD_CLOSURE_REDUCED_MODEL',
                'swarm.closure=local_field uses the electrical backend reduced-field proxy; verify or calibrate zone_reduced_field_Td for production studies.',
                'swarm.closure',
            )
        )

    if run_config.case.schema_version != 2:
        messages.append(ConfigMessage('ERROR', 'UNSUPPORTED_SCHEMA_VERSION', 'Only case schema_version 2 is supported.', 'schema'))

    return ConfigValidationReport(messages=messages)


def _merged_power_port_configs(chamber: Any, recipe: Any, messages: list[ConfigMessage]) -> list[tuple[Any, str, dict[str, Any]]]:
    merged: list[tuple[Any, str, dict[str, Any]]] = []
    for step in recipe.steps:
        for port_id, step_cfg in step.power_ports.items():
            port = chamber.power_port_by_id.get(port_id)
            if port is None:
                messages.append(
                    ConfigMessage(
                        'ERROR',
                        'UNKNOWN_POWER_PORT',
                        f'Recipe step {step.step_id} references unknown power port {port_id}',
                        port_id,
                    )
                )
                continue
            cfg = dict(port.parameters or {})
            cfg.update(step_cfg or {})
            cfg.setdefault('kind', getattr(port, 'kind', ''))
            merged.append((step, port_id, cfg))
    return merged


def validate_loaded_inputs(run_config: RunConfig, chamber: Any, recipe: Any) -> ConfigValidationReport:
    messages: list[ConfigMessage] = []
    port_configs = _merged_power_port_configs(chamber, recipe, messages)
    if run_config.physics.electrical_backend == 'dc_series_circuit':
        for step, port_id, cfg in port_configs:
            try:
                DCSeriesCircuitConfig.from_mapping(dc_series_config_mapping_at_time(cfg, float(step.t_start_s)))
            except Exception as exc:
                messages.append(
                    ConfigMessage(
                        'ERROR',
                        'DC_SERIES_CIRCUIT_CONFIG_INVALID',
                        f'dc_series_circuit port {port_id} in step {step.step_id} is missing or has invalid parameters: {exc}',
                        port_id,
                    )
                )
    if run_config.physics.electrical_backend == 'external_circuit_table':
        for step, port_id, cfg in port_configs:
            try:
                table = read_circuit_table(resolve_circuit_table_path(run_config, cfg))
                validate_circuit_table_columns(table, cfg)
            except Exception as exc:
                messages.append(
                    ConfigMessage(
                        'ERROR',
                        'EXTERNAL_CIRCUIT_TABLE_CONFIG_INVALID',
                        f'external_circuit_table port {port_id} in step {step.step_id} is missing or has invalid table settings: {exc}',
                        port_id,
                    )
                )
    if run_config.physics.electrical_backend == 'rf_envelope':
        for step, port_id, cfg in port_configs:
            try:
                validate_rf_envelope_port(cfg)
                for warning in rf_envelope_calibration_warnings(cfg):
                    messages.append(
                        ConfigMessage(
                            'WARNING',
                            'RF_ENVELOPE_CALIBRATION_HINT',
                            f'rf_envelope port {port_id} in step {step.step_id}: {warning}',
                            port_id,
                        )
                    )
            except Exception as exc:
                messages.append(
                    ConfigMessage(
                        'ERROR',
                        'RF_ENVELOPE_CONFIG_INVALID',
                        f'rf_envelope port {port_id} in step {step.step_id} is missing or has invalid parameters: {exc}',
                        port_id,
                    )
                )
    return ConfigValidationReport(messages=messages)
