from __future__ import annotations

from typing import Any

from pathlib import Path

from plasma_global.config.models import ResolvedPaths, RunConfig
from plasma_global.eedf.registry import EEDF_REGISTRY
from plasma_global.electrical.registry import ELECTRICAL_REGISTRY
from plasma_global.numerics.registry import INTEGRATOR_REGISTRY
from plasma_global.physics.prescribed_electrons import build_prescribed_electron_profile
from plasma_global.validation import ValidationIssue, ValidationReport

SWARM_CLOSURES = ('auto', 'mean_energy', 'local_field')


def _surface_ion_loss_messages(chamber: Any) -> list[ValidationIssue]:
    from plasma_global.reactor.surface_validation import validate_surface_ion_loss_config

    return validate_surface_ion_loss_config(chamber)


def _reactor_structure_messages(chamber: Any, recipe: Any) -> list[ValidationIssue]:
    from plasma_global.reactor.validation import validate_reactor_config

    return validate_reactor_config(chamber, recipe)


def _eedf_backend_messages(run_config: RunConfig) -> list[ValidationIssue]:
    from plasma_global.eedf.swarm_backend import SWARM_MODEL_NAMES

    backend = str(run_config.physics.eedf_backend)
    if backend not in EEDF_REGISTRY.entries:
        return [
            ValidationIssue(
                'ERROR',
                'EEDF_BACKEND_UNRECOGNIZED',
                f'physics.eedf_backend must be one of {EEDF_REGISTRY.names()}, got {backend!r}.',
                'physics.eedf_backend',
            )
        ]
    if backend != 'swarm':
        return []

    swarm = run_config.swarm
    model_name = str(getattr(swarm, 'model_name', 'table') or 'table')
    closure = str(getattr(swarm, 'closure', 'auto') or 'auto').lower()
    messages: list[ValidationIssue] = []
    if model_name not in SWARM_MODEL_NAMES:
        messages.append(
            ValidationIssue(
                'ERROR',
                'SWARM_MODEL_UNRECOGNIZED',
                f'swarm.model_name must be one of {list(SWARM_MODEL_NAMES)}, got {model_name!r}.',
                'swarm.model_name',
            )
        )
    if closure not in SWARM_CLOSURES:
        messages.append(
            ValidationIssue(
                'ERROR',
                'SWARM_CLOSURE_UNRECOGNIZED',
                f'swarm.closure must be one of {list(SWARM_CLOSURES)}, got {closure!r}.',
                'swarm.closure',
            )
        )
    if model_name == 'table' and not getattr(swarm.table, 'file', None):
        messages.append(
            ValidationIssue(
                'ERROR',
                'SWARM_TABLE_FILE_MISSING',
                'swarm.model_name table requires swarm.table.file.',
                'swarm.table.file',
            )
        )
    return messages


def _integrator_messages(run_config: RunConfig) -> list[ValidationIssue]:
    integrator = str(run_config.physics.integrator)
    if integrator in INTEGRATOR_REGISTRY.entries:
        return []
    return [
        ValidationIssue(
            'ERROR',
            'INTEGRATOR_UNSUPPORTED',
            f'physics.integrator must be one of {INTEGRATOR_REGISTRY.names()}, got {integrator!r}.',
            'physics.integrator',
        )
    ]


def validate_run_config(run_config: RunConfig, resolved_paths: ResolvedPaths) -> ValidationReport:
    messages: list[ValidationIssue] = []
    required_files = {
        'chamber': resolved_paths.chamber_file,
        'recipe': resolved_paths.recipe_file,
    }
    if resolved_paths.chemistry_manifest is not None:
        required_files['chemistry_manifest'] = resolved_paths.chemistry_manifest
    else:
        messages.append(ValidationIssue('ERROR', 'CHEMISTRY_PATH_MISSING', 'files.chemistry.manifest must be set.'))

    for name, p_str in required_files.items():
        p = Path(p_str)
        if not p.exists():
            messages.append(ValidationIssue('ERROR', 'FILE_NOT_FOUND', f'Required {name} path does not exist: {p}', name))

    messages.extend(_integrator_messages(run_config))

    if not 0.0 <= run_config.physics.gas_heating_fraction <= 1.0:
        messages.append(ValidationIssue('ERROR', 'GAS_HEATING_FRACTION_RANGE', 'physics.gas_heating_fraction must be between 0 and 1.', 'physics.gas_heating_fraction'))

    if run_config.physics.wall_relaxation_s_inv < 0.0:
        messages.append(ValidationIssue('ERROR', 'WALL_RELAXATION_RANGE', 'physics.wall_relaxation_s_inv must be non-negative.', 'physics.wall_relaxation_s_inv'))

    electron_closure = str(getattr(run_config.physics, 'electron_density_closure', 'quasi_neutral') or 'quasi_neutral').lower()
    allowed_electron_closures = {'quasi_neutral', 'prescribed_profile'}
    if electron_closure not in allowed_electron_closures:
        messages.append(
            ValidationIssue(
                'ERROR',
                'ELECTRON_DENSITY_CLOSURE_UNRECOGNIZED',
                f'physics.electron_density_closure must be one of {sorted(allowed_electron_closures)}, got {electron_closure!r}.',
                'physics.electron_density_closure',
            )
        )
    elif electron_closure == 'prescribed_profile':
        try:
            zone_columns = run_config.swarm.prescribed_electron_profile.zone_columns
            validation_zones = [str(zone_id) for zone_id in zone_columns] if zone_columns else ['*']
            build_prescribed_electron_profile(run_config, resolved_paths, validation_zones)
        except Exception as exc:
            messages.append(
                ValidationIssue(
                    'ERROR',
                    'PRESCRIBED_ELECTRON_PROFILE_INVALID',
                    f'prescribed electron density profile is missing or invalid: {exc}',
                    'swarm.prescribed_electron_profile',
                )
            )

    messages.extend(_eedf_backend_messages(run_config))

    if run_config.case.schema_version != 2:
        messages.append(ValidationIssue('ERROR', 'UNSUPPORTED_SCHEMA_VERSION', 'Only case schema_version 2 is supported.', 'schema'))

    return ValidationReport(messages=messages)


def _merged_power_port_configs(chamber: Any, recipe: Any) -> list[tuple[Any, str, dict[str, Any]]]:
    merged: list[tuple[Any, str, dict[str, Any]]] = []
    for step in recipe.steps:
        for port_id, step_cfg in step.power_ports.items():
            port = chamber.power_port_by_id.get(port_id)
            if port is None:
                continue
            cfg = dict(port.parameters or {})
            cfg.update(step_cfg or {})
            cfg.setdefault('kind', getattr(port, 'kind', ''))
            merged.append((step, port_id, cfg))
    return merged


def validate_loaded_inputs(run_config: RunConfig, resolved_paths: ResolvedPaths, chamber: Any, recipe: Any) -> ValidationReport:
    messages: list[ValidationIssue] = []
    messages.extend(_reactor_structure_messages(chamber, recipe))
    messages.extend(_surface_ion_loss_messages(chamber))
    port_configs = _merged_power_port_configs(chamber, recipe)
    backend = str(run_config.physics.electrical_backend)
    spec = ELECTRICAL_REGISTRY.entries.get(backend)
    if spec is None:
        messages.append(
            ValidationIssue(
                'ERROR',
                'ELECTRICAL_BACKEND_UNRECOGNIZED',
                f'physics.electrical_backend must be one of {ELECTRICAL_REGISTRY.names()}, got {backend!r}.',
                'physics.electrical_backend',
            )
        )
        return ValidationReport(messages=messages)
    validator = spec.config_validator
    if validator is None:
        return ValidationReport(messages=messages)
    for step, port_id, cfg in port_configs:
        try:
            validator(step, port_id, cfg, resolved_paths)
        except Exception as exc:
            messages.append(
                ValidationIssue(
                    'ERROR',
                    f'{backend.upper()}_CONFIG_INVALID',
                    f'{backend} port {port_id} in step {step.step_id} is missing or invalid: {exc}',
                    port_id,
                )
            )
    return ValidationReport(messages=messages)
