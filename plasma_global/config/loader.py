from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from plasma_global.config.models import (
    Boltzmann2TermConfig,
    CaseMetadata,
    ChemistryFilesConfig,
    FilesConfig,
    NumericsConfig,
    OutputFormatsConfig,
    OutputPlotsConfig,
    OutputsConfig,
    PhysicsConfig,
    PrescribedElectronProfileConfig,
    ResolvedPaths,
    RunConfig,
    RuntimeConfig,
    SwarmCacheConfig,
    SwarmConfig,
    SwarmEnergyGridConfig,
    SwarmReducedFieldGridConfig,
    SwarmTableConfig,
)


def _expand_env(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        out = {k: v for k, v in base.items()}
        for key, value in override.items():
            if key in out:
                out[key] = _deep_merge(out[key], value)
            else:
                out[key] = value
        return out
    return override


def _load_yaml_with_includes(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    if seen is None:
        seen = set()
    if path in seen:
        raise ValueError(f'Include cycle detected while loading {path}')
    seen.add(path)
    with path.open('r', encoding='utf-8') as fh:
        raw = yaml.safe_load(fh) or {}
    raw = _expand_env(raw)
    includes = raw.pop('include', []) or raw.pop('includes', []) or []
    if isinstance(includes, (str, Path)):
        includes = [includes]
    merged: dict[str, Any] = {}
    for inc in includes:
        inc_path = Path(inc)
        if not inc_path.is_absolute():
            inc_path = (path.parent / inc_path).resolve()
        merged = _deep_merge(merged, _load_yaml_with_includes(inc_path, seen=set(seen)))
    merged = _deep_merge(merged, raw)
    return merged


def _normalize_raw(raw: dict[str, Any]) -> dict[str, Any]:
    if 'files' in raw or 'physics' in raw or 'case' in raw:
        allowed = {'case', 'files', 'runtime', 'physics', 'numerics', 'outputs', 'swarm', 'schema_version', 'kind'}
        unknown = sorted(str(k) for k in raw if str(k) not in allowed)
        if unknown:
            raise ValueError(f'Unsupported configuration keys: {", ".join(unknown)}')
        case = raw.get('case', {})
        case.setdefault('schema_version', int(raw.get('schema_version', 2)))
        case.setdefault('kind', raw.get('kind', 'plasma_global_case'))
        return {
            'case': case,
            'files': raw.get('files', {}),
            'runtime': _normalize_runtime(raw.get('runtime', {})),
            'physics': raw.get('physics', {}),
            'numerics': raw.get('numerics', {}),
            'outputs': raw.get('outputs', {}),
            'swarm': raw.get('swarm', {}),
        }
    raise ValueError('Unsupported configuration schema. Use schema_version 2 with case/files/physics sections.')


def _normalize_runtime(raw_runtime: dict[str, Any] | None) -> dict[str, Any]:
    return dict(raw_runtime or {})


def _reject_unknown(raw: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(str(k) for k in raw if str(k) not in allowed)
    if unknown:
        raise ValueError(f'Unsupported {section} keys: {", ".join(unknown)}')


def _output_formats(raw: dict[str, Any] | None) -> OutputFormatsConfig:
    data = dict(raw or {})
    _reject_unknown(data, {'solution_h5', 'observables_csv', 'summary_yaml'}, 'outputs.formats')
    return OutputFormatsConfig(**data)


def _output_plots(raw: dict[str, Any] | None) -> OutputPlotsConfig:
    data = dict(raw or {})
    _reject_unknown(data, {'enabled', 'format', 'dpi', 'items'}, 'outputs.plots')
    if isinstance(data.get('format'), str):
        data['format'] = [data['format']]
    return OutputPlotsConfig(**data)


def _swarm_config(raw: dict[str, Any] | None) -> SwarmConfig:
    data = dict(raw or {})
    _reject_unknown(
        data,
        {'model_name', 'closure', 'mixture_key_species', 'cache', 'boltzmann_2term', 'table', 'prescribed_electron_profile'},
        'swarm',
    )
    cache = dict(data.pop('cache', {}) or {})
    _reject_unknown(cache, {'max_entries', 'fraction_decimals'}, 'swarm.cache')

    boltzmann = dict(data.pop('boltzmann_2term', {}) or {})
    _reject_unknown(
        boltzmann,
        {'energy_grid', 'reduced_field_grid_Td', 'max_shape_iterations', 'max_field_iterations'},
        'swarm.boltzmann_2term',
    )
    energy_grid = dict(boltzmann.pop('energy_grid', {}) or {})
    _reject_unknown(energy_grid, {'min_eV', 'max_eV', 'n'}, 'swarm.boltzmann_2term.energy_grid')
    field_grid = dict(boltzmann.pop('reduced_field_grid_Td', {}) or {})
    _reject_unknown(field_grid, {'min', 'max', 'n'}, 'swarm.boltzmann_2term.reduced_field_grid_Td')

    table = dict(data.pop('table', {}) or {})
    _reject_unknown(table, {'file', 'lookup', 'electron_energy_mode', 'energy_relaxation_time_s'}, 'swarm.table')

    profile = dict(data.pop('prescribed_electron_profile', {}) or {})
    _reject_unknown(profile, {'file', 'file_key', 'zone_columns', 'interpolation', 'hold'}, 'swarm.prescribed_electron_profile')
    if profile.get('zone_columns') is None:
        profile['zone_columns'] = {}

    return SwarmConfig(
        cache=SwarmCacheConfig(**cache),
        boltzmann_2term=Boltzmann2TermConfig(
            energy_grid=SwarmEnergyGridConfig(**energy_grid),
            reduced_field_grid_Td=SwarmReducedFieldGridConfig(**field_grid),
            **boltzmann,
        ),
        table=SwarmTableConfig(**table),
        prescribed_electron_profile=PrescribedElectronProfileConfig(**profile),
        **data,
    )


def load_run_config(path: str | Path) -> RunConfig:
    path = Path(path).resolve()
    raw = _load_yaml_with_includes(path)
    norm = _normalize_raw(raw)
    outputs_raw = norm.get('outputs', {}) or {}
    _reject_unknown(outputs_raw, {'formats', 'plots'}, 'outputs')
    formats_raw = outputs_raw.get('formats', {}) or {}
    chemistry_raw = norm['files'].get('chemistry', {}) or {}
    _reject_unknown(chemistry_raw, {'manifest'}, 'files.chemistry')
    if not chemistry_raw.get('manifest'):
        raise ValueError('files.chemistry.manifest is required')
    return RunConfig(
        case=CaseMetadata(**norm.get('case', {})),
        files=FilesConfig(
            chamber=norm['files']['chamber'],
            recipe=norm['files']['recipe'],
            chemistry=ChemistryFilesConfig(**chemistry_raw),
            output_dir=norm['files'].get('output_dir', './outputs'),
            external_inputs=norm['files'].get('external_inputs', {}) or {},
        ),
        runtime=RuntimeConfig(**norm.get('runtime', {})),
        physics=PhysicsConfig(**norm.get('physics', {})),
        numerics=NumericsConfig(**norm.get('numerics', {})),
        outputs=OutputsConfig(
            formats=_output_formats(formats_raw),
            plots=_output_plots(outputs_raw.get('plots', {})),
        ),
        swarm=_swarm_config(norm.get('swarm', {})),
    )


def resolve_run_paths(run_config: RunConfig, source_path: str | Path) -> ResolvedPaths:
    source_path = Path(source_path).resolve()
    base_dir = source_path.parent

    def _resolve(opt: str | None) -> str | None:
        if not opt:
            return None
        p = Path(opt)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        return str(p)

    chemistry_manifest = _resolve(run_config.files.chemistry.manifest)
    chemistry_dir = str(Path(chemistry_manifest).parent) if chemistry_manifest is not None else None

    return ResolvedPaths(
        source_config=str(source_path),
        base_dir=str(base_dir),
        chamber_file=_resolve(run_config.files.chamber) or '',
        recipe_file=_resolve(run_config.files.recipe) or '',
        chemistry_manifest=chemistry_manifest,
        chemistry_dir=chemistry_dir,
        output_dir=_resolve(run_config.files.output_dir) or '',
        external_inputs={k: _resolve(v) for k, v in (run_config.files.external_inputs or {}).items()},
    )
