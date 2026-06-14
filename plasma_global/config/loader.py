from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from plasma_global.config.models import (
    CaseMetadata,
    ChemistryFilesConfig,
    FilesConfig,
    LoggingConfig,
    NumericsConfig,
    OutputsConfig,
    PhysicsConfig,
    ResolvedPaths,
    RunConfig,
    RuntimeConfig,
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


def _ns(data: Any) -> Any:
    if isinstance(data, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in data.items()})
    if isinstance(data, list):
        return [_ns(v) for v in data]
    return data


def _normalize_raw(raw: dict[str, Any]) -> dict[str, Any]:
    if 'files' in raw or 'physics' in raw or 'case' in raw:
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
            'logging': raw.get('logging', {}),
            'swarm': raw.get('swarm', {}),
            'imports': raw.get('imports', {}),
        }
    raise ValueError('Unsupported configuration schema. Use schema_version 2 with case/files/physics sections.')


def _normalize_runtime(raw_runtime: dict[str, Any] | None) -> dict[str, Any]:
    runtime = dict(raw_runtime or {})
    runtime.pop('dry_run', None)
    runtime.pop('continue_from_checkpoint', None)
    return runtime


def load_run_config(path: str | Path) -> RunConfig:
    path = Path(path).resolve()
    raw = _load_yaml_with_includes(path)
    norm = _normalize_raw(raw)
    outputs_raw = norm.get('outputs', {}) or {}
    return RunConfig(
        case=CaseMetadata(**norm.get('case', {})),
        files=FilesConfig(
            chamber=norm['files']['chamber'],
            recipe=norm['files']['recipe'],
            chemistry=ChemistryFilesConfig(**(norm['files'].get('chemistry', {}) or {})),
            output_dir=norm['files'].get('output_dir', './outputs'),
            external_inputs=norm['files'].get('external_inputs', {}) or {},
        ),
        runtime=RuntimeConfig(**norm.get('runtime', {})),
        physics=PhysicsConfig(**norm.get('physics', {})),
        numerics=NumericsConfig(**norm.get('numerics', {})),
        outputs=OutputsConfig(
            formats=_ns(outputs_raw.get('formats', {})),
            plots=_ns(outputs_raw.get('plots', {})),
            save=_ns(outputs_raw.get('save', {})),
        ),
        logging=LoggingConfig(**norm.get('logging', {})),
        swarm=_ns(norm.get('swarm', {})),
        imports=norm.get('imports', {}) or {},
        raw=raw,
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
    chemistry_dir = _resolve(run_config.files.chemistry.directory)
    if chemistry_dir is None and chemistry_manifest is not None:
        chemistry_dir = str(Path(chemistry_manifest).parent)

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
