from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_chemistry_manifest(path: Path) -> dict[str, Any]:
    raw = _load_yaml(path)
    allowed = {
        'species_file',
        'gas_reactions_file',
        'surface_reactions_file',
        'cross_sections_manifest',
        'model_files',
        'extensions',
    }
    unsupported = sorted(str(k) for k in raw if str(k) not in allowed)
    if unsupported:
        raise ValueError(f'Unsupported chemistry manifest keys: {", ".join(unsupported)}. Use model_files instead.')

    required = ['species_file', 'gas_reactions_file', 'surface_reactions_file']
    missing = [key for key in required if not raw.get(key)]
    if missing:
        raise ValueError(f'Chemistry manifest missing required keys: {", ".join(missing)}')

    extensions = raw.get('extensions', {}) or {}
    if not isinstance(extensions, dict):
        raise ValueError('chemistry manifest extensions must be a mapping')
    allowed_extensions = {'state_variables', 'processes'}
    bad_extensions = sorted(str(k) for k in extensions if str(k) not in allowed_extensions)
    if bad_extensions:
        raise ValueError(f'Unsupported chemistry extension keys: {", ".join(bad_extensions)}')

    return {
        'species': _resolve(path.parent, raw.get('species_file')),
        'gas_reactions': _resolve(path.parent, raw.get('gas_reactions_file')),
        'surface_reactions': _resolve(path.parent, raw.get('surface_reactions_file')),
        'model_files': {
            str(k): _resolve(path.parent, v)
            for k, v in (raw.get('model_files', {}) or {}).items()
        },
        'cross_sections_manifest': _resolve(path.parent, raw.get('cross_sections_manifest')),
        'state_variables': _resolve(path.parent, extensions.get('state_variables')),
        'processes': _resolve(path.parent, extensions.get('processes')),
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as fh:
        return yaml.safe_load(fh) or {}


def _resolve(base_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path
