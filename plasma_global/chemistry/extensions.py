from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from plasma_global.chemistry.models import ProcessSpec, StateVariableSpec
from plasma_global.chemistry.parser import parse_csv_bool, parse_pipe_list


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f'Extension file must contain a mapping: {path}')
    return data


def _entry_items(raw: Any, *, key_name: str) -> list[tuple[str, dict[str, Any]]]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [(str(key), dict(value or {})) for key, value in raw.items()]
    if isinstance(raw, list):
        items: list[tuple[str, dict[str, Any]]] = []
        for entry in raw:
            if not isinstance(entry, dict):
                raise ValueError(f'{key_name} entries must be mappings')
            data = dict(entry)
            id_key = 'state_id' if key_name == 'state_variables' else 'process_id'
            entry_id = str(data.pop('id', data.pop(id_key, ''))).strip()
            if not entry_id:
                raise ValueError(f'{key_name} list entries require id')
            items.append((entry_id, data))
        return items
    raise ValueError(f'{key_name} must be a mapping or list')


def _float_or_none(value: Any, default: float | None) -> float | None:
    if value is None:
        return None
    if value == '':
        return default
    return float(value)


def _list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return parse_pipe_list(value)
    return [str(item) for item in value]


def load_state_variables(path: str | Path | None) -> list[StateVariableSpec]:
    if path is None:
        return []
    raw = _load_yaml(Path(path))
    unsupported = sorted(str(k) for k in raw if str(k) != 'state_variables')
    if unsupported:
        raise ValueError(f'Unsupported state variable extension keys: {", ".join(unsupported)}')
    specs: list[StateVariableSpec] = []
    for state_id, data in _entry_items(raw.get('state_variables'), key_name='state_variables'):
        specs.append(
            StateVariableSpec(
                state_id=state_id,
                scope=str(data.get('scope', '')).lower(),
                unit=str(data.get('unit', '') or ''),
                initial=float(data.get('initial', 0.0) or 0.0),
                lower_bound=_float_or_none(data.get('lower_bound', 0.0), 0.0),
                scale=float(data.get('scale', 1.0) or 1.0),
                output=parse_csv_bool(data.get('output', True)),
                zones=_list(data.get('zones')),
                surfaces=_list(data.get('surfaces')),
            )
        )
    return specs


def load_processes(path: str | Path | None) -> list[ProcessSpec]:
    if path is None:
        return []
    raw = _load_yaml(Path(path))
    unsupported = sorted(str(k) for k in raw if str(k) != 'processes')
    if unsupported:
        raise ValueError(f'Unsupported process extension keys: {", ".join(unsupported)}')
    specs: list[ProcessSpec] = []
    for process_id, data in _entry_items(raw.get('processes'), key_name='processes'):
        parameters = dict(data)
        kind = str(parameters.pop('kind', '')).lower()
        target = str(parameters.pop('target', '')).strip()
        zones = _list(parameters.pop('zones', None))
        surfaces = _list(parameters.pop('surfaces', None))
        specs.append(
            ProcessSpec(
                process_id=process_id,
                kind=kind,
                target=target,
                parameters=parameters,
                zones=zones,
                surfaces=surfaces,
            )
        )
    return specs
