from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from plasma_global.chemistry.rate_model_schema import MODEL_FILE_BACKENDS
from plasma_global.chemistry.rate_tables import attach_ion_yield_table, attach_tabulated_1d


def load_rate_models_from_file(path: Path, category: str) -> dict[str, dict[str, Any]]:
    raw = _load_yaml(path)
    models = raw.get('rate_models', {}) or {}
    _validate_model_category(path, category, models)
    _attach_tabular_models(models, category, path.parent)
    return models


def merge_rate_models(target: dict[str, dict[str, Any]], incoming: dict[str, dict[str, Any]], source: Path) -> None:
    duplicates = sorted(set(target) & set(incoming))
    if duplicates:
        raise ValueError(f'Duplicate rate/energy model keys in {source}: {duplicates}')
    target.update(incoming)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as fh:
        return yaml.safe_load(fh) or {}


def _validate_model_category(path: Path, category: str, models: dict[str, dict[str, Any]]) -> None:
    allowed = MODEL_FILE_BACKENDS.get(category)
    if allowed is None:
        known = ', '.join(sorted(MODEL_FILE_BACKENDS))
        raise ValueError(f'unsupported chemistry model_files category {category!r}; expected one of {known}')
    bad = [
        (key, str(model.get('backend', '')).lower())
        for key, model in models.items()
        if str(model.get('backend', '')).lower() not in allowed
    ]
    if bad:
        formatted = ', '.join(f'{key}:{backend}' for key, backend in bad)
        allowed_s = ', '.join(sorted(allowed))
        raise ValueError(
            f'Model file {path} is declared as {category!r}, but contains unsupported backends '
            f'{formatted}. Allowed backends: {allowed_s}'
        )


def _attach_tabular_models(models: dict[str, dict[str, Any]], category: str, base_dir: Path) -> None:
    for model_key, model in models.items():
        backend = str(model.get('backend', '')).lower()
        if category == 'gas_rate' and backend == 'tabulated_1d':
            attach_tabulated_1d(str(model_key), model, base_dir)
        if category == 'surface_rate' and backend == 'ion_yield_table':
            attach_ion_yield_table(str(model_key), model, base_dir)
