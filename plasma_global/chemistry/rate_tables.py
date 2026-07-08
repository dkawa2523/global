from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

BOUNDS_POLICIES = {'clip', 'error'}
INTERPOLATIONS = {'linear'}
TABULATED_1D_AXES = {'gas_temperature_K', 'mean_energy_eV', 'reduced_field_Td', 'pressure_Pa'}


def _resolve(base_dir: Path, value: str | Path | None) -> Path:
    if value is None or not str(value).strip():
        raise ValueError('tabulated rate model requires file')
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _read_numeric_columns(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f'Tabulated rate file not found: {path}')
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        if not columns:
            raise ValueError(f'Tabulated rate file has no header: {path}')
        rows = list(reader)
    if not rows:
        raise ValueError(f'Tabulated rate file has no data rows: {path}')
    return columns, rows


def _column(rows: list[dict[str, str]], path: Path, column: str, columns: list[str]) -> np.ndarray:
    if column not in columns:
        available = ', '.join(columns)
        raise ValueError(f'Tabulated rate file {path} is missing column {column!r}. Available columns: {available}')
    try:
        values = np.asarray([float(row[column]) for row in rows], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Tabulated rate file {path} column {column!r} must be numeric') from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f'Tabulated rate file {path} column {column!r} contains non-finite values')
    return values


def _validate_table(path: Path, axis: np.ndarray, values: np.ndarray, *, value_label: str) -> None:
    if axis.size < 2:
        raise ValueError(f'Tabulated rate file {path} needs at least two rows')
    if np.any(axis[1:] <= axis[:-1]):
        raise ValueError(f'Tabulated rate file {path} axis must be strictly increasing')
    if np.any(values < 0.0):
        raise ValueError(f'Tabulated rate file {path} contains negative {value_label} values')


def _validate_options(model: dict[str, Any], model_key: str, *, require_axis: bool) -> None:
    bounds = str(model.get('bounds_policy', 'clip') or 'clip').lower()
    if bounds not in BOUNDS_POLICIES:
        raise ValueError(f'Rate model {model_key} bounds_policy must be one of {sorted(BOUNDS_POLICIES)}, got {bounds!r}')
    interpolation = str(model.get('interpolation', 'linear') or 'linear').lower()
    if interpolation not in INTERPOLATIONS:
        raise ValueError(f'Rate model {model_key} interpolation must be one of {sorted(INTERPOLATIONS)}, got {interpolation!r}')
    if require_axis:
        axis_name = str(model.get('x', '')).strip()
        if axis_name not in TABULATED_1D_AXES:
            raise ValueError(f'Rate model {model_key} x must be one of {sorted(TABULATED_1D_AXES)}, got {axis_name!r}')


def attach_tabulated_1d(model_key: str, model: dict[str, Any], base_dir: Path) -> None:
    _validate_options(model, model_key, require_axis=True)
    path = _resolve(base_dir, model.get('file'))
    columns, rows = _read_numeric_columns(path)
    x_column = str(model.get('x_column', '')).strip()
    value_column = str(model.get('value_column', '')).strip()
    axis = _column(rows, path, x_column, columns)
    values = _column(rows, path, value_column, columns)
    _validate_table(path, axis, values, value_label='rate')
    model['_table_1d'] = {'x': axis, 'values': values, 'path': str(path)}


def attach_ion_yield_table(model_key: str, model: dict[str, Any], base_dir: Path) -> None:
    _validate_options(model, model_key, require_axis=False)
    path = _resolve(base_dir, model.get('file'))
    columns, rows = _read_numeric_columns(path)
    energy_column = str(model.get('energy_column', '')).strip()
    yield_column = str(model.get('yield_column', '')).strip()
    axis = _column(rows, path, energy_column, columns)
    values = _column(rows, path, yield_column, columns)
    _validate_table(path, axis, values, value_label='yield')
    model['_table_1d'] = {'x': axis, 'values': values, 'path': str(path)}


def lookup_table_1d(model: dict[str, Any], value: float) -> float:
    table = model.get('_table_1d')
    if not isinstance(table, dict):
        raise ValueError(f'Rate model {model.get("backend", "")!r} was not prepared with a 1D table')
    axis = np.asarray(table['x'], dtype=float)
    values = np.asarray(table['values'], dtype=float)
    x0 = float(value)
    lower = float(axis[0])
    upper = float(axis[-1])
    outside = x0 < lower or x0 > upper
    if outside and str(model.get('bounds_policy', 'clip')).lower() == 'error':
        raise ValueError(
            f'1D table lookup outside bounds for {table.get("path")}: value={x0}, axis_min={lower}, axis_max={upper}'
        )
    return float(np.interp(float(np.clip(x0, lower, upper)), axis, values))
