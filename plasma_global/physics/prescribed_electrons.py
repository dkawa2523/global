from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DENSITY_COLUMNS = ('electron_density_m3', 'ne_m3', 'electrons_m3')
DEFAULT_DENSITY_COLUMNS_CM3 = ('electron_density_cm3', 'ne_cm3', 'Electrons_cm-3')


@dataclass
class PrescribedElectronProfile:
    path: Path
    time_s: np.ndarray
    columns_m3: dict[str, np.ndarray]
    zone_columns: dict[str, str]
    interpolation: str = 'linear'
    hold: str = 'edge'

    def _value(self, column: str, time_s: float) -> float:
        values = self.columns_m3[column]
        if self.hold == 'error' and (time_s < self.time_s[0] or time_s > self.time_s[-1]):
            raise ValueError(f'time_s={time_s} is outside prescribed electron profile range for {self.path}')
        if self.interpolation in {'previous', 'zoh', 'zero_order_hold'}:
            idx = int(np.searchsorted(self.time_s, time_s, side='right') - 1)
            idx = min(max(idx, 0), len(self.time_s) - 1)
            return float(values[idx])
        return float(np.interp(time_s, self.time_s, values))

    def density_by_zone(self, time_s: float, zone_ids: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for zone_id in zone_ids:
            column = self.zone_columns.get(zone_id) or self.zone_columns.get('*')
            if column is None:
                raise ValueError(f'No prescribed electron-density column configured for zone {zone_id!r}')
            out[zone_id] = max(self._value(column, time_s), 0.0)
        return out

    def metadata(self) -> dict[str, Any]:
        return {
            'file': str(self.path),
            'time_s_min': float(self.time_s[0]),
            'time_s_max': float(self.time_s[-1]),
            'columns_m3': sorted(self.columns_m3),
            'zone_columns': dict(self.zone_columns),
            'interpolation': self.interpolation,
            'hold': self.hold,
        }


def _cfg_dict(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    return dict(vars(cfg))


def _resolve_profile_path(run_config: Any, cfg: dict[str, Any]) -> Path:
    raw_file = cfg.get('file') or cfg.get('csv_file')
    if raw_file:
        path = Path(str(raw_file))
        if not path.is_absolute():
            resolved = getattr(run_config, '_resolved_paths', None)
            base_dir = Path(getattr(resolved, 'base_dir', Path(getattr(run_config.paths, 'chamber_file', '.')).parent))
            path = base_dir / path
        return path.resolve()
    file_key = str(cfg.get('file_key') or cfg.get('external_input_key') or 'electron_profile_csv')
    external_inputs = dict(getattr(run_config.paths, 'external_inputs', {}) or {})
    if file_key not in external_inputs or not external_inputs[file_key]:
        keys = ', '.join(sorted(external_inputs)) or '<none>'
        raise ValueError(f'Prescribed electron profile key {file_key!r} is not defined in files.external_inputs. Available keys: {keys}')
    return Path(str(external_inputs[file_key])).resolve()


def read_prescribed_electron_profile(path: str | Path, cfg: dict[str, Any], zone_ids: list[str]) -> PrescribedElectronProfile:
    profile_path = Path(path).resolve()
    if not profile_path.is_file():
        raise FileNotFoundError(f'Prescribed electron profile not found: {profile_path}')
    with profile_path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or 'time_s' not in reader.fieldnames:
            raise ValueError(f'Prescribed electron profile requires a time_s column: {profile_path}')
        rows = [row for row in reader if row.get('time_s') not in (None, '')]
    if not rows:
        raise ValueError(f'Prescribed electron profile contains no data rows: {profile_path}')
    raw_time = np.asarray([float(row['time_s']) for row in rows], dtype=float)
    order = np.argsort(raw_time)
    time_s = raw_time[order]
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError(f'Prescribed electron profile time_s must be strictly increasing after sorting: {profile_path}')

    fieldnames = set(reader.fieldnames or [])
    columns_m3: dict[str, np.ndarray] = {}
    for name in fieldnames - {'time_s'}:
        values = np.asarray([float(row[name]) for row in rows], dtype=float)[order]
        if name in DEFAULT_DENSITY_COLUMNS_CM3 or str(cfg.get('unit', '')).lower() in {'cm-3', 'cm^-3', 'cm3'}:
            values = values * 1.0e6
        columns_m3[name] = values

    zone_columns_raw = _cfg_dict(cfg.get('zone_columns'))
    zone_columns = {str(zone): str(column) for zone, column in zone_columns_raw.items()}
    density_column = cfg.get('density_column')
    if density_column:
        zone_columns.setdefault('*', str(density_column))
    if not zone_columns:
        for candidate in DEFAULT_DENSITY_COLUMNS + DEFAULT_DENSITY_COLUMNS_CM3:
            if candidate in columns_m3:
                zone_columns['*'] = candidate
                break
    for zone_id in zone_ids:
        column = zone_columns.get(zone_id) or zone_columns.get('*')
        if column is None or column not in columns_m3:
            raise ValueError(f'No valid prescribed electron-density column for zone {zone_id!r} in {profile_path}')

    return PrescribedElectronProfile(
        path=profile_path,
        time_s=time_s,
        columns_m3=columns_m3,
        zone_columns=zone_columns,
        interpolation=str(cfg.get('interpolation', 'linear')).lower(),
        hold=str(cfg.get('hold', 'edge')).lower(),
    )


def build_prescribed_electron_profile(run_config: Any, zone_ids: list[str]) -> PrescribedElectronProfile | None:
    closure = str(getattr(run_config.physics, 'electron_density_closure', 'quasi_neutral') or 'quasi_neutral').lower()
    if closure not in {'prescribed_profile', 'external_profile', 'profile'}:
        return None
    cfg = _cfg_dict(getattr(getattr(run_config, 'swarm', None), 'prescribed_electron_profile', None))
    path = _resolve_profile_path(run_config, cfg)
    return read_prescribed_electron_profile(path, cfg, zone_ids)
