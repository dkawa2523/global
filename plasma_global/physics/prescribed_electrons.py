from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from plasma_global.config.models import PrescribedElectronProfileConfig, ResolvedPaths, RunConfig
from plasma_global.io.time_series import TimeSeriesTable, read_time_series_csv


DEFAULT_DENSITY_COLUMN = 'electron_density_m3'


@dataclass
class PrescribedElectronProfile:
    table: TimeSeriesTable
    zone_columns: dict[str, str]
    interpolation: str = 'linear'
    hold: str = 'edge'

    @property
    def path(self) -> Path:
        return self.table.path

    @property
    def time_s(self):
        return self.table.time_s

    @property
    def columns_m3(self):
        return self.table.columns

    def _value(self, column: str, time_s: float) -> float:
        try:
            return self.table.value(column, time_s, interpolation=self.interpolation, hold=self.hold)
        except ValueError as exc:
            if 'outside time-series range' in str(exc):
                raise ValueError(f'time_s={time_s} is outside prescribed electron profile range for {self.path}') from exc
            raise

    def density_by_zone(self, time_s: float, zone_ids: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for zone_id in zone_ids:
            column = self.zone_columns.get(zone_id) or self.zone_columns.get('*')
            if column is None:
                raise ValueError(f'No prescribed electron-density column configured for zone {zone_id!r}')
            out[zone_id] = max(self._value(column, time_s), 0.0)
        return out


def _resolve_profile_path(resolved_paths: ResolvedPaths, cfg: PrescribedElectronProfileConfig) -> Path:
    if cfg.file:
        path = Path(str(cfg.file))
        if not path.is_absolute():
            base_dir = Path(resolved_paths.base_dir)
            path = base_dir / path
        return path.resolve()
    if not cfg.file_key:
        raise ValueError('prescribed electron profile requires file or file_key')
    file_key = str(cfg.file_key)
    external_inputs = dict(resolved_paths.external_inputs or {})
    if file_key not in external_inputs or not external_inputs[file_key]:
        keys = ', '.join(sorted(external_inputs)) or '<none>'
        raise ValueError(f'Prescribed electron profile key {file_key!r} is not defined in files.external_inputs. Available keys: {keys}')
    return Path(str(external_inputs[file_key])).resolve()


def read_prescribed_electron_profile(path: str | Path, cfg: PrescribedElectronProfileConfig, zone_ids: list[str]) -> PrescribedElectronProfile:
    table = read_time_series_csv(path, label='Prescribed electron profile', drop_empty_columns=False)

    zone_columns = {str(zone): str(column) for zone, column in cfg.zone_columns.items()}
    if not zone_columns:
        zone_columns['*'] = DEFAULT_DENSITY_COLUMN
    for zone_id in zone_ids:
        column = zone_columns.get(zone_id) or zone_columns.get('*')
        if column is None or column not in table.columns:
            raise ValueError(f'No valid prescribed electron-density column for zone {zone_id!r} in {table.path}')

    return PrescribedElectronProfile(
        table=table,
        zone_columns=zone_columns,
        interpolation=str(cfg.interpolation).lower(),
        hold=str(cfg.hold).lower(),
    )


def build_prescribed_electron_profile(
    run_config: RunConfig,
    resolved_paths: ResolvedPaths,
    zone_ids: list[str],
) -> PrescribedElectronProfile | None:
    closure = str(run_config.physics.electron_density_closure or 'quasi_neutral').lower()
    if closure != 'prescribed_profile':
        return None
    cfg = run_config.swarm.prescribed_electron_profile
    path = _resolve_profile_path(resolved_paths, cfg)
    return read_prescribed_electron_profile(path, cfg, zone_ids)
