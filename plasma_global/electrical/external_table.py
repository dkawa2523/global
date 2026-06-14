from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from plasma_global.electrical.base import ElectricalBackend, PowerRequest, PowerResult


POWER_COLUMNS = ('absorbed_power_W', 'power_W', 'plasma_power_W', 'P_abs_W')
VOLTAGE_COLUMNS = ('voltage_V', 'gap_voltage_V', 'plasma_voltage_V')
CURRENT_COLUMNS = ('current_A', 'plasma_current_A')
REDUCED_FIELD_COLUMNS = ('reduced_field_Td', 'EoverN_Td', 'E_over_N_Td')


@dataclass
class CircuitTable:
    path: Path
    time_s: np.ndarray
    columns: dict[str, np.ndarray]

    @property
    def names(self) -> set[str]:
        return set(self.columns)


@dataclass(frozen=True)
class CircuitTableColumns:
    power: str | None
    voltage: str | None
    current: str | None
    reduced_field: str | None

    @property
    def power_source(self) -> str:
        return 'column' if self.power else 'voltage_current_product'


def _available_columns(table: CircuitTable) -> str:
    return ', '.join(['time_s', *sorted(table.columns)]) or '<none>'


def _first_existing(names: tuple[str, ...], table: CircuitTable, cfg: dict[str, Any], cfg_key: str) -> str | None:
    configured = cfg.get(cfg_key)
    if configured:
        name = str(configured)
        if name not in table.columns:
            raise ValueError(
                f'Column {name!r} configured by {cfg_key} is not present in {table.path}. '
                f'Available columns: {_available_columns(table)}'
            )
        return name
    for name in names:
        if name in table.columns:
            return name
    return None


def resolve_circuit_table_columns(table: CircuitTable, cfg: dict[str, Any]) -> CircuitTableColumns:
    columns = CircuitTableColumns(
        power=_first_existing(POWER_COLUMNS, table, cfg, 'power_column'),
        voltage=_first_existing(VOLTAGE_COLUMNS, table, cfg, 'voltage_column'),
        current=_first_existing(CURRENT_COLUMNS, table, cfg, 'current_column'),
        reduced_field=_first_existing(REDUCED_FIELD_COLUMNS, table, cfg, 'reduced_field_column'),
    )
    if columns.power is None and (columns.voltage is None or columns.current is None):
        raise ValueError(
            'External circuit table needs either a power column '
            f'{POWER_COLUMNS} or both voltage and current columns '
            f'{VOLTAGE_COLUMNS} and {CURRENT_COLUMNS}. '
            f'Available columns: {_available_columns(table)}'
        )
    for name in (columns.power, columns.voltage, columns.current, columns.reduced_field):
        if name is not None and not np.all(np.isfinite(table.columns[name])):
            raise ValueError(f'External circuit table column {name!r} in {table.path} contains missing or non-finite values.')
    return columns


def resolve_circuit_table_path(run_config: Any, cfg: dict[str, Any]) -> Path:
    raw_file = cfg.get('file') or cfg.get('csv_file')
    if raw_file:
        path = Path(str(raw_file))
        if not path.is_absolute():
            resolved = getattr(run_config, '_resolved_paths', None)
            base_dir = Path(getattr(resolved, 'base_dir', '.'))
            path = base_dir / path
        return path.resolve()

    file_key = str(cfg.get('file_key') or cfg.get('external_input_key') or 'circuit_result_csv')
    external_inputs = dict(getattr(run_config.paths, 'external_inputs', {}) or {})
    if file_key not in external_inputs or not external_inputs[file_key]:
        keys = ', '.join(sorted(external_inputs)) or '<none>'
        raise ValueError(f'External circuit table key {file_key!r} is not defined in files.external_inputs. Available keys: {keys}')
    return Path(str(external_inputs[file_key])).resolve()


def read_circuit_table(path: str | Path) -> CircuitTable:
    table_path = Path(path).resolve()
    if not table_path.is_file():
        raise FileNotFoundError(f'External circuit table not found: {table_path}')

    with table_path.open('r', encoding='utf-8-sig', newline='') as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f'External circuit table has no header: {table_path}')
        if 'time_s' not in reader.fieldnames:
            raise ValueError(f'External circuit table requires a time_s column: {table_path}')

        times: list[float] = []
        values: dict[str, list[float]] = {name: [] for name in reader.fieldnames if name != 'time_s'}
        for row in reader:
            if row.get('time_s') in (None, ''):
                continue
            times.append(float(row['time_s']))
            for name in values:
                cell = row.get(name)
                values[name].append(float(cell) if cell not in (None, '') else math.nan)

    if len(times) < 1:
        raise ValueError(f'External circuit table contains no data rows: {table_path}')

    raw_time = np.asarray(times, dtype=float)
    if not np.all(np.isfinite(raw_time)):
        raise ValueError(f'External circuit table time_s column contains non-finite values: {table_path}')
    order = np.argsort(raw_time)
    time_arr = raw_time[order]
    if np.any(np.diff(time_arr) <= 0.0):
        raise ValueError(f'External circuit table time_s column must be strictly increasing after sorting: {table_path}')

    columns: dict[str, np.ndarray] = {}
    for name, vals in values.items():
        arr = np.asarray(vals, dtype=float)[order]
        if np.any(np.isfinite(arr)):
            columns[name] = arr
    return CircuitTable(path=table_path, time_s=time_arr, columns=columns)


def validate_circuit_table_columns(table: CircuitTable, cfg: dict[str, Any]) -> None:
    resolve_circuit_table_columns(table, cfg)


class ExternalCircuitTableBackend(ElectricalBackend):
    """One-way loose coupling from measured or SPICE-generated circuit CSV data."""

    def prepare(self, chamber: Any, recipe: Any, run_config: Any) -> None:
        super().prepare(chamber=chamber, recipe=recipe, run_config=run_config)
        self._tables: dict[Path, CircuitTable] = {}

    def _table_for(self, cfg: dict[str, Any]) -> CircuitTable:
        path = resolve_circuit_table_path(self.run_config, cfg)
        if path not in self._tables:
            self._tables[path] = read_circuit_table(path)
        return self._tables[path]

    def _value(self, table: CircuitTable, column: str, time_s: float, cfg: dict[str, Any]) -> float:
        arr = table.columns[column]
        hold = str(cfg.get('hold', 'edge')).lower()
        interpolation = str(cfg.get('interpolation', 'linear')).lower()
        if hold == 'error' and (time_s < table.time_s[0] or time_s > table.time_s[-1]):
            raise ValueError(f'time_s={time_s} is outside external circuit table range for {table.path}')
        if interpolation in {'previous', 'zoh', 'zero_order_hold'}:
            idx = int(np.searchsorted(table.time_s, time_s, side='right') - 1)
            idx = min(max(idx, 0), len(table.time_s) - 1)
            return float(arr[idx])
        return float(np.interp(time_s, table.time_s, arr))

    def evaluate(self, request: PowerRequest) -> PowerResult:
        p_zone: dict[str, float] = {z.zone_id: 0.0 for z in self.chamber.zones}
        p_port: dict[str, float] = {}
        port_details: dict[str, dict[str, float | str]] = {}
        table_sources: dict[str, dict[str, Any]] = {}
        zone_reduced_field: dict[str, float] = {z.zone_id: 0.0 for z in self.chamber.zones}
        plasma_potential = 0.0

        for port_id, step_cfg in request.recipe_step.power_ports.items():
            port = self.chamber.power_port_by_id[port_id]
            cfg = dict(port.parameters or {})
            cfg.update(step_cfg or {})
            zone_id = str(cfg.get('zone_id') or port.zone_id)
            table = self._table_for(cfg)
            columns = resolve_circuit_table_columns(table, cfg)

            voltage = self._value(table, columns.voltage, request.time_s, cfg) if columns.voltage else 0.0
            current = self._value(table, columns.current, request.time_s, cfg) if columns.current else 0.0
            if columns.power:
                absorbed = self._value(table, columns.power, request.time_s, cfg)
            else:
                absorbed = voltage * current

            absorbed *= float(cfg.get('power_scale', 1.0))
            voltage *= float(cfg.get('voltage_scale', 1.0))
            current *= float(cfg.get('current_scale', 1.0))
            absorbed = max(absorbed, 0.0)

            reduced_field = 0.0
            reduced_field_source = 'none'
            if columns.reduced_field:
                reduced_field = max(self._value(table, columns.reduced_field, request.time_s, cfg), 0.0)
                reduced_field_source = 'column'
            elif 'gap_m' in cfg and columns.voltage:
                total_density = None
                if 'total_density_m3' in cfg:
                    total_density = float(cfg['total_density_m3'])
                    reduced_field_source = 'voltage_gap_config_density'
                else:
                    zone_density = (request.metadata or {}).get('zone_total_density_m3', {})
                    if zone_id in zone_density:
                        total_density = float(zone_density[zone_id])
                        reduced_field_source = 'voltage_gap_runtime_density'
                if total_density is not None:
                    electric_field = abs(voltage) / max(float(cfg['gap_m']), 1.0e-30)
                    reduced_field = electric_field / max(total_density, 1.0e-30) / 1.0e-21
                else:
                    reduced_field_source = 'none'

            p_zone[zone_id] = p_zone.get(zone_id, 0.0) + absorbed
            p_port[port_id] = absorbed
            zone_reduced_field[zone_id] = max(zone_reduced_field.get(zone_id, 0.0), reduced_field)
            plasma_potential = max(plasma_potential, float(cfg.get('plasma_potential_V', 0.0)))
            port_details[port_id] = {
                'backend': 'external_circuit_table',
                'zone_id': zone_id,
                'file': str(table.path),
                'absorbed_power_W': absorbed,
                'voltage_V': voltage,
                'current_A': current,
                'reduced_field_Td': reduced_field,
            }
            table_sources[port_id] = {
                'file': str(table.path),
                'time_start_s': float(table.time_s[0]),
                'time_end_s': float(table.time_s[-1]),
                'n_rows': int(table.time_s.size),
                'power_column': columns.power,
                'voltage_column': columns.voltage,
                'current_column': columns.current,
                'reduced_field_column': columns.reduced_field,
                'reduced_field_source': reduced_field_source,
                'power_source': columns.power_source,
                'interpolation': str(cfg.get('interpolation', 'linear')).lower(),
                'hold': str(cfg.get('hold', 'edge')).lower(),
            }

        return PowerResult(
            absorbed_power_W_by_zone=p_zone,
            port_power_W=p_port,
            self_bias_V=0.0,
            plasma_potential_V=plasma_potential,
            metadata={
                'port_details': port_details,
                'surface_ied': {},
                'zone_reduced_field_Td': zone_reduced_field,
                'circuit_interface': {
                    'kind': 'external_table_one_way',
                    'model': 'external_circuit_table',
                    'version': 1,
                    'external_circuit_ready': True,
                    'table_sources': table_sources,
                },
            },
        )
