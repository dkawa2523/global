from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from plasma_global.electrical.base import ElectricalBackend, ElectricalPortSnapshot, PowerRequest, PowerResult
from plasma_global.io.time_series import TimeSeriesTable, read_time_series_csv


POWER_COLUMNS = ('absorbed_power_W',)
VOLTAGE_COLUMNS = ('voltage_V',)
CURRENT_COLUMNS = ('current_A',)
REDUCED_FIELD_COLUMNS = ('reduced_field_Td',)


CircuitTable = TimeSeriesTable


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


def _first_existing(names: tuple[str, ...], table: CircuitTable) -> str | None:
    for name in names:
        if name in table.columns:
            return name
    return None


def resolve_circuit_table_columns(table: CircuitTable, cfg: dict[str, Any]) -> CircuitTableColumns:
    columns = CircuitTableColumns(
        power=_first_existing(POWER_COLUMNS, table),
        voltage=_first_existing(VOLTAGE_COLUMNS, table),
        current=_first_existing(CURRENT_COLUMNS, table),
        reduced_field=_first_existing(REDUCED_FIELD_COLUMNS, table),
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


def resolve_circuit_table_path(resolved_paths: Any, cfg: dict[str, Any]) -> Path:
    raw_file = cfg.get('file')
    if raw_file:
        path = Path(str(raw_file))
        if not path.is_absolute():
            base_dir = Path(getattr(resolved_paths, 'base_dir', '.'))
            path = base_dir / path
        return path.resolve()

    file_key = str(cfg.get('file_key') or 'circuit_result_csv')
    external_inputs = dict(getattr(resolved_paths, 'external_inputs', {}) or {})
    if file_key not in external_inputs or not external_inputs[file_key]:
        keys = ', '.join(sorted(external_inputs)) or '<none>'
        raise ValueError(f'External circuit table key {file_key!r} is not defined in files.external_inputs. Available keys: {keys}')
    return Path(str(external_inputs[file_key])).resolve()


def read_circuit_table(path: str | Path) -> CircuitTable:
    return read_time_series_csv(path, label='External circuit table')


def validate_circuit_table_columns(table: CircuitTable, cfg: dict[str, Any]) -> None:
    resolve_circuit_table_columns(table, cfg)


class ExternalCircuitTableBackend(ElectricalBackend):
    """One-way loose coupling from measured or SPICE-generated circuit CSV data."""

    def prepare(self, chamber: Any, recipe: Any, run_config: Any, resolved_paths: Any) -> None:
        super().prepare(chamber=chamber, recipe=recipe, run_config=run_config, resolved_paths=resolved_paths)
        self._tables: dict[Path, CircuitTable] = {}

    def _table_for(self, cfg: dict[str, Any]) -> CircuitTable:
        path = resolve_circuit_table_path(self.resolved_paths, cfg)
        if path not in self._tables:
            self._tables[path] = read_circuit_table(path)
        return self._tables[path]

    def _value(self, table: CircuitTable, column: str, time_s: float, cfg: dict[str, Any]) -> float:
        hold = str(cfg.get('hold', 'edge')).lower()
        interpolation = str(cfg.get('interpolation', 'linear')).lower()
        try:
            return table.value(column, time_s, interpolation=interpolation, hold=hold)
        except ValueError as exc:
            if 'outside time-series range' in str(exc):
                raise ValueError(f'time_s={time_s} is outside external circuit table range for {table.path}') from exc
            raise

    def evaluate(self, request: PowerRequest) -> PowerResult:
        p_zone: dict[str, float] = {z.zone_id: 0.0 for z in self.chamber.zones}
        p_port: dict[str, float] = {}
        port_observables: dict[str, ElectricalPortSnapshot] = {}
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
            if columns.reduced_field:
                reduced_field = max(self._value(table, columns.reduced_field, request.time_s, cfg), 0.0)
            elif 'gap_m' in cfg and columns.voltage:
                total_density = None
                if 'total_density_m3' in cfg:
                    total_density = float(cfg['total_density_m3'])
                else:
                    state = request.zone_state.get(zone_id)
                    if state is not None:
                        total_density = float(state.total_density_m3)
                if total_density is not None:
                    electric_field = abs(voltage) / max(float(cfg['gap_m']), 1.0e-30)
                    reduced_field = electric_field / max(total_density, 1.0e-30) / 1.0e-21

            p_zone[zone_id] = p_zone.get(zone_id, 0.0) + absorbed
            p_port[port_id] = absorbed
            port_observables[port_id] = ElectricalPortSnapshot({
                'source_voltage_V': float(voltage),
                'gap_voltage_V': float(voltage),
                'current_A': float(current),
                'absorbed_power_W': float(absorbed),
                'delivered_power_W': float(absorbed),
                'reduced_field_Td': float(reduced_field),
            })
            zone_reduced_field[zone_id] = max(zone_reduced_field.get(zone_id, 0.0), reduced_field)
            plasma_potential = max(plasma_potential, float(cfg.get('plasma_potential_V', 0.0)))

        return PowerResult(
            absorbed_power_W_by_zone=p_zone,
            port_power_W=p_port,
            port_observables=port_observables,
            self_bias_V=0.0,
            plasma_potential_V=plasma_potential,
            zone_reduced_field_Td=zone_reduced_field,
        )
