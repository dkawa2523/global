from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TimeSeriesTable:
    path: Path
    time_s: np.ndarray
    columns: dict[str, np.ndarray]

    @property
    def names(self) -> set[str]:
        return set(self.columns)

    def value(self, column: str, time_s: float, *, interpolation: str = 'linear', hold: str = 'edge') -> float:
        arr = self.columns[column]
        if hold == 'error' and (time_s < self.time_s[0] or time_s > self.time_s[-1]):
            raise ValueError(f'time_s={time_s} is outside time-series range for {self.path}')
        if interpolation in {'previous', 'zoh', 'zero_order_hold'}:
            idx = int(np.searchsorted(self.time_s, time_s, side='right') - 1)
            idx = min(max(idx, 0), len(self.time_s) - 1)
            return float(arr[idx])
        return float(np.interp(time_s, self.time_s, arr))


def read_time_series_csv(
    path: str | Path,
    *,
    label: str,
    drop_empty_columns: bool = True,
) -> TimeSeriesTable:
    table_path = Path(path).resolve()
    if not table_path.is_file():
        raise FileNotFoundError(f'{label} not found: {table_path}')

    with table_path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f'{label} has no header: {table_path}')
        if 'time_s' not in reader.fieldnames:
            raise ValueError(f'{label} requires a time_s column: {table_path}')

        times: list[float] = []
        values: dict[str, list[float]] = {name: [] for name in reader.fieldnames if name != 'time_s'}
        for row in reader:
            if row.get('time_s') in (None, ''):
                continue
            times.append(float(row['time_s']))
            for name in values:
                cell = row.get(name)
                values[name].append(float(cell) if cell not in (None, '') else math.nan)

    if not times:
        raise ValueError(f'{label} contains no data rows: {table_path}')

    raw_time = np.asarray(times, dtype=float)
    if not np.all(np.isfinite(raw_time)):
        raise ValueError(f'{label} time_s column contains non-finite values: {table_path}')
    order = np.argsort(raw_time)
    time_s = raw_time[order]
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError(f'{label} time_s column must be strictly increasing after sorting: {table_path}')

    columns: dict[str, np.ndarray] = {}
    for name, vals in values.items():
        arr = np.asarray(vals, dtype=float)[order]
        if not drop_empty_columns or np.any(np.isfinite(arr)):
            columns[name] = arr
    return TimeSeriesTable(path=table_path, time_s=time_s, columns=columns)
