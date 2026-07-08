from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import yaml


GRID_COLUMNS = ('mean_energy_eV', 'EoverN_Td', 'effective_field_Td')


def _read_csv(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with path.open('r', encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'{path} has no data rows')
    columns = list(rows[0])
    data = {col: np.asarray([float(row[col]) for row in rows], dtype=float) for col in columns}
    return columns, data


def _first_column(columns: list[str], choices: tuple[str, ...]) -> str | None:
    for col in choices:
        if col in columns:
            return col
    return None


def build_rate_table_h5(table_dir: Path, output_path: Path) -> Path:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError('h5py is required to write rate table HDF5 files.') from exc

    rates_path = table_dir / 'rates.csv'
    transport_path = table_dir / 'transport.csv'
    metadata_path = table_dir / 'metadata.yaml'
    rate_columns, rates = _read_csv(rates_path)
    transport_columns, transport = _read_csv(transport_path)
    grid_col = _first_column(rate_columns, GRID_COLUMNS)
    if grid_col is None:
        raise ValueError(f'{rates_path} needs one grid column from {GRID_COLUMNS}')
    if grid_col not in transport:
        raise ValueError(f'{transport_path} must include the same grid column {grid_col!r}')

    raw_grid = rates[grid_col]
    order = np.argsort(raw_grid)
    grid = raw_grid[order]
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = yaml.safe_load(metadata_path.read_text(encoding='utf-8')) or {}

    mean_energy = transport.get('mean_energy_eV')
    effective_field = transport.get('effective_field_Td')
    if effective_field is None:
        effective_field = transport.get('EoverN_Td')
    if grid_col == 'mean_energy_eV':
        mean_energy = raw_grid
    if grid_col in {'EoverN_Td', 'effective_field_Td'}:
        effective_field = raw_grid
    if mean_energy is None:
        raise ValueError('transport.csv needs mean_energy_eV unless rates.csv is gridded by mean_energy_eV')
    if effective_field is None:
        raise ValueError('transport.csv needs effective_field_Td or EoverN_Td unless rates.csv is field-gridded')

    mobility = transport.get('mobility_m2_V_s')
    diffusion = transport.get('diffusion_m2_s')
    if mobility is None:
        mobility = np.zeros_like(grid)
    if diffusion is None:
        diffusion = np.zeros_like(grid)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, 'w') as h5:
        h5.create_dataset('mean_energy_eV', data=np.asarray(mean_energy, dtype=float)[order])
        h5.create_dataset('effective_field_Td', data=np.asarray(effective_field, dtype=float)[order])
        h5.create_dataset('mobility_m2_V_s', data=np.asarray(mobility, dtype=float)[order])
        h5.create_dataset('diffusion_m2_s', data=np.asarray(diffusion, dtype=float)[order])
        group = h5.create_group('rate_coefficients')
        for col in rate_columns:
            if col == grid_col:
                continue
            group.create_dataset(col, data=np.asarray(rates[col], dtype=float)[order])
        h5.attrs['source_directory'] = str(table_dir)
        h5.attrs['grid_column'] = grid_col
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                h5.attrs[str(key)] = value
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description='Build a table-model HDF5 file from external BOLSIG/rate-table CSV files.')
    parser.add_argument('table_dir', type=Path, help='Directory containing rates.csv, transport.csv, and optional metadata.yaml.')
    parser.add_argument('--output', type=Path, required=True, help='Output HDF5 path.')
    args = parser.parse_args()
    print(build_rate_table_h5(args.table_dir.resolve(), args.output.resolve()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
