from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

GRID_COLUMNS = ("mean_energy_eV", "EoverN_Td", "effective_field_Td")


def convert_v2_rate_table_h5(
    source_path: Path,
    output_path: Path,
) -> tuple[Path, tuple[str, ...]]:
    """Convert the legacy electron table container to the strict v3 layout.

    The v2 ZDPlaskin fixture contains one transport diagnostic which the v3
    runtime deliberately does not consume.  This one-way importer recognizes
    only that named legacy dataset; every other unknown entry remains an error.
    Numerical axes and rate data are copied verbatim, without sorting, clipping,
    or filling invalid values.
    """

    source = Path(source_path).resolve()
    output = Path(output_path).resolve()
    required = {
        "mean_energy_eV",
        "effective_field_Td",
        "mobility_m2_V_s",
        "rate_coefficients",
    }
    recognized_legacy_only = {"diffusion_m2_s"}
    with h5py.File(source, "r") as source_h5:
        entries = set(source_h5)
        missing = sorted(required - entries)
        if missing:
            raise ValueError(
                f"{source} is missing legacy electron-table entries: "
                f"{', '.join(missing)}"
            )
        unknown = sorted(entries - required - recognized_legacy_only)
        if unknown:
            raise ValueError(
                f"{source} has unsupported legacy electron-table entries: "
                f"{', '.join(unknown)}"
            )
        rate_group = source_h5["rate_coefficients"]
        if not isinstance(rate_group, h5py.Group) or not rate_group:
            raise ValueError(f"{source} needs a non-empty rate_coefficients group")
        if any(not isinstance(rate_group[name], h5py.Dataset) for name in rate_group):
            raise ValueError(f"{source} rate_coefficients must contain only datasets")

        output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output, "w") as output_h5:
            output_h5.attrs["format"] = "plasma_global_electron_kinetics"
            output_h5.attrs["format_version"] = 1
            output_h5.attrs["migrated_from"] = str(source)
            for name in (
                "mean_energy_eV",
                "effective_field_Td",
                "mobility_m2_V_s",
            ):
                output_h5.create_dataset(name, data=source_h5[name][...])
            output_rates = output_h5.create_group("rate_coefficients")
            for name in rate_group:
                output_rates.create_dataset(name, data=rate_group[name][...])

    dropped = tuple(sorted(entries & recognized_legacy_only))
    return output, dropped


def _read_csv(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or ())
        rows = list(reader)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError(f"{path} needs a unique, non-empty CSV header")
    if not rows:
        raise ValueError(f"{path} has no data rows")
    try:
        data = {
            column: np.asarray([float(row[column]) for row in rows], dtype=float)
            for column in columns
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} contains a non-numeric cell") from exc
    if any(not np.all(np.isfinite(values)) for values in data.values()):
        raise ValueError(f"{path} contains a non-finite value")
    return columns, data


def _first_column(columns: list[str], choices: tuple[str, ...]) -> str | None:
    for col in choices:
        if col in columns:
            return col
    return None


def build_rate_table_h5(table_dir: Path, output_path: Path) -> Path:
    """Convert strict CSV inputs to the HDF5 read by v3 electron kinetics."""

    table_dir = Path(table_dir)
    output_path = Path(output_path)
    rates_path = table_dir / "rates.csv"
    transport_path = table_dir / "transport.csv"
    metadata_path = table_dir / "metadata.yaml"
    rate_columns, rates = _read_csv(rates_path)
    transport_columns, transport = _read_csv(transport_path)
    grid_col = _first_column(rate_columns, GRID_COLUMNS)
    if grid_col is None:
        raise ValueError(f"{rates_path} needs one grid column from {GRID_COLUMNS}")
    if grid_col not in transport:
        raise ValueError(
            f"{transport_path} must include the same grid column {grid_col!r}"
        )
    allowed_transport = {
        grid_col,
        "mean_energy_eV",
        "effective_field_Td",
        "EoverN_Td",
        "mobility_m2_V_s",
    }
    unknown_transport = sorted(set(transport_columns) - allowed_transport)
    if unknown_transport:
        raise ValueError(
            f"{transport_path} has unsupported columns: {', '.join(unknown_transport)}"
        )

    rate_order = np.argsort(rates[grid_col])
    transport_order = np.argsort(transport[grid_col])
    grid = rates[grid_col][rate_order]
    transport_grid = transport[grid_col][transport_order]
    if grid.shape != transport_grid.shape or not np.array_equal(grid, transport_grid):
        raise ValueError("rates.csv and transport.csv must contain the same grid")
    if grid.size < 2 or np.any(grid < 0.0) or np.any(np.diff(grid) <= 0.0):
        raise ValueError("rate-table grid must be nonnegative, unique, and increasing")

    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        loaded = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{metadata_path} must contain a mapping")
        metadata = loaded

    mean_energy = transport.get("mean_energy_eV")
    effective_field = transport.get("effective_field_Td")
    if effective_field is None:
        effective_field = transport.get("EoverN_Td")
    if grid_col == "mean_energy_eV":
        mean_energy = transport[grid_col]
    if grid_col in {"EoverN_Td", "effective_field_Td"}:
        effective_field = transport[grid_col]
    if mean_energy is None:
        raise ValueError(
            "transport.csv needs mean_energy_eV unless rates.csv is gridded by "
            "mean_energy_eV"
        )
    if effective_field is None:
        raise ValueError(
            "transport.csv needs effective_field_Td or EoverN_Td unless rates.csv "
            "is field-gridded"
        )

    mobility = transport.get("mobility_m2_V_s")
    if mobility is None:
        raise ValueError("transport.csv needs mobility_m2_V_s")

    mean_energy = np.asarray(mean_energy, dtype=float)[transport_order]
    effective_field = np.asarray(effective_field, dtype=float)[transport_order]
    mobility = np.asarray(mobility, dtype=float)[transport_order]
    if np.any(mean_energy < 0.0) or np.any(effective_field < 0.0):
        raise ValueError("mean energy and effective field must be nonnegative")
    if np.any(mobility <= 0.0):
        raise ValueError("mobility must be positive")
    rate_names = [column for column in rate_columns if column != grid_col]
    if not rate_names:
        raise ValueError("rates.csv must contain at least one rate coefficient")
    if any(np.any(rates[name] < 0.0) for name in rate_names):
        raise ValueError("rate coefficients must be nonnegative")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        h5.attrs["format"] = "plasma_global_electron_kinetics"
        h5.attrs["format_version"] = 1
        h5.attrs["grid_column"] = grid_col
        h5.create_dataset("mean_energy_eV", data=mean_energy)
        h5.create_dataset("effective_field_Td", data=effective_field)
        h5.create_dataset("mobility_m2_V_s", data=mobility)
        group = h5.create_group("rate_coefficients")
        for name in rate_names:
            group.create_dataset(name, data=np.asarray(rates[name])[rate_order])
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                h5.attrs[str(key)] = value
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a v3 electron-kinetics HDF5 file from external rate-table CSV files."
        )
    )
    parser.add_argument(
        "table_dir",
        type=Path,
        help="Directory containing rates.csv, transport.csv, and optional metadata.yaml.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output HDF5 path.")
    args = parser.parse_args()
    print(build_rate_table_h5(args.table_dir.resolve(), args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
