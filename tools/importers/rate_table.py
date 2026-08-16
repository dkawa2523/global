"""Developer CLI for runtime-owned electron-table conversion helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

from plasma_global.input._migrate_v2_rate_table import (
    GRID_COLUMNS,
    build_rate_table_h5,
    convert_v2_rate_table_h5,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a v3 electron-kinetics HDF5 file from external rate-table CSV files."
        )
    )
    parser.add_argument(
        "table_dir",
        type=Path,
        help=(
            "Directory containing rates.csv, transport.csv, and optional metadata.yaml."
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="Output HDF5 path.")
    arguments = parser.parse_args(argv)
    print(
        build_rate_table_h5(arguments.table_dir.resolve(), arguments.output.resolve())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GRID_COLUMNS",
    "build_rate_table_h5",
    "convert_v2_rate_table_h5",
    "main",
]
