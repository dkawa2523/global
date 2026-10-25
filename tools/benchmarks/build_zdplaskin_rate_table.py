"""Regenerate the ZDPlaskin-derived electron-coefficient CSV.

The committed CSV inputs are the immutable raw layer.  This builder performs no
network access. It explicitly sorts and aligns the source coordinates without
interpolating or repairing missing values.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from plasma_global._numeric_csv import read_numeric_csv

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = ROOT / "benchmarks" / "raw" / "zdplaskin_example2_eovern"
DEFAULT_OUTPUT = (
    ROOT
    / "plasma_global"
    / "assets"
    / "examples"
    / "data"
    / "zdplaskin_example2"
    / "electrons.csv"
)


def build_zdplaskin_rate_table(
    output: Path = DEFAULT_OUTPUT,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
) -> Path:
    """Prepare the canonical CSV, sorting both source grids explicitly."""
    rates = read_numeric_csv(Path(raw_dir) / "rates.csv")
    transport = read_numeric_csv(Path(raw_dir) / "transport.csv")
    rate_grid = rates.pop("EoverN_Td")
    rate_order = np.argsort(rate_grid)
    transport_order = np.argsort(transport["EoverN_Td"])
    grid = rate_grid[rate_order]
    if (
        grid.size < 2
        or np.any(np.diff(grid) <= 0.0)
        or not np.array_equal(grid, transport["EoverN_Td"][transport_order])
    ):
        raise ValueError("rates and transport must share one unique grid")
    columns = {
        "mean_energy_eV": transport["mean_energy_eV"][transport_order],
        "mobility_m2_V_s": transport["mobility_m2_V_s"][transport_order],
        **{
            ("momentum:Ar" if key == "xs_zdp_ar_effective" else f"k:{key}"): value[
                rate_order
            ]
            for key, value in sorted(rates.items())
        },
        "reduced_field_Td": transport["EoverN_Td"][transport_order],
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(zip(*columns.values(), strict=True))
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the ZDPlaskin-derived E/N coefficient CSV."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(
        build_zdplaskin_rate_table(
            args.output.resolve(), raw_dir=args.raw_dir.resolve()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
