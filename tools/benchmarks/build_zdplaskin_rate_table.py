"""Regenerate the canonical v3 ZDPlaskin-derived kinetics table.

The committed CSV inputs are the immutable raw layer.  This builder performs no
network access and delegates all schema checks to the strict v3 rate-table
importer, so the output is reproducible from repository contents alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.importers.rate_table import build_rate_table_h5

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = ROOT / "benchmarks" / "raw" / "zdplaskin_example2_eovern"
DEFAULT_OUTPUT = (
    ROOT
    / "examples"
    / "v3"
    / "chemistry"
    / "zdplaskin_example2"
    / "tables"
    / "zdplaskin_example2_eovern_rates.h5"
)


def build_zdplaskin_rate_table(
    output: Path = DEFAULT_OUTPUT,
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
) -> Path:
    """Build the runtime HDF5 from committed, strictly validated CSV inputs."""

    return build_rate_table_h5(Path(raw_dir), Path(output))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the canonical v3 ZDPlaskin-derived E/N table."
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
