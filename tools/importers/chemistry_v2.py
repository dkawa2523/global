"""Developer CLI for the runtime-owned schema-v2 chemistry converter."""

from __future__ import annotations

import argparse
from pathlib import Path

from plasma_global.input._migrate_v2_chemistry import convert_v2_chemistry
from plasma_global.input._migrate_v2_chemistry_models import read_yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--boundary-products",
        type=Path,
        help="YAML mapping from incident ion ID to an explicit neutral product side",
    )
    parser.add_argument(
        "--cross-section-segments",
        type=Path,
        help="YAML mapping from cross-section ID to a zero-based concatenated segment",
    )
    arguments = parser.parse_args(argv)
    products: dict[str, str] | None = None
    if arguments.boundary_products is not None:
        products = dict(read_yaml(arguments.boundary_products))
    segments: dict[str, int] | None = None
    if arguments.cross_section_segments is not None:
        segments = {
            cross_section_id: int(segment)
            for cross_section_id, segment in read_yaml(
                arguments.cross_section_segments
            ).items()
        }
    print(
        convert_v2_chemistry(
            arguments.manifest,
            arguments.output,
            boundary_products=products,
            cross_section_segments=segments,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["convert_v2_chemistry", "main"]
