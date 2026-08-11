"""Thin command-line composition layer over the public application API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, get_args

import yaml

from plasma_global.api import load_case, simulate, write_result
from plasma_global.audit import audit_case
from plasma_global.output import export_result_csv, plot_result_h5


def _compile_case(case: Any) -> Any:
    """Lazy boundary kept separate so validation tests need no finished solver."""

    from plasma_global.build import compile_case

    return compile_case(case)


def _kind_ids(annotation: Any) -> tuple[str, ...]:
    """Extract Literal discriminator values from an Annotated model union."""

    args = get_args(annotation)
    union = args[0] if args else annotation
    members = get_args(union) or (union,)
    values: list[str] = []
    for member in members:
        field = getattr(member, "model_fields", {}).get("kind")
        if field is None:
            continue
        values.extend(str(value) for value in get_args(field.annotation))
    return tuple(dict.fromkeys(values))


def _model_catalog() -> dict[str, tuple[str, ...]]:
    from plasma_global.input.schema import (
        ElectronClosure,
        ElectronDensityModel,
        ElectronModel,
        GasEnergyModel,
        PowerModel,
        WallTransport,
    )

    return {
        "electrons": _kind_ids(ElectronModel),
        "electron_closure": _kind_ids(ElectronClosure),
        "electron_density": _kind_ids(ElectronDensityModel),
        "gas_energy": _kind_ids(GasEnergyModel),
        "power": _kind_ids(PowerModel),
        "wall_transport": _kind_ids(WallTransport),
    }


def _cmd_validate(args: argparse.Namespace) -> int:
    case = load_case(args.case)
    _compile_case(case)
    print("Validation OK")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    case = load_case(args.case)
    result = simulate(case)
    paths = write_result(result, args.output)
    print(paths.result_h5)
    print(paths.summary_yaml)
    return 0 if result.status.success else 1


def _cmd_audit(args: argparse.Namespace) -> int:
    report = audit_case(load_case(args.case))
    print(yaml.safe_dump(report.to_dict(), sort_keys=False, allow_unicode=True).strip())
    return 0 if report.passed else 1


def _cmd_migrate_v2(args: argparse.Namespace) -> int:
    from plasma_global.input.migrate_v2 import migrate_v2_to_yaml

    def optional_mapping(path: Path | None) -> dict[str, Any] | None:
        if path is None:
            return None
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError(f"{path} must contain a YAML mapping")
        return {str(key): item for key, item in value.items()}

    products = optional_mapping(args.boundary_products)
    raw_segments = optional_mapping(args.cross_section_segments)
    segments = (
        None
        if raw_segments is None
        else {key: int(value) for key, value in raw_segments.items()}
    )
    result = migrate_v2_to_yaml(
        args.source,
        args.output,
        boundary_products=products,
        cross_section_segments=segments,
    )
    print(args.output)
    print(args.output.with_name(f"{args.output.stem}.migration.yaml"))
    for warning in result.report.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    for key in result.report.unused_keys:
        print(f"UNUSED: {key}", file=sys.stderr)
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    catalog = _model_catalog()
    if args.json:
        print(json.dumps(catalog, indent=2, ensure_ascii=False))
        return 0
    for category, identifiers in catalog.items():
        print(f"{category}:")
        for identifier in identifiers:
            print(f"  - {identifier}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    path = export_result_csv(args.result, args.csv)
    print(path)
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    output_dir = args.output or args.result.parent / "plots"
    paths = plot_result_h5(
        args.result,
        output_dir,
        series=args.series,
        image_format=args.format,
        dpi=args.dpi,
    )
    for path in paths:
        print(path)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="plasma-global")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Load and compile a case")
    validate.add_argument("case", type=Path)
    validate.set_defaults(handler=_cmd_validate)

    run = commands.add_parser("run", help="Simulate a case and write fixed outputs")
    run.add_argument("case", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.set_defaults(handler=_cmd_run)

    audit = commands.add_parser(
        "audit", help="Simulate a case and report provenance and physical balances"
    )
    audit.add_argument("case", type=Path)
    audit.set_defaults(handler=_cmd_audit)

    migrate = commands.add_parser("migrate-v2", help="Convert a v2 case to v3 YAML")
    migrate.add_argument("source", type=Path)
    migrate.add_argument("--output", type=Path, required=True)
    migrate.add_argument(
        "--boundary-products",
        type=Path,
        help="YAML mapping from ambiguous positive-ion IDs to product sides",
    )
    migrate.add_argument(
        "--cross-section-segments",
        type=Path,
        help="YAML mapping selecting one axis from a concatenated legacy curve",
    )
    migrate.set_defaults(handler=_cmd_migrate_v2)

    models = commands.add_parser("models", help="List accepted model IDs")
    models.add_argument("--json", action="store_true")
    models.set_defaults(handler=_cmd_models)

    export = commands.add_parser("export", help="Export result.h5 to CSV")
    export.add_argument("result", type=Path)
    export.add_argument("--csv", type=Path, required=True)
    export.set_defaults(handler=_cmd_export)

    plot = commands.add_parser("plot", help="Plot result series")
    plot.add_argument("result", type=Path)
    plot.add_argument("series", nargs="*")
    plot.add_argument("--output", type=Path)
    plot.add_argument("--format", default="png")
    plot.add_argument("--dpi", type=int, default=150)
    plot.set_defaults(handler=_cmd_plot)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
