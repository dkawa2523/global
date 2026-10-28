"""Thin command-line composition layer over the public application API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, get_args

import yaml

from plasma_global._yaml import YamlLoadError
from plasma_global.api import load_case, simulate, write_result
from plasma_global.audit import audit_case
from plasma_global.output import (
    export_result_csv,
    network_result_h5,
    plot_result_h5,
    preflight_output,
)


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
    from plasma_global.input._resolved_case import (
        ElectronDensityModel,
        GasEnergyModel,
        PowerModel,
        WallTransport,
    )
    from plasma_global.input.schema import ElectronModel

    return {
        "electrons": _kind_ids(ElectronModel),
        "electron_density": _kind_ids(ElectronDensityModel),
        "gas_energy": _kind_ids(GasEnergyModel),
        "power": _kind_ids(PowerModel),
        "wall_transport": _kind_ids(WallTransport),
    }


def _cmd_validate(args: argparse.Namespace) -> int:
    from plasma_global.build import compile_case

    case = load_case(args.case)
    compiled = compile_case(case)
    print("Validation OK (input and compilation; integration has not run)")
    print(
        f"Case: {case.case.name}; zones: {len(case.reactor.zones)}; "
        f"duration: {case.recipe.duration_s:g} s"
    )
    print(
        f"Electrons: {case.models.electron_closure.kind}; "
        f"coefficients: {case.models.electrons.kind}"
    )
    for zone in case.reactor.zones:
        print(
            f"Initial {zone.zone_id}: {zone.pressure_Pa:g} Pa, "
            f"{zone.gas_temperature_K:g} K"
        )
    for table in dict.fromkeys(
        id(table) for table in compiled.model.electron_kinetics_by_zone.values()
    ):
        value = next(
            item
            for item in compiled.model.electron_kinetics_by_zone.values()
            if id(item) == table
        )
        print(
            f"Table: {value.source}; {value.lookup}: "
            f"[{value.axis[0]:g}, {value.axis[-1]:g}]"
        )
    for issue in compiled.assessment.issues:
        print(f"{issue.level}: {issue.message}", file=sys.stderr)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(
            f"Output directory already exists: {args.output}; "
            "choose a new run directory"
        )
    case = load_case(args.case)
    preflight_output(case.output)
    result = simulate(case)
    paths = write_result(result, args.output)
    print(paths.result_h5)
    print(paths.summary_yaml)
    if paths.result_csv is not None:
        print(paths.result_csv)
    for path in paths.analysis_files:
        print(path)
    for issue in result.metadata.get("assessment", {}).get("issues", ()):
        print(f"{issue['level']}: {issue['message']}", file=sys.stderr)
    return int(not result.status.success)


def _cmd_audit(args: argparse.Namespace) -> int:
    report = audit_case(load_case(args.case))
    print(yaml.safe_dump(report.to_dict(), sort_keys=False, allow_unicode=True).strip())
    return 0 if report.passed else 1


def _cmd_migrate_v3(args: argparse.Namespace) -> int:
    from plasma_global.input.migrate_v3 import migrate_v3

    print(migrate_v3(args.source, args.output))
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    from plasma_global.examples import initialize_example, list_examples

    if args.list:
        if args.example is not None or args.output is not None:
            raise ValueError("init --list does not take an example or --output")
        for name, description in list_examples().items():
            summary = " ".join(description.split()).split(". ", 1)[0].split("。", 1)[0]
            print(f"{name}  {summary}")
        return 0
    if args.output is None:
        raise ValueError(
            "init requires --output; use init --list to see available cases"
        )
    print(initialize_example(args.example or "argon_maxwellian", args.output))
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


def _cmd_outputs(args: argparse.Namespace) -> int:
    from plasma_global._output_values import PORT_UNITS, port_values
    from plasma_global.build import compile_case
    from plasma_global.core.result import series_unit
    from plasma_global.postprocess import available_observables

    compiled = compile_case(load_case(args.case))
    plan = compiled.output_plan
    model = compiled.model
    initial = model.initial_state(compiled.initial_state)
    segment = model.segments[0]
    ports = port_values(model.evaluate(segment.start_s, initial, segment))
    names = tuple(
        dict.fromkeys(
            (
                *compiled.model.layout.labels,
                *available_observables(compiled.model),
                *plan.port_names,
                *ports,
                *plan.aliases,
            )
        )
    )
    units = {
        **plan.metadata["series_units"],
        **{
            name: PORT_UNITS[name.partition("[")[0].removeprefix("port_")]
            for name in ports
        },
    }
    rows = {
        name: {
            "unit": series_unit(name, units),
            "stored": name in compiled.model.layout.labels
            or name in plan.stored
            or name in plan.aliases,
        }
        for name in names
    }
    print(
        json.dumps(rows, indent=2, ensure_ascii=False)
        if args.json
        else yaml.safe_dump(rows, sort_keys=False, allow_unicode=True).strip()
    )
    return 0


def _cmd_network(args: argparse.Namespace) -> int:
    from plasma_global.input._schema_output import NetworkConfig

    values = {
        "mode": args.mode,
        "measure": args.measure,
        "time_s": args.time_s,
        "domains": args.domains,
        "zones": args.zones,
        "surfaces": args.surfaces,
        "min_fraction": args.min_fraction,
        "species": args.species,
    }
    overrides = {key: value for key, value in values.items() if value is not None}
    config = NetworkConfig.model_validate(overrides) if overrides else None
    for path in network_result_h5(
        args.result, args.output, config=config, image_format=args.format
    ):
        print(path)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="plasma-global")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Load and compile a case")
    validate.add_argument("case", type=Path)
    validate.set_defaults(handler=_cmd_validate)

    run = commands.add_parser(
        "run", help="Simulate a case and write its requested outputs"
    )
    run.add_argument("case", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.set_defaults(handler=_cmd_run)

    audit = commands.add_parser(
        "audit", help="Simulate a case and report model assessment and provenance"
    )
    audit.add_argument("case", type=Path)
    audit.set_defaults(handler=_cmd_audit)

    initialize = commands.add_parser(
        "init", help="Copy a working example with its input data"
    )
    initialize.add_argument("example", nargs="?")
    initialize.add_argument(
        "--list", action="store_true", help="List installed examples and benchmarks"
    )
    initialize.add_argument("--output", type=Path)
    initialize.set_defaults(handler=_cmd_init)

    migrate = commands.add_parser(
        "migrate-v3", help="Convert v3 inputs into a new v4 bundle"
    )
    migrate.add_argument("source", type=Path)
    migrate.add_argument("--output", type=Path, required=True)
    migrate.set_defaults(handler=_cmd_migrate_v3)

    models = commands.add_parser("models", help="List accepted model IDs")
    models.add_argument("--json", action="store_true")
    models.set_defaults(handler=_cmd_models)

    outputs = commands.add_parser(
        "outputs", help="List available physical series and stored selections"
    )
    outputs.add_argument("case", type=Path)
    outputs.add_argument("--json", action="store_true")
    outputs.set_defaults(handler=_cmd_outputs)

    network = commands.add_parser(
        "network",
        help="Draw a Graphviz reaction mechanism or event network from saved results",
    )
    network.add_argument("result", type=Path)
    network.add_argument("--output", type=Path, required=True)
    network.add_argument("--mode", choices=("mechanism", "events"))
    network.add_argument("--measure", choices=("cumulative", "rate"))
    network.add_argument("--time-s", type=float)
    network.add_argument("--domains", nargs="+", choices=("gas", "boundary", "surface"))
    network.add_argument("--zones", nargs="+")
    network.add_argument("--surfaces", nargs="+")
    network.add_argument("--species", nargs="+")
    network.add_argument("--min-fraction", type=float)
    network.add_argument("--format", choices=("svg", "png", "pdf", "dot"))
    network.set_defaults(handler=_cmd_network)

    export = commands.add_parser("export", help="Export result.h5 to CSV")
    export.add_argument("result", type=Path)
    export.add_argument("--csv", type=Path, required=True)
    export.set_defaults(handler=_cmd_export)

    plot = commands.add_parser("plot", help="Plot result series")
    plot.add_argument("result", type=Path)
    plot.add_argument("series", nargs="*")
    plot.add_argument("--output", type=Path)
    plot.add_argument("--format")
    plot.add_argument("--dpi", type=int)
    plot.set_defaults(handler=_cmd_plot)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        YamlLoadError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
