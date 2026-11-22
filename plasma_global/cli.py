"""Thin command-line composition layer over the public application API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from plasma_global._yaml import YamlLoadError
from plasma_global.api import (
    compile_case,
    list_models,
    load_case,
    simulate,
    write_result,
)
from plasma_global.errors import IntegrationError
from plasma_global.output import (
    ResultPaths,
    audit_result_h5,
    export_result_csv,
    inspect_result_h5,
    network_result_h5,
    plot_result_h5,
)


def _cmd_validate(args: argparse.Namespace) -> int:
    case = load_case(args.case)
    compiled = compile_case(case)
    print("Validation OK (input and compilation; integration has not run)")
    print(
        f"Case: {case.case.name}; zones: {len(case.reactor.zones)}; "
        f"duration: {case.run.total_duration_s:g} s"
    )
    print(
        f"Electrons: {case.models.electrons.kind}; "
        f"coefficients: {getattr(case.models.electrons.coefficients, 'kind', 'table')}"
    )
    for identity, zone in case.reactor.zones.items():
        print(
            f"Initial {identity}: {zone.initial.pressure_Pa:g} Pa, "
            f"{zone.initial.gas_temperature_K:g} K"
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
    _print_output_omissions(compiled.output_plan.metadata)
    return 0


def _print_output_omissions(metadata: Any) -> None:
    for item in metadata.get("default_output_omissions", ()):
        interval = item["unavailable_intervals"][0]
        print(
            f"WARNING: basic output omits {item['series']!r}; "
            f"undefined in segment {interval['segment_id']!r}: {interval['reason']}",
            file=sys.stderr,
        )


def _report_saved_paths(paths: ResultPaths) -> None:
    print(paths.result_h5)
    print(paths.summary_yaml)
    if paths.result_csv is not None:
        print(paths.result_csv)
    for path in paths.analysis_files:
        print(path)
    for item in paths.artifacts.requests:
        if item.status == "failed":
            print(
                f"ARTIFACT ERROR [{item.kind}/{item.name}]: {item.message}",
                file=sys.stderr,
            )


def _cmd_run(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(
            f"Output directory already exists: {args.output}; "
            "choose a new run directory"
        )
    case = compile_case(load_case(args.case))
    _print_output_omissions(case.output_plan.metadata)
    try:
        result = simulate(case)
    except IntegrationError as exc:
        if exc.partial_result is None:
            print(f"INTEGRATION ERROR: {exc}", file=sys.stderr)
            return 1
        result = exc.partial_result
    paths = write_result(result, args.output)
    _report_saved_paths(paths)
    for issue in result.metadata.get("assessment", {}).get("issues", ()):
        print(f"{issue['level']}: {issue['message']}", file=sys.stderr)
    if not result.status.success:
        print(f"INTEGRATION ERROR: {result.status.message}", file=sys.stderr)
        return 1
    return 0 if paths.artifacts.success else 3


def _conservation_tolerances(values: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for value in values:
        name, separator, raw = value.partition("=")
        if not separator or not name or name != name.strip():
            raise ValueError("conservation constraints require SERIES=TOLERANCE")
        if name in result:
            raise ValueError(f"duplicate conservation constraint {name!r}")
        result[name] = float(raw)
    return result


def _cmd_audit(args: argparse.Namespace) -> int:
    tolerances = _conservation_tolerances(args.conservation)
    report = audit_result_h5(
        args.result,
        conservation_observables=tuple(tolerances),
        conservation_tolerances=tolerances,
        nonnegative_series=args.nonnegative,
        negative_tolerance=args.negative_tolerance,
    )
    print(yaml.safe_dump(report.to_dict(), sort_keys=False, allow_unicode=True).strip())
    return 0 if report.passed else 1


def _cmd_inspect(args: argparse.Namespace) -> int:
    document = inspect_result_h5(args.result)
    print(
        json.dumps(document, indent=2, ensure_ascii=False)
        if args.json
        else yaml.safe_dump(document, sort_keys=False, allow_unicode=True).strip()
    )
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
    catalog = list_models(args.category, args.kind)
    if args.json:
        print(json.dumps(catalog, indent=2, ensure_ascii=False))
        return 0
    if args.kind is not None:
        print(yaml.safe_dump(catalog, sort_keys=False, allow_unicode=True).strip())
        return 0
    for category, entries in catalog.items():
        print(f"{category}:")
        for kind, entry in entries.items():
            print(f"  {kind}: {entry['purpose']}")
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
    from plasma_global._output_plan import build_output_catalog, compile_output
    from plasma_global.build import compile_physics

    prepared = compile_physics(load_case(args.case))
    rows = build_output_catalog(prepared.case, prepared.model, prepared.chemistry_data)
    selection_error = None
    try:
        rows = dict(
            compile_output(
                prepared.case, prepared.model, prepared.chemistry_data
            ).catalog
        )
    except (ValueError, KeyError) as exc:
        selection_error = str(exc)
    print(
        json.dumps(rows, indent=2, ensure_ascii=False)
        if args.json
        else yaml.safe_dump(rows, sort_keys=False, allow_unicode=True).strip()
    )
    if selection_error:
        print(f"OUTPUT SELECTION ERROR: {selection_error}", file=sys.stderr)
    return 0


def _cmd_network(args: argparse.Namespace) -> int:
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
    for path in network_result_h5(
        args.result,
        args.output,
        name=args.name,
        overrides=overrides,
        image_format=args.format,
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
        "audit", help="Check saved numerical results without integration"
    )
    audit.add_argument("result", type=Path)
    audit.add_argument("--nonnegative", nargs="*", default=[])
    audit.add_argument(
        "--conservation", action="append", default=[], metavar="SERIES=TOLERANCE"
    )
    audit.add_argument("--negative-tolerance", type=float, default=0.0)
    audit.set_defaults(handler=_cmd_audit)

    inspect = commands.add_parser(
        "inspect", help="Inspect saved status, series, units and provenance"
    )
    inspect.add_argument("result", type=Path)
    inspect.add_argument("--json", action="store_true")
    inspect.set_defaults(handler=_cmd_inspect)

    initialize = commands.add_parser(
        "init", help="Copy a working example with its input data"
    )
    initialize.add_argument("example", nargs="?")
    initialize.add_argument(
        "--list", action="store_true", help="List installed examples and benchmarks"
    )
    initialize.add_argument("--output", type=Path)
    initialize.set_defaults(handler=_cmd_init)

    models = commands.add_parser(
        "models", help="Discover models, required inputs and working examples"
    )
    models.add_argument("category", nargs="?")
    models.add_argument("kind", nargs="?")
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
    network.add_argument("--name", help="Select one saved network request")
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
