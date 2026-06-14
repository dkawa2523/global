from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from plasma_global.config import load_run_config, resolve_run_paths, write_effective_config, write_resolved_paths
from plasma_global.numerics.jacobian_check import check_jacobian
from plasma_global.workflows.context import EEDF_REGISTRY, ELECTRICAL_REGISTRY, INTEGRATOR_REGISTRY, build_case, load_case_from_yaml
from plasma_global.workflows.runner import run_from_yaml


def _registry_payload() -> dict[str, dict[str, dict[str, Any]]]:
    return {
        'eedf': EEDF_REGISTRY.details(),
        'electrical': ELECTRICAL_REGISTRY.details(),
        'integrator': INTEGRATOR_REGISTRY.details(),
    }


def _print_registry(payload: dict[str, dict[str, dict[str, Any]]]) -> None:
    labels = {
        'eedf': 'EEDF / Swarm backends',
        'electrical': 'Electrical backends',
        'integrator': 'Integrator backends',
    }
    for category, title in labels.items():
        print(f'{title}:')
        for name, detail in payload[category].items():
            description = detail.get('description') or ''
            print(f'  - {name}: {description}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='plasma-global')
    sub = parser.add_subparsers(dest='command', required=True)

    validate_parser = sub.add_parser('validate', help='Validate a case without running the solver')
    validate_parser.add_argument('case', type=Path)

    run_parser = sub.add_parser('run', help='Run a case and write configured outputs')
    run_parser.add_argument('case', type=Path)

    export_parser = sub.add_parser('export-config', help='Write effective_case.yaml and resolved_paths.yaml')
    export_parser.add_argument('case', type=Path)
    export_parser.add_argument('output_dir', type=Path)

    backends_parser = sub.add_parser('list-backends', help='List available model backends')
    backends_parser.add_argument('--json', action='store_true', help='Emit machine-readable JSON')

    jac_parser = sub.add_parser('check-jacobian', help='Finite-difference check the analytic Jacobian at a case state')
    jac_parser.add_argument('case', type=Path)
    jac_parser.add_argument('--advance-s', type=float, default=0.0, help='Integrate from the first recipe time before checking')
    jac_parser.add_argument('--top', type=int, default=10, help='Number of largest mismatches to print')
    jac_parser.add_argument('--fail-threshold', type=float, default=None, help='Return nonzero if max relative error exceeds this value')

    args = parser.parse_args(argv)

    if args.command == 'validate':
        loaded = load_case_from_yaml(args.case)
        print('Validation OK')
        for msg in loaded.validation_messages:
            print(f"[{msg['level']}] {msg['code']}: {msg['message']}")
        return 0

    if args.command == 'run':
        try:
            result = run_from_yaml(args.case)
        except RuntimeError as exc:
            print(f'ERROR: {exc}', file=sys.stderr)
            return 1
        summary = result['summary']
        print(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True).strip())
        print(f"Outputs: {result['output_dir']}")
        return 0 if bool(summary.get('success', False)) else 1

    if args.command == 'export-config':
        case_path = args.case.resolve()
        out_dir = args.output_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        run_config = load_run_config(case_path)
        resolved = resolve_run_paths(run_config, case_path)
        write_effective_config(out_dir / 'effective_case.yaml', run_config)
        write_resolved_paths(out_dir / 'resolved_paths.yaml', resolved)
        print(out_dir)
        return 0

    if args.command == 'list-backends':
        payload = _registry_payload()
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            _print_registry(payload)
        return 0

    if args.command == 'check-jacobian':
        loaded = load_case_from_yaml(args.case)
        built = build_case(loaded)
        system = built.system
        y = system.initial_state()
        t0 = loaded.recipe.steps[0].t_start_s
        time_s = t0
        if args.advance_s > 0.0:
            time_s = t0 + float(args.advance_s)
            seg = built.integrator.solve(system=system, y0=y, t_span=(t0, time_s), t_eval=[time_s])
            if not seg.success:
                print(f'ERROR: Solver failed before Jacobian check: {seg.message}', file=sys.stderr)
                return 1
            y = system.project_state(seg.y[:, -1])
        result = check_jacobian(system, time_s, y, top_n=args.top)
        print(f'max_relative_error: {result.max_relative_error:.6g}')
        for item in result.mismatches:
            print(
                f'{item.relative_error:.6g} row={item.row_label} col={item.column_label} '
                f'fd={item.finite_difference:.6g} jac={item.jacobian_value:.6g}'
            )
        if result.group_summaries:
            print('group_max_relative_error:')
            for item in result.group_summaries[:max(int(args.top), 0)]:
                print(
                    f'  {item.group}: {item.max_relative_error:.6g} '
                    f'row={item.row_label} col={item.column_label} col_group={item.column_group} '
                    f'fd={item.finite_difference:.6g} jac={item.jacobian_value:.6g} '
                    f'checked={item.checked_entry_count}'
                )
        if args.fail_threshold is not None and result.max_relative_error > args.fail_threshold:
            return 2
        return 0

    raise AssertionError(f'Unhandled command: {args.command}')


if __name__ == '__main__':
    raise SystemExit(main())
