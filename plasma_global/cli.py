from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from plasma_global.config import load_run_config, resolve_run_paths, write_effective_config, write_resolved_paths
from plasma_global.workflows.context import EEDF_REGISTRY, ELECTRICAL_REGISTRY, INTEGRATOR_REGISTRY, load_case_from_yaml
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

    raise AssertionError(f'Unhandled command: {args.command}')


if __name__ == '__main__':
    raise SystemExit(main())
