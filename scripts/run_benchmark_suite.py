from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plasma_global.workflows.runner import run_from_yaml


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _expectation_passed(value: float, spec: dict[str, Any]) -> tuple[bool, str]:
    if 'min' in spec and value < float(spec['min']):
        return False, f'{value} < min {spec["min"]}'
    if 'max' in spec and value > float(spec['max']):
        return False, f'{value} > max {spec["max"]}'
    if 'target' in spec:
        target = float(spec['target'])
        rtol = float(spec.get('rtol', 0.0))
        atol = float(spec.get('atol', 0.0))
        if abs(value - target) > atol + rtol * abs(target):
            return False, f'{value} differs from target {target}'
    return True, 'ok'


def evaluate_expectations(summary: dict[str, Any], expectations: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    passed = True
    for key, spec in expectations.items():
        if key not in summary:
            checks[key] = {'passed': False, 'reason': 'missing summary key'}
            passed = False
            continue
        ok, reason = _expectation_passed(float(summary[key]), spec or {})
        checks[key] = {'passed': ok, 'value': float(summary[key]), 'reason': reason}
        passed = passed and ok
    return {'passed': passed, 'checks': checks}


def run_suite(manifest_path: Path, *, output: Path | None = None) -> dict[str, Any]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding='utf-8')) or {}
    base = manifest_path.parent
    results = []
    for bench in manifest.get('benchmarks', []) or []:
        if not bool(bench.get('enabled', True)):
            continue
        case_path = _resolve(base, str(bench['case']))
        run = run_from_yaml(case_path)
        expectation_report = evaluate_expectations(run['summary'], bench.get('expectations', {}) or {})
        results.append(
            {
                'id': bench.get('id', case_path.stem),
                'case': str(case_path),
                'output_dir': run['output_dir'],
                'passed': bool(run['summary'].get('success', False)) and expectation_report['passed'],
                'expectations': expectation_report['checks'],
            }
        )
    report = {'benchmark_count': len(results), 'passed': all(item['passed'] for item in results), 'benchmarks': results}
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml.safe_dump(report, sort_keys=False), encoding='utf-8')
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description='Run benchmark cases and compare summary metrics against manifest tolerances.')
    parser.add_argument('manifest', type=Path, nargs='?', default=ROOT / 'benchmarks' / 'manifest.yaml')
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks' / 'last_report.yaml')
    args = parser.parse_args()
    report = run_suite(args.manifest.resolve(), output=args.output.resolve())
    print(yaml.safe_dump(report, sort_keys=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
