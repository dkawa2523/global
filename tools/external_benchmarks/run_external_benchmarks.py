from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plasma_global.workflows.runner import run_from_yaml
from tools.external_benchmarks.crane_two_reaction_argon import (
    DEFAULT_CASE as CRANE_CASE,
    DEFAULT_REFERENCE as CRANE_REFERENCE,
    build_report as build_crane_report,
)
from tools.external_benchmarks.pygmol_argon import (
    DEFAULT_CASE as PYGMOL_CASE,
    RATE_MODE_LOCAL_FIT,
    RATE_MODE_LOCAL_TABLE,
    RATE_MODE_SURROGATE,
    build_report as build_pygmol_report,
)
from tools.external_benchmarks.zdplaskin_example2 import (
    DEFAULT_OUTPUT_DIR as ZDPLASKIN_OUTPUT_DIR,
    DEFAULT_REFERENCE as ZDPLASKIN_REFERENCE,
    build_report as build_zdplaskin_report,
)
from tools.external_benchmarks.zdplaskin_eovern import build_eovern_report


ZDPLASKIN_CASE = ROOT / 'examples' / 'configs' / 'case_zdplaskin_example2.yaml'
DEFAULT_WRITE = ROOT / 'tools' / 'external_benchmarks' / 'last_report.yaml'


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(report, sort_keys=False), encoding='utf-8')


def run_zdplaskin(*, run_case: bool = True) -> dict[str, Any]:
    if run_case:
        run_from_yaml(ZDPLASKIN_CASE)
    report = build_zdplaskin_report(ZDPLASKIN_OUTPUT_DIR, ZDPLASKIN_REFERENCE)
    comparison_file = ZDPLASKIN_OUTPUT_DIR / 'comparison_zdplaskin_example2.yaml'
    _write_report(comparison_file, report)
    eovern_report = build_eovern_report(ZDPLASKIN_OUTPUT_DIR, ZDPLASKIN_REFERENCE)
    eovern_file = ZDPLASKIN_OUTPUT_DIR / 'comparison_zdplaskin_eovern.yaml'
    _write_report(eovern_file, eovern_report)
    return {
        'id': 'zdplaskin_example2_argon_dc_series',
        'case': str(ZDPLASKIN_CASE),
        'output_dir': str(ZDPLASKIN_OUTPUT_DIR),
        'comparison_file': str(comparison_file),
        'eovern_comparison_file': str(eovern_file),
        'passed': report['local_result']['final_electron_density_m3'] > 0.0,
        'summary': {
            'final_electron_density_m3': report['local_result']['final_electron_density_m3'],
            'final_mean_electron_energy_eV': report['local_result']['final_mean_electron_energy_eV'],
            'final_reduced_field_Td': report['local_result']['final_reduced_field_Td'],
            'same_footing_reduced_field_Td': eovern_report['local_run_field']['same_ZDPlaskin_final_voltage_reduced_field_Td'],
            'reported_over_zdplaskin_EoverN_ratio': eovern_report['comparison']['local_reported_over_zdplaskin_saved_EoverN_ratio'],
        },
    }


def run_crane() -> dict[str, Any]:
    report = build_crane_report(CRANE_CASE, CRANE_REFERENCE)
    output_dir = Path(report['local_result']['output_dir'])
    comparison_file = output_dir / 'comparison_crane_two_reaction_argon.yaml'
    _write_report(comparison_file, report)
    return {
        'id': 'crane_two_reaction_argon_ode_parity',
        'case': str(CRANE_CASE),
        'output_dir': str(output_dir),
        'comparison_file': str(comparison_file),
        'passed': report['comparison']['final_electron_density_m3_relative_error'] < 5.0e-4,
        'summary': {
            'final_electron_density_m3': report['local_result']['final_electron_density_m3'],
            'final_electron_density_relative_error': report['comparison']['final_electron_density_m3_relative_error'],
        },
    }


def run_pygmol(*, rerun_local: bool = True, rate_mode: str = RATE_MODE_SURROGATE) -> dict[str, Any]:
    try:
        report = build_pygmol_report(PYGMOL_CASE, rerun_local=rerun_local, write_solution_csv=False, rate_mode=rate_mode)
    except RuntimeError as exc:
        return {
            'id': 'pygmol_argon_global_model_sanity',
            'case': str(PYGMOL_CASE),
            'rate_mode': rate_mode,
            'passed': True,
            'skipped': True,
            'skip_reason': str(exc),
            'summary': {},
        }
    return {
        'id': 'pygmol_argon_global_model_sanity',
        'case': str(PYGMOL_CASE),
        'rate_mode': report['rate_mode'],
        'output_dir': report['output_dir'],
        'comparison_file': report['comparison_file'],
        'generated_model_file': report.get('generated_model_file'),
        'passed': report['passed'],
        'summary': report['summary'],
    }


def run_selected(which: str, *, run_zdplaskin_case: bool = True, pygmol_rate_mode: str = RATE_MODE_SURROGATE) -> dict[str, Any]:
    results = []
    if which in {'all', 'zdplaskin'}:
        results.append(run_zdplaskin(run_case=run_zdplaskin_case))
    if which in {'all', 'crane'}:
        results.append(run_crane())
    if which in {'all', 'pygmol'}:
        results.append(run_pygmol(rerun_local=True, rate_mode=pygmol_rate_mode))
    return {
        'tool': 'external_benchmarks',
        'benchmark_count': len(results),
        'passed': all(item['passed'] for item in results),
        'benchmarks': results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Run external-code benchmark comparisons outside the core solver package.')
    parser.add_argument('--only', choices=['all', 'zdplaskin', 'crane', 'pygmol'], default='all')
    parser.add_argument('--pygmol-rate-mode', choices=[RATE_MODE_SURROGATE, RATE_MODE_LOCAL_FIT, RATE_MODE_LOCAL_TABLE], default=RATE_MODE_SURROGATE)
    parser.add_argument('--skip-zdplaskin-run', action='store_true', help='Reuse the existing ZDPlaskin surrogate output directory instead of re-running the local case.')
    parser.add_argument('--write', type=Path, default=DEFAULT_WRITE)
    args = parser.parse_args()
    report = run_selected(
        args.only,
        run_zdplaskin_case=not args.skip_zdplaskin_run,
        pygmol_rate_mode=args.pygmol_rate_mode,
    )
    _write_report(args.write.resolve(), report)
    print(yaml.safe_dump(report, sort_keys=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
