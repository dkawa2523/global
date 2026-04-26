from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.external_benchmarks.pygmol_argon import (
    DEFAULT_CASE,
    POWER_MODE_LOCAL_ABSORBED,
    POWER_MODE_RECIPE,
    RATE_MODE_LOCAL_TABLE,
    WALL_LOSS_MODE_LOCAL_COEFFICIENT,
    WALL_LOSS_MODE_PYGMOL,
    build_report,
)


DEFAULT_REPORT = 'comparison_pygmol_same_footing_decomposition'


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def stage_specs() -> list[dict[str, str]]:
    return [
        {
            'stage_id': 'same_rate_recipe_power_pygmol_wall',
            'description': 'same local electron-impact rates; PyGMol recipe power and PyGMol wall loss remain',
            'rate_mode': RATE_MODE_LOCAL_TABLE,
            'power_mode': POWER_MODE_RECIPE,
            'wall_loss_mode': WALL_LOSS_MODE_PYGMOL,
        },
        {
            'stage_id': 'same_rate_local_power_pygmol_wall',
            'description': 'same local electron-impact rates and same local absorbed-power waveform; PyGMol wall loss remains',
            'rate_mode': RATE_MODE_LOCAL_TABLE,
            'power_mode': POWER_MODE_LOCAL_ABSORBED,
            'wall_loss_mode': WALL_LOSS_MODE_PYGMOL,
        },
        {
            'stage_id': 'same_rate_local_power_local_wall',
            'description': 'same local electron-impact rates, same local absorbed-power waveform, and local global ion wall-loss coefficient',
            'rate_mode': RATE_MODE_LOCAL_TABLE,
            'power_mode': POWER_MODE_LOCAL_ABSORBED,
            'wall_loss_mode': WALL_LOSS_MODE_LOCAL_COEFFICIENT,
        },
    ]


def _row_for_step(report: dict[str, Any], step_id: str) -> dict[str, Any]:
    return next((row for row in report.get('rows', []) if row.get('step_id') == step_id), {})


def _compact_stage_row(spec: dict[str, str], report: dict[str, Any]) -> dict[str, Any]:
    ignition = _row_for_step(report, 'ignition')
    production = _row_for_step(report, 'production')
    afterglow = _row_for_step(report, 'afterglow')
    return {
        'stage_id': spec['stage_id'],
        'rate_mode': spec['rate_mode'],
        'power_mode': spec['power_mode'],
        'wall_loss_mode': spec['wall_loss_mode'],
        'passed': bool(report.get('passed')),
        'comparison_file': report.get('comparison_file'),
        'generated_model_file': report.get('generated_model_file'),
        'ignition_final_ne_ratio': ignition.get('ratio_final_ne_pygmol_over_local'),
        'production_final_ne_ratio': production.get('ratio_final_ne_pygmol_over_local'),
        'afterglow_final_ne_ratio': afterglow.get('ratio_final_ne_pygmol_over_local'),
        'production_mean_ne_ratio': production.get('ratio_mean_ne_pygmol_over_local'),
        'production_power_ratio': production.get('ratio_mean_absorbed_power_pygmol_over_local'),
        'production_wall_loss_rate_ratio': production.get('ratio_final_wall_loss_rate_pygmol_over_local'),
        'production_energy_ratio': production.get('ratio_final_mean_energy_equiv_pygmol_over_local'),
        'production_local_power_W': production.get('local_mean_absorbed_power_W'),
        'production_pygmol_power_W': production.get('pygmol_mean_absorbed_power_W'),
        'production_local_wall_loss_rate_s': production.get('local_final_global_ion_loss_rate_s'),
        'production_pygmol_wall_loss_rate_s': production.get('pygmol_final_Arplus_wall_loss_rate_s'),
    }


def build_decomposition_report(case_path: Path = DEFAULT_CASE, *, rerun_local: bool = False) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    output_dir: Path | None = None
    for idx, spec in enumerate(stage_specs()):
        report = build_report(
            case_path,
            rerun_local=rerun_local and idx == 0,
            write_solution_csv=False,
            rate_mode=spec['rate_mode'],
            power_mode=spec['power_mode'],
            wall_loss_mode=spec['wall_loss_mode'],
        )
        output_dir = Path(report['output_dir'])
        reports.append({'stage': spec, 'report': report})
        summary_rows.append(_compact_stage_row(spec, report))
    assert output_dir is not None
    summary_csv = output_dir / f'{DEFAULT_REPORT}.csv'
    report_yaml = output_dir / f'{DEFAULT_REPORT}.yaml'
    fields = [
        'stage_id',
        'rate_mode',
        'power_mode',
        'wall_loss_mode',
        'passed',
        'comparison_file',
        'generated_model_file',
        'ignition_final_ne_ratio',
        'production_final_ne_ratio',
        'afterglow_final_ne_ratio',
        'production_mean_ne_ratio',
        'production_power_ratio',
        'production_wall_loss_rate_ratio',
        'production_energy_ratio',
        'production_local_power_W',
        'production_pygmol_power_W',
        'production_local_wall_loss_rate_s',
        'production_pygmol_wall_loss_rate_s',
    ]
    _write_csv(summary_csv, summary_rows, fields)
    out = {
        'tool': 'pygmol_same_footing',
        'scope': 'external_benchmark_only',
        'case': str(case_path.resolve()),
        'output_dir': str(output_dir),
        'summary_csv': str(summary_csv),
        'passed': all(bool(row['passed']) for row in summary_rows),
        'interpretation': [
            'Stages are cumulative: same electron-impact rates, then same absorbed power, then same global ion wall-loss coefficient.',
            'Only the PyGMol external adapter is modified; plasma_global core physics and input chemistry are not changed.',
            'Remaining differences after the final stage indicate geometry, flow/pressure regulation, electron-energy equation, and other global-model equation-form differences.',
        ],
        'summary': summary_rows,
        'reports': reports,
    }
    _write_yaml(report_yaml, out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Decompose PyGMol differences by aligning rate, absorbed power, and wall-loss coefficient cumulatively.')
    parser.add_argument('--case', type=Path, default=DEFAULT_CASE)
    parser.add_argument('--rerun-local', action='store_true')
    args = parser.parse_args(argv)
    report = build_decomposition_report(args.case, rerun_local=args.rerun_local)
    print(yaml.safe_dump(report, sort_keys=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
