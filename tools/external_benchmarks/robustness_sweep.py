from __future__ import annotations

import argparse
import csv
import math
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
from tools.external_benchmarks.build_zdplaskin_rate_table import (
    DEFAULT_OUTPUT as ZDPLASKIN_WIDE_RATE_TABLE,
    build_zdplaskin_rate_table,
)
from tools.external_benchmarks.same_footing_assessment import build_same_footing_report


OUT_ROOT = ROOT / 'examples' / 'outputs' / 'robustness_sweep'
GENERATED_CONFIG_DIR = OUT_ROOT / '_generated_configs'
DEFAULT_WRITE = OUT_ROOT / 'robustness_report.yaml'
DEFAULT_SUMMARY_CSV = OUT_ROOT / 'robustness_summary.csv'
TORR_TO_PA = 133.3223684
ZDPLASKIN_WIDE_TABLE_RELATIVE = 'tables/zdplaskin_example2_wide_eovern_rates.h5'


class _NoAliasSafeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, Dumper=_NoAliasSafeDumper, sort_keys=False), encoding='utf-8')


def _posix(path: Path) -> str:
    return path.resolve().as_posix()


def _finite_positive(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def _summary_subset(summary: dict[str, Any]) -> dict[str, Any]:
    keys = [
        'success',
        'final_electron_density_m3',
        'final_mean_electron_energy_eV',
        'final_self_bias_V',
        'final_plasma_potential_V',
    ]
    out = {key: summary[key] for key in keys if key in summary}
    for key, value in summary.items():
        if not str(key).startswith('final_port_'):
            continue
        if any(str(key).endswith(suffix) for suffix in ('_absorbed_power_W', '_reduced_field_Td', '_gap_voltage_V', '_current_A')):
            out[key] = value
    if 'warning_counts' in summary:
        out['warning_counts'] = summary['warning_counts']
    peak_ne = 0.0
    max_power = 0.0
    max_field = 0.0
    for block in (summary.get('step_summary') or {}).values():
        if isinstance(block, dict):
            peak_ne = max(peak_ne, float(block.get('max_electron_density_m3', 0.0) or 0.0))
            for key, value in block.items():
                if str(key).endswith('_absorbed_power_W'):
                    max_power = max(max_power, float(value or 0.0))
                if str(key).endswith('_reduced_field_Td'):
                    max_field = max(max_field, float(value or 0.0))
    if peak_ne > 0.0:
        out['peak_electron_density_m3'] = peak_ne
    if max_power > 0.0:
        out['max_absorbed_power_W'] = max_power
    if max_field > 0.0:
        out['max_reduced_field_Td'] = max_field
    return out


def assess_summary(summary: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    checks['solver_success'] = {'passed': bool(summary.get('success', False))}
    ne_ok = _finite_positive(summary.get('final_electron_density_m3'))
    checks['positive_final_electron_density'] = {'passed': ne_ok, 'value': summary.get('final_electron_density_m3')}
    mean_e = summary.get('final_mean_electron_energy_eV')
    mean_e_ok = _finite_positive(mean_e) and float(mean_e) < 100.0
    checks['finite_reasonable_mean_energy'] = {'passed': mean_e_ok, 'value': mean_e}

    numeric_bad = []
    for key, value in _summary_subset(summary).items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            numeric_bad.append(key)
    checks['finite_selected_summary_values'] = {'passed': not numeric_bad, 'bad_keys': numeric_bad}

    return {'passed': all(item['passed'] for item in checks.values()), 'checks': checks}


def _copy_case(case_path: Path, *, case_id: str, chamber_path: Path | None, output_dir: Path) -> dict[str, Any]:
    raw = _read_yaml(case_path)
    raw['include'] = _posix(ROOT / 'examples' / 'configs' / 'base_case.yaml')
    raw.setdefault('case', {})
    raw['case']['name'] = case_id
    tags = list(raw['case'].get('tags', []))
    if 'robustness_sweep' not in tags:
        tags.append('robustness_sweep')
    raw['case']['tags'] = tags
    raw.setdefault('files', {})
    if chamber_path is not None:
        raw['files']['chamber'] = _posix(chamber_path)
    else:
        raw['files']['chamber'] = _posix((case_path.parent / str(raw['files']['chamber'])).resolve())
    raw['files']['recipe'] = _posix((case_path.parent / str(raw['files']['recipe'])).resolve())
    chemistry = raw['files'].setdefault('chemistry', {})
    if chemistry.get('manifest'):
        chemistry['manifest'] = _posix((case_path.parent / str(chemistry['manifest'])).resolve())
    raw['files']['output_dir'] = _posix(output_dir)
    return raw


def ensure_wide_zdplaskin_rate_table() -> Path:
    if not ZDPLASKIN_WIDE_RATE_TABLE.exists():
        build_zdplaskin_rate_table(output=ZDPLASKIN_WIDE_RATE_TABLE)
    return ZDPLASKIN_WIDE_RATE_TABLE


def make_zdplaskin_variant(
    case_id: str,
    *,
    pressure_torr: float | None = None,
    source_voltage_V: float | None = None,
    rate_table_file: str | None = None,
) -> Path:
    base_case = ROOT / 'examples' / 'configs' / 'case_zdplaskin_example2.yaml'
    base_chamber = ROOT / 'examples' / 'configs' / 'chamber_zdplaskin_example2.yaml'
    chamber = _read_yaml(base_chamber)
    if pressure_torr is not None:
        pressure_pa = pressure_torr * TORR_TO_PA
        for zone in chamber.get('zones', []) or []:
            zone['pressure_Pa'] = pressure_pa
        chamber.setdefault('metadata', {})['robustness_pressure_torr'] = pressure_torr
    if source_voltage_V is not None:
        for port in chamber.get('power_ports', []) or []:
            if port.get('port_id') == 'dc_series_drive':
                port.setdefault('parameters', {})['source_voltage_V'] = float(source_voltage_V)
        chamber.setdefault('metadata', {})['robustness_source_voltage_V'] = float(source_voltage_V)

    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    chamber_path = GENERATED_CONFIG_DIR / f'chamber_{case_id}.yaml'
    case_path = GENERATED_CONFIG_DIR / f'case_{case_id}.yaml'
    _write_yaml(chamber_path, chamber)
    case = _copy_case(base_case, case_id=case_id, chamber_path=chamber_path, output_dir=OUT_ROOT / case_id)
    if rate_table_file is not None:
        case.setdefault('swarm', {}).setdefault('table', {})['file'] = rate_table_file
    _write_yaml(case_path, case)
    return case_path


def make_argon_lxcat_pressure_variant(case_id: str, *, pressure_Pa: float) -> Path:
    base_case = ROOT / 'examples' / 'configs' / 'case_argon_lxcat.yaml'
    base_chamber = ROOT / 'examples' / 'configs' / 'chamber_argon_icp.yaml'
    chamber = _read_yaml(base_chamber)
    for zone in chamber.get('zones', []) or []:
        zone['pressure_Pa'] = float(pressure_Pa)
    for pump in chamber.get('pumps', []) or []:
        pump['target_pressure_Pa'] = float(pressure_Pa)
    chamber.setdefault('metadata', {})['robustness_pressure_Pa'] = float(pressure_Pa)

    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    chamber_path = GENERATED_CONFIG_DIR / f'chamber_{case_id}.yaml'
    case_path = GENERATED_CONFIG_DIR / f'case_{case_id}.yaml'
    _write_yaml(chamber_path, chamber)
    case = _copy_case(base_case, case_id=case_id, chamber_path=chamber_path, output_dir=OUT_ROOT / case_id)
    _write_yaml(case_path, case)
    return case_path


def generated_stress_cases() -> list[dict[str, Any]]:
    ensure_wide_zdplaskin_rate_table()
    return [
        {
            'id': 'zdp_pressure_50torr',
            'axis': 'pressure',
            'scope': 'stress_no_external_reference',
            'case': make_zdplaskin_variant(
                'zdp_pressure_50torr',
                pressure_torr=50.0,
                rate_table_file=ZDPLASKIN_WIDE_TABLE_RELATIVE,
            ),
            'notes': ['Uses the wide diagnostic ZDPlaskin-cross-section E/N table; no external 50 torr reference target.'],
        },
        {
            'id': 'zdp_pressure_200torr',
            'axis': 'pressure',
            'scope': 'stress_no_external_reference',
            'case': make_zdplaskin_variant(
                'zdp_pressure_200torr',
                pressure_torr=200.0,
                rate_table_file=ZDPLASKIN_WIDE_TABLE_RELATIVE,
            ),
            'notes': ['Uses the wide diagnostic ZDPlaskin-cross-section E/N table; no external 200 torr reference target.'],
        },
        {
            'id': 'zdp_voltage_500V',
            'axis': 'dc_source_voltage',
            'scope': 'stress_no_external_reference',
            'case': make_zdplaskin_variant(
                'zdp_voltage_500V',
                source_voltage_V=500.0,
                rate_table_file=ZDPLASKIN_WIDE_TABLE_RELATIVE,
            ),
            'notes': ['Uses the wide diagnostic ZDPlaskin-cross-section E/N table; no external 500 V reference target.'],
        },
        {
            'id': 'zdp_voltage_1500V',
            'axis': 'dc_source_voltage',
            'scope': 'stress_no_external_reference',
            'case': make_zdplaskin_variant(
                'zdp_voltage_1500V',
                source_voltage_V=1500.0,
                rate_table_file=ZDPLASKIN_WIDE_TABLE_RELATIVE,
            ),
            'notes': ['Uses the wide diagnostic ZDPlaskin-cross-section E/N table; no external 1500 V reference target.'],
        },
        {
            'id': 'argon_lxcat_pressure_16Pa',
            'axis': 'pressure',
            'scope': 'stress_no_external_reference',
            'case': make_argon_lxcat_pressure_variant('argon_lxcat_pressure_16Pa', pressure_Pa=16.0),
            'notes': ['Pure-Ar LXCat/two-term ICP pressure perturbation; no external reference target.'],
        },
    ]


def fixed_stress_cases() -> list[dict[str, Any]]:
    return [
        {
            'id': 'argon_lxcat_icp_baseline',
            'axis': 'electrical_backend_and_chemistry',
            'scope': 'production_baseline_no_strict_external_reference',
            'case': ROOT / 'examples' / 'configs' / 'case_argon_lxcat.yaml',
            'notes': ['Pure Ar, direct absorbed-power ICP/CCP-style backend, public LXCat-style cross sections.'],
        },
        {
            'id': 'argon_rf_envelope_calibration',
            'axis': 'electrical_backend',
            'scope': 'stress_no_external_reference',
            'case': ROOT / 'examples' / 'configs' / 'case_rf_envelope_calibration.yaml',
            'notes': ['HF/LF RF-envelope backend with calibrated coefficients.'],
        },
        {
            'id': 'cf4_o2_smoke_swarm',
            'axis': 'chemistry',
            'scope': 'smoke_no_external_reference',
            'case': ROOT / 'examples' / 'configs' / 'case_smoke.yaml',
            'notes': ['Mixed Ar/CF4/O2 chemistry smoke test with surfaces and multiple zones.'],
        },
    ]


def _read_observables(output_dir: Path) -> list[dict[str, str]]:
    path = output_dir / 'observables.csv'
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def _rate_table_grid_range(case_path: Path) -> tuple[float, float] | None:
    raw = _read_yaml(case_path)
    table_file = (((raw.get('swarm') or {}).get('table') or {}).get('file'))
    chemistry_manifest = (((raw.get('files') or {}).get('chemistry') or {}).get('manifest'))
    if not table_file or not chemistry_manifest:
        return None
    chemistry_dir = Path(chemistry_manifest).parent
    table_path = Path(table_file)
    if not table_path.is_absolute():
        table_path = chemistry_dir / table_path
    if not table_path.exists():
        return None
    try:
        import h5py
    except ImportError:
        return None
    with h5py.File(table_path, 'r') as h5:
        grid = h5['effective_field_Td'][:]
    return float(min(grid)), float(max(grid))


def _field_coverage(case_path: Path, output_dir: Path) -> dict[str, Any] | None:
    grid_range = _rate_table_grid_range(case_path)
    if grid_range is None:
        return None
    rows = _read_observables(output_dir)
    values: list[float] = []
    for row in rows:
        for key, raw in row.items():
            if key == 'EoverN_plasma_Td' or key.endswith('_reduced_field_Td'):
                try:
                    values.append(float(raw))
                except (TypeError, ValueError):
                    pass
    if not values:
        return {
            'rate_table_EoverN_range_Td': {'min': grid_range[0], 'max': grid_range[1]},
            'observed_EoverN_range_Td': None,
            'inside_table_range': None,
        }
    observed = (min(values), max(values))
    inside = observed[0] >= grid_range[0] and observed[1] <= grid_range[1]
    return {
        'rate_table_EoverN_range_Td': {'min': grid_range[0], 'max': grid_range[1]},
        'observed_EoverN_range_Td': {'min': observed[0], 'max': observed[1]},
        'inside_table_range': inside,
    }


def run_case_record(case_def: dict[str, Any]) -> dict[str, Any]:
    case_path = Path(case_def['case']).resolve()
    try:
        result = run_from_yaml(case_path)
        output_dir = Path(result['output_dir'])
        assessment = assess_summary(result['summary'])
        coverage = _field_coverage(case_path, output_dir)
        warnings: list[str] = []
        if coverage and coverage.get('inside_table_range') is False:
            warnings.append('Observed E/N leaves the active rate-table range; numerical result is a stress check, not a physics validation.')
        return {
            'id': case_def['id'],
            'axis': case_def['axis'],
            'scope': case_def['scope'],
            'case': str(case_path),
            'output_dir': str(output_dir),
            'status': 'pass' if assessment['passed'] else 'fail',
            'passed': assessment['passed'],
            'summary': _summary_subset(result['summary']),
            'checks': assessment['checks'],
            'rate_table_coverage': coverage,
            'notes': list(case_def.get('notes', [])),
            'warnings': warnings,
        }
    except Exception as exc:
        return {
            'id': case_def['id'],
            'axis': case_def['axis'],
            'scope': case_def['scope'],
            'case': str(case_path),
            'status': 'fail',
            'passed': False,
            'error': f'{type(exc).__name__}: {exc}',
            'notes': list(case_def.get('notes', [])),
            'warnings': ['Case execution failed.'],
        }


def _write_summary_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'id',
        'axis',
        'scope',
        'status',
        'passed',
        'final_electron_density_m3',
        'peak_electron_density_m3',
        'final_mean_electron_energy_eV',
        'total_absorbed_power_W',
        'max_absorbed_power_W',
        'field_min_Td',
        'field_max_Td',
        'max_reduced_field_Td',
        'inside_rate_table_range',
    ]
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            summary = rec.get('summary', {}) or {}
            coverage = rec.get('rate_table_coverage') or {}
            observed = coverage.get('observed_EoverN_range_Td') or {}
            powers = [
                float(value)
                for key, value in summary.items()
                if str(key).startswith('final_port_') and str(key).endswith('_absorbed_power_W')
            ]
            writer.writerow(
                {
                    'id': rec.get('id'),
                    'axis': rec.get('axis'),
                    'scope': rec.get('scope'),
                    'status': rec.get('status'),
                    'passed': rec.get('passed'),
                    'final_electron_density_m3': summary.get('final_electron_density_m3'),
                    'peak_electron_density_m3': summary.get('peak_electron_density_m3'),
                    'final_mean_electron_energy_eV': summary.get('final_mean_electron_energy_eV'),
                    'total_absorbed_power_W': sum(powers) if powers else '',
                    'max_absorbed_power_W': summary.get('max_absorbed_power_W', ''),
                    'field_min_Td': observed.get('min', ''),
                    'field_max_Td': observed.get('max', ''),
                    'max_reduced_field_Td': summary.get('max_reduced_field_Td', ''),
                    'inside_rate_table_range': coverage.get('inside_table_range', ''),
                }
            )


def build_robustness_report(*, run_cases: bool = True) -> dict[str, Any]:
    references: list[dict[str, Any]] = []
    if run_cases:
        same_footing = build_same_footing_report(run_cases=True)
        zdp = next(item for item in same_footing['benchmarks'] if item['benchmark'] == 'ZDPlaskin example2')
        references.append(
            {
                'id': 'zdplaskin_example2_same_footing',
                'axis': 'pressure_electrical_chemistry_base_point',
                'scope': 'external_reference',
                'passed': bool(zdp['overall_assessment']['current_local_run_matches_zdplaskin']),
                'summary': zdp['simple_verdict'],
                'outputs': same_footing['outputs'],
            }
        )
        crane = build_crane_report(CRANE_CASE, CRANE_REFERENCE)
        references.append(
            {
                'id': 'crane_two_reaction_argon_ode_parity',
                'axis': 'chemistry',
                'scope': 'external_reference',
                'passed': crane['comparison']['final_electron_density_m3_relative_error'] < 5.0e-4,
                'summary': {
                    'final_electron_density_relative_error': crane['comparison']['final_electron_density_m3_relative_error'],
                    'final_Ar_plus_density_relative_error': crane['comparison']['final_Ar_plus_density_m3_relative_error'],
                    'final_Ar_density_relative_error': crane['comparison']['final_Ar_density_m3_relative_error'],
                },
            }
        )

    stress_cases = generated_stress_cases() + fixed_stress_cases()
    stress_results = [run_case_record(case_def) for case_def in stress_cases] if run_cases else []
    passed = all(item.get('passed', False) for item in references + stress_results)
    warning_cases = [
        {
            'id': item['id'],
            'warnings': item.get('warnings', []),
            'rate_table_coverage': item.get('rate_table_coverage'),
        }
        for item in stress_results
        if item.get('warnings')
    ]
    return {
        'tool': 'robustness_sweep',
        'scope': (
            'External references are strict where available. Pressure/electrical stress cases without external reference check '
            'solver health and closure range only; they are not accuracy validation.'
        ),
        'passed': passed,
        'external_reference_results': references,
        'stress_results': stress_results,
        'warning_cases': warning_cases,
        'conclusion': {
            'external_reference_count': len(references),
            'stress_case_count': len(stress_results),
            'failed_cases': [item['id'] for item in references + stress_results if not item.get('passed', False)],
            'main_limitation': (
                'The ZDPlaskin parity table remains tied to one saved discharge. Wide-table stress cases now avoid immediate E/N range clipping, '
                'but they still need external reference outputs before they can be accuracy benchmarks.'
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Run external-reference and stress benchmarks across pressure, electrical backend, and chemistry axes.')
    parser.add_argument('--write', type=Path, default=DEFAULT_WRITE)
    parser.add_argument('--summary-csv', type=Path, default=DEFAULT_SUMMARY_CSV)
    args = parser.parse_args()
    report = build_robustness_report(run_cases=True)
    _write_yaml(args.write.resolve(), report)
    _write_summary_csv(args.summary_csv.resolve(), report.get('stress_results', []))
    print(yaml.safe_dump(report, sort_keys=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
