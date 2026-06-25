from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plasma_global.eedf.base import EEDFRequest
from plasma_global.eedf.table import TabulatedSwarmModel
from plasma_global.workflows.context import load_case_from_yaml
from plasma_global.workflows.runner import run_from_yaml
from tools.external_benchmarks.crane_two_reaction_argon import (
    DEFAULT_CASE as CRANE_CASE,
    DEFAULT_REFERENCE as CRANE_REFERENCE,
    build_report as build_crane_report,
)
from tools.external_benchmarks.pygmol_argon import (
    DEFAULT_CASE as PYGMOL_CASE,
    build_report as build_pygmol_report,
)
from tools.external_benchmarks.pygmol_precision import build_precision_report as build_pygmol_precision_report
from tools.external_benchmarks.plot_pygmol_benchmarks import build_pygmol_graphs
from tools.external_benchmarks.zdplaskin_eovern import build_eovern_report
from tools.external_benchmarks.zdplaskin_example2 import (
    DEFAULT_OUTPUT_DIR as ZDP_OUTPUT_DIR,
    DEFAULT_REFERENCE as ZDP_REFERENCE,
    build_report as build_zdp_report,
)


DEFAULT_MANIFEST = ROOT / 'tools' / 'external_benchmarks' / 'manifest.yaml'
DEFAULT_OUTPUT_DIR = ROOT / 'examples' / 'outputs' / 'external_benchmarks' / 'diagnostic_suite'
ZDP_CASE = ROOT / 'examples' / 'configs' / 'case_zdplaskin_example2.yaml'
ZDP_CIRCUIT_CSV = ROOT / 'examples' / 'external' / 'zdplaskin_example2_circuit.csv'
ZDP_RATE_TABLE = ROOT / 'examples' / 'chemistry_zdplaskin_example2' / 'tables' / 'zdplaskin_example2_eovern_rates.h5'


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _nested_get(mapping: dict[str, Any], dotted_path: str) -> Any:
    current: Any = mapping
    for part in dotted_path.split('.'):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_path)
        current = current[part]
    return current


def _ratio(local: float, reference: float) -> float | None:
    if reference == 0.0:
        return None
    return local / reference


def _relative_error_from_ratio(ratio: float | None) -> float | None:
    if ratio is None:
        return None
    return abs(float(ratio) - 1.0)


def _finite_positive(value: Any) -> bool:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(candidate) and candidate >= 0.0


def _status_for_value(value: Any, threshold: dict[str, Any] | None) -> str:
    if threshold is None:
        return 'info'
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(threshold.get('status_on_fail', 'fail'))
    if not math.isfinite(number):
        return str(threshold.get('status_on_fail', 'fail'))
    if 'min' in threshold and number < float(threshold['min']):
        return str(threshold.get('status_on_fail', 'fail'))
    if 'max' in threshold and number > float(threshold['max']):
        return str(threshold.get('status_on_fail', 'fail'))
    return 'pass'


def _format_threshold(threshold: dict[str, Any] | None) -> str:
    if threshold is None:
        return ''
    parts = []
    if 'min' in threshold:
        parts.append(f">={threshold['min']}")
    if 'max' in threshold:
        parts.append(f"<={threshold['max']}")
    return ' and '.join(parts)


def _severity(status: str, claim_level: str) -> str:
    if status == 'pass':
        return 'none'
    if status == 'info':
        return 'info'
    if status == 'action_required':
        return 'medium'
    if claim_level == 'strong':
        return 'high'
    if claim_level == 'scoped':
        return 'medium'
    return 'low'


def load_benchmark_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest = _load_yaml(path)
    if int(manifest.get('version', 0)) != 2:
        raise ValueError(f'External benchmark manifest must use version 2: {path}')
    benchmarks = manifest.get('benchmarks') or []
    ids = [str(item.get('id')) for item in benchmarks]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise ValueError(f'Duplicate benchmark ids in manifest: {duplicates}')
    categories = set((manifest.get('diagnostic_categories') or {}).keys())
    for bench in benchmarks:
        metric_ids = [str(metric.get('id')) for metric in bench.get('metrics') or []]
        dup_metrics = sorted({item for item in metric_ids if metric_ids.count(item) > 1})
        if dup_metrics:
            raise ValueError(f'Duplicate metric ids in {bench.get("id")}: {dup_metrics}')
        for metric in bench.get('metrics') or []:
            category = metric.get('diagnostic_category')
            if category not in categories:
                raise ValueError(f'Metric {bench.get("id")}.{metric.get("id")} references unknown diagnostic category {category!r}')
    return manifest


def _benchmark_by_id(manifest: dict[str, Any], benchmark_id: str) -> dict[str, Any]:
    for benchmark in manifest['benchmarks']:
        if benchmark['id'] == benchmark_id:
            return benchmark
    raise KeyError(benchmark_id)


def _diagnostic(manifest: dict[str, Any], category: str) -> dict[str, Any]:
    return manifest['diagnostic_categories'][category]


def _metric_row(
    manifest: dict[str, Any],
    benchmark: dict[str, Any],
    metric: dict[str, Any],
    value: Any,
    *,
    observed: Any = '',
    reference: Any = '',
    note: str = '',
) -> dict[str, Any]:
    threshold = metric.get('threshold')
    status = _status_for_value(value, threshold)
    category = str(metric['diagnostic_category'])
    diag = _diagnostic(manifest, category)
    return {
        'benchmark_id': benchmark['id'],
        'software': benchmark['software'],
        'problem': benchmark['problem'],
        'metric_id': metric['id'],
        'group': metric.get('group', ''),
        'claim_level': benchmark.get('claim_level', ''),
        'value': value,
        'observed': observed,
        'reference': reference,
        'threshold': _format_threshold(threshold),
        'status': status,
        'severity': _severity(status, str(benchmark.get('claim_level', ''))),
        'diagnostic_category': category,
        'suspected_modules': '; '.join(diag.get('suspected_modules') or []),
        'recommendation': diag.get('recommendation', ''),
        'note': note,
    }


def _evaluate_manifest_metrics(
    manifest: dict[str, Any],
    benchmark: dict[str, Any],
    report: dict[str, Any],
    generated: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for metric in benchmark.get('metrics') or []:
        if metric.get('value_path'):
            value = _nested_get(report, str(metric['value_path']))
        else:
            value = generated.get(str(metric.get('generated')))
        rows.append(_metric_row(manifest, benchmark, metric, value))
    return rows


def evaluate_crane(manifest: dict[str, Any], output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    benchmark = _benchmark_by_id(manifest, 'CRANE-1')
    report = build_crane_report(CRANE_CASE, CRANE_REFERENCE)
    report_path = output_dir / 'reports' / 'CRANE-1_report.yaml'
    _write_yaml(report_path, report)
    local = report['local_result']
    charge_ref = max(abs(float(local['final_electron_density_m3'])), 1.0)
    generated = {
        'crane_charge_balance_relative_error': abs(float(local['final_Ar_plus_density_m3']) - float(local['final_electron_density_m3'])) / charge_ref,
    }
    rows = _evaluate_manifest_metrics(manifest, benchmark, report, generated)
    return {'report_path': str(report_path), 'report': report}, rows


def _nrmse(reference: np.ndarray, candidate: np.ndarray) -> float:
    mask = np.isfinite(reference) & np.isfinite(candidate)
    if not np.any(mask):
        return float('nan')
    ref = reference[mask]
    cand = candidate[mask]
    denom = max(float(np.nanmax(ref) - np.nanmin(ref)), abs(float(np.nanmean(ref))), 1.0e-30)
    return float(np.sqrt(np.mean((cand - ref) ** 2)) / denom)


def _zdp_circuit_nrmse(output_dir: Path) -> float:
    external = pd.read_csv(ZDP_CIRCUIT_CSV)
    local = pd.read_csv(output_dir / 'observables.csv')
    t = external['time_s'].to_numpy(dtype=float)
    local_t = local['time_s'].to_numpy(dtype=float)
    local_eovern = np.interp(t, local_t, local['EoverN_plasma_Td'].to_numpy(dtype=float))
    return _nrmse(external['reduced_field_Td'].to_numpy(dtype=float), local_eovern)


def _write_zdp_peak_window_budget(output_dir: Path, *, center_s: float, half_width_s: float = 2.0e-5) -> Path:
    observables = pd.read_csv(ZDP_OUTPUT_DIR / 'observables.csv')
    mask = (observables['time_s'] >= center_s - half_width_s) & (observables['time_s'] <= center_s + half_width_s)
    window = observables.loc[mask].copy()
    if window.empty:
        nearest = int((observables['time_s'] - center_s).abs().idxmin())
        window = observables.iloc[[nearest]].copy()
    prefixes = (
        'time_s',
        'ne_',
        'mean_energy_',
        'EoverN_',
        'pabs_',
        'rate_table_',
        'electrical_',
        'reaction_rate_',
        'species_source_',
        'species_loss_',
        'wall_loss_',
        'electron_energy_loss_',
    )
    selected = [column for column in window.columns if column == 'time_s' or column.startswith(prefixes)]
    path = output_dir / 'tables' / 'ZDPlaskin-1_peak_window_budget.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    window[selected].to_csv(path, index=False)
    return path


def evaluate_zdplaskin_1(manifest: dict[str, Any], output_dir: Path, *, run_case: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    benchmark = _benchmark_by_id(manifest, 'ZDPlaskin-1')
    if run_case:
        run_from_yaml(ZDP_CASE)
    report = build_zdp_report(ZDP_OUTPUT_DIR, ZDP_REFERENCE)
    eovern_report = build_eovern_report(ZDP_OUTPUT_DIR, ZDP_REFERENCE)
    report['reduced_field_physical_comparison'] = eovern_report
    peak_budget_path = _write_zdp_peak_window_budget(
        output_dir,
        center_s=float(report['zdplaskin_reference']['peak_saved_electron_density_time_s']),
    )
    report['peak_window_budget_csv'] = str(peak_budget_path)
    report_path = output_dir / 'reports' / 'ZDPlaskin-1_report.yaml'
    _write_yaml(report_path, report)
    comparison = report['comparison']
    eovern_comparison = eovern_report['comparison']
    local_columns = set(pd.read_csv(ZDP_OUTPUT_DIR / 'observables.csv', nrows=0).columns)
    expected_electrical = {
        'electrical_dc_series_drive_source_voltage_V',
        'electrical_dc_series_drive_gap_voltage_V',
        'electrical_dc_series_drive_current_A',
        'electrical_dc_series_drive_plasma_conductance_S',
        'electrical_dc_series_drive_reduced_field_Td',
    }
    generated = {
        'zdp_final_e_relative_error': _relative_error_from_ratio(comparison['final_electron_density_ratio_local_over_zdplaskin']),
        'zdp_final_Ar_star_relative_error': _relative_error_from_ratio(comparison['final_Ar_star_ratio_local_over_zdplaskin']),
        'zdp_final_Ar_plus_relative_error': _relative_error_from_ratio(comparison['final_Ar_plus_ratio_local_over_zdplaskin']),
        'zdp_final_Ar2_plus_relative_error': _relative_error_from_ratio(comparison['final_Ar2_plus_ratio_local_over_zdplaskin']),
        'zdp_peak_e_relative_error': _relative_error_from_ratio(comparison['peak_electron_density_ratio_local_over_zdplaskin']),
        'zdp_final_EoverN_relative_error': _relative_error_from_ratio(comparison['final_reduced_field_ratio_local_over_zdplaskin']),
        'zdp_final_power_relative_error': _relative_error_from_ratio(eovern_comparison['absorbed_power_over_zdplaskin_VI_power_ratio']),
        'zdp_circuit_EoverN_waveform_nrmse': _zdp_circuit_nrmse(ZDP_OUTPUT_DIR),
        'zdp_reaction_budget_columns_present': float(
            any(col.startswith('reaction_rate_plasma_') for col in local_columns)
            and any(col.startswith('species_source_plasma_') for col in local_columns)
            and any(col.startswith('species_loss_plasma_') for col in local_columns)
        ),
        'zdp_electrical_waveform_columns_present': float(expected_electrical.issubset(local_columns)),
        'zdp_peak_window_budget_generated': float(peak_budget_path.exists() and peak_budget_path.stat().st_size > 0),
    }
    rows = _evaluate_manifest_metrics(manifest, benchmark, report, generated)
    return {'report_path': str(report_path), 'report': report}, rows


def _required_cross_section_ids(loaded: Any) -> set[str]:
    required: set[str] = set()
    for reaction in loaded.mechanism.gas_reactions:
        if not getattr(reaction, 'enabled', True):
            continue
        model = loaded.mechanism.rate_models.get(reaction.rate_model_key, {})
        if str(model.get('backend', '')).lower() == 'electron_impact_xsec' and model.get('cross_section_id'):
            required.add(str(model['cross_section_id']))
    return required


def _rate_table_report() -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        return {'error': f'h5py missing: {exc}', 'metrics': {}}

    loaded = load_case_from_yaml(ZDP_CASE)
    required = _required_cross_section_ids(loaded)
    metrics: dict[str, Any] = {}
    details: dict[str, Any] = {'table': str(ZDP_RATE_TABLE), 'required_cross_section_ids': sorted(required)}
    with h5py.File(ZDP_RATE_TABLE, 'r') as h5:
        required_datasets = {'mean_energy_eV', 'effective_field_Td', 'mobility_m2_V_s', 'diffusion_m2_s', 'rate_coefficients'}
        metrics['zdp_table_required_datasets_present'] = float(required_datasets.issubset(h5.keys()))
        field = np.asarray(h5['effective_field_Td'][:], dtype=float)
        metrics['zdp_table_field_axis_strictly_increasing'] = float(bool(np.all(np.diff(field) > 0.0)))
        available_rates = set(h5['rate_coefficients'].keys()) if 'rate_coefficients' in h5 else set()
        metrics['zdp_table_required_rates_present'] = float(required.issubset(available_rates))
        final_eovern = float(_load_yaml(ZDP_REFERENCE)['saved_output']['final_saved_reduced_field_Td'])
        metrics['zdp_table_final_field_inside_table'] = float(float(np.nanmin(field)) <= final_eovern <= float(np.nanmax(field)))
        details.update({
            'field_min_Td': float(np.nanmin(field)),
            'field_max_Td': float(np.nanmax(field)),
            'final_reference_EoverN_Td': final_eovern,
            'available_rate_ids': sorted(available_rates),
        })

    model = TabulatedSwarmModel()
    model.prepare(
        mechanism=loaded.mechanism,
        chamber=loaded.chamber,
        run_config=loaded.run_config,
        resolved_paths=loaded.resolved_paths,
        swarm_config=loaded.run_config.swarm,
    )
    result = model.evaluate(
        EEDFRequest(
            time_s=0.0,
            zone_id='plasma',
            composition={'Ar': 1.0},
            electron_density_m3=1.0e16,
            mean_energy_eV=3.0,
            reduced_field_Td=float(details['final_reference_EoverN_Td']),
            gas_temperature_K=300.0,
            pressure_Pa=13332.0,
        )
    )
    clipped_result = model.evaluate(
        EEDFRequest(
            time_s=0.0,
            zone_id='plasma',
            composition={'Ar': 1.0},
            electron_density_m3=1.0e16,
            mean_energy_eV=3.0,
            reduced_field_Td=float(details['field_max_Td']) * 10.0,
            gas_temperature_K=300.0,
            pressure_Pa=13332.0,
        )
    )
    finite_rates = all(_finite_positive(value) for value in result.rate_coefficients.values())
    finite_transport = all(
        _finite_positive(value)
        for value in [
            result.transport.mean_energy_eV,
            result.transport.mobility_m2_V_s,
            result.transport.diffusion_m2_s,
            result.transport.effective_field_Td,
        ]
    )
    metrics['zdp_table_local_field_interpolation_finite'] = float(finite_rates and finite_transport)
    diagnostics = result.diagnostics
    clipping_diagnostics = clipped_result.diagnostics
    required_diag_keys = {
        'backend',
        'table_path',
        'grid_column',
        'lookup_mode',
        'lookup_value',
        'lookup_clipped_value',
        'axis_min',
        'axis_max',
        'lookup_clipped',
    }
    metrics['zdp_table_lookup_diagnostics_present'] = float(required_diag_keys.issubset(diagnostics))
    metrics['zdp_table_clipping_diagnostics_present'] = float(
        bool(clipping_diagnostics.get('lookup_clipped'))
        and bool(clipping_diagnostics.get('lookup_clipped_high'))
        and float(clipping_diagnostics.get('lookup_clipped_value', 0.0)) == float(details['field_max_Td'])
    )
    details['interpolated_lookup_mode'] = result.transport.lookup_mode
    details['interpolated_transport'] = {
        'mean_energy_eV': result.transport.mean_energy_eV,
        'mobility_m2_V_s': result.transport.mobility_m2_V_s,
        'diffusion_m2_s': result.transport.diffusion_m2_s,
        'effective_field_Td': result.transport.effective_field_Td,
    }
    details['interpolated_lookup_diagnostics'] = diagnostics
    details['out_of_range_lookup_diagnostics'] = clipping_diagnostics
    details['interpolated_rate_ids'] = sorted(result.rate_coefficients)
    return {'metrics': metrics, 'details': details}


def evaluate_zdplaskin_2(manifest: dict[str, Any], output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    benchmark = _benchmark_by_id(manifest, 'ZDPlaskin-2')
    report = _rate_table_report()
    report_path = output_dir / 'reports' / 'ZDPlaskin-2_report.yaml'
    _write_yaml(report_path, report)
    rows = _evaluate_manifest_metrics(manifest, benchmark, report, report.get('metrics', {}))
    return {'report_path': str(report_path), 'report': report}, rows


def evaluate_pygmol_1(manifest: dict[str, Any], output_dir: Path, *, rerun_local: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    benchmark = _benchmark_by_id(manifest, 'PyGMol-1')
    try:
        report = build_pygmol_report(PYGMOL_CASE, rerun_local=rerun_local, write_solution_csv=True)
        powered = [row for row in report['rows'] if float(row.get('local_mean_absorbed_power_W') or 0.0) > 0.0]
        ratios = [
            float(row['ratio_final_ne_pygmol_over_local'])
            for row in powered
            if row.get('ratio_final_ne_pygmol_over_local') not in ('', None)
        ]
        generated = {
            'pygmol_powered_final_ne_ratio_min': min(ratios) if ratios else float('nan'),
            'pygmol_powered_final_ne_ratio_max': max(ratios) if ratios else float('nan'),
            'pygmol_precision_claim_scoped_out': 1.0,
        }
        note = 'Current PyGMol comparison is a sanity comparison; precision parity is intentionally not claimed.'
    except RuntimeError as exc:
        report = {'skipped': True, 'skip_reason': str(exc)}
        generated = {
            'pygmol_powered_final_ne_ratio_min': float('nan'),
            'pygmol_powered_final_ne_ratio_max': float('nan'),
            'pygmol_precision_claim_scoped_out': 1.0,
        }
        note = str(exc)
    report_path = output_dir / 'reports' / 'PyGMol-1_report.yaml'
    _write_yaml(report_path, report)
    rows = []
    for metric in benchmark.get('metrics') or []:
        rows.append(_metric_row(manifest, benchmark, metric, generated.get(metric.get('generated')), note=note))
    return {'report_path': str(report_path), 'report': report}, rows


def evaluate_pygmol_precision_1(manifest: dict[str, Any], output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    benchmark = _benchmark_by_id(manifest, 'PyGMol-Precision-1')
    try:
        report = build_pygmol_precision_report(PYGMOL_CASE, output_dir=output_dir / 'pygmol_precision')
        metrics = report.get('metrics', {})
        generated = {f'pygmol_precision_{key}': value for key, value in metrics.items()}
        note = report.get('interpretation', '')
    except RuntimeError as exc:
        report = {'skipped': True, 'skip_reason': str(exc)}
        generated = {
            'pygmol_precision_same_footing_axes_aligned': 0.0,
            'pygmol_precision_max_final_relative_error': float('nan'),
            'pygmol_precision_max_waveform_nrmse': float('nan'),
            'pygmol_precision_e_waveform_nrmse': float('nan'),
            'pygmol_precision_final_T_e_relative_error': float('nan'),
        }
        note = str(exc)
    report_path = output_dir / 'reports' / 'PyGMol-Precision-1_report.yaml'
    _write_yaml(report_path, report)
    rows = _evaluate_manifest_metrics(manifest, benchmark, report, generated)
    for row in rows:
        row['note'] = note
    return {'report_path': str(report_path), 'report': report}, rows


def evaluate_swarm_1(manifest: dict[str, Any], output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    benchmark = _benchmark_by_id(manifest, 'SWARM-1')
    report = _rate_table_report()
    metrics = report.get('metrics', {})
    metrics['swarm_table_reader_prepares'] = metrics.get('zdp_table_local_field_interpolation_finite', 0.0)
    metrics['swarm_lookup_mode_is_field'] = float(report.get('details', {}).get('interpolated_lookup_mode') == 'field')
    transport = report.get('details', {}).get('interpolated_transport', {})
    metrics['swarm_transport_values_finite'] = float(all(_finite_positive(value) for value in transport.values()))
    metrics['swarm_rate_values_finite'] = metrics.get('zdp_table_local_field_interpolation_finite', 0.0)
    metrics['swarm_lookup_diagnostics_present'] = metrics.get('zdp_table_lookup_diagnostics_present', 0.0)
    metrics['swarm_clipping_diagnostics_present'] = metrics.get('zdp_table_clipping_diagnostics_present', 0.0)
    report_path = output_dir / 'reports' / 'SWARM-1_report.yaml'
    _write_yaml(report_path, report)
    rows = _evaluate_manifest_metrics(manifest, benchmark, report, metrics)
    return {'report_path': str(report_path), 'report': report}, rows


def _runtime_case(case_id: str, case_path: Path, repetitions: int) -> dict[str, Any]:
    runs = []
    for rep in range(repetitions):
        start = time.perf_counter()
        result = run_from_yaml(case_path)
        wall_time = time.perf_counter() - start
        diagnostics = result['solution'].diagnostics
        runs.append({
            'rep_index': rep + 1,
            'run_kind': 'cold' if rep == 0 else 'warm',
            'wall_time_s': wall_time,
            'success': bool(result['solution'].success),
            'nfev': diagnostics.get('nfev'),
            'njev': diagnostics.get('njev'),
            'nlu': diagnostics.get('nlu'),
            'n_times': int(len(result['solution'].t)),
        })
    wall_times = [float(run['wall_time_s']) for run in runs]
    nfev = [float(run['nfev'] or 0.0) for run in runs]
    nlu = [float(run['nlu'] or 0.0) for run in runs]
    return {
        'case_id': case_id,
        'case_path': str(case_path),
        'runs': runs,
        'aggregate': {
            'median_wall_time_s': statistics.median(wall_times),
            'min_wall_time_s': min(wall_times),
            'max_wall_time_s': max(wall_times),
            'warm_median_wall_time_s': statistics.median(wall_times[1:]) if len(wall_times) > 1 else None,
            'nfev_median': statistics.median(nfev),
            'nlu_median': statistics.median(nlu),
        },
    }


def evaluate_runtime_1(manifest: dict[str, Any], output_dir: Path, *, repetitions: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    benchmark = _benchmark_by_id(manifest, 'Runtime-1')
    cases = [
        ('CRANE-1', CRANE_CASE),
        ('ZDPlaskin-1', ZDP_CASE),
        ('PyGMol-local', PYGMOL_CASE),
    ]
    report = {
        'repetitions': repetitions,
        'cases': [_runtime_case(case_id, case_path, repetitions) for case_id, case_path in cases],
    }
    report_path = output_dir / 'runtime_profile.yaml'
    _write_yaml(report_path, report)
    rows = []
    for case in report['cases']:
        generated = {
            'runtime_median_wall_time_s': case['aggregate']['median_wall_time_s'],
            'runtime_nfev_median': case['aggregate']['nfev_median'],
        }
        for metric in benchmark.get('metrics') or []:
            row = _metric_row(
                manifest,
                benchmark,
                metric,
                generated.get(metric.get('generated')),
                note=f"runtime case: {case['case_id']}",
            )
            row['benchmark_id'] = f"Runtime-1:{case['case_id']}"
            rows.append(row)
    return {'report_path': str(report_path), 'report': report}, rows


def _plot_metric_bar(path: Path, title: str, rows: list[dict[str, Any]], *, xlabel: str = 'metric value') -> None:
    plot_rows = [row for row in rows if row['status'] not in {'info'}]
    if not plot_rows:
        return
    labels = [str(row['metric_id']) for row in plot_rows]
    values = [max(abs(float(row['value'])), 1.0e-15) for row in plot_rows]
    colors = [
        '#59a14f' if row['status'] == 'pass' else '#f28e2b' if row['status'] == 'warn' else '#e15759'
        for row in plot_rows
    ]
    fig, ax = plt.subplots(figsize=(10.5, max(4.0, 0.42 * len(labels) + 1.5)))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis='x', which='both', alpha=0.25)
    for bar, row in zip(bars, plot_rows, strict=True):
        ax.text(
            max(abs(float(row['value'])), 1.0e-15) * 1.1,
            bar.get_y() + bar.get_height() / 2.0,
            f"{float(row['value']):.3g} [{row['status']}]",
            va='center',
            fontsize=8.5,
        )
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _threshold_max(row: dict[str, Any]) -> float | None:
    for part in str(row.get('threshold', '')).split('and'):
        stripped = part.strip()
        if not stripped.startswith('<='):
            continue
        try:
            return float(stripped[2:])
        except ValueError:
            return None
    return None


def _as_finite_float(value: Any) -> float | None:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    return candidate if math.isfinite(candidate) else None


def _positive_array(values: Any) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=float), 1.0e-300)


def _read_solution_h5(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    try:
        import h5py
    except ImportError:  # pragma: no cover
        return None
    if not path.exists():
        return None
    with h5py.File(path, 'r') as h5:
        labels = [label.decode() if isinstance(label, bytes) else str(label) for label in h5['state_labels'][:]]
        time_s = np.asarray(h5['t_s'][:], dtype=float)
        values = np.asarray(h5['y'][:], dtype=float)
    return time_s, {label: values[index] for index, label in enumerate(labels)}


def _metric_label(row: dict[str, Any]) -> str:
    metric = str(row['metric_id'])
    metric = metric.replace('_relative_error', ' rel. error')
    metric = metric.replace('_waveform_nrmse', ' waveform NRMSE')
    metric = metric.replace('_', ' ')
    return f"{row['benchmark_id']}\n{metric}"


def _plot_acceptance_margin(path: Path, rows: list[dict[str, Any]]) -> None:
    plot_rows = []
    for row in rows:
        metric_id = str(row['metric_id']).lower()
        if 'relative_error' not in metric_id and 'nrmse' not in metric_id:
            continue
        threshold = _threshold_max(row)
        value = _as_finite_float(row.get('value'))
        if threshold is None or threshold <= 0.0 or value is None:
            continue
        plot_rows.append((row, max(abs(value) / threshold, 1.0e-12), threshold))
    if not plot_rows:
        return

    plot_rows.sort(key=lambda item: item[1], reverse=True)
    labels = [_metric_label(row) for row, _usage, _threshold in plot_rows]
    values = [usage for _row, usage, _threshold in plot_rows]
    colors = [
        '#59a14f' if row['status'] == 'pass' else '#f28e2b' if row['status'] == 'warn' else '#e15759'
        for row, _usage, _threshold in plot_rows
    ]

    fig, ax = plt.subplots(figsize=(11.4, max(5.0, 0.38 * len(labels) + 1.8)))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors)
    ax.axvspan(1.0e-12, 1.0, color='#59a14f', alpha=0.10, label='within acceptance threshold')
    ax.axvline(1.0, color='#222222', lw=1.2, ls='--', label='threshold')
    ax.set_xscale('log')
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel('Observed error / acceptance threshold')
    ax.set_title('Cross-Benchmark Validity Margin')
    ax.grid(axis='x', which='both', alpha=0.24)
    ax.legend(loc='lower right', fontsize=8)
    for bar, (row, usage, threshold) in zip(bars, plot_rows, strict=True):
        value = float(row['value'])
        ax.text(
            max(usage, 1.0e-12) * 1.12,
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.3g} / {threshold:.3g} [{row['status']}]",
            va='center',
            fontsize=8.2,
        )
    fig.text(
        0.01,
        0.01,
        'Only error and NRMSE metrics with explicit upper thresholds are shown; lower is better.',
        fontsize=8,
        color='#555555',
    )
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _base_benchmark_id(row: dict[str, Any]) -> str:
    return str(row['benchmark_id']).split(':')[0]


def _claim_rank(claim_level: str) -> int:
    return {'context': 1, 'sanity': 2, 'scoped': 3, 'strong': 4}.get(str(claim_level), 1)


def _facet_for(row: dict[str, Any]) -> str | None:
    benchmark = _base_benchmark_id(row)
    group = str(row.get('group', ''))
    category = str(row.get('diagnostic_category', ''))
    if benchmark == 'CRANE-1' and group == 'species_final':
        return 'ODE/SI final species'
    if group == 'conservation':
        return 'charge conservation'
    if benchmark == 'ZDPlaskin-1' and group == 'species_final':
        return 'final species'
    if benchmark == 'ZDPlaskin-1' and group in {'species_transient', 'circuit_transient'}:
        return 'transient / waveform'
    if benchmark == 'ZDPlaskin-1' and group == 'circuit_final':
        return 'circuit final'
    if group in {'schema', 'grid'}:
        return 'rate-table schema/grid'
    if group in {'diagnostics', 'model_diagnostics'}:
        return 'diagnostics'
    if group in {'backend', 'interpolation'} or category in {'zdplaskin_rate_table', 'swarm_table_ingestion'}:
        return 'table backend/interp.'
    if benchmark == 'PyGMol-1' and group == 'powered_sanity':
        return 'PyGMol production sanity'
    if benchmark == 'PyGMol-1' and group == 'claim_scope':
        return 'claim scoping'
    if benchmark == 'PyGMol-Precision-1':
        return 'PyGMol same-footing parity'
    return None


def _plot_claim_support_matrix(path: Path, rows: list[dict[str, Any]]) -> None:
    facets = [
        'ODE/SI final species',
        'charge conservation',
        'final species',
        'transient / waveform',
        'circuit final',
        'rate-table schema/grid',
        'table backend/interp.',
        'diagnostics',
        'PyGMol production sanity',
        'claim scoping',
        'PyGMol same-footing parity',
    ]
    benchmarks = [
        benchmark
        for benchmark in dict.fromkeys(_base_benchmark_id(row) for row in rows)
        if benchmark != 'Runtime-1'
    ]
    if not benchmarks:
        return

    values = np.zeros((len(benchmarks), len(facets)), dtype=float)
    labels = np.full((len(benchmarks), len(facets)), '', dtype=object)
    benchmark_index = {name: i for i, name in enumerate(benchmarks)}
    facet_index = {name: i for i, name in enumerate(facets)}
    software_by_benchmark = {
        _base_benchmark_id(row): str(row['software'])
        for row in rows
        if _base_benchmark_id(row) in benchmark_index
    }

    for row in rows:
        benchmark = _base_benchmark_id(row)
        facet = _facet_for(row)
        if benchmark not in benchmark_index or facet not in facet_index:
            continue
        i = benchmark_index[benchmark]
        j = facet_index[facet]
        if row['status'] not in {'pass', 'info'}:
            values[i, j] = 5.0
            labels[i, j] = str(row['status'])
            continue
        if values[i, j] == 5.0:
            continue
        rank = _claim_rank(str(row.get('claim_level', '')))
        if rank > values[i, j]:
            values[i, j] = float(rank)
            labels[i, j] = str(row.get('claim_level', ''))

    cmap = matplotlib.colors.ListedColormap(['#eeeeee', '#d9e8f5', '#f1ce63', '#8cd17d', '#59a14f', '#e15759'])
    fig, ax = plt.subplots(figsize=(13.5, max(4.2, 0.58 * len(benchmarks) + 1.8)))
    ax.imshow(values, cmap=cmap, vmin=0, vmax=5, aspect='auto')
    x_labels = [label.replace(' ', '\n') for label in facets]
    y_labels = [f'{benchmark}\n{software_by_benchmark.get(benchmark, "")}' for benchmark in benchmarks]
    ax.set_xticks(np.arange(len(facets)), x_labels, fontsize=8)
    ax.set_yticks(np.arange(len(benchmarks)), y_labels)
    ax.set_title('External-Benchmark Evidence Map')
    ax.set_xticks(np.arange(-0.5, len(facets), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(benchmarks), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.8)
    ax.tick_params(which='minor', bottom=False, left=False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text = str(labels[i, j])
            if text:
                ax.text(j, i, text, ha='center', va='center', fontsize=7.4, color='#222222')
    fig.text(
        0.01,
        0.01,
        'Cell text is the supported claim level; red cells mark threshold misses. Blank cells are not claimed by that benchmark.',
        fontsize=8,
        color='#555555',
    )
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _plot_software_status(path: Path, rows: list[dict[str, Any]]) -> None:
    software_order = [
        software
        for software in dict.fromkeys(str(row['software']) for row in rows)
        if software != 'this code'
    ]
    if not software_order:
        return
    statuses = ['pass', 'fail', 'warn', 'action_required']
    counts = {software: {status: 0 for status in statuses} for software in software_order}
    for row in rows:
        software = str(row['software'])
        status = str(row['status'])
        if software not in counts or status == 'info':
            continue
        if status not in counts[software]:
            counts[software][status] = 0
        counts[software][status] += 1

    colors = {'pass': '#59a14f', 'fail': '#e15759', 'warn': '#f28e2b', 'action_required': '#b07aa1'}
    x = np.arange(len(software_order))
    bottom = np.zeros(len(software_order), dtype=float)
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    for status in statuses:
        values = np.asarray([counts[software].get(status, 0) for software in software_order], dtype=float)
        ax.bar(x, values, bottom=bottom, color=colors[status], label=status.replace('_', ' '))
        bottom += values
    x_labels = [software.replace('-compatible table', '\ncompatible table') for software in software_order]
    ax.set_xticks(x, x_labels, rotation=10, ha='right')
    ax.set_ylabel('Metric count')
    ax.set_title('Benchmark Result Count by External Software')
    ax.grid(axis='y', alpha=0.24)
    ax.legend(frameon=False, ncols=4, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 1.02))
    ax.set_ylim(0.0, max(float(np.max(bottom)) * 1.18, 1.0))
    for idx, total in enumerate(bottom):
        ax.text(idx, total + max(float(np.max(bottom)) * 0.025, 0.18), f'{int(total)} metrics', ha='center', va='bottom', fontsize=8.3)
    fig.subplots_adjust(bottom=0.25, top=0.86)
    fig.text(
        0.01,
        0.01,
        'Counts include scoped sanity, schema, interpolation, final-value, waveform, and diagnostic metrics from the current manifest.',
        fontsize=8,
        color='#555555',
    )
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _plot_crane_density_timeseries(path: Path) -> None:
    solution = _read_solution_h5(ROOT / 'examples' / 'outputs' / 'crane_two_reaction_argon' / 'solution.h5')
    if solution is None:
        return
    time_s, data = solution
    ar_plus = data.get('n[plasma,Ar_plus]')
    if ar_plus is None:
        return

    reference = _load_yaml(ROOT / 'examples' / 'external' / 'crane_two_reaction_argon_reference.yaml')
    final = reference['converted_output_final']
    final_t_us = float(reference['committed_output_final']['time_s']) * 1.0e6
    time_us = time_s * 1.0e6

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True)
    panels = [
        (
            axes[0],
            'Electron Density',
            ar_plus,
            float(final['electron_density_m3']),
            'This code e, quasineutral',
            'External CRANE final e',
            '#4e79a7',
        ),
        (
            axes[1],
            'Ar+ Density',
            ar_plus,
            float(final['Ar_plus_density_m3']),
            'This code Ar+',
            'External CRANE final Ar+',
            '#59a14f',
        ),
    ]
    for ax, title, values, ref_value, local_label, ref_label, color in panels:
        ax.plot(time_us, _positive_array(values), color=color, lw=1.8, label=local_label)
        ax.scatter([final_t_us], [ref_value], color='#e15759', marker='x', s=76, label=ref_label, zorder=3)
        ax.set_yscale('log')
        ax.set_title(title)
        ax.set_xlabel('Time [us]')
        ax.grid(True, which='both', alpha=0.23)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel('Density [m$^{-3}$]')
    fig.suptitle('CRANE-1: Electron and Ar+ Density Time Series', y=1.02)
    fig.subplots_adjust(bottom=0.23)
    fig.text(
        0.01,
        0.01,
        'CRANE TwoReactionArgon stores no independent electron state in this local case; e is inferred from Ar+ charge balance. External reference is a committed final value.',
        fontsize=8,
        color='#555555',
    )
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _plot_zdplaskin_density_timeseries(path: Path) -> None:
    solution = _read_solution_h5(ROOT / 'examples' / 'outputs' / 'zdplaskin_example2_surrogate' / 'solution.h5')
    if solution is None:
        return
    time_s, data = solution
    ar_plus = data.get('n[plasma,Ar_plus]')
    if ar_plus is None:
        return
    ar2_plus = data.get('n[plasma,Ar2_plus]', np.zeros_like(ar_plus))
    electron = ar_plus + ar2_plus

    reference = _load_yaml(ROOT / 'examples' / 'external' / 'zdplaskin_example2_summary.yaml')
    saved = reference['saved_output']
    final_t_ms = float(saved['final_saved_time_s']) * 1.0e3
    peak_t_ms = float(saved['peak_saved_electron_density_time_s']) * 1.0e3
    time_ms = time_s * 1.0e3

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True)
    axes[0].plot(time_ms, _positive_array(electron), color='#4e79a7', lw=1.8, label='This code e = Ar+ + Ar2+')
    axes[0].scatter(
        [final_t_ms],
        [float(saved['final_saved_electron_density_m3'])],
        color='#e15759',
        marker='x',
        s=76,
        label='External ZDPlaskin final e',
        zorder=3,
    )
    axes[0].scatter(
        [peak_t_ms],
        [float(saved['peak_saved_electron_density_m3'])],
        color='#222222',
        marker='*',
        s=110,
        label='External ZDPlaskin peak e',
        zorder=3,
    )
    axes[0].set_title('Electron Density')

    axes[1].plot(time_ms, _positive_array(ar_plus), color='#59a14f', lw=1.8, label='This code Ar+')
    axes[1].scatter(
        [final_t_ms],
        [float(saved['final_saved_Ar_plus_density_m3'])],
        color='#e15759',
        marker='x',
        s=76,
        label='External ZDPlaskin final Ar+',
        zorder=3,
    )
    axes[1].set_title('Ar+ Density')

    for ax in axes:
        ax.set_yscale('log')
        ax.set_xlabel('Time [ms]')
        ax.grid(True, which='both', alpha=0.23)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel('Density [m$^{-3}$]')
    fig.suptitle('ZDPlaskin-1: Electron and Ar+ Density Time Series', y=1.02)
    fig.subplots_adjust(bottom=0.23)
    fig.text(
        0.01,
        0.01,
        'External ZDPlaskin species waveforms are not stored here; markers show committed final values and the committed electron-density peak.',
        fontsize=8,
        color='#555555',
    )
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _plot_zdp_rate_table(path: Path) -> None:
    try:
        import h5py
    except ImportError:  # pragma: no cover
        return
    with h5py.File(ZDP_RATE_TABLE, 'r') as h5:
        field = np.asarray(h5['effective_field_Td'][:], dtype=float)
        rates = h5['rate_coefficients']
        selected = [name for name in ['xs_zdp_ar_ionization', 'xs_zdp_ar_excitation', 'xs_zdp_ar_star_ionization'] if name in rates]
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        for name in selected:
            values = np.asarray(rates[name][:], dtype=float)
            ax.plot(field, np.maximum(values, 1.0e-300), label=name)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('E/N (Td)')
    ax.set_ylabel('rate coefficient (m3/s)')
    ax.set_title('ZDPlaskin-2 output-derived E/N rate table')
    ax.grid(which='both', alpha=0.24)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _plot_runtime(path: Path, report: dict[str, Any]) -> None:
    cases = report.get('cases') or []
    labels = [case['case_id'] for case in cases]
    wall = [float(case['aggregate']['median_wall_time_s']) for case in cases]
    nfev = [float(case['aggregate']['nfev_median']) for case in cases]
    fig, (ax_time, ax_nfev) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    x = np.arange(len(labels))
    ax_time.bar(x, wall, color='#4c78a8')
    ax_time.set_xticks(x, labels, rotation=20, ha='right')
    ax_time.set_ylabel('median wall time (s)')
    ax_time.set_title('Runtime-1 local wall time')
    ax_time.grid(axis='y', alpha=0.25)
    ax_nfev.bar(x, nfev, color='#f28e2b')
    ax_nfev.set_xticks(x, labels, rotation=20, ha='right')
    ax_nfev.set_ylabel('median nfev')
    ax_nfev.set_title('Runtime-1 solver RHS evaluations')
    ax_nfev.grid(axis='y', alpha=0.25)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _plot_failure_heatmap(path: Path, rows: list[dict[str, Any]]) -> None:
    categories = sorted({row['diagnostic_category'] for row in rows if row['status'] not in {'pass', 'info'}})
    benchmarks = sorted({row['benchmark_id'].split(':')[0] for row in rows if row['status'] not in {'pass', 'info'}})
    if not categories or not benchmarks:
        categories = ['no_findings']
        benchmarks = ['all']
        values = np.zeros((1, 1))
    else:
        values = np.zeros((len(categories), len(benchmarks)))
        cat_idx = {name: i for i, name in enumerate(categories)}
        bench_idx = {name: i for i, name in enumerate(benchmarks)}
        for row in rows:
            if row['status'] not in {'pass', 'info'}:
                values[cat_idx[row['diagnostic_category']], bench_idx[row['benchmark_id'].split(':')[0]]] += 1.0
    fig, ax = plt.subplots(figsize=(max(6.5, 1.3 * len(benchmarks)), max(3.8, 0.45 * len(categories) + 1.3)))
    image = ax.imshow(values, cmap='YlOrRd', vmin=0)
    ax.set_xticks(np.arange(len(benchmarks)), benchmarks, rotation=25, ha='right')
    ax.set_yticks(np.arange(len(categories)), categories)
    ax.set_title('Failure-to-module diagnostic heatmap')
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, int(values[i, j]), ha='center', va='center', color='#222222')
    fig.colorbar(image, ax=ax, fraction=0.036, pad=0.04)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def write_plots(output_dir: Path, rows: list[dict[str, Any]], reports: dict[str, Any]) -> list[Path]:
    figure_dir = output_dir / 'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for path, plotter in [
        (figure_dir / 'benchmark_threshold_margin.png', _plot_acceptance_margin),
        (figure_dir / 'benchmark_claim_support_matrix.png', _plot_claim_support_matrix),
        (figure_dir / 'benchmark_status_by_software.png', _plot_software_status),
    ]:
        plotter(path, rows)
        if path.exists():
            paths.append(path)
    for benchmark_id, title in [
        ('CRANE-1', 'CRANE-1 species/conservation relative errors'),
        ('ZDPlaskin-1', 'ZDPlaskin-1 final/transient/circuit errors'),
        ('ZDPlaskin-2', 'ZDPlaskin-2 rate-table schema and interpolation checks'),
        ('PyGMol-1', 'PyGMol-1 sanity ratios and precision-claim scope'),
        ('PyGMol-Precision-1', 'PyGMol-Precision-1 same-footing final/waveform errors'),
        ('SWARM-1', 'SWARM-1 table ingestion checks'),
    ]:
        path = figure_dir / f'{benchmark_id}_metrics.png'
        _plot_metric_bar(path, title, [row for row in rows if row['benchmark_id'] == benchmark_id])
        if path.exists():
            paths.append(path)
    for path, plotter in [
        (figure_dir / 'CRANE-1_density_timeseries.png', _plot_crane_density_timeseries),
        (figure_dir / 'ZDPlaskin-1_density_timeseries.png', _plot_zdplaskin_density_timeseries),
    ]:
        plotter(path)
        if path.exists():
            paths.append(path)
    zdp_rate_path = figure_dir / 'ZDPlaskin-2_rate_table_vs_EoverN.png'
    _plot_zdp_rate_table(zdp_rate_path)
    if zdp_rate_path.exists():
        paths.append(zdp_rate_path)
    runtime_report = reports.get('Runtime-1', {}).get('report')
    if runtime_report:
        runtime_path = figure_dir / 'Runtime-1_solver_stats.png'
        _plot_runtime(runtime_path, runtime_report)
        paths.append(runtime_path)
    if 'PyGMol-1' in reports or 'PyGMol-Precision-1' in reports:
        paths.extend(path for path in build_pygmol_graphs(output_dir=figure_dir / 'pygmol') if path.suffix.lower() == '.png')
    heatmap_path = figure_dir / 'failure_to_module_heatmap.png'
    _plot_failure_heatmap(heatmap_path, rows)
    paths.append(heatmap_path)
    return paths


def write_findings(path: Path, rows: list[dict[str, Any]]) -> None:
    findings = [row for row in rows if row['status'] not in {'pass', 'info'}]
    lines = [
        '# Benchmark Findings',
        '',
        'This report lists benchmark threshold failures, warnings, and action-required items that should guide future core-code fixes.',
        '',
    ]
    if not findings:
        lines.extend(['No threshold failures or action-required items were found.', ''])
    else:
        for row in sorted(findings, key=lambda item: (item['severity'], item['benchmark_id'], item['metric_id'])):
            lines.extend(
                [
                    f"## {row['benchmark_id']} / {row['metric_id']}",
                    '',
                    f"- Status: `{row['status']}`",
                    f"- Severity: `{row['severity']}`",
                    f"- Value: `{row['value']}`; threshold: `{row['threshold']}`",
                    f"- Diagnostic category: `{row['diagnostic_category']}`",
                    f"- Suspected modules: {row['suspected_modules']}",
                    f"- Recommended next check: {row['recommendation']}",
                ]
            )
            if row.get('note'):
                lines.append(f"- Note: {row['note']}")
            lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')


def write_summary(path: Path, manifest: dict[str, Any], rows: list[dict[str, Any]], figures: list[Path], output_dir: Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row['benchmark_id'].split(':')[0]].append(row)
    lines = [
        '# External Benchmark Diagnostic Suite',
        '',
        'This suite separates benchmark claims by external software and problem setup, then maps threshold misses to suspected core-code areas.',
        '',
        '## Benchmark Status',
        '',
        '| Benchmark | Software | Claim | Status | Findings |',
        '| --- | --- | --- | --- | --- |',
    ]
    for benchmark in manifest['benchmarks']:
        bench_rows = grouped.get(benchmark['id'], [])
        bad = [row for row in bench_rows if row['status'] not in {'pass', 'info'}]
        status = 'pass' if not bad else 'attention'
        lines.append(f"| {benchmark['id']} | {benchmark['software']} | {benchmark['claim_level']} | {status} | {len(bad)} |")
    lines.extend(['', '## Generated Figures', ''])
    for figure in figures:
        lines.append(f"- `{figure.relative_to(ROOT).as_posix()}`")
    lines.extend(
        [
            '',
            '## Generated Tables',
            '',
            f"- `{(output_dir / 'benchmark_metrics.csv').relative_to(ROOT).as_posix()}`",
            f"- `{(output_dir / 'benchmark_findings.md').relative_to(ROOT).as_posix()}`",
            f"- `{(output_dir / 'benchmark_report.md').relative_to(ROOT).as_posix()}`",
            f"- `{(output_dir / 'diagnostic_report.yaml').relative_to(ROOT).as_posix()}`",
            '',
        ]
    )
    path.write_text('\n'.join(lines), encoding='utf-8')


def run_suite(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    only: set[str] | None = None,
    run_cases: bool = True,
    repetitions: int = 1,
) -> dict[str, Any]:
    manifest = load_benchmark_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    selected = only or {benchmark['id'] for benchmark in manifest['benchmarks']}

    if 'CRANE-1' in selected:
        reports['CRANE-1'], new_rows = evaluate_crane(manifest, output_dir)
        rows.extend(new_rows)
    if 'ZDPlaskin-1' in selected:
        reports['ZDPlaskin-1'], new_rows = evaluate_zdplaskin_1(manifest, output_dir, run_case=run_cases)
        rows.extend(new_rows)
    if 'ZDPlaskin-2' in selected:
        reports['ZDPlaskin-2'], new_rows = evaluate_zdplaskin_2(manifest, output_dir)
        rows.extend(new_rows)
    if 'PyGMol-1' in selected:
        reports['PyGMol-1'], new_rows = evaluate_pygmol_1(manifest, output_dir, rerun_local=run_cases)
        rows.extend(new_rows)
    if 'PyGMol-Precision-1' in selected:
        reports['PyGMol-Precision-1'], new_rows = evaluate_pygmol_precision_1(manifest, output_dir)
        rows.extend(new_rows)
    if 'SWARM-1' in selected:
        reports['SWARM-1'], new_rows = evaluate_swarm_1(manifest, output_dir)
        rows.extend(new_rows)
    if 'Runtime-1' in selected:
        reports['Runtime-1'], new_rows = evaluate_runtime_1(manifest, output_dir, repetitions=max(1, repetitions))
        rows.extend(new_rows)

    metric_fields = [
        'benchmark_id',
        'software',
        'problem',
        'metric_id',
        'group',
        'claim_level',
        'value',
        'observed',
        'reference',
        'threshold',
        'status',
        'severity',
        'diagnostic_category',
        'suspected_modules',
        'recommendation',
        'note',
    ]
    _write_csv(output_dir / 'benchmark_metrics.csv', rows, metric_fields)
    figures = write_plots(output_dir, rows, reports)
    write_findings(output_dir / 'benchmark_findings.md', rows)
    write_summary(output_dir / 'benchmark_summary.md', manifest, rows, figures, output_dir)
    report = {
        'tool': 'external_benchmark_diagnostic_suite',
        'manifest': str(manifest_path),
        'output_dir': str(output_dir),
        'benchmark_count': len(selected),
        'metric_count': len(rows),
        'attention_count': sum(1 for row in rows if row['status'] not in {'pass', 'info'}),
        'artifacts': {
            'metrics_csv': str(output_dir / 'benchmark_metrics.csv'),
            'findings_md': str(output_dir / 'benchmark_findings.md'),
            'summary_md': str(output_dir / 'benchmark_summary.md'),
            'figures': [str(path) for path in figures],
        },
        'reports': reports,
    }
    _write_yaml(output_dir / 'diagnostic_report.yaml', report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run problem-scoped external benchmark diagnostics and map misses to core-code suspects.')
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--only', action='append', default=None, help='Benchmark id to run; repeatable. Defaults to all.')
    parser.add_argument('--skip-case-runs', action='store_true', help='Reuse existing local outputs for ZDPlaskin/PyGMol where possible.')
    parser.add_argument('--repetitions', type=int, default=1, help='Runtime repetitions for Runtime-1.')
    args = parser.parse_args(argv)
    report = run_suite(
        manifest_path=args.manifest.resolve(),
        output_dir=args.output_dir.resolve(),
        only=set(args.only) if args.only else None,
        run_cases=not args.skip_case_runs,
        repetitions=args.repetitions,
    )
    print(yaml.safe_dump({
        'tool': report['tool'],
        'output_dir': report['output_dir'],
        'metric_count': report['metric_count'],
        'attention_count': report['attention_count'],
        'artifacts': report['artifacts'],
    }, sort_keys=False))
    return 0 if int(report['attention_count']) == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
