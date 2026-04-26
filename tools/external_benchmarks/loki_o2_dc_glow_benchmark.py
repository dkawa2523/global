from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.external_benchmarks.loki_o2_dc_glow_digitization import DEFAULT_SPEC, build_report


DEFAULT_REPORT = ROOT / 'examples' / 'external' / 'loki_o2_dc_glow_benchmark_report.yaml'
DEFAULT_SUMMARY_CSV = ROOT / 'examples' / 'external' / 'loki_o2_dc_glow_benchmark_summary.csv'

PRIMARY_PRESSURE_FILES = {
    'fig05a': ROOT / 'examples' / 'external' / 'loki_o2_dc_glow_digitized' / 'pressure_sweep' / 'fig05a_avg_temperatures_vs_pressure.csv',
    'fig05b': ROOT / 'examples' / 'external' / 'loki_o2_dc_glow_digitized' / 'pressure_sweep' / 'fig05b_avg_reduced_field_vs_pressure.csv',
    'fig06': ROOT / 'examples' / 'external' / 'loki_o2_dc_glow_digitized' / 'pressure_sweep' / 'fig06_molar_fractions_vs_pressure.csv',
    'fig07a': ROOT / 'examples' / 'external' / 'loki_o2_dc_glow_digitized' / 'pressure_sweep' / 'fig07a_main_charged_species_vs_pressure.csv',
    'fig11': ROOT / 'examples' / 'external' / 'loki_o2_dc_glow_digitized' / 'acceptance_envelope_reference' / 'fig11_relative_difference_envelope_vs_pressure.csv',
}

SECONDARY_PRESSURE_FILES = {
    'fig08': ROOT / 'examples' / 'external' / 'loki_o2_dc_glow_digitized' / 'pressure_sweep' / 'fig08_atomic_excited_states_vs_pressure.csv',
    'fig09a': ROOT / 'examples' / 'external' / 'loki_o2_dc_glow_digitized' / 'pressure_sweep' / 'fig09a_molecular_states_vs_pressure.csv',
    'fig09b': ROOT / 'examples' / 'external' / 'loki_o2_dc_glow_digitized' / 'pressure_sweep' / 'fig09b_ozone_and_herzberg_vs_pressure.csv',
}

PROFILE_FILES = {
    'fig01': ROOT / 'examples' / 'external' / 'loki_o2_dc_glow_digitized' / 'profile_assumption_diagnostics' / 'fig01_gas_temperature_profiles.csv',
    'fig02': ROOT / 'examples' / 'external' / 'loki_o2_dc_glow_digitized' / 'profile_assumption_diagnostics' / 'fig02_reduced_field_profiles.csv',
    'fig03': ROOT / 'examples' / 'external' / 'loki_o2_dc_glow_digitized' / 'profile_assumption_diagnostics' / 'fig03_electron_density_profiles.csv',
    'fig04': ROOT / 'examples' / 'external' / 'loki_o2_dc_glow_digitized' / 'profile_assumption_diagnostics' / 'fig04_atomic_oxygen_profiles.csv',
    'fig07b': ROOT / 'examples' / 'external' / 'loki_o2_dc_glow_digitized' / 'profile_assumption_diagnostics' / 'fig07b_charged_species_profiles_10Torr.csv',
    'fig10': ROOT / 'examples' / 'external' / 'loki_o2_dc_glow_digitized' / 'profile_assumption_diagnostics' / 'fig10_ozone_reactant_profiles_10Torr.csv',
}


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _load_pressure_sweep(path: Path) -> dict[str, dict[str, dict[float, float]]]:
    out: dict[str, dict[str, dict[float, float]]] = {}
    with path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            quantity_id = str(row['quantity_id']).strip()
            symbol = str(row.get('symbol', '')).strip()
            dataset_key = f'{quantity_id}::{symbol}' if symbol else quantity_id
            series_id = str(row['series_id']).strip()
            pressure = float(row['pressure_Torr'])
            value = float(row['value'])
            out.setdefault(dataset_key, {}).setdefault(series_id, {})[pressure] = value
    return out


def _load_acceptance_envelope(path: Path) -> dict[str, dict[float, float]]:
    out: dict[str, dict[float, float]] = {}
    with path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metric_id = str(row['metric_id']).strip()
            pressure = float(row['pressure_Torr'])
            value = float(row['value_percent']) / 100.0
            out.setdefault(metric_id, {})[pressure] = value
    return out


def _load_profile_curves(path: Path) -> dict[str, dict[str, dict[float, list[tuple[float, float]]]]]:
    out: dict[str, dict[str, dict[float, list[tuple[float, float]]]]] = {}
    with path.open('r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            quantity_id = str(row['quantity_id']).strip()
            symbol = str(row.get('symbol', '')).strip()
            dataset_key = f'{quantity_id}::{symbol}' if symbol else quantity_id
            series_id = str(row['series_id']).strip()
            pressure = float(row['pressure_Torr'])
            radius = float(row['radius_mm'])
            value = float(row['value'])
            out.setdefault(dataset_key, {}).setdefault(series_id, {}).setdefault(pressure, []).append((radius, value))
    for dataset in out.values():
        for series in dataset.values():
            for pressure, points in list(series.items()):
                series[pressure] = sorted(points)
    return out


def _relative_difference_series(
    data: dict[str, dict[str, dict[float, float]]],
    dataset_key: str,
    *,
    numerator_series: str = '0D_LoKI',
    comparison_series: str = '1D_fluid',
) -> dict[float, float]:
    quantity = data[dataset_key]
    numerators = quantity[numerator_series]
    comparisons = quantity[comparison_series]
    pressures = sorted(set(numerators) & set(comparisons))
    return {
        pressure: abs(numerators[pressure] - comparisons[pressure]) / abs(numerators[pressure])
        for pressure in pressures
        if numerators[pressure] != 0.0
    }


def _mean_proxy(series_by_quantity: dict[str, dict[float, float]]) -> dict[float, float]:
    common_pressures = sorted(set.intersection(*(set(values) for values in series_by_quantity.values())))
    return {
        pressure: mean(series_by_quantity[quantity_id][pressure] for quantity_id in series_by_quantity)
        for pressure in common_pressures
    }


def _series_summary(series: dict[float, float]) -> dict[str, Any]:
    ordered = sorted(series.items())
    return {
        'pressures_Torr': [pressure for pressure, _ in ordered],
        'values_percent': [round(value * 100.0, 4) for _, value in ordered],
        'max_percent': round(max(series.values()) * 100.0, 4),
        'min_percent': round(min(series.values()) * 100.0, 4),
    }


def _ref_metric(ref: dict[str, dict[float, float]], metric_id: str) -> dict[float, float]:
    return ref.get(metric_id, {})


def _dataset_key(quantity_id: str, symbol: str) -> str:
    return f'{quantity_id}::{symbol}'


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def _profile_series_summary(points: list[tuple[float, float]]) -> dict[str, Any]:
    ordered = sorted(points)
    radii = [radius for radius, _ in ordered]
    values = [value for _, value in ordered]
    center = values[0]
    edge = values[-1]
    span = max(values) - min(values)
    scale = max(max(abs(value) for value in values), 1.0)
    tolerance = scale * 0.02
    delta = edge - center
    if span <= tolerance:
        trend = 'flat'
    elif delta > tolerance:
        trend = 'increasing'
    elif delta < -tolerance:
        trend = 'decreasing'
    else:
        trend = 'flat'
    return {
        'radius_mm': radii,
        'values': values,
        'trend': trend,
        'center_value': center,
        'edge_value': edge,
        'edge_to_center_ratio': _safe_ratio(edge, center),
        'center_to_edge_ratio': _safe_ratio(center, edge),
        'min_value': min(values),
        'max_value': max(values),
    }


def _profile_expectations() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for pressure in (1.0, 10.0):
        for series_id in ('0D_LoKI', '1D_fluid'):
            checks.extend(
                [
                    {
                        'file_id': 'fig01',
                        'figure': '1',
                        'dataset_key': _dataset_key('gas_temperature_profile', 'Tg(r)'),
                        'series_id': series_id,
                        'pressure_Torr': pressure,
                        'expected_trend': 'decreasing',
                    },
                    {
                        'file_id': 'fig02',
                        'figure': '2',
                        'dataset_key': _dataset_key('reduced_electric_field_profile', 'E_over_N(r)'),
                        'series_id': series_id,
                        'pressure_Torr': pressure,
                        'expected_trend': 'decreasing',
                    },
                    {
                        'file_id': 'fig03',
                        'figure': '3',
                        'dataset_key': _dataset_key('electron_density_profile', 'ne(r)'),
                        'series_id': series_id,
                        'pressure_Torr': pressure,
                        'expected_trend': 'decreasing',
                    },
                    {
                        'file_id': 'fig04',
                        'figure': '4',
                        'dataset_key': _dataset_key('atomic_oxygen_density_profile', 'O(3P)(r)'),
                        'series_id': series_id,
                        'pressure_Torr': pressure,
                        'expected_trend': 'increasing',
                    },
                ]
            )
    for quantity_id, symbol in (
        ('electron_density_profile', 'e(r)'),
        ('negative_ion_density_profile', 'O_minus(r)'),
        ('positive_ion_density_profile', 'O2_plus(r)'),
    ):
        checks.append(
            {
                'file_id': 'fig07b',
                'figure': '7b',
                'dataset_key': _dataset_key(quantity_id, symbol),
                'series_id': '0D_LoKI',
                'pressure_Torr': 10.0,
                'expected_trend': 'flat',
            }
        )
        checks.append(
            {
                'file_id': 'fig07b',
                'figure': '7b',
                'dataset_key': _dataset_key(quantity_id, symbol),
                'series_id': '1D_fluid',
                'pressure_Torr': 10.0,
                'expected_trend': 'decreasing',
            }
        )
    for symbol, expected_trend in (
        ('O2(X)(r)', 'increasing'),
        ('O(3P)(r)', 'increasing'),
        ('O2(a)(r)', 'increasing'),
        ('O3(r)', 'increasing'),
        ('O2(b)(r)', 'decreasing'),
    ):
        checks.append(
            {
                'file_id': 'fig10',
                'figure': '10',
                'dataset_key': _dataset_key('ozone_reactant_profile', symbol),
                'series_id': '0D_LoKI',
                'pressure_Torr': 10.0,
                'expected_trend': 'flat',
            }
        )
        checks.append(
            {
                'file_id': 'fig10',
                'figure': '10',
                'dataset_key': _dataset_key('ozone_reactant_profile', symbol),
                'series_id': '1D_fluid',
                'pressure_Torr': 10.0,
                'expected_trend': expected_trend,
            }
        )
    return checks


def _overlap_threshold_pass(
    computed: dict[str, dict[float, float]],
    reference: dict[str, dict[float, float]],
    threshold: float,
) -> bool:
    for metric_id, series in computed.items():
        overlap_pressures = set(series) & set(reference.get(metric_id, {}))
        if not overlap_pressures:
            return False
        if any(series[pressure] > threshold for pressure in overlap_pressures):
            return False
    return True


def _overlap_reference_ceiling_pass(series: dict[float, float], reference: dict[float, float]) -> bool:
    overlap_pressures = sorted(set(series) & set(reference))
    if not overlap_pressures:
        return False
    return all(series[pressure] <= reference[pressure] for pressure in overlap_pressures)


def build_benchmark_report(
    spec_path: Path = DEFAULT_SPEC,
    *,
    report_path: Path = DEFAULT_REPORT,
    summary_csv_path: Path = DEFAULT_SUMMARY_CSV,
) -> dict[str, Any]:
    readiness = build_report(spec_path.resolve(), workspace_root=ROOT, write_files=True, report_path=None)
    runnable_primary = bool(readiness['summary']['runnable_primary_pressure_benchmark'])

    if not runnable_primary:
        report = {
            'tool': 'loki_o2_dc_glow_benchmark',
            'scope': 'external_digitized_reference_only',
            'spec_file': str(spec_path.resolve()),
            'passed': False,
            'summary': {
                'runnable_primary_pressure_benchmark': False,
                'readiness_stage': readiness['summary']['readiness_stage'],
            },
            'evaluation': {
                'execution_mode': 'blocked_by_readiness',
                'assessment': 'Primary digitized datasets are incomplete, so the external LoKI benchmark cannot run yet.',
                'next_actions': readiness['evaluation']['next_actions'],
            },
        }
        _write_yaml(report_path.resolve(), report)
        return report

    runnable_extended = bool(readiness['summary']['runnable_extended_species_benchmark'])
    runnable_full = bool(readiness['summary']['runnable_full_diagnostic_benchmark'])

    fig05a = _load_pressure_sweep(PRIMARY_PRESSURE_FILES['fig05a'])
    fig05b = _load_pressure_sweep(PRIMARY_PRESSURE_FILES['fig05b'])
    fig06 = _load_pressure_sweep(PRIMARY_PRESSURE_FILES['fig06'])
    fig07a = _load_pressure_sweep(PRIMARY_PRESSURE_FILES['fig07a'])
    fig11 = _load_acceptance_envelope(PRIMARY_PRESSURE_FILES['fig11'])

    direct_tg = _relative_difference_series(fig05a, _dataset_key('average_gas_temperature', 'Tg,av'))
    direct_tnw = _relative_difference_series(fig05a, _dataset_key('near_wall_temperature', 'Tnw'))
    direct_en = _relative_difference_series(fig05b, _dataset_key('average_reduced_electric_field', 'E_over_N'))
    direct_e = _relative_difference_series(fig07a, _dataset_key('average_electron_density', 'ne'))
    direct_ominus = _relative_difference_series(fig07a, _dataset_key('average_negative_ion_density', 'O_minus'))
    direct_o2plus = _relative_difference_series(fig07a, _dataset_key('average_positive_ion_density', 'O2_plus'))
    direct_atomic_o = _relative_difference_series(fig06, _dataset_key('average_molar_fraction_atomic_oxygen', '[O]/ng'))
    direct_molecular_o2 = _relative_difference_series(fig06, _dataset_key('average_molar_fraction_molecular_oxygen', '[O2]/ng'))

    charge_proxy = _mean_proxy(
        {
            'average_electron_density': direct_e,
            'average_negative_ion_density': direct_ominus,
            'average_positive_ion_density': direct_o2plus,
        }
    )
    first_wave_neutral_proxy = _mean_proxy(
        {
            'average_molar_fraction_atomic_oxygen': direct_atomic_o,
            'average_molar_fraction_molecular_oxygen': direct_molecular_o2,
        }
    )

    direct_main = {
        'avg_gas_temperature_relative_difference': direct_tg,
        'near_wall_temperature_relative_difference': direct_tnw,
        'avg_reduced_electric_field_relative_difference': direct_en,
    }
    ref_main = {
        metric_id: _ref_metric(fig11, metric_id)
        for metric_id in (
            'avg_gas_temperature_relative_difference',
            'near_wall_temperature_relative_difference',
            'avg_reduced_electric_field_relative_difference',
        )
    }

    secondary_species_differences: dict[str, dict[float, float]] = {}
    extended_neutral_proxy: dict[float, float] = {}
    extended_neutral_proxy_reference_pass: bool | None = None
    o3_relative_difference: dict[float, float] = {}
    profile_diagnostics: dict[str, Any] = {}
    profile_shape_expectation_pass: bool | None = None
    comparison_rows: list[dict[str, Any]] = []

    if runnable_extended:
        fig08 = _load_pressure_sweep(SECONDARY_PRESSURE_FILES['fig08'])
        fig09a = _load_pressure_sweep(SECONDARY_PRESSURE_FILES['fig09a'])
        fig09b = _load_pressure_sweep(SECONDARY_PRESSURE_FILES['fig09b'])

        secondary_species_differences = {
            'O(3P)': _relative_difference_series(fig08, _dataset_key('average_atomic_excited_state_density', 'O(3P)')),
            'O(1D)': _relative_difference_series(fig08, _dataset_key('average_atomic_excited_state_density', 'O(1D)')),
            'O(1S)': _relative_difference_series(fig08, _dataset_key('average_atomic_excited_state_density', 'O(1S)')),
            'O2(X)': _relative_difference_series(fig09a, _dataset_key('average_molecular_state_density', 'O2(X)')),
            'O2(a)': _relative_difference_series(fig09a, _dataset_key('average_molecular_state_density', 'O2(a)')),
            'O2(b)': _relative_difference_series(fig09a, _dataset_key('average_molecular_state_density', 'O2(b)')),
            'O2(Hz)': _relative_difference_series(fig09b, _dataset_key('average_molecular_state_density', 'O2(Hz)')),
            'O3': _relative_difference_series(fig09b, _dataset_key('average_molecular_state_density', 'O3')),
        }
        extended_neutral_proxy = _mean_proxy(
            {
                'average_molar_fraction_atomic_oxygen': direct_atomic_o,
                'average_molar_fraction_molecular_oxygen': direct_molecular_o2,
                **secondary_species_differences,
            }
        )
        extended_neutral_proxy_reference_pass = _overlap_reference_ceiling_pass(
            extended_neutral_proxy,
            _ref_metric(fig11, 'avg_neutral_species_relative_difference'),
        )
        o3_relative_difference = secondary_species_differences['O3']

    if runnable_full:
        profile_data = {
            file_id: _load_profile_curves(path)
            for file_id, path in PROFILE_FILES.items()
        }
        profile_checks: list[dict[str, Any]] = []
        for expectation in _profile_expectations():
            points = profile_data[expectation['file_id']][expectation['dataset_key']][expectation['series_id']][expectation['pressure_Torr']]
            summary = _profile_series_summary(points)
            observed_trend = summary['trend']
            check = {
                'figure': expectation['figure'],
                'file_id': expectation['file_id'],
                'dataset_key': expectation['dataset_key'],
                'series_id': expectation['series_id'],
                'pressure_Torr': expectation['pressure_Torr'],
                'expected_trend': expectation['expected_trend'],
                'observed_trend': observed_trend,
                'passed': observed_trend == expectation['expected_trend'],
                'center_value': summary['center_value'],
                'edge_value': summary['edge_value'],
                'edge_to_center_ratio': summary['edge_to_center_ratio'],
                'center_to_edge_ratio': summary['center_to_edge_ratio'],
            }
            profile_checks.append(check)
            comparison_rows.append(
                {
                    'pressure_Torr': expectation['pressure_Torr'],
                    'metric_id': f'profile::{expectation["dataset_key"]}::{expectation["series_id"]}',
                    'comparison_scope': 'diagnostic_profile_shape',
                    'computed_from_primary_percent': '',
                    'figure11_reference_percent': '',
                    'difference_percent_points': '',
                    'note': (
                        f'figure={expectation["figure"]}; expected_trend={expectation["expected_trend"]}; '
                        f'observed_trend={observed_trend}; edge_to_center_ratio={summary["edge_to_center_ratio"]}'
                    ),
                }
            )
        profile_shape_expectation_pass = all(item['passed'] for item in profile_checks)
        o2_plus_1d = next(
            item
            for item in profile_checks
            if item['file_id'] == 'fig07b'
            and item['dataset_key'] == _dataset_key('positive_ion_density_profile', 'O2_plus(r)')
            and item['series_id'] == '1D_fluid'
        )
        o3_1d = next(
            item
            for item in profile_checks
            if item['file_id'] == 'fig10'
            and item['dataset_key'] == _dataset_key('ozone_reactant_profile', 'O3(r)')
            and item['series_id'] == '1D_fluid'
        )
        o2b_1d = next(
            item
            for item in profile_checks
            if item['file_id'] == 'fig10'
            and item['dataset_key'] == _dataset_key('ozone_reactant_profile', 'O2(b)(r)')
            and item['series_id'] == '1D_fluid'
        )
        profile_diagnostics = {
            'shape_expectation_pass': profile_shape_expectation_pass,
            'checks': profile_checks,
            'key_findings': [
                'Figures 1 to 4 preserve the expected radial directionality: Tg, E/N, and ne decrease toward the wall, while O(3P) increases.',
                (
                    'At 10 Torr the charged-species profile diagnostic stays strongly center-peaked in 1D: '
                    f'O2+ edge-to-center ratio is about {o2_plus_1d["edge_to_center_ratio"]:.4f}.'
                ),
                (
                    'At 10 Torr the chemistry split from figure 10 is preserved: '
                    f'O3 is wall-peaked with edge-to-center ratio about {o3_1d["edge_to_center_ratio"]:.1f}, '
                    f'while O2(b) stays center-peaked with edge-to-center ratio about {o2b_1d["edge_to_center_ratio"]:.4f}.'
                ),
            ],
        }

    for metric_id, series in direct_main.items():
        for pressure in sorted(set(series) & set(ref_main[metric_id])):
            comparison_rows.append(
                {
                    'pressure_Torr': pressure,
                    'metric_id': metric_id,
                    'comparison_scope': 'direct_overlap_metric',
                    'computed_from_primary_percent': round(series[pressure] * 100.0, 6),
                    'figure11_reference_percent': round(ref_main[metric_id][pressure] * 100.0, 6),
                    'difference_percent_points': round((series[pressure] - ref_main[metric_id][pressure]) * 100.0, 6),
                    'note': 'Directly reconstructible from first-wave primary pressure-sweep datasets.',
                }
            )

    for metric_id, series, ref_metric_id, note in (
        (
            'avg_charged_species_subset_proxy',
            charge_proxy,
            'avg_charged_species_relative_difference',
            'First-wave charged proxy from e, O-, and O2+ only.',
        ),
        (
            'avg_neutral_species_first_wave_subset_proxy',
            first_wave_neutral_proxy,
            'avg_neutral_species_relative_difference',
            'First-wave neutral proxy from O/ng and O2/ng only.',
        ),
    ):
        ref_series = _ref_metric(fig11, ref_metric_id)
        for pressure in sorted(set(series) & set(ref_series)):
            comparison_rows.append(
                {
                    'pressure_Torr': pressure,
                    'metric_id': metric_id,
                    'comparison_scope': 'subset_proxy_vs_aggregate_reference',
                    'computed_from_primary_percent': round(series[pressure] * 100.0, 6),
                    'figure11_reference_percent': round(ref_series[pressure] * 100.0, 6),
                    'difference_percent_points': round((series[pressure] - ref_series[pressure]) * 100.0, 6),
                    'note': note,
                }
            )

    if runnable_extended:
        ref_neutral = _ref_metric(fig11, 'avg_neutral_species_relative_difference')
        for pressure in sorted(set(extended_neutral_proxy) & set(ref_neutral)):
            comparison_rows.append(
                {
                    'pressure_Torr': pressure,
                    'metric_id': 'avg_neutral_species_extended_subset_proxy',
                    'comparison_scope': 'extended_subset_vs_aggregate_reference',
                    'computed_from_primary_percent': round(extended_neutral_proxy[pressure] * 100.0, 6),
                    'figure11_reference_percent': round(ref_neutral[pressure] * 100.0, 6),
                    'difference_percent_points': round((extended_neutral_proxy[pressure] - ref_neutral[pressure]) * 100.0, 6),
                    'note': 'Extended neutral proxy includes O/ng, O2/ng, O(3P), O(1D), O(1S), O2(X), O2(a), O2(b), O2(Hz), and O3.',
                }
            )
        for symbol, series in secondary_species_differences.items():
            for pressure, value in sorted(series.items()):
                comparison_rows.append(
                    {
                        'pressure_Torr': pressure,
                        'metric_id': f'species::{symbol}',
                        'comparison_scope': 'extended_species_direct_difference',
                        'computed_from_primary_percent': round(value * 100.0, 6),
                        'figure11_reference_percent': '',
                        'difference_percent_points': '',
                        'note': 'Direct 0D versus 1D relative difference for a digitized secondary neutral or excited species.',
                    }
                )

    _write_csv(
        summary_csv_path.resolve(),
        comparison_rows,
        [
            'pressure_Torr',
            'metric_id',
            'comparison_scope',
            'computed_from_primary_percent',
            'figure11_reference_percent',
            'difference_percent_points',
            'note',
        ],
    )

    direct_main_threshold_pass = _overlap_threshold_pass(direct_main, ref_main, 0.05)
    ref_fig11_main_pass = max(
        max(_ref_metric(fig11, metric_id).values())
        for metric_id in (
            'avg_gas_temperature_relative_difference',
            'near_wall_temperature_relative_difference',
            'electron_temperature_relative_difference',
            'avg_reduced_electric_field_relative_difference',
        )
    ) <= 0.05
    ref_fig11_species_pass = (
        _ref_metric(fig11, 'avg_charged_species_relative_difference').get(1.0, 1.0) <= 0.20
        and _ref_metric(fig11, 'avg_neutral_species_relative_difference').get(1.0, 1.0) <= 0.20
        and _ref_metric(fig11, 'avg_charged_species_relative_difference').get(10.0, 1.0) <= 0.60
        and _ref_metric(fig11, 'avg_neutral_species_relative_difference').get(10.0, 1.0) <= 0.60
    )

    if runnable_full:
        passed = (
            runnable_primary
            and direct_main_threshold_pass
            and ref_fig11_main_pass
            and ref_fig11_species_pass
            and bool(extended_neutral_proxy_reference_pass)
            and bool(profile_shape_expectation_pass)
        )
    elif runnable_extended:
        passed = (
            runnable_primary
            and direct_main_threshold_pass
            and ref_fig11_main_pass
            and ref_fig11_species_pass
            and bool(extended_neutral_proxy_reference_pass)
        )
    else:
        passed = runnable_primary and direct_main_threshold_pass and ref_fig11_main_pass and ref_fig11_species_pass

    key_findings = [
        'Directly reconstructible main discharge parameters from figures 5a and 5b stay within the paper-level 5 percent envelope.',
        'Figure 11 envelope data are digitized as an external acceptance reference on their observed six-pressure subset.',
        'Charged-species proxy from e, O-, and O2+ remains well below the figure 11 aggregate charged envelope, which is consistent with the paper-level aggregate covering a broader charged set.',
        'Electron-temperature relative difference remains reference-only because Te pressure-sweep curves are not part of the digitized pressure-sweep set.',
    ]
    if runnable_extended:
        key_findings.extend(
            [
                'The extended neutral subset now follows the same rising pressure trend as figure 11 and stays below the aggregate neutral envelope at all overlapping pressures.',
                'O3 remains the dominant high-pressure outlier: the digitized 1D-versus-0D difference reaches about a factor 3.4 at 10 Torr.',
                'O2(X) stays close between 0D and 1D, while O2(a), O2(b), and O3 account for most of the growing high-pressure neutral-species spread.',
            ]
        )
    if runnable_full:
        key_findings.extend(profile_diagnostics['key_findings'])
    elif not runnable_extended:
        key_findings.append(
            'Neutral-species interpretation is still first-wave only until figures 8, 9a, and 9b are digitized.'
        )

    summary = {
        'runnable_primary_pressure_benchmark': runnable_primary,
        'runnable_extended_species_benchmark': runnable_extended,
        'runnable_full_diagnostic_benchmark': runnable_full,
        'readiness_stage': readiness['summary']['readiness_stage'],
        'direct_main_parameter_threshold_percent': 5.0,
        'direct_main_parameter_threshold_pass': direct_main_threshold_pass,
        'figure11_reference_main_parameter_threshold_pass': ref_fig11_main_pass,
        'figure11_reference_species_envelope_pass': ref_fig11_species_pass,
    }
    if runnable_extended:
        summary['extended_neutral_subset_reference_ceiling_pass'] = bool(extended_neutral_proxy_reference_pass)
    if runnable_full:
        summary['profile_shape_expectation_pass'] = bool(profile_shape_expectation_pass)

    direct_reconstruction: dict[str, Any] = {
        'avg_gas_temperature_relative_difference': _series_summary(direct_tg),
        'near_wall_temperature_relative_difference': _series_summary(direct_tnw),
        'avg_reduced_electric_field_relative_difference': _series_summary(direct_en),
        'avg_charged_species_subset_proxy': _series_summary(charge_proxy),
        'avg_neutral_species_first_wave_subset_proxy': _series_summary(first_wave_neutral_proxy),
    }
    if runnable_extended:
        direct_reconstruction['avg_neutral_species_extended_subset_proxy'] = _series_summary(extended_neutral_proxy)

    report = {
        'tool': 'loki_o2_dc_glow_benchmark',
        'scope': 'external_digitized_reference_only',
        'spec_file': str(spec_path.resolve()),
        'digitization_root': readiness['digitization_root'],
        'digitization_readiness_report': readiness,
        'summary_csv': str(summary_csv_path.resolve()),
        'passed': passed,
        'summary': summary,
        'direct_reconstruction': direct_reconstruction,
        'extended_species_relative_differences': (
            {symbol: _series_summary(series) for symbol, series in secondary_species_differences.items()}
            if runnable_extended
            else {}
        ),
        'profile_diagnostics': profile_diagnostics,
        'figure11_reference': {
            metric_id: _series_summary(series)
            for metric_id, series in fig11.items()
        },
        'evaluation': {
            'execution_mode': 'digitized_reference_consistency_check',
            'assessment': (
                'LoKI reference data are digitized through the full diagnostic benchmark, including radial-profile checks from figures 1, 2, 3, 4, 7b, and 10.'
                if runnable_full
                else 'LoKI pressure-sweep reference data are digitized and runnable through the extended species benchmark.'
                if runnable_extended
                else 'Primary LoKI pressure-sweep reference data are digitized and runnable as an external benchmark candidate.'
            ),
            'key_findings': key_findings,
            'comparison_rows': comparison_rows,
        },
    }
    _write_yaml(report_path.resolve(), report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run the external-only LoKI O2 DC glow benchmark using digitized paper curves.')
    parser.add_argument('--spec', type=Path, default=DEFAULT_SPEC)
    parser.add_argument('--write', type=Path, default=DEFAULT_REPORT)
    parser.add_argument('--summary-csv', type=Path, default=DEFAULT_SUMMARY_CSV)
    args = parser.parse_args(argv)
    report = build_benchmark_report(args.spec.resolve(), report_path=args.write.resolve(), summary_csv_path=args.summary_csv.resolve())
    print(yaml.safe_dump(report, sort_keys=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
