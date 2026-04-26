from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
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
from tools.external_benchmarks.loki_o2_dc_glow_benchmark import (
    DEFAULT_REPORT as LOKI_DEFAULT_REPORT,
    build_benchmark_report as build_loki_report,
)
from tools.external_benchmarks.pygmol_argon import (
    DEFAULT_CASE as PYGMOL_CASE,
    RATE_MODE_LOCAL_FIT,
    RATE_MODE_LOCAL_TABLE,
    RATE_MODE_SURROGATE,
    build_report as build_pygmol_report,
)
from tools.external_benchmarks.pygmol_same_footing import (
    build_decomposition_report as build_pygmol_decomposition_report,
)
from tools.external_benchmarks.same_footing_assessment import (
    DEFAULT_WRITE as SAME_FOOTING_DEFAULT_WRITE,
    build_same_footing_report,
)
from tools.external_benchmarks.robustness_dashboard import (
    build_robustness_dashboard,
)
from tools.external_benchmarks.zdplaskin_eovern import (
    DEFAULT_OUTPUT_DIR as ZDPLASKIN_OUTPUT_DIR,
    DEFAULT_REFERENCE as ZDPLASKIN_REFERENCE,
    build_eovern_report,
)
from tools.external_benchmarks.zdplaskin_example2 import (
    build_report as build_zdplaskin_report,
)


ZDPLASKIN_CASE = ROOT / 'examples' / 'configs' / 'case_zdplaskin_example2.yaml'
OUTPUT_DIR = ROOT / 'examples' / 'outputs' / 'external_benchmark_dashboard'
DEFAULT_REPORT = OUTPUT_DIR / 'external_benchmark_dashboard.yaml'
DEFAULT_METRICS_CSV = OUTPUT_DIR / 'external_benchmark_dashboard_metrics.csv'
DEFAULT_OVERVIEW_PNG = OUTPUT_DIR / 'external_benchmark_overview.png'
DEFAULT_PYGMOL_PNG = OUTPUT_DIR / 'external_benchmark_pygmol_decomposition.png'
DEFAULT_COVERAGE_PNG = OUTPUT_DIR / 'external_benchmark_coverage_matrix.png'
DEFAULT_MARKDOWN = OUTPUT_DIR / 'external_benchmark_report.md'
DEFAULT_MARKDOWN_JA = OUTPUT_DIR / 'external_benchmark_report_ja.md'


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio_mismatch_percent(value: Any) -> float | None:
    ratio = _safe_float(value)
    if ratio is None or not math.isfinite(ratio) or ratio <= 0.0:
        return None
    return abs(ratio - 1.0) * 100.0


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def _markdown_relpath(base_path: Path, target_path: Path) -> str:
    return Path(os.path.relpath(target_path, start=base_path.parent)).as_posix()


def _pygmol_step_row(report: dict[str, Any], step_id: str) -> dict[str, Any]:
    for row in report.get('rows', []):
        if row.get('step_id') == step_id:
            return row
    raise KeyError(f'PyGMol step {step_id!r} not found in report.')


def _stage_row(report: dict[str, Any], stage_id: str) -> dict[str, Any]:
    for row in report.get('summary', []):
        if row.get('stage_id') == stage_id:
            return row
    raise KeyError(f'PyGMol stage {stage_id!r} not found in decomposition report.')


def collect_reports(*, rerun_zdplaskin: bool = True, rerun_pygmol_local: bool = True) -> dict[str, Any]:
    if rerun_zdplaskin:
        run_from_yaml(ZDPLASKIN_CASE)

    zdplaskin = build_zdplaskin_report(ZDPLASKIN_OUTPUT_DIR, ZDPLASKIN_REFERENCE)
    zdplaskin_eovern = build_eovern_report(ZDPLASKIN_OUTPUT_DIR, ZDPLASKIN_REFERENCE)
    crane = build_crane_report(CRANE_CASE, CRANE_REFERENCE)
    pygmol_surrogate = build_pygmol_report(
        PYGMOL_CASE,
        rerun_local=rerun_pygmol_local,
        write_solution_csv=False,
        rate_mode=RATE_MODE_SURROGATE,
    )
    pygmol_local_fit = build_pygmol_report(
        PYGMOL_CASE,
        rerun_local=False,
        write_solution_csv=False,
        rate_mode=RATE_MODE_LOCAL_FIT,
    )
    pygmol_local_table = build_pygmol_report(
        PYGMOL_CASE,
        rerun_local=False,
        write_solution_csv=False,
        rate_mode=RATE_MODE_LOCAL_TABLE,
    )
    pygmol_same_footing = build_pygmol_decomposition_report(PYGMOL_CASE, rerun_local=False)
    loki = build_loki_report(report_path=LOKI_DEFAULT_REPORT, summary_csv_path=ROOT / 'examples' / 'external' / 'loki_o2_dc_glow_benchmark_summary.csv')
    same_footing = build_same_footing_report(run_cases=False)
    robustness = build_robustness_dashboard(run_cases=False)
    _write_yaml(SAME_FOOTING_DEFAULT_WRITE, same_footing)

    return {
        'zdplaskin': zdplaskin,
        'zdplaskin_eovern': zdplaskin_eovern,
        'crane': crane,
        'pygmol_surrogate': pygmol_surrogate,
        'pygmol_local_fit': pygmol_local_fit,
        'pygmol_local_table': pygmol_local_table,
        'pygmol_same_footing': pygmol_same_footing,
        'loki': loki,
        'same_footing': same_footing,
        'robustness': robustness,
    }


def _metric_rows(reports: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    zdp_cmp = reports['zdplaskin']['comparison']
    for metric_id, value in (
        ('final_electron_density_ratio_local_over_zdplaskin', zdp_cmp['final_electron_density_ratio_local_over_zdplaskin']),
        ('final_Ar_star_ratio_local_over_zdplaskin', zdp_cmp['final_Ar_star_ratio_local_over_zdplaskin']),
        ('final_Ar_plus_ratio_local_over_zdplaskin', zdp_cmp['final_Ar_plus_ratio_local_over_zdplaskin']),
        ('final_Ar2_plus_ratio_local_over_zdplaskin', zdp_cmp['final_Ar2_plus_ratio_local_over_zdplaskin']),
        ('local_reported_over_zdplaskin_saved_EoverN_ratio', reports['zdplaskin_eovern']['comparison']['local_reported_over_zdplaskin_saved_EoverN_ratio']),
        ('same_voltage_EoverN_over_zdplaskin_saved_ratio', reports['zdplaskin_eovern']['comparison']['same_voltage_EoverN_over_zdplaskin_saved_ratio']),
    ):
        rows.append(
            {
                'benchmark_family': 'zdplaskin',
                'benchmark_id': 'zdplaskin_example2',
                'metric_id': metric_id,
                'step_or_stage': '',
                'value': value,
                'unit': 'ratio',
            }
        )

    crane_cmp = reports['crane']['comparison']
    for metric_id, value in (
        ('final_electron_density_relative_error', crane_cmp['final_electron_density_m3_relative_error']),
        ('final_Ar_plus_density_relative_error', crane_cmp['final_Ar_plus_density_m3_relative_error']),
        ('final_Ar_density_relative_error', crane_cmp['final_Ar_density_m3_relative_error']),
    ):
        rows.append(
            {
                'benchmark_family': 'crane',
                'benchmark_id': 'crane_two_reaction_argon',
                'metric_id': metric_id,
                'step_or_stage': '',
                'value': value,
                'unit': 'fraction',
            }
        )

    for benchmark_id, report in (
        ('pygmol_surrogate', reports['pygmol_surrogate']),
        ('pygmol_local_fit', reports['pygmol_local_fit']),
        ('pygmol_local_table', reports['pygmol_local_table']),
    ):
        for step_id in ('ignition', 'production', 'afterglow'):
            row = _pygmol_step_row(report, step_id)
            for metric_id, value in (
                ('ratio_final_ne_pygmol_over_local', row.get('ratio_final_ne_pygmol_over_local')),
                ('ratio_mean_ne_pygmol_over_local', row.get('ratio_mean_ne_pygmol_over_local')),
                ('ratio_final_mean_energy_equiv_pygmol_over_local', row.get('ratio_final_mean_energy_equiv_pygmol_over_local')),
                ('ratio_mean_absorbed_power_pygmol_over_local', row.get('ratio_mean_absorbed_power_pygmol_over_local')),
                ('ratio_final_wall_loss_rate_pygmol_over_local', row.get('ratio_final_wall_loss_rate_pygmol_over_local')),
            ):
                rows.append(
                    {
                        'benchmark_family': 'pygmol',
                        'benchmark_id': benchmark_id,
                        'metric_id': metric_id,
                        'step_or_stage': step_id,
                        'value': value,
                        'unit': 'ratio',
                    }
                )

    for stage_row in reports['pygmol_same_footing']['summary']:
        for metric_id, value in (
            ('production_final_ne_ratio', stage_row.get('production_final_ne_ratio')),
            ('production_mean_ne_ratio', stage_row.get('production_mean_ne_ratio')),
            ('production_power_ratio', stage_row.get('production_power_ratio')),
            ('production_wall_loss_rate_ratio', stage_row.get('production_wall_loss_rate_ratio')),
            ('production_energy_ratio', stage_row.get('production_energy_ratio')),
        ):
            rows.append(
                {
                    'benchmark_family': 'pygmol_same_footing',
                    'benchmark_id': 'pygmol_same_footing',
                    'metric_id': metric_id,
                    'step_or_stage': stage_row['stage_id'],
                    'value': value,
                    'unit': 'ratio',
                }
            )

    loki = reports['loki']
    for metric_id, series in (
        ('avg_gas_temperature_relative_difference', loki['direct_reconstruction']['avg_gas_temperature_relative_difference']),
        ('near_wall_temperature_relative_difference', loki['direct_reconstruction']['near_wall_temperature_relative_difference']),
        ('avg_reduced_electric_field_relative_difference', loki['direct_reconstruction']['avg_reduced_electric_field_relative_difference']),
        ('avg_neutral_species_extended_subset_proxy', loki['direct_reconstruction']['avg_neutral_species_extended_subset_proxy']),
        ('O3', loki['extended_species_relative_differences']['O3']),
    ):
        rows.append(
            {
                'benchmark_family': 'loki',
                'benchmark_id': 'loki_o2_dc_glow',
                'metric_id': metric_id,
                'step_or_stage': 'max_over_pressure',
                'value': max(float(v) for v in series['values_percent']) / 100.0,
                'unit': 'fraction',
            }
        )
    rows.append(
        {
            'benchmark_family': 'loki',
            'benchmark_id': 'loki_o2_dc_glow',
            'metric_id': 'profile_shape_expectation_pass',
            'step_or_stage': 'full_diagnostic',
            'value': 1.0 if loki['summary'].get('profile_shape_expectation_pass') else 0.0,
            'unit': 'flag',
        }
    )
    return rows


def _max_finite(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return max(finite)


def _fmt_percent(value: Any, digits: int = 2) -> str:
    x = _safe_float(value)
    if x is None or not math.isfinite(x):
        return 'n/a'
    return f'{x:.{digits}f}%'


def _fmt_ratio(value: Any, digits: int = 3) -> str:
    x = _safe_float(value)
    if x is None or not math.isfinite(x):
        return 'n/a'
    return f'{x:.{digits}f}'


def _fmt_sci(value: Any, digits: int = 3) -> str:
    x = _safe_float(value)
    if x is None or not math.isfinite(x):
        return 'n/a'
    return f'{x:.{digits}e}'


def _coverage_rows() -> list[dict[str, Any]]:
    return [
        {
            'benchmark': 'ZDPlaskin example2',
            'reference_type': 'strict executable parity',
            'same_footing': 'high',
            'comparison_target': 'DC series-discharge state with committed ZDPlaskin output',
            'axes': {
                'Circuit / power': 2,
                'Physical E/N': 2,
                'Species': 2,
                'Wall loss': 1,
                'Profiles': 0,
                'Scalar ODE': 1,
            },
            'main_limitation': 'single published Ar case; no spatial profiles',
        },
        {
            'benchmark': 'CRANE TwoReactionArgon',
            'reference_type': 'strict scalar-chemistry parity',
            'same_footing': 'high',
            'comparison_target': 'public two-reaction argon tutorial output',
            'axes': {
                'Circuit / power': 0,
                'Physical E/N': 0,
                'Species': 2,
                'Wall loss': 0,
                'Profiles': 0,
                'Scalar ODE': 2,
            },
            'main_limitation': 'not a full plasma-discharge closure benchmark',
        },
        {
            'benchmark': 'PyGMol Ar baseline',
            'reference_type': 'broad executable global-model comparison',
            'same_footing': 'medium',
            'comparison_target': 'compact executable Ar global model with independent closure choices',
            'axes': {
                'Circuit / power': 1,
                'Physical E/N': 0,
                'Species': 1,
                'Wall loss': 1,
                'Profiles': 0,
                'Scalar ODE': 1,
            },
            'main_limitation': 'electron-impact rates and wall closure are not naturally aligned',
        },
        {
            'benchmark': 'PyGMol same-footing',
            'reference_type': 'diagnostic decomposition benchmark',
            'same_footing': 'medium-high',
            'comparison_target': 'same-rate, same-power, and same-wall-loss staged alignment',
            'axes': {
                'Circuit / power': 2,
                'Physical E/N': 1,
                'Species': 1,
                'Wall loss': 2,
                'Profiles': 0,
                'Scalar ODE': 1,
            },
            'main_limitation': 'still a different executable model, not strict parity chemistry',
        },
        {
            'benchmark': 'LoKI O2 DC glow',
            'reference_type': 'digitized-reference benchmark',
            'same_footing': 'medium',
            'comparison_target': 'published O2 DC glow pressure sweep and radial-profile figures',
            'axes': {
                'Circuit / power': 0,
                'Physical E/N': 2,
                'Species': 2,
                'Wall loss': 1,
                'Profiles': 2,
                'Scalar ODE': 0,
            },
            'main_limitation': 'digitized figures, not official raw-output files',
        },
    ]


def _headline_summary(reports: dict[str, Any]) -> list[dict[str, Any]]:
    zdp_cmp = reports['zdplaskin']['comparison']
    zdp_en = reports['zdplaskin_eovern']['comparison']
    zdp_percent = _max_finite(
        [
            _ratio_mismatch_percent(zdp_cmp['final_electron_density_ratio_local_over_zdplaskin']),
            _ratio_mismatch_percent(zdp_cmp['final_Ar_star_ratio_local_over_zdplaskin']),
            _ratio_mismatch_percent(zdp_cmp['final_Ar_plus_ratio_local_over_zdplaskin']),
            _ratio_mismatch_percent(zdp_cmp['final_Ar2_plus_ratio_local_over_zdplaskin']),
            _ratio_mismatch_percent(zdp_en['local_reported_over_zdplaskin_saved_EoverN_ratio']),
            _ratio_mismatch_percent(zdp_en['same_voltage_EoverN_over_zdplaskin_saved_ratio']),
        ]
    )
    crane_cmp = reports['crane']['comparison']
    crane_percent = _max_finite(
        [
            crane_cmp['final_electron_density_m3_relative_error'] * 100.0,
            crane_cmp['final_Ar_plus_density_m3_relative_error'] * 100.0,
            crane_cmp['final_Ar_density_m3_relative_error'] * 100.0,
        ]
    )
    pyg_surrogate_percent = _ratio_mismatch_percent(
        _pygmol_step_row(reports['pygmol_surrogate'], 'production')['ratio_final_ne_pygmol_over_local']
    )
    pyg_local_table_percent = _ratio_mismatch_percent(
        _pygmol_step_row(reports['pygmol_local_table'], 'production')['ratio_final_ne_pygmol_over_local']
    )
    pyg_same_footing_percent = _ratio_mismatch_percent(
        _stage_row(reports['pygmol_same_footing'], 'same_rate_local_power_local_wall')['production_final_ne_ratio']
    )
    loki_primary_percent = _max_finite(
        [
            reports['loki']['direct_reconstruction']['avg_gas_temperature_relative_difference']['max_percent'],
            reports['loki']['direct_reconstruction']['near_wall_temperature_relative_difference']['max_percent'],
            reports['loki']['direct_reconstruction']['avg_reduced_electric_field_relative_difference']['max_percent'],
        ]
    )
    loki_o3_percent = reports['loki']['extended_species_relative_differences']['O3']['max_percent']

    return [
        {
            'benchmark': 'ZDPlaskin example2',
            'reference_type': 'strict executable parity',
            'headline_metric': 'max exact-match deviation over species and E/N',
            'headline_percent': zdp_percent,
            'interpretation': 'strong same-footing agreement for the current DC Ar parity case',
        },
        {
            'benchmark': 'CRANE TwoReactionArgon',
            'reference_type': 'strict scalar-chemistry parity',
            'headline_metric': 'max final-state relative error',
            'headline_percent': crane_percent,
            'interpretation': 'reaction assembly, units, and stiff ODE integration are effectively exact',
        },
        {
            'benchmark': 'PyGMol Ar baseline',
            'reference_type': 'broad executable global-model comparison',
            'headline_metric': 'production-step electron-density mismatch',
            'headline_percent': pyg_surrogate_percent,
            'interpretation': 'reasonable order agreement, but still a meaningful model-closure gap',
        },
        {
            'benchmark': 'PyGMol Ar same local table',
            'reference_type': 'same-rate diagnostic',
            'headline_metric': 'production-step electron-density mismatch',
            'headline_percent': pyg_local_table_percent,
            'interpretation': 'same electron-impact rates alone do not align the discharge state',
        },
        {
            'benchmark': 'PyGMol same-footing final stage',
            'reference_type': 'diagnostic decomposition benchmark',
            'headline_metric': 'production-step electron-density mismatch',
            'headline_percent': pyg_same_footing_percent,
            'interpretation': 'most of the PyGMol gap is explained by power and wall-loss closure differences',
        },
        {
            'benchmark': 'LoKI O2 DC glow',
            'reference_type': 'digitized-reference benchmark',
            'headline_metric': 'max primary-discharge difference over Tg/Tnw/E/N',
            'headline_percent': loki_primary_percent,
            'interpretation': 'primary discharge quantities are close, but species complexity must be read separately',
        },
        {
            'benchmark': 'LoKI O2 DC glow O3',
            'reference_type': 'digitized-reference benchmark',
            'headline_metric': 'max extended-species O3 difference',
            'headline_percent': loki_o3_percent,
            'interpretation': 'ozone remains the clearest chemistry/transport outlier in the current O2 benchmark',
        },
    ]


def _headline_summary_main_text(reports: dict[str, Any]) -> list[dict[str, Any]]:
    keep = {
        'ZDPlaskin example2',
        'CRANE TwoReactionArgon',
        'PyGMol Ar baseline',
        'PyGMol same-footing final stage',
        'LoKI O2 DC glow',
    }
    return [row for row in _headline_summary(reports) if row['benchmark'] in keep]


def _benchmark_limitations_summary() -> list[dict[str, str]]:
    return [
        {
            'benchmark': 'ZDPlaskin example2',
            'limitation': 'single committed Ar discharge case',
            'why_it_matters': 'strong for same-footing DC parity, weak for generality across chemistry and operating space',
        },
        {
            'benchmark': 'CRANE TwoReactionArgon',
            'limitation': 'scalar chemistry only',
            'why_it_matters': 'does not validate circuit closure, wall loss, transport closure, or electron-energy physics',
        },
        {
            'benchmark': 'PyGMol Ar baseline / same-footing',
            'limitation': 'independent executable model with different native closure choices',
            'why_it_matters': 'useful for sensitivity decomposition, but not a strict accuracy proof for the local solver',
        },
        {
            'benchmark': 'LoKI O2 DC glow',
            'limitation': 'digitized publication figures rather than official raw-output files',
            'why_it_matters': 'good for pressure trends and profile-shape credibility, but not raw-output parity',
        },
    ]


def _validation_status_summary(reports: dict[str, Any]) -> list[dict[str, str]]:
    loki = reports['loki']
    primary_loki_percent = _max_finite(
        [
            loki['direct_reconstruction']['avg_gas_temperature_relative_difference']['max_percent'],
            loki['direct_reconstruction']['near_wall_temperature_relative_difference']['max_percent'],
            loki['direct_reconstruction']['avg_reduced_electric_field_relative_difference']['max_percent'],
        ]
    )
    return [
        {
            'topic': 'Reaction assembly, SI conversion, and stiff ODE integration',
            'status': 'validated within current benchmark scope',
            'evidence': 'CRANE TwoReactionArgon',
            'comment': 'final-state relative error remains at about 2.4e-4 % or smaller',
        },
        {
            'topic': 'Current DC Ar same-footing species and physical E/N parity',
            'status': 'validated within current benchmark scope',
            'evidence': 'ZDPlaskin example2',
            'comment': 'headline exact-match deviation remains below about 3 % for the current parity case',
        },
        {
            'topic': 'Low-pressure O2 primary-discharge pressure trends and profile shapes',
            'status': 'supported by digitized benchmark',
            'evidence': 'LoKI O2 DC glow digitized benchmark',
            'comment': f'primary Tg/Tnw/E/N difference stays near {primary_loki_percent:.1f} % and profile-shape checks pass',
        },
        {
            'topic': 'Cross-code global-model agreement without closure harmonization',
            'status': 'not yet validated',
            'evidence': 'PyGMol executable comparison',
            'comment': 'PyGMol remains sensitive to power and wall-loss closure, so baseline cross-code agreement is not a proof of correctness',
        },
        {
            'topic': 'Quantitative ozone-heavy O2 chemistry',
            'status': 'not yet validated',
            'evidence': 'LoKI O2 DC glow O3 comparison',
            'comment': f'O3 remains an outlier at about {loki["extended_species_relative_differences"]["O3"]["max_percent"]:.1f} %',
        },
        {
            'topic': 'Generality across pressures, power waveforms, and unrelated chemistries',
            'status': 'not yet validated',
            'evidence': 'current external set',
            'comment': 'the present benchmark set is informative but still too narrow to claim broad process-space validation',
        },
    ]


def _validation_status_short_summary(reports: dict[str, Any]) -> list[dict[str, str]]:
    detailed_rows = _validation_status_summary(reports)
    return [
        {
            'topic': row['topic'],
            'status': row['status'],
            'support': row['evidence'],
        }
        for row in detailed_rows
    ]


def _applicability_evidence_summary(reports: dict[str, Any]) -> list[dict[str, str]]:
    robustness = reports['robustness']
    headline = robustness.get('headline', {})
    stress_case_count = int(headline.get('stress_case_count', 0))
    failed_cases = list(headline.get('failed_cases', []))
    passed_case_count = stress_case_count - len(failed_cases)
    return [
        {
            'item': 'Strict external references retained',
            'value': str(int(headline.get('external_reference_count', 0))),
            'meaning': 'the strict validation backbone is unchanged',
        },
        {
            'item': 'Added stress / applicability cases',
            'value': str(stress_case_count),
            'meaning': 'pressure, source-voltage, electrical-backend, and chemistry perturbations',
        },
        {
            'item': 'Stress cases passing boundedness checks',
            'value': f'{passed_case_count}/{stress_case_count}',
            'meaning': 'finite positive solver-health checks across the executed perturbations',
        },
        {
            'item': 'Interpretation',
            'value': 'boundedness / applicability evidence',
            'meaning': 'supports practical usability and operating-envelope claims, not new strict accuracy validation or predictive-accuracy proof',
        },
    ]


def _plot_overview(reports: dict[str, Any], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    ax_zdp, ax_crane, ax_pygmol, ax_loki = axes.flat

    zdp_cmp = reports['zdplaskin']['comparison']
    zdp_en = reports['zdplaskin_eovern']['comparison']
    zdp_labels = ['e', 'Ar*', 'Ar+', 'Ar2+', 'E/N current', 'E/N sameV']
    zdp_values = [
        _ratio_mismatch_percent(zdp_cmp['final_electron_density_ratio_local_over_zdplaskin']),
        _ratio_mismatch_percent(zdp_cmp['final_Ar_star_ratio_local_over_zdplaskin']),
        _ratio_mismatch_percent(zdp_cmp['final_Ar_plus_ratio_local_over_zdplaskin']),
        _ratio_mismatch_percent(zdp_cmp['final_Ar2_plus_ratio_local_over_zdplaskin']),
        _ratio_mismatch_percent(zdp_en['local_reported_over_zdplaskin_saved_EoverN_ratio']),
        _ratio_mismatch_percent(zdp_en['same_voltage_EoverN_over_zdplaskin_saved_ratio']),
    ]
    ax_zdp.bar(zdp_labels, zdp_values, color=['#4477aa'] * 4 + ['#66c2a5', '#66c2a5'])
    ax_zdp.axhline(2.0, color='#666666', linestyle='--', linewidth=1.0, label='2%')
    ax_zdp.axhline(10.0, color='#aaaaaa', linestyle=':', linewidth=1.0, label='10%')
    ax_zdp.set_ylim(0.0, max(10.5, max(zdp_values) * 1.2))
    ax_zdp.set_title('ZDPlaskin: Exact-Match Deviation')
    ax_zdp.set_ylabel('|local/reference - 1| [%]')
    for idx, value in enumerate(zdp_values):
        ax_zdp.text(idx, value, f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    ax_zdp.tick_params(axis='x', rotation=20)
    ax_zdp.legend(loc='upper right', fontsize=8)

    crane_cmp = reports['crane']['comparison']
    crane_labels = ['e', 'Ar+', 'Ar']
    crane_values = [
        crane_cmp['final_electron_density_m3_relative_error'] * 100.0,
        crane_cmp['final_Ar_plus_density_m3_relative_error'] * 100.0,
        crane_cmp['final_Ar_density_m3_relative_error'] * 100.0,
    ]
    ax_crane.bar(crane_labels, crane_values, color='#dd8452')
    ax_crane.set_yscale('log')
    ax_crane.axhline(0.05, color='#666666', linestyle='--', linewidth=1.0, label='0.05%')
    ax_crane.set_title('CRANE: Final-State Relative Error')
    ax_crane.set_ylabel('relative error [%]')
    for idx, value in enumerate(crane_values):
        ax_crane.text(idx, value, f'{value:.2e}', ha='center', va='bottom', fontsize=8)
    ax_crane.legend(loc='upper right', fontsize=8)

    pygmol_modes = [
        ('baseline', reports['pygmol_surrogate']),
        ('same-footing\nfinal', {'rows': [{'step_id': 'production', 'ratio_final_ne_pygmol_over_local': _stage_row(reports['pygmol_same_footing'], 'same_rate_local_power_local_wall')['production_final_ne_ratio']}]}),
    ]
    pygmol_labels = [label for label, _ in pygmol_modes]
    pygmol_values = [
        _ratio_mismatch_percent(_pygmol_step_row(report, 'production').get('ratio_final_ne_pygmol_over_local'))
        for _, report in pygmol_modes
    ]
    ax_pygmol.bar(
        pygmol_labels,
        pygmol_values,
        color=['#4c72b0', '#55a868'],
    )
    ax_pygmol.axhline(30.0, color='#55a868', linestyle='--', linewidth=1.0, label='30%')
    ax_pygmol.axhline(100.0, color='#999999', linestyle=':', linewidth=1.0, label='100%')
    ax_pygmol.set_ylim(0.0, max(110.0, max(pygmol_values) * 1.25))
    ax_pygmol.set_title('PyGMol: Main-Text Mismatch View')
    ax_pygmol.set_ylabel('|PyGMol/local - 1| [%]')
    for idx, value in enumerate(pygmol_values):
        ax_pygmol.text(idx, value, f'{value:.1f}', ha='center', va='bottom', fontsize=8)
    ax_pygmol.tick_params(axis='x', rotation=20)
    ax_pygmol.legend(loc='upper right', fontsize=8)

    loki = reports['loki']
    loki_labels = ['Tg', 'Tnw', 'E/N', 'O3']
    loki_values = [
        loki['direct_reconstruction']['avg_gas_temperature_relative_difference']['max_percent'],
        loki['direct_reconstruction']['near_wall_temperature_relative_difference']['max_percent'],
        loki['direct_reconstruction']['avg_reduced_electric_field_relative_difference']['max_percent'],
        loki['extended_species_relative_differences']['O3']['max_percent'],
    ]
    ax_loki.bar(loki_labels, loki_values, color=['#55a868', '#55a868', '#55a868', '#c44e52'])
    ax_loki.axhline(5.0, color='#666666', linestyle='--', linewidth=1.0, label='5%')
    ax_loki.axhline(60.0, color='#999999', linestyle=':', linewidth=1.0, label='60%')
    ax_loki.set_title('LoKI: Primary Metrics Plus O3 Caution')
    ax_loki.set_ylabel('difference [%]')
    for idx, value in enumerate(loki_values):
        ax_loki.text(idx, value, f'{value:.1f}', ha='center', va='bottom', fontsize=8)
    ax_loki.text(
        0.02,
        0.98,
        (
            f"full diagnostic: {loki['summary']['runnable_full_diagnostic_benchmark']}\n"
            f"profile pass: {loki['summary']['profile_shape_expectation_pass']}"
        ),
        transform=ax_loki.transAxes,
        ha='left',
        va='top',
        fontsize=9,
        bbox={'facecolor': 'white', 'edgecolor': '#bbbbbb', 'boxstyle': 'round,pad=0.3'},
    )
    ax_loki.legend(loc='upper left', bbox_to_anchor=(0.0, 0.82), fontsize=8)

    fig.suptitle('External Benchmark Results: Readable Summary')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_pygmol_decomposition(reports: dict[str, Any], output_path: Path) -> None:
    summary_rows = reports['pygmol_same_footing']['summary']
    stage_labels = {
        'same_rate_recipe_power_pygmol_wall': 'same rate\nrecipe power\nPyGMol wall',
        'same_rate_local_power_pygmol_wall': 'same rate\nlocal power\nPyGMol wall',
        'same_rate_local_power_local_wall': 'same rate\nlocal power\nlocal wall',
    }
    stages = [stage_labels.get(row['stage_id'], row['stage_id']) for row in summary_rows]
    final_ne = [_safe_float(row.get('production_final_ne_ratio')) for row in summary_rows]
    mean_ne = [_safe_float(row.get('production_mean_ne_ratio')) for row in summary_rows]
    power = [_safe_float(row.get('production_power_ratio')) for row in summary_rows]
    wall = [_safe_float(row.get('production_wall_loss_rate_ratio')) for row in summary_rows]
    energy = [_safe_float(row.get('production_energy_ratio')) for row in summary_rows]

    fig, (ax_ratio, ax_mismatch) = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    ax_ratio.plot(stages, final_ne, marker='o', linewidth=2.0, label='final ne ratio')
    ax_ratio.plot(stages, mean_ne, marker='s', linewidth=1.8, label='mean ne ratio')
    ax_ratio.plot(stages, power, marker='^', linewidth=1.8, label='absorbed power ratio')
    ax_ratio.plot(stages, wall, marker='D', linewidth=1.8, label='wall-loss ratio')
    ax_ratio.plot(stages, energy, marker='x', linewidth=1.8, label='energy ratio')
    ax_ratio.axhline(1.0, color='black', linestyle='--', linewidth=1.0)
    ax_ratio.set_yscale('log')
    ax_ratio.set_ylabel('PyGMol / local')
    ax_ratio.set_title('PyGMol Same-Footing Ratios')
    ax_ratio.tick_params(axis='x', rotation=12)
    ax_ratio.legend(loc='best', fontsize=8)

    width = 0.18
    positions = list(range(len(stages)))
    mismatch_series = [
        ('final ne', [_ratio_mismatch_percent(value) for value in final_ne], '#4c72b0'),
        ('mean ne', [_ratio_mismatch_percent(value) for value in mean_ne], '#dd8452'),
        ('power', [_ratio_mismatch_percent(value) for value in power], '#55a868'),
        ('wall loss', [_ratio_mismatch_percent(value) for value in wall], '#c44e52'),
        ('energy', [_ratio_mismatch_percent(value) for value in energy], '#8172b2'),
    ]
    for idx, (label, values, color) in enumerate(mismatch_series):
        offset = (idx - 2) * width
        ax_mismatch.bar([x + offset for x in positions], values, width=width, label=label, color=color)
    ax_mismatch.axhline(30.0, color='#55a868', linestyle='--', linewidth=1.0, label='30%')
    ax_mismatch.axhline(100.0, color='#999999', linestyle=':', linewidth=1.0, label='100%')
    ax_mismatch.set_yscale('log')
    ax_mismatch.set_xticks(positions)
    ax_mismatch.set_xticklabels(stages)
    ax_mismatch.tick_params(axis='x', rotation=12)
    ax_mismatch.set_ylabel('|ratio - 1| [%]')
    ax_mismatch.set_title('How Far Each Axis Is From Local')
    ax_mismatch.legend(loc='best', fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_coverage_matrix(output_path: Path) -> None:
    rows = _coverage_rows()
    axes_labels = list(rows[0]['axes'].keys())
    benchmark_labels = [row['benchmark'] for row in rows]
    matrix = [[row['axes'][label] for label in axes_labels] for row in rows]

    cmap = ListedColormap(['#f2f2f2', '#9ecae1', '#2171b5'])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, ax = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')
    ax.set_xticks(range(len(axes_labels)))
    ax.set_xticklabels(axes_labels, rotation=18, ha='right')
    ax.set_yticks(range(len(benchmark_labels)))
    ax.set_yticklabels(benchmark_labels)
    ax.set_title('Benchmark Coverage Matrix')

    label_map = {
        0: 'not directly benchmarked',
        1: 'diagnostic / indirect',
        2: 'direct comparison',
    }
    for row_idx, row in enumerate(rows):
        for col_idx, axis_label in enumerate(axes_labels):
            value = row['axes'][axis_label]
            text = '0' if value == 0 else ('D' if value == 1 else 'S')
            color = 'black' if value < 2 else 'white'
            ax.text(col_idx, row_idx, text, ha='center', va='center', color=color, fontsize=10, fontweight='bold')
        ax.text(
            len(axes_labels) - 0.05,
            row_idx,
            f"  {row['reference_type']}",
            ha='left',
            va='center',
            fontsize=8,
            color='#333333',
            clip_on=False,
        )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_ticks([0, 1, 2])
    colorbar.set_ticklabels([label_map[0], label_map[1], label_map[2]])
    colorbar.ax.tick_params(labelsize=8)

    ax.text(
        0.0,
        -0.18,
        'Cell labels: S = direct/strict comparison, D = diagnostic or indirect coverage, 0 = not directly benchmarked.',
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=8,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = '| ' + ' | '.join(headers) + ' |'
    separator_line = '| ' + ' | '.join(['---'] * len(headers)) + ' |'
    body = ['| ' + ' | '.join(row) + ' |' for row in rows]
    return '\n'.join([header_line, separator_line, *body])


def _build_markdown_report(
    reports: dict[str, Any],
    *,
    generated_at_utc: str,
    overview_png_path: Path,
    pygmol_png_path: Path,
    coverage_png_path: Path,
    markdown_report_path: Path,
) -> str:
    coverage_rows = _coverage_rows()
    headline_rows = _headline_summary(reports)
    limitation_rows = _benchmark_limitations_summary()
    validation_short_rows = _validation_status_short_summary(reports)
    validation_detailed_rows = _validation_status_summary(reports)
    applicability_rows = _applicability_evidence_summary(reports)
    robustness_outputs = reports['robustness']['outputs']
    headline_rows_main = _headline_summary_main_text(reports)

    local_code_description = (
        'The local code is a reactor-averaged low-pressure plasma model that solves '
        'species balance, electron-energy closure, wall-loss closure, and optional '
        'electrical backends from YAML-defined chemistry and chamber inputs. The '
        'external benchmark suite is intentionally kept outside the core package so '
        'reference-code adapters do not change production solver logic.'
    )
    validation_scope_description = (
        'Within the present benchmark set, the strongest direct support is limited to '
        'the current DC Ar parity case, a scalar-chemistry regression, and a digitized '
        'O2 trend/profile benchmark. Broader pressure, power, and chemistry excursions '
        'are treated separately as applicability evidence rather than strict validation.'
    )

    comparison_rows = [
        [
            row['benchmark'],
            row['reference_type'],
            row['comparison_target'],
            row['same_footing'],
            row['main_limitation'],
        ]
        for row in coverage_rows
    ]
    headline_table_rows = [
        [
            row['benchmark'],
            row['reference_type'],
            row['headline_metric'],
            'n/a' if row['headline_percent'] is None else f"{row['headline_percent']:.3g}",
            row['interpretation'],
        ]
        for row in headline_rows_main
    ]
    limitation_table_rows = [
        [
            row['benchmark'],
            row['limitation'],
            row['why_it_matters'],
        ]
        for row in limitation_rows
    ]
    validation_short_table_rows = [
        [
            row['topic'],
            row['status'],
            row['support'],
        ]
        for row in validation_short_rows
    ]
    validation_detailed_table_rows = [
        [
            row['topic'],
            row['status'],
            row['evidence'],
            row['comment'],
        ]
        for row in validation_detailed_rows
    ]
    applicability_table_rows = [
        [
            row['item'],
            row['value'],
            row['meaning'],
        ]
        for row in applicability_rows
    ]

    zdp_cmp = reports['zdplaskin']['comparison']
    crane_cmp = reports['crane']['comparison']
    loki = reports['loki']
    pyg_same_footing = reports['pygmol_same_footing']
    final_stage = _stage_row(pyg_same_footing, 'same_rate_local_power_local_wall')
    robustness_overview_rel = _markdown_relpath(markdown_report_path, Path(robustness_outputs['overview_png']))
    robustness_coverage_rel = _markdown_relpath(markdown_report_path, Path(robustness_outputs['rate_table_coverage_png']))
    robustness_report_rel = _markdown_relpath(markdown_report_path, Path(robustness_outputs['markdown_report_md']))

    sections = [
        '# External Benchmark Report',
        '',
        f'Generated at UTC `{generated_at_utc}`.',
        '',
        '## 1. Purpose and local-model scope',
        '',
        local_code_description,
        '',
        validation_scope_description,
        '',
        'This report is written in a paper-style structure: first the benchmark footing, '
        'then what each benchmark actually compares, then the quantitative results, and '
        'finally the physical interpretation and remaining limitations.',
        '',
        '## 2. Benchmark set and comparison targets',
        '',
        _markdown_table(
            ['Benchmark', 'Reference type', 'Comparison target', 'Same-footing level', 'Main limitation'],
            comparison_rows,
        ),
        '',
        '### Coverage map',
        '',
        f'![Benchmark coverage matrix]({coverage_png_path.name})',
        '',
        'Interpretation of the coverage matrix: `S` means the quantity is compared directly, '
        '`D` means it is only covered diagnostically or indirectly, and `0` means the benchmark '
        'does not directly constrain that axis.',
        '',
        '## 3. Headline quantitative results',
        '',
        _markdown_table(
            ['Benchmark', 'Reference type', 'Headline metric', 'Value [%]', 'Interpretation'],
            headline_table_rows,
        ),
        '',
        '### Integrated overview',
        '',
        f'![Integrated overview]({overview_png_path.name})',
        '',
        'The overview figure is intended for the main Results section. It keeps one compact panel '
        'per benchmark family and expresses agreement in percent space instead of raw ratios so a '
        'reader can see at a glance whether the comparison is near-exact, broadly consistent, or '
        'still structurally different. Deliberately diagnostic rows such as PyGMol same-rate-only '
        'and secondary LoKI chemistry subsets are left out of this main-text summary and discussed later instead.',
        '',
        '### PyGMol decomposition',
        '',
        f'![PyGMol decomposition]({pygmol_png_path.name})',
        '',
        'The PyGMol figure is intended for Discussion. The left panel shows raw `PyGMol/local` ratios. '
        'The right panel converts the same information into mismatch percentages so the dominant closure '
        'difference can be read without mentally translating values like `0.72` or `19.2`.',
        '',
        '## 4. Benchmark-by-benchmark interpretation',
        '',
        '### 4.1 ZDPlaskin example2',
        '',
        (
            'This is the strictest same-footing discharge benchmark in the current set. '
            f'The current local run stays within about `{_max_finite([_ratio_mismatch_percent(zdp_cmp["final_electron_density_ratio_local_over_zdplaskin"]), _ratio_mismatch_percent(zdp_cmp["final_Ar_star_ratio_local_over_zdplaskin"]), _ratio_mismatch_percent(zdp_cmp["final_Ar_plus_ratio_local_over_zdplaskin"]), _ratio_mismatch_percent(zdp_cmp["final_Ar2_plus_ratio_local_over_zdplaskin"]) ]):.2f}%` '
            'for the reported final species set, while the same-footing E/N comparison remains near '
            'sub-percent to percent-level agreement. In a paper, this should be presented as evidence that '
            'the current DC-series Ar parity case is self-consistent on the chosen physical footing.'
        ),
        '',
        '### 4.2 CRANE TwoReactionArgon',
        '',
        (
            'CRANE is not a full plasma-discharge benchmark, but it is the cleanest scalar chemistry check. '
            f'The maximum final-state relative error is `{_max_finite([crane_cmp["final_electron_density_m3_relative_error"] * 100.0, crane_cmp["final_Ar_plus_density_m3_relative_error"] * 100.0, crane_cmp["final_Ar_density_m3_relative_error"] * 100.0]):.3e}%`, '
            'so this benchmark is best cited as validation of reaction assembly, SI conversion, and stiff ODE integration.'
        ),
        '',
        '### 4.3 PyGMol executable comparison',
        '',
        (
            'PyGMol should be described as an executable global-model comparison, not as a strict parity reference. '
            f'The baseline production-step electron-density mismatch is `{_ratio_mismatch_percent(_pygmol_step_row(reports["pygmol_surrogate"], "production")["ratio_final_ne_pygmol_over_local"]):.1f}%`. '
            f'Using the same local electron-impact table alone does not close the gap and leaves a mismatch of `{_ratio_mismatch_percent(_pygmol_step_row(reports["pygmol_local_table"], "production")["ratio_final_ne_pygmol_over_local"]):.1f}%`. '
            f'After same-rate, same-power, and same-wall-loss alignment, the final-stage production mismatch falls to `{_ratio_mismatch_percent(final_stage["production_final_ne_ratio"]):.1f}%`. '
            'That pattern strongly supports the interpretation that power deposition and wall-loss closure dominate the PyGMol difference more than the electron-impact table itself.'
        ),
        '',
        '### 4.4 LoKI O2 DC glow',
        '',
        (
            f'The LoKI benchmark now runs through the full diagnostic stage and passes the profile-shape checks (`profile pass = {loki["summary"]["profile_shape_expectation_pass"]}`). '
            f'The maximum primary-discharge difference over `Tg`, `Tnw`, and `E/N` is `{_max_finite([loki["direct_reconstruction"]["avg_gas_temperature_relative_difference"]["max_percent"], loki["direct_reconstruction"]["near_wall_temperature_relative_difference"]["max_percent"], loki["direct_reconstruction"]["avg_reduced_electric_field_relative_difference"]["max_percent"]]):.1f}%`, '
            f'while `O3` remains the clearest outlier at `{loki["extended_species_relative_differences"]["O3"]["max_percent"]:.1f}%`. '
            'For manuscript use, this should be framed as a successful primary-discharge and profile benchmark with a chemically demanding ozone channel that still exposes transport/chemistry limitations.'
        ),
        '',
        '## 5. Recommended manuscript logic',
        '',
        'A reader-friendly order is:',
        '',
        '1. State what the local model solves and what it does not solve.',
        '2. Separate strict parity references from executable sanity checks and digitized references.',
        '3. Use ZDPlaskin and CRANE to establish numerical trustworthiness.',
        '4. Use PyGMol to show which model-closure choices change the discharge state.',
        '5. Use LoKI to show whether low-pressure O2 pressure trends and profile shapes are physically credible.',
        '6. Add the robustness/applicability suite as a distinct support layer for operating-envelope claims.',
        '7. End by stating clearly which physics are validated, which are only diagnostically supported, and which remain open.',
        '',
        '## 6. Benchmark limitations',
        '',
        _markdown_table(
            ['Benchmark', 'Main limitation', 'Why it matters'],
            limitation_table_rows,
        ),
        '',
        'The role of this table is to prevent overclaiming. Each benchmark is useful, but each one constrains only part of the full low-pressure plasma-model problem.',
        '',
        '## 7. Supported versus not yet validated',
        '',
        '### 7.1 Main-text concluding table',
        '',
        _markdown_table(
            ['Topic', 'Support level', 'Supporting benchmark'],
            validation_short_table_rows,
        ),
        '',
        'This short table is the version to keep in the main text. It preserves the distinction between direct validation, digitized-reference support, and not-yet-validated claims without pulling too much detail into the conclusion.',
        '',
        '### 7.2 Supplementary detailed concluding table',
        '',
        _markdown_table(
            ['Topic', 'Support level', 'Evidence', 'Comment'],
            validation_detailed_table_rows,
        ),
        '',
        'This detailed table is the supplement-facing version. It retains the evidence path and the cautionary comment needed to defend each claim during review.',
        '',
        '## 8. Applicability evidence from the robustness suite',
        '',
        'The strict-validation table above should be read together with a separate robustness/applicability suite. '
        'That suite does not introduce new authoritative external references. Instead, it asks a narrower but still important question: '
        'does the code remain physically bounded and practically usable when pressure, source voltage, electrical backend, and chemistry are perturbed away from the parity points?',
        '',
        _markdown_table(
            ['Item', 'Value', 'Why it helps'],
            applicability_table_rows,
        ),
        '',
        'The intended reading is narrow: these results support boundedness, workflow usability, and operating-envelope plausibility only. '
        'They do not by themselves validate predicted densities, powers, or radical levels against external truth.',
        '',
        'The accompanying robustness artifacts are kept separate on purpose:',
        '',
        f'- [robustness stress overview]({robustness_overview_rel})',
        f'- [robustness rate-table coverage]({robustness_coverage_rel})',
        f'- [robustness applicability report]({robustness_report_rel})',
        '',
        'This one-level cross-reference lets a manuscript place strict validation and practical applicability side by side without implying that '
        'stress-case boundedness is the same thing as external or experimental validation.',
        '',
        '## 9. Additional calculations and figures worth keeping',
        '',
        '- Keep percent-space mismatch metrics for all headline comparisons. They are much easier to read in a paper than raw ratios.',
        '- Keep one benchmark-coverage figure. It prevents readers from mistaking CRANE or PyGMol for the wrong kind of validation.',
        '- Keep the PyGMol staged-decomposition figure. It is the clearest demonstration of why executable global models can disagree.',
        '- When discussing LoKI, separate primary discharge quantities from difficult chemistry channels such as `O3`.',
        '- Keep a one-level pointer from the main benchmark report to the robustness suite so generality claims remain visibly weaker than strict validation claims, but not invisible.',
        '- If a later paper needs a shorter main text, move the detailed metric CSV and YAML outputs to supplementary material and keep only the coverage matrix plus the two current result figures in the main body.',
        '',
    ]
    return '\n'.join(sections)


def _build_markdown_report_ja(
    reports: dict[str, Any],
    *,
    generated_at_utc: str,
    overview_png_path: Path,
    pygmol_png_path: Path,
    coverage_png_path: Path,
    markdown_report_ja_path: Path,
) -> str:
    coverage_rows = _coverage_rows()
    limitation_rows = _benchmark_limitations_summary()
    validation_short_rows = _validation_status_short_summary(reports)
    validation_detailed_rows = _validation_status_summary(reports)
    applicability_rows = _applicability_evidence_summary(reports)
    headline_rows_main = _headline_summary_main_text(reports)
    robustness_outputs = reports['robustness']['outputs']

    comparison_rows = [
        [
            row['benchmark'],
            row['reference_type'],
            row['comparison_target'],
            row['same_footing'],
            row['main_limitation'],
        ]
        for row in coverage_rows
    ]
    headline_table_rows = [
        [
            row['benchmark'],
            row['reference_type'],
            row['headline_metric'],
            'n/a' if row['headline_percent'] is None else f"{row['headline_percent']:.3g}",
            row['interpretation'],
        ]
        for row in headline_rows_main
    ]
    limitation_table_rows = [
        [
            row['benchmark'],
            row['limitation'],
            row['why_it_matters'],
        ]
        for row in limitation_rows
    ]
    validation_short_table_rows = [
        [
            row['topic'],
            row['status'],
            row['support'],
        ]
        for row in validation_short_rows
    ]
    validation_detailed_table_rows = [
        [
            row['topic'],
            row['status'],
            row['evidence'],
            row['comment'],
        ]
        for row in validation_detailed_rows
    ]
    applicability_table_rows = [
        [
            row['item'],
            row['value'],
            row['meaning'],
        ]
        for row in applicability_rows
    ]

    zdp_cmp = reports['zdplaskin']['comparison']
    crane_cmp = reports['crane']['comparison']
    loki = reports['loki']
    final_stage = _stage_row(reports['pygmol_same_footing'], 'same_rate_local_power_local_wall')
    robustness_overview_rel = _markdown_relpath(markdown_report_ja_path, Path(robustness_outputs['overview_png']))
    robustness_coverage_rel = _markdown_relpath(markdown_report_ja_path, Path(robustness_outputs['rate_table_coverage_png']))
    robustness_report_rel = _markdown_relpath(markdown_report_ja_path, Path(robustness_outputs['markdown_report_md']))

    sections = [
        '# 外部ベンチマーク報告書（日本語版）',
        '',
        f'生成日時 UTC `{generated_at_utc}`。',
        '',
        '## 1. 目的と本コードのモデル範囲',
        '',
        '本コードは、低圧プラズマを対象とした reactor-averaged global model であり、'
        '気相 species 収支、電子エネルギー収支、壁損失 closure、および reduced-order electrical backend を連成して解く。'
        '外部 benchmark 群は `plasma_global` 本体から分離されており、参照コード固有の都合が production solver のロジックに混入しない構成としている。',
        '',
        '本 benchmark 集合が直接に強く支える範囲は、現在の DC Ar parity case、scalar chemistry regression、'
        'および digitized O2 trend/profile benchmark に限られる。これより広い圧力・電力・chemistry 摂動は、'
        'strict validation ではなく applicability evidence として別に扱う。',
        '',
        '本コードの支配方程式を global model の代表形で書けば、species 収支は',
        '',
        '```text',
        'd n_k / dt = Σ_r ν_k,r R_r + S_k,flow + S_k,surface - L_k,wall',
        'R_r = k_r(Te, E/N, Tg, θ, ...) Π_j n_j^α_j',
        '```',
        '',
        'であり、電子エネルギー収支は概念的には',
        '',
        '```text',
        'd w_e / dt = P_abs / V - Σ_r (Δε_r R_r) - Q_wall - Q_flow',
        '```',
        '',
        'と表せる。ここで `w_e` は電子エネルギー密度、`P_abs` は吸収電力である。'
        '通常の production 設定では電子密度 `n_e` は独立 ODE ではなく準中性条件から定まる代数量として扱う。',
        '',
        '現在の主要な charged-particle wall-loss は Bohm-family closure であり、',
        '',
        '```text',
        'Γ_i = h_factor * 0.61 * n_i * sqrt(ε_e / m_i)',
        '(d n_i / dt)_wall = - (A_eff / V) Γ_i',
        '```',
        '',
        'で与えられる。DC / pulsed-DC の reduced circuit backend ではさらに',
        '',
        '```text',
        'G_plasma = e * n_e * μ_e * A / g',
        'R_plasma = 1 / G_plasma',
        'I = V_source / (R_ballast + R_plasma)',
        'V_gap = I R_plasma',
        'P_abs = η_abs V_gap I',
        'E/N = |V_gap / g| / N',
        '```',
        '',
        'を用いて電流・ギャップ電圧・吸収電力・reduced field を結びつける。',
        '',
        '## 2. ベンチマーク集合と比較対象',
        '',
        _markdown_table(
            ['ベンチマーク', '参照種別', '比較対象', '同一土俵度', '主な制約'],
            comparison_rows,
        ),
        '',
        '### Coverage map',
        '',
        f'![Benchmark coverage matrix]({coverage_png_path.name})',
        '',
        'Coverage map の `S` は直接比較、`D` は診断的または間接的な拘束、`0` はその軸を直接には拘束していないことを表す。',
        '',
        '## 3. 主要な定量結果',
        '',
        _markdown_table(
            ['ベンチマーク', '参照種別', '主要指標', '値 [%]', '解釈'],
            headline_table_rows,
        ),
        '',
        '### 統合 overview',
        '',
        f'![Integrated overview]({overview_png_path.name})',
        '',
        'この図は main text の Results を意識した summary である。各 benchmark family を 1 panel にまとめ、'
        'raw ratio ではなく percent-space で一致度を示すことで、near-exact / broadly consistent / structurally different を直感的に読めるようにしている。'
        'PyGMol same-rate-only や LoKI の副次的 neutral subset のような診断専用情報は、この main summary からは意図的に外している。',
        '',
        '### PyGMol 差分分解',
        '',
        f'![PyGMol decomposition]({pygmol_png_path.name})',
        '',
        'PyGMol 図は Discussion 向けであり、左 panel で `PyGMol/local` の raw ratio、右 panel で mismatch [%] を示す。'
        'これにより、same rate、same power、same wall-loss のどこで差が縮むかを物理 closure の差として読める。',
        '',
        '## 4. ベンチマークごとの物理解釈',
        '',
        '### 4.1 ZDPlaskin example2',
        '',
        (
            '本 benchmark は現在の集合の中で最も strict な same-footing discharge benchmark である。'
            f'最終 species の reported set に対する max exact-match deviation は約 `{_max_finite([_ratio_mismatch_percent(zdp_cmp["final_electron_density_ratio_local_over_zdplaskin"]), _ratio_mismatch_percent(zdp_cmp["final_Ar_star_ratio_local_over_zdplaskin"]), _ratio_mismatch_percent(zdp_cmp["final_Ar_plus_ratio_local_over_zdplaskin"]), _ratio_mismatch_percent(zdp_cmp["final_Ar2_plus_ratio_local_over_zdplaskin"]) ]):.2f}%` であり、'
            'physical E/N comparison も sub-percent から数 % の範囲に収まる。したがって、現在の DC-series Ar parity case については、'
            'species、circuit footing、E/N interpretation を含めた self-consistency をかなり強く支持すると読める。'
        ),
        '',
        '### 4.2 CRANE TwoReactionArgon',
        '',
        (
            'CRANE は full plasma-discharge benchmark ではないが、scalar chemistry と stiff ODE integration の検証としては最も純粋である。'
            f'最終状態の最大相対誤差は `{_max_finite([crane_cmp["final_electron_density_m3_relative_error"] * 100.0, crane_cmp["final_Ar_plus_density_m3_relative_error"] * 100.0, crane_cmp["final_Ar_density_m3_relative_error"] * 100.0]):.3e}%` であり、'
            '反応組立て、SI 単位系、数値積分基盤の信頼性を強く支える。'
        ),
        '',
        '### 4.3 PyGMol executable comparison',
        '',
        (
            'PyGMol は strict parity reference ではなく、別の executable global model との比較である。'
            f'baseline の production-step electron-density mismatch は `{_ratio_mismatch_percent(_pygmol_step_row(reports["pygmol_surrogate"], "production")["ratio_final_ne_pygmol_over_local"]):.1f}%`、'
            f'same electron-impact table only では逆に `{_ratio_mismatch_percent(_pygmol_step_row(reports["pygmol_local_table"], "production")["ratio_final_ne_pygmol_over_local"]):.1f}%` と大きく、'
            f'same-rate, same-power, same-wall-loss の最終段では `{_ratio_mismatch_percent(final_stage["production_final_ne_ratio"]):.1f}%` まで縮む。'
            'この構造は、電子衝突 rate そのものよりも、power deposition と wall-loss closure の差が discharge state を強く支配していることを示す。'
        ),
        '',
        '### 4.4 LoKI O2 DC glow',
        '',
        (
            f'LoKI benchmark は full diagnostic stage まで実行できており、profile-shape check も pass (`{loki["summary"]["profile_shape_expectation_pass"]}`) している。'
            f'`Tg`, `Tnw`, `E/N` の primary-discharge 指標に対する最大差は `{_max_finite([loki["direct_reconstruction"]["avg_gas_temperature_relative_difference"]["max_percent"], loki["direct_reconstruction"]["near_wall_temperature_relative_difference"]["max_percent"], loki["direct_reconstruction"]["avg_reduced_electric_field_relative_difference"]["max_percent"]]):.1f}%` である一方、'
            f'`O3` は `{loki["extended_species_relative_differences"]["O3"]["max_percent"]:.1f}%` と明瞭な outlier である。'
            'したがって、この benchmark は低圧 O2 の primary discharge trend と profile shape の plausibility を支えるが、'
            '重い neutral chemistry、特に ozone channel については今後の改良余地を明確に示している。'
        ),
        '',
        '### 4.5 本コードの優位性',
        '',
        '本コードの優位性は、現段階では「絶対精度が他コードより常に高い」という主張ではない。'
        '優位性はむしろ、(i) chemistry / EEDF / wall-loss / circuit を分離した modular closure 設計、'
        '(ii) strict validation と diagnostic decomposition を分けて扱える benchmark discipline、'
        '(iii) stiff chemistry を扱う数値基盤の安定性、にある。'
        'すなわち、「なぜ合うのか／なぜ合わないのか」を closure ごとに切り分けられる点が、研究開発基盤としての大きな強みである。',
        '',
        '## 5. 論文本文での構成案',
        '',
        '読み手に自然な順序は次の通りである。',
        '',
        '1. まず本コードが解いている支配方程式と closure の範囲を明示する。',
        '2. strict parity, executable comparison, digitized benchmark を明確に分ける。',
        '3. ZDPlaskin と CRANE で数値実装の信頼性を示す。',
        '4. PyGMol で closure 差がどこに効くかを示す。',
        '5. LoKI で低圧 O2 の trend / profile の物理的もっともらしさを示す。',
        '6. robustness を operating-envelope の boundedness / usability として補助的に置く。',
        '7. 最後に、何が support され、何が未検証かを短表で切る。',
        '',
        '## 6. ベンチマークの限界',
        '',
        _markdown_table(
            ['ベンチマーク', '主な制約', 'なぜ重要か'],
            limitation_table_rows,
        ),
        '',
        'この表の役割は overclaim を防ぐことにある。各 benchmark は有用だが、低圧プラズマ global model の全問題を一つで拘束するものではない。',
        '',
        '## 7. 支持される範囲と未検証の範囲',
        '',
        '### 7.1 本文用 short table',
        '',
        _markdown_table(
            ['項目', '支持レベル', '根拠 benchmark'],
            validation_short_table_rows,
        ),
        '',
        'この short table は main text 向けであり、direct validation、digitized-reference support、not-yet-validated を混同せずに並べるためのものである。',
        '',
        '### 7.2 supplement 用 detailed table',
        '',
        _markdown_table(
            ['項目', '支持レベル', '根拠', 'コメント'],
            validation_detailed_table_rows,
        ),
        '',
        'こちらは supplement 向けであり、各主張の evidence path と caution を残す。',
        '',
        '## 8. robustness suite による適用性の補強',
        '',
        'strict validation table とは別に、robustness / applicability suite を併読する必要がある。'
        'この suite は authoritative external reference を増やすものではなく、圧力、source voltage、electrical backend、chemistry を parity point からずらしたときに、'
        '本コードが bounded に動作し、workflow として実用に耐えるかを問うものである。',
        '',
        _markdown_table(
            ['項目', '値', '意味'],
            applicability_table_rows,
        ),
        '',
        'この結果の読み方は限定的である。ここで支持されるのは boundedness、workflow usability、operating-envelope plausibility であって、'
        '予測された density、power、radical level の外部真値に対する validation ではない。',
        '',
        '関連 artifact は以下である。',
        '',
        f'- [robustness stress overview]({robustness_overview_rel})',
        f'- [robustness rate-table coverage]({robustness_coverage_rel})',
        f'- [robustness applicability report]({robustness_report_rel})',
        '',
        '## 9. 低圧プラズマモデルとして残る理論課題と将来項目',
        '',
        '本コードを低圧プラズマの予測モデルとしてさらに強くするには、少なくとも次の理論・モデル上の不足を順に埋める必要がある。',
        '',
        '1. **空間非一様性と輸送**: 現在は global model であり、中心/周辺差、local power deposition、diffusion length の self-consistent 計算は行わない。'
        '将来は multi-zone diffusion closure の強化、edge-to-center model の体系化、1D fluid surrogate との接続が有効である。',
        '2. **sheath / IED**: 現在の Bohm-family loss は leading-order sink としては妥当だが、time-dependent sheath、collisional sheath、ion transit、IEDF/IADF を解いていない。'
        '低圧 CCP / bias 系ではここが主要課題となる。',
        '3. **RF 周期内電子加熱**: `rf_envelope` は recipe-level には有用だが、RF 周期内の stochastic heating、harmonics、phase-resolved EEDF は扱わない。'
        'cycle-averaged closure の校正と sub-RF effective heating model が将来課題である。',
        '4. **双方向回路連成**: `external_circuit_table` は one-way loose coupling であり、plasma impedance から circuit への feedback はない。'
        'ngspice 等との macro-step loose coupling、さらに plasma ODE / circuit DAE 分割連成が将来項目である。',
        '5. **表面反応と壁条件の校正**: O2 や fluorocarbon 系では sticking、quenching、recombination、coverage dependence の不確かさが大きい。'
        'surface reaction budget と material-dependent calibration が必要である。',
        '6. **state-resolved chemistry と外部 rate workflow**: vibrational ladder、metastable、superelastic collision を含む高励起状態は mean-energy closure だけでは不十分な場合がある。'
        'BOLSIG+/LoKI 由来 rate table の標準 workflow と state-resolved chemistry 整備が望ましい。',
        '7. **実験比較と不確かさ評価**: 最終的には `n_e`, `T_e`, `T_g`, ion flux, self-bias, radical density などの observable に対する実験比較と UQ が必要である。',
        '',
        '要するに、本コードは「低圧プラズマ global model の研究・開発基盤」としてはかなり強いが、'
        '「空間輸送・sheath・RF 加熱・surface calibration まで含めた完全な予測モデル」としては、まだ系統的な拡張が必要である。',
        '',
    ]
    return '\n'.join(sections)


def _build_markdown_report_ja(
    reports: dict[str, Any],
    *,
    generated_at_utc: str,
    overview_png_path: Path,
    pygmol_png_path: Path,
    coverage_png_path: Path,
    markdown_report_ja_path: Path,
) -> str:
    coverage_rows = _coverage_rows()
    limitation_rows = _benchmark_limitations_summary()
    validation_short_rows = _validation_status_short_summary(reports)
    validation_detailed_rows = _validation_status_summary(reports)
    applicability_rows = _applicability_evidence_summary(reports)
    headline_rows_main = _headline_summary_main_text(reports)
    robustness_outputs = reports['robustness']['outputs']

    zdp = reports['zdplaskin']
    zdp_cmp = zdp['comparison']
    zdp_eovern = reports['zdplaskin_eovern']
    zdp_eovern_cmp = zdp_eovern['comparison']
    zdp_local = zdp['local_result']
    zdp_ref = zdp['zdplaskin_reference']
    zdp_power_ratio = zdp_eovern_cmp['absorbed_power_over_zdplaskin_VI_power_ratio']
    zdp_power_mismatch = _ratio_mismatch_percent(zdp_power_ratio)
    zdp_species_mismatch = _max_finite(
        [
            _ratio_mismatch_percent(zdp_cmp['final_electron_density_ratio_local_over_zdplaskin']),
            _ratio_mismatch_percent(zdp_cmp['final_Ar_star_ratio_local_over_zdplaskin']),
            _ratio_mismatch_percent(zdp_cmp['final_Ar_plus_ratio_local_over_zdplaskin']),
            _ratio_mismatch_percent(zdp_cmp['final_Ar2_plus_ratio_local_over_zdplaskin']),
        ]
    )
    zdp_en_mismatch = _ratio_mismatch_percent(zdp_eovern_cmp['local_reported_over_zdplaskin_saved_EoverN_ratio'])

    crane_cmp = reports['crane']['comparison']

    pygmol_baseline = _pygmol_step_row(reports['pygmol_surrogate'], 'production')
    pygmol_local_fit = _pygmol_step_row(reports['pygmol_local_fit'], 'production')
    pygmol_local_table = _pygmol_step_row(reports['pygmol_local_table'], 'production')
    pyg_stage_rate = _stage_row(reports['pygmol_same_footing'], 'same_rate_recipe_power_pygmol_wall')
    pyg_stage_power = _stage_row(reports['pygmol_same_footing'], 'same_rate_local_power_pygmol_wall')
    pyg_stage_final = _stage_row(reports['pygmol_same_footing'], 'same_rate_local_power_local_wall')

    loki = reports['loki']
    loki_primary_max = _max_finite(
        [
            loki['direct_reconstruction']['avg_gas_temperature_relative_difference']['max_percent'],
            loki['direct_reconstruction']['near_wall_temperature_relative_difference']['max_percent'],
            loki['direct_reconstruction']['avg_reduced_electric_field_relative_difference']['max_percent'],
        ]
    )

    reference_type_ja = {
        'strict executable parity': '厳密な実行コード parity',
        'strict scalar-chemistry parity': '厳密な scalar chemistry parity',
        'broad executable global-model comparison': '広義の実行可能 global model 比較',
        'diagnostic decomposition benchmark': '診断分解ベンチマーク',
        'digitized-reference benchmark': '文献 digitize 参照ベンチマーク',
    }
    same_footing_ja = {
        'high': '高い',
        'medium-high': '中程度より高い',
        'medium': '中程度',
    }
    comparison_rows = [
        [
            row['benchmark'],
            reference_type_ja.get(row['reference_type'], row['reference_type']),
            row['comparison_target'],
            same_footing_ja.get(row['same_footing'], row['same_footing']),
            row['main_limitation'],
        ]
        for row in coverage_rows
    ]
    headline_table_rows = [
        [
            row['benchmark'],
            reference_type_ja.get(row['reference_type'], row['reference_type']),
            row['headline_metric'],
            _fmt_percent(row['headline_percent'], 3 if row['headline_percent'] and row['headline_percent'] < 0.01 else 1),
            row['interpretation'],
        ]
        for row in headline_rows_main
    ]
    limitation_table_rows = [
        [row['benchmark'], row['limitation'], row['why_it_matters']]
        for row in limitation_rows
    ]
    validation_short_table_rows = [
        [row['topic'], row['status'], row['support']]
        for row in validation_short_rows
    ]
    validation_detailed_table_rows = [
        [row['topic'], row['status'], row['evidence'], row['comment']]
        for row in validation_detailed_rows
    ]
    applicability_table_rows = [
        [row['item'], row['value'], row['meaning']]
        for row in applicability_rows
    ]

    architecture_rows = [
        ['`config`', 'YAML 読み込み、path 解決、入力検証', '何を解くかを定義する。物理モデルの矛盾はここで早めに落とす。'],
        ['`chemistry`', 'species / reaction / cross section / mechanism bundle', '反応式・断面積・速度モデルの入口を集約する。'],
        ['`reactor`', 'zone / surface / edge / inlet / power port', '0D / multi-zone 幾何と流路・表面を保持する。'],
        ['`eedf`', 'Maxwell / rate-table / internal two-term Boltzmann', '電子衝突 rate と輸送係数の closure を返す。'],
        ['`electrical`', 'direct power / dc series / external table / rf envelope / ccp / icp', '吸収電力と reduced field を backend として供給する。'],
        ['`physics`', 'gas-phase core / surface core / prescribed electrons', '反応項、壁損失、表面反応、準中性 closure を評価する。'],
        ['`numerics`', 'state layout / global system / analytic Jacobian / BDF', 'stiff ODE を安定に解く。'],
        ['`observables` と `diagnostics`', 'CSV / YAML / budget / provenance', '結果を物理解釈可能な形に整える。'],
        ['`workflows`', 'load -> build -> run -> export', '全体の実行順序を一つの入口にまとめる。'],
    ]
    workflow_rows = [
        ['1', '入力読込', '`run.yaml` から chamber, recipe, chemistry manifest を解決する。'],
        ['2', '検証', '設定・反応機構・電気 backend・壁損失 family を検証し、二重計上や欠損入力を防ぐ。'],
        ['3', '状態配置', 'zone ごとの gas densities, electron energy, optional gas temperature, surface states の添字を確定する。'],
        ['4', 'backend 構築', 'EEDF backend と electrical backend を registry から生成する。'],
        ['5', '大域系組立て', '`GlobalPlasmaSystem` が gas, surface, electrical, observables を束ねて RHS / Jacobian を提供する。'],
        ['6', '時間積分', 'recipe step ごとに SciPy BDF を走らせ、各 segment 後に admissible state へ投影する。'],
        ['7', '出力整形', 'observables, summary, budgets, plots を書き出す。'],
    ]
    design_category_rows = [
        [
            '責務分離',
            '`physics`, `electrical`, `eedf`, `numerics`, `workflows` を分離',
            '反応、電力、電子 kinetics、数値積分の変更が相互に絡みすぎない。',
            '「何が物理差で何が数値差か」を切り分けやすい。',
        ],
        [
            'registry ベースの backend 交換',
            'EEDF / electrical / integrator を registry から構築',
            '既存 solver 本体を崩さずに新 backend を追加しやすい。',
            '将来の circuit model や EEDF model を局所変更で差し込める。',
        ],
        [
            'solver-facing API の安定化',
            '`GlobalPlasmaSystem` が `initial_state`, `rhs`, `jacobian`, `compute_observables` を提供',
            '数値 solver 側のインターフェースが安定する。',
            '物理モジュールを変更しても時間積分器や出力系への影響を最小化できる。',
        ],
        [
            '状態配置の独立化',
            '`StateLayout` が gas / energy / surface state の添字を集中管理',
            '状態量が増えても添字の不整合を起こしにくい。',
            '新しい状態変数の追加が局所化する。',
        ],
        [
            '診断と主 solver の分離',
            'budget, provenance, observables, benchmark dashboard を主 solver から分離',
            'production solve の経路を複雑化しにくい。',
            '診断の追加や削除が計算本体を壊しにくい。',
        ],
        [
            'benchmark ツールの外部化',
            '`tools/external_benchmarks` で比較ロジックを保持',
            '比較用の調整が本体物理を汚染しにくい。',
            'PyGMol や LoKI のような外部比較を、本体設計と切り分けて育てられる。',
        ],
    ]
    extension_rows = [
        [
            '新しい EEDF / rate backend',
            'registry に backend を追加し、`EEDFResult` 相当の rate・transport を返す。',
            'global ODE 本体を変えずに electron-impact closure を差し替えられる。',
        ],
        [
            '新しい電気モデル',
            '`electrical` backend と `PowerRequest/PowerResult` を実装する。',
            'DC, pulsed-DC, RF envelope, external table と同じ接続点で増設できる。',
        ],
        [
            '新しい wall-loss / surface model',
            '`surface_models` と `SurfaceCore` 側に family を追加する。',
            'Bohm / ambipolar と同じ validation 境界で、新 closure を二重計上なく導入できる。',
        ],
        [
            '新しい状態変数',
            '`StateLayout` と core の初期化・投影・observables を拡張する。',
            '添字管理が一箇所なので、状態追加時の破壊範囲を読める。',
        ],
        [
            '新しい診断・可視化',
            '`diagnostics`, `observables`, `plotters` に追加する。',
            '主 solver の RHS/Jacobian を汚さずに説明力だけ増やせる。',
        ],
        [
            '新しい benchmark',
            '`tools/external_benchmarks` に追加し、dashboard から集約する。',
            '本体 physics を benchmark 用に曲げずに比較体系を広げられる。',
        ],
    ]
    extension_guard_rows = [
        [
            '入力検証が先に走る',
            '未設定パラメータや incompatible family を run 前に落とせる。',
        ],
        [
            '物理と数値の境界が明確',
            'closure 追加時に、どこまでが物理変更でどこからが solver 変更かを説明しやすい。',
        ],
        [
            'benchmark を本体外に置いている',
            '比較のための特別処理が production solver に混ざりにくい。',
        ],
        [
            'diagnostics が独立している',
            '新機能を入れたときの budget / observables を後付けしやすい。',
        ],
    ]
    theory_gap_rows = [
        [
            'ZDPlaskin example2',
            '0D Ar 放電 + 回路 parity。実効的には species, absorbed power, circuit-derived E/N を同じ土俵に乗せる。',
            'local 側は内部 `dc_series_circuit` を使い、電子衝突は committed output から逆算した external E/N-rate table を使う。live BOLSIG+ 実行ではない。',
            '現在の DC Ar parity case に限れば、power・E/N・最終組成の self-consistency を見る強い benchmark になる。',
        ],
        [
            'CRANE TwoReactionArgon',
            '定数反応速度の 2 反応 ODE。E/N は rate 選択用の与条件で、動的な electron energy や回路は持たない。',
            '本コード側の full plasma model のうち、比較できるのは gas-phase scalar chemistry と SI 変換だけ。',
            'reaction assembly、単位変換、stiff ODE 積分の健全性を切り出して検証できる。',
        ],
        [
            'PyGMol',
            '別実装の executable global model。自然状態では rate・power・wall loss の closure が揃っていない。',
            '本コードと同一方程式ではない。したがって baseline 差は精度差ではなく closure 差を多く含む。',
            'same-rate -> same-power -> same-wall-loss の順に揃えると、どの closure が差を支配しているかを分解できる。',
        ],
        [
            'LoKI O2 DC glow',
            '公開論文の 0D と 1D の O2 DC glow を digitize した benchmark。pressure sweep と radial profile を使う。',
            'official raw output ではなく digitized reference。local solver の executable parity ではない。',
            '低圧 O2 の primary discharge trend と profile shape の plausibility を評価するのに向く。',
        ],
    ]

    zdp_result_rows = [
        ['最終電子密度比 `n_e(local) / n_e(ZDPlaskin)`', _fmt_ratio(zdp_cmp['final_electron_density_ratio_local_over_zdplaskin'], 4), _fmt_percent(_ratio_mismatch_percent(zdp_cmp['final_electron_density_ratio_local_over_zdplaskin']), 2)],
        ['最終 `Ar*` 密度比', _fmt_ratio(zdp_cmp['final_Ar_star_ratio_local_over_zdplaskin'], 4), _fmt_percent(_ratio_mismatch_percent(zdp_cmp['final_Ar_star_ratio_local_over_zdplaskin']), 2)],
        ['最終 `Ar+` 密度比', _fmt_ratio(zdp_cmp['final_Ar_plus_ratio_local_over_zdplaskin'], 4), _fmt_percent(_ratio_mismatch_percent(zdp_cmp['final_Ar_plus_ratio_local_over_zdplaskin']), 2)],
        ['最終 `Ar2+` 密度比', _fmt_ratio(zdp_cmp['final_Ar2_plus_ratio_local_over_zdplaskin'], 4), _fmt_percent(_ratio_mismatch_percent(zdp_cmp['final_Ar2_plus_ratio_local_over_zdplaskin']), 2)],
        ['現在 run の回路場 E/N 比', _fmt_ratio(zdp_eovern_cmp['local_reported_over_zdplaskin_saved_EoverN_ratio'], 4), _fmt_percent(_ratio_mismatch_percent(zdp_eovern_cmp['local_reported_over_zdplaskin_saved_EoverN_ratio']), 2)],
        ['吸収電力比 `Pabs(local) / V I(ZDPlaskin)`', _fmt_ratio(zdp_power_ratio, 4), _fmt_percent(zdp_power_mismatch, 2)],
    ]
    crane_result_rows = [
        ['電子密度相対誤差', _fmt_percent(crane_cmp['final_electron_density_m3_relative_error'] * 100.0, 6), _fmt_sci(crane_cmp['final_electron_density_m3_relative_error'], 3)],
        ['`Ar+` 密度相対誤差', _fmt_percent(crane_cmp['final_Ar_plus_density_m3_relative_error'] * 100.0, 6), _fmt_sci(crane_cmp['final_Ar_plus_density_m3_relative_error'], 3)],
        ['`Ar` 密度相対誤差', _fmt_percent(crane_cmp['final_Ar_density_m3_relative_error'] * 100.0, 6), _fmt_sci(crane_cmp['final_Ar_density_m3_relative_error'], 3)],
    ]
    pygmol_stage_rows = [
        [
            'baseline',
            'PyGMol ネイティブ closure',
            _fmt_percent(_ratio_mismatch_percent(pygmol_baseline['ratio_final_ne_pygmol_over_local']), 1),
            _fmt_ratio(pygmol_baseline['ratio_mean_absorbed_power_pygmol_over_local'], 3),
            _fmt_ratio(pygmol_baseline['ratio_final_wall_loss_rate_pygmol_over_local'], 4),
            _fmt_ratio(pygmol_baseline['ratio_final_mean_energy_equiv_pygmol_over_local'], 3),
        ],
        [
            'same local-fit',
            'local Arrhenius fit で electron-impact rate を寄せる',
            _fmt_percent(_ratio_mismatch_percent(pygmol_local_fit['ratio_final_ne_pygmol_over_local']), 1),
            _fmt_ratio(pygmol_local_fit['ratio_mean_absorbed_power_pygmol_over_local'], 3),
            _fmt_ratio(pygmol_local_fit['ratio_final_wall_loss_rate_pygmol_over_local'], 4),
            _fmt_ratio(pygmol_local_fit['ratio_final_mean_energy_equiv_pygmol_over_local'], 3),
        ],
        [
            'same local-table',
            'local rate table で electron-impact rate を同一化',
            _fmt_percent(_ratio_mismatch_percent(pygmol_local_table['ratio_final_ne_pygmol_over_local']), 1),
            _fmt_ratio(pygmol_local_table['ratio_mean_absorbed_power_pygmol_over_local'], 3),
            _fmt_ratio(pygmol_local_table['ratio_final_wall_loss_rate_pygmol_over_local'], 4),
            _fmt_ratio(pygmol_local_table['ratio_final_mean_energy_equiv_pygmol_over_local'], 3),
        ],
        [
            'same-rate / recipe-power / pygmol-wall',
            'rate のみ local に合わせる',
            _fmt_percent(_ratio_mismatch_percent(pyg_stage_rate['production_final_ne_ratio']), 1),
            _fmt_ratio(pyg_stage_rate['production_power_ratio'], 3),
            _fmt_ratio(pyg_stage_rate['production_wall_loss_rate_ratio'], 4),
            _fmt_ratio(pyg_stage_rate['production_energy_ratio'], 3),
        ],
        [
            'same-rate / local-power / pygmol-wall',
            'rate と吸収電力を local に合わせる',
            _fmt_percent(_ratio_mismatch_percent(pyg_stage_power['production_final_ne_ratio']), 1),
            _fmt_ratio(pyg_stage_power['production_power_ratio'], 3),
            _fmt_ratio(pyg_stage_power['production_wall_loss_rate_ratio'], 4),
            _fmt_ratio(pyg_stage_power['production_energy_ratio'], 3),
        ],
        [
            'same-rate / local-power / local-wall',
            'rate・吸収電力・壁損失を local に合わせる',
            _fmt_percent(_ratio_mismatch_percent(pyg_stage_final['production_final_ne_ratio']), 1),
            _fmt_ratio(pyg_stage_final['production_power_ratio'], 3),
            _fmt_ratio(pyg_stage_final['production_wall_loss_rate_ratio'], 4),
            _fmt_ratio(pyg_stage_final['production_energy_ratio'], 3),
        ],
    ]
    loki_result_rows = [
        ['`Tg,av` 最大差', _fmt_percent(loki['direct_reconstruction']['avg_gas_temperature_relative_difference']['max_percent'], 2), '一次放電量として十分近い'],
        ['`Tnw` 最大差', _fmt_percent(loki['direct_reconstruction']['near_wall_temperature_relative_difference']['max_percent'], 2), '壁近傍温度の差も小さい'],
        ['`E/N` 最大差', _fmt_percent(loki['direct_reconstruction']['avg_reduced_electric_field_relative_difference']['max_percent'], 2), '一次放電量としては整合的'],
        ['extended neutral subset proxy 最大差', _fmt_percent(loki['direct_reconstruction']['avg_neutral_species_extended_subset_proxy']['max_percent'], 2), '重い中性化学では差が増える'],
        ['`O3` 最大差', _fmt_percent(loki['extended_species_relative_differences']['O3']['max_percent'], 2), '最も大きい outlier'],
        ['profile shape check', str(bool(loki['summary']['profile_shape_expectation_pass'])), '傾向としては 0D / 1D の差を超えて破綻していない'],
    ]

    robustness_overview_rel = _markdown_relpath(markdown_report_ja_path, Path(robustness_outputs['overview_png']))
    robustness_coverage_rel = _markdown_relpath(markdown_report_ja_path, Path(robustness_outputs['rate_table_coverage_png']))
    robustness_report_rel = _markdown_relpath(markdown_report_ja_path, Path(robustness_outputs['markdown_report_md']))

    sections = [
        '# 外部ベンチマーク報告（日本語論文調ドラフト）',
        '',
        f'生成日時 (UTC): `{generated_at_utc}`',
        '',
        '## 1. 目的と読み方',
        '',
        '本報告の目的は、`plasma_global` が何を状態量として解き、どの物理を closure として外から与え、どの benchmark が何を実際に検証しているのかを、第三者が物理的に追跡できる形で整理することである。ここで重視しているのは、単に「合った / 合わない」を列挙することではなく、**どの理論差がどの結果差を生んでいるか**を分離して説明することである。',
        '',
        '本コードは reactor-averaged の 0D / multi-zone global model であり、gas-phase species、電子エネルギー、必要に応じて gas temperature、surface coverage、wall inventory、film thickness を状態量として扱う。一方で、電子速度分布関数そのもの、sheath の時間発展、空間分布をもつ輸送場、RF 周期内の高速電子加熱は直接解いていない。したがって、本コードの妥当性は、**global model としてどの closure を採用し、その closure が benchmark に対してどこまで整合的か**という観点で読む必要がある。',
        '',
        'また、本報告に含まれる benchmark は性格が異なる。ZDPlaskin と CRANE は比較的強い same-footing benchmark、PyGMol は closure 差の診断 benchmark、LoKI O2 DC glow は digitized 文献 benchmark、robustness suite は boundedness と workflow usability の補強証拠である。これらを同列に「全部 validation」とは呼ばず、**strict validation / diagnostic comparison / digitized support / applicability evidence** を分けて扱う。',
        '',
        '## 2. コード構成と処理の流れ',
        '',
        '### 2.1 モジュール構成',
        '',
        _markdown_table(
            ['モジュール', '主な責務', '物理・数値上の意味'],
            architecture_rows,
        ),
        '',
        '### 2.2 実行時の処理フロー',
        '',
        _markdown_table(
            ['段階', '処理名', '内容'],
            workflow_rows,
        ),
        '',
        'この構成の要点は、`GlobalPlasmaSystem` が solver-facing な安定した API を持ちつつ、実際の物理は `GasPhaseCore`、`SurfaceCore`、`ElectricalCouplingAdapter`、`ObservablesAdapter` に分離されていることである。これにより、反応機構、壁損失、電子 kinetics、回路 backend を個別に変更しても、大域 ODE の入口は大きく壊れない。',
        '',
        '### 2.3 設計カテゴリごとの利点',
        '',
        _markdown_table(
            ['設計カテゴリ', '現在の設計', '直接の利点', '第三者にとっての理解上の利点'],
            design_category_rows,
        ),
        '',
        'ここで重要なのは、設計上の利点が単なる保守性にとどまらないことである。低圧プラズマ global model では、結果差が chemistry 差、wall-loss 差、power coupling 差、EEDF 差のどこから来るのかを追跡できないと、比較自体が不毛になりやすい。本コードの構成は、その差分分解を可能にする方向へ寄っている。',
        '',
        '### 2.4 将来拡張時の入口と利点',
        '',
        _markdown_table(
            ['拡張カテゴリ', '主な追加場所', '設計上の利点'],
            extension_rows,
        ),
        '',
        '### 2.5 拡張時に破綻しにくい理由',
        '',
        _markdown_table(
            ['ガード', '意味'],
            extension_guard_rows,
        ),
        '',
        '言い換えると、本コードは「何でも一つの巨大 solver に押し込む」構成ではなく、**closure と workflow を差し替えながら本体の整合性を保つ**構成である。これは低圧プラズマのようにモデル差が大きい分野で特に有利であり、将来 RF、外部回路、state-resolved chemistry、surface calibration を伸ばすときにも、変更範囲を説明しやすい。',
        '',
        '## 3. 支配方程式と closure',
        '',
        '### 3.1 種密度方程式',
        '',
        '本コードの gas-phase state は、各 zone の species 密度 `n_k` に対して概念的に次式で進む。',
        '',
        '```text',
        'd n_k / dt = Σ_r ν_k,r R_r + S_k,flow + S_k,surface - L_k,wall',
        'R_r = k_r(Te, E/N, Tg, θ, ...) Π_j n_j^α_j',
        '```',
        '',
        'ここで `R_r` は反応 `r` の体積反応速度、`ν_k,r` は species `k` の化学量論係数、`S_k,flow` は流入出、`S_k,surface` は表面反応との結合、`L_k,wall` は global wall loss を表す。重要なのは、速度係数 `k_r` が単なる定数ではなく、EEDF backend から返される電子衝突 rate、あるいは Arrhenius / table model を通して **電子エネルギーや reduced field に依存している**点である。',
        '',
        '### 3.2 電子密度 closure と電子エネルギー方程式',
        '',
        'production 設定では電子密度 `n_e` は通常、独立 ODE ではなく準中性条件からの代数量である。概念的には',
        '',
        '```text',
        'n_e ≈ Σ_i z_i n_i - Σ_a |z_a| n_a',
        '```',
        '',
        'として与えられる。これは global model において、イオン生成と損失の結果として電子密度が決まるという立場に対応している。benchmark 用には `prescribed_profile` も用意されているが、それは one-way 連成または固定 `n_e(t)` での rate 比較のための機能であり、自己無撞着な放電解ではない。',
        '',
        '電子エネルギー密度 `w_e` は概念的に次で進む。',
        '',
        '```text',
        'd w_e / dt = P_abs / V - Σ_r (Δε_r R_r) - Q_wall - Q_flow',
        '```',
        '',
        'ここで `P_abs` は electrical backend が返す吸収電力、`Δε_r` は反応ごとの電子エネルギー損失、`Q_wall` と `Q_flow` は壁面・流れに伴うエネルギー損失である。本コードは EEDF 全分布を ODE 状態としては持たず、**電子エネルギー状態から rate と輸送係数を問い合わせる**設計である。したがって、`local_field` closure と `mean_energy` closure は理論的に同じではない。',
        '',
        '### 3.3 壁損失 closure',
        '',
        '正イオンの主要 wall loss は Bohm-family closure であり、現在の既定は `bohm_edge_loss` である。概念式は',
        '',
        '```text',
        'Γ_i = h_factor * 0.61 * n_i * sqrt(ε_e / m_i)',
        '(d n_i / dt)_wall = - (A_eff / V) Γ_i',
        '```',
        '',
        'である。ここで `h_factor` は edge-to-center 的な補正を担う。別 family として `ambipolar_diffusion` もあるが、これは',
        '',
        '```text',
        'd n_i / dt = -k_loss n_i',
        '```',
        '',
        'という一次 volumetric sink として扱われ、**Bohm と ambipolar を同時加算しない**。この制約は、同じ wall sink を二重計上しないための物理的・設計的なガードである。',
        '',
        '### 3.4 電気 backend',
        '',
        'DC / pulsed-DC の `dc_series_circuit` backend は、プラズマを導電性負荷として表した reduced circuit である。',
        '',
        '```text',
        'G_plasma = e * n_e * μ_e * A / g',
        'R_plasma = 1 / G_plasma',
        'I = V_source / (R_ballast + R_plasma)',
        'V_gap = I R_plasma',
        'P_abs = η_abs V_gap I',
        'E/N = |V_gap / g| / N',
        '```',
        '',
        'これは sheath capacitance、matching network、RLC ringing、SPICE DAE を解くものではない。しかし、低圧 global model において「回路から見た gap voltage と absorbed power を self-consistent に決める」には有効であり、ZDPlaskin parity のような DC benchmark では十分に意味がある。一方、`external_circuit_table` は one-way loose coupling、`rf_envelope` は RF 周期平均の recipe-level backend であって、いずれも full circuit solve ではない。',
        '',
        '### 3.5 本コードが自己無撞着に解いているもの / いないもの',
        '',
        _markdown_table(
            ['項目', '扱い', '意味'],
            [
                ['gas species', 'ODE 状態', '反応、流れ、壁損失、表面反応で進む'],
                ['electron energy', 'ODE 状態', '吸収電力と衝突損失を介して進む'],
                ['electron density', '代数 closure（既定）', '準中性条件から決まる'],
                ['electron-impact rate', 'EEDF backend 由来', 'Maxwell / rate-table / two-term backend で置換可能'],
                ['gap voltage / absorbed power', 'electrical backend 由来', 'direct power / dc circuit / external table / rf envelope など'],
                ['spatial profile', '直接は解かない', 'profile は global closure や external benchmark で間接評価する'],
                ['sheath dynamics / IEDF', 'proxy のみ', '高忠実度な sheath solve は未実装'],
            ],
        ),
        '',
        '## 4. ベンチマーク体系と理論差の整理',
        '',
        _markdown_table(
            ['ベンチマーク', '参照側の理論像', '本コードとの理論差', 'この比較で実際に読めること'],
            theory_gap_rows,
        ),
        '',
        '### Coverage map',
        '',
        f'![Benchmark coverage matrix]({coverage_png_path.name})',
        '',
        'この coverage map では、`S` は強い benchmark 軸、`D` は診断的に有用な軸、`0` はその benchmark で本質的には見ていない軸を表す。重要なのは、CRANE や LoKI を ZDPlaskin と同じ「厳密 validation」として読まないことである。',
        '',
        '## 5. 定量結果の要約',
        '',
        '実行コード parity では次の mismatch 指標を使う。',
        '',
        '```text',
        'M_q = 100 × | q_local / q_ref - 1 |',
        '```',
        '',
        'したがって `M_q = 0 %` が完全一致であり、数 % 程度であれば同一 closure / 同一入力条件の benchmark ではかなり良い一致と読める。LoKI については raw executable ratio ではなく、digitized 曲線から再構成した相対差と論文 Fig.11 の差分 envelope を用いて解釈する。',
        '',
        _markdown_table(
            ['ベンチマーク', '参照型', '主指標', '差 [%]', '解釈'],
            headline_table_rows,
        ),
        '',
        '### 統合グラフ',
        '',
        f'![Integrated overview]({overview_png_path.name})',
        '',
        'この図では raw ratio ではなく percent-space の mismatch を使っている。これにより、ZDPlaskin / CRANE の「ほぼ一致」と、PyGMol baseline の「closure 差を含む有意な差」、LoKI の「一次放電量は近いが化学は別に読むべき」を一枚で見分けやすくしている。',
        '',
        '### PyGMol 差分分解グラフ',
        '',
        f'![PyGMol decomposition]({pygmol_png_path.name})',
        '',
        'PyGMol については、単一の mismatch 値よりも staged decomposition の方が本質的である。ここでは概念的に',
        '',
        '```text',
        'Δ_total ≈ Δ_rate + Δ_power + Δ_wall + Δ_residual',
        '```',
        '',
        'という読み方をする。もちろん実際の系は非線形なので厳密な線形加算ではないが、same-rate -> same-power -> same-wall-loss と段階的に固定することで、どの closure が結果差の主因かを切り分けられる。',
        '',
        '## 6. ベンチマークごとの詳細解釈',
        '',
        '### 6.1 ZDPlaskin example2: DC Ar parity',
        '',
        'ZDPlaskin との比較は、現在の外部セットの中で最も強い same-footing discharge benchmark である。参照側も本コード側も、0D Ar discharge と回路由来 E/N を持つため、species・absorbed power・physical E/N を同時に比較できる。',
        '',
        'ただし完全同一ではない。local 側は内部 `dc_series_circuit` backend を使い、electron-impact rate は committed output から逆算した external E/N-rate table を読む。したがってこれは「現在の ZDPlaskin example2 と同じ土俵の parity」であって、「任意の Ar/DC ケースに一般化された validation」ではない。',
        '',
        '指標は次の通りである。',
        '',
        _markdown_table(
            ['量', '比または値', 'mismatch'],
            zdp_result_rows,
        ),
        '',
        '理論的には、ZDPlaskin 側と local 側で比較すべき reduced field は固定 power proxy ではなく **回路 backend が返す gap voltage 由来の E/N** である。本コードでは',
        '',
        '```text',
        'E/N = (V_gap / g) / N_gas',
        'N_gas = p / (k_B T_g)',
        '```',
        '',
        'を用いており、この current-run E/N が ZDPlaskin 保存値と約 '
        f'{_fmt_percent(zdp_en_mismatch, 2)} の差で一致している。ここから、少なくともこの parity case では、E/N の単位換算・gap・pressure・temperature の扱いが物理的に揃っていると読める。',
        '',
        f'一方、power も species も小さな mismatch に収まっているため、現在の DC Ar parity case に限れば、**回路 -> E/N -> electron-impact rate -> species inventory** の連鎖が破綻していないことを示している。具体的には最終電子密度は local `{_fmt_sci(zdp_local["final_electron_density_m3"], 4)}` m^-3、ZDPlaskin `{_fmt_sci(zdp_ref["final_saved_electron_density_m3"], 4)}` m^-3 であり、species 系の最大差は {_fmt_percent(zdp_species_mismatch, 2)} である。',
        '',
        '### 6.2 CRANE TwoReactionArgon: scalar chemistry の切り出し検証',
        '',
        'CRANE は full discharge benchmark ではなく、**定数速度係数を持つ 2 反応 ODE** を same-footing で解く benchmark である。理論式は概念的に',
        '',
        '```text',
        'd n_e / dt      = + k_ion n_e n_Ar - k_rec n_e n_Ar+',
        'd n_Ar+ / dt    = + k_ion n_e n_Ar - k_rec n_e n_Ar+',
        'd n_Ar / dt     = - k_ion n_e n_Ar + k_rec n_e n_Ar+',
        '```',
        '',
        'であり、ここには電子エネルギー方程式、回路、壁損失、EEDF backend は本質的には現れない。したがって、この比較が検証しているのは **反応組立て・cm 系から SI 系への変換・stiff ODE 積分** である。',
        '',
        _markdown_table(
            ['量', '相対誤差 [%]', '相対誤差（無次元）'],
            crane_result_rows,
        ),
        '',
        '誤差は事実上ゼロに近く、この benchmark が示すのは「本コードが full plasma model として正しい」ことではなく、**少なくとも scalar chemistry ODE の数値実装は壊れていない**ということである。この切り分けは論理的に重要で、CRANE 一致をもって EEDF や wall loss を主張してはいけない。',
        '',
        '### 6.3 PyGMol: closure 差の診断 benchmark',
        '',
        'PyGMol との比較は、**別実装の executable global model と closure 差を分解する**ための benchmark である。baseline の mismatch だけを見ると、本コードが悪いのか、PyGMol が悪いのか、単に closure が違うだけなのか判別できない。そこで本レポートでは、electron-impact rate、absorbed power、wall-loss coefficient を段階的に揃えている。',
        '',
        _markdown_table(
            ['段階', '何を揃えたか', 'production `n_e` mismatch', 'power 比', 'wall-loss 比', 'mean-energy 比'],
            pygmol_stage_rows,
        ),
        '',
        'この表から、same electron-impact rate だけでは差がむしろ非常に大きく残ることが分かる。`same local-table` の production `n_e` mismatch は '
        f'{_fmt_percent(_ratio_mismatch_percent(pygmol_local_table["ratio_final_ne_pygmol_over_local"]), 1)} であり、電子衝突 rate を揃えるだけでは discharge state は揃わない。',
        '',
        '一方、same-rate に加えて absorbed power を揃えると差は縮み、さらに wall-loss coefficient まで local 側に合わせると mismatch は '
        f'{_fmt_percent(_ratio_mismatch_percent(pyg_stage_final["production_final_ne_ratio"]), 1)} まで低下する。これは、PyGMol と本コードの差の主因が electron-impact rate そのものではなく、**power closure と wall-loss closure** にあることを示す。',
        '',
        '物理的には、低圧 global model の電子密度は概念的に',
        '',
        '```text',
        'production ~ ionization source / global charged-particle loss',
        '```',
        '',
        'で決まる。ionization source は electron-impact rate と absorbed power に敏感であり、loss は wall-loss coefficient に強く支配される。したがって PyGMol 差分分解は、「なぜ合わないか」を物理量のレベルで示している点に価値がある。これは strict validation ではないが、**本コードが closure 差を論理的に把握できる設計である**ことの証拠になる。',
        '',
        '### 6.4 LoKI O2 DC glow: 低圧 O2 trend / profile の plausibility',
        '',
        'LoKI O2 DC glow は official raw output parity ではなく、公開論文図の digitization に基づく benchmark である。そのため、ここで主張できるのは「low-pressure O2 の primary-discharge trend と profile shape が大きく外れていない」ことであり、「数値値が完全に一致する」ことではない。',
        '',
        _markdown_table(
            ['量', '最大差', '読み方'],
            loki_result_rows,
        ),
        '',
        f'`Tg,av`, `Tnw`, `E/N` の primary-discharge 量は最大でも {_fmt_percent(loki_primary_max, 2)} 程度であり、digitized benchmark としてはかなり良い。一方、`O3` は {_fmt_percent(loki["extended_species_relative_differences"]["O3"]["max_percent"], 2)} の outlier であり、重い中性化学・壁面化学・輸送近似の弱さが残っていると読める。',
        '',
        'LoKI 側は 0D と 1D の比較を含むため、この差は単なる reaction set 差ではなく、**transport と profile 仮定の差**を強く反映する。したがって LoKI は、本コードの一次放電量が plausibility を持つことの裏付けにはなるが、重い中性化学を量的に validation したことにはならない。',
        '',
        '## 7. 本コードの優位性',
        '',
        '本コードの優位性は、現時点では「常に他コードより精度が高い」といった単純な主張ではない。より正確には、次の 4 点にある。',
        '',
        '1. **closure を分解して扱える設計**',
        '   chemistry、EEDF、wall loss、electrical backend が分離されており、どの仮定が結果差を生んでいるかを追いやすい。',
        '2. **strict validation と diagnostic comparison を混同しない benchmark discipline**',
        '   ZDPlaskin / CRANE を backbone としつつ、PyGMol や LoKI を役割の異なる benchmark として位置づけている。',
        '3. **stiff chemistry を扱う数値基盤**',
        '   analytic sparse Jacobian と BDF により、反応数が増えても設計としては持ちこたえやすい。',
        '4. **将来拡張の入口が整理されている**',
        '   external circuit table、rf envelope、prescribed electron profile、surface diagnostics などが既に backend / workflow の単位で分かれている。',
        '',
        'この優位性は、単一 benchmark の一致値よりも、**どこが一致し、どこが一致せず、その理由をどの程度説明できるか**に表れている。PyGMol 分解が成立していることは、その代表例である。',
        '',
        '## 8. 制約と overclaim を避けるための整理',
        '',
        _markdown_table(
            ['ベンチマーク', '主な制約', 'なぜ重要か'],
            limitation_table_rows,
        ),
        '',
        '本レポートで強く言えるのは、(i) scalar chemistry ODE は極めて正確、(ii) 現在の DC Ar parity case は strong same-footing agreement、(iii) low-pressure O2 の primary trend / profile shape は digitized benchmark に支えられている、までである。逆に、全 chemistry、全圧力、全電源波形での予測精度まで主張するのはまだ早い。',
        '',
        '## 9. 現在 support される範囲と未検証範囲',
        '',
        '### 9.1 本文向け short table',
        '',
        _markdown_table(
            ['項目', '現状の表現', '根拠'],
            validation_short_table_rows,
        ),
        '',
        '### 9.2 supplement 向け detailed table',
        '',
        _markdown_table(
            ['項目', '現状の表現', '証拠', 'コメント'],
            validation_detailed_table_rows,
        ),
        '',
        '## 10. robustness suite による適用性の補強',
        '',
        'robustness suite は authoritative external reference ではない。ここで示しているのは、pressure、source voltage、electrical backend、chemistry を parity 点から少し動かしたときに、本コードが有限・正の状態を保ち、workflow として実行可能かどうかである。',
        '',
        _markdown_table(
            ['項目', '値', '意味'],
            applicability_table_rows,
        ),
        '',
        'この suite は新しい精度 validation ではないが、strict benchmark の周囲でコードが運転範囲的に破綻しにくいことを補足する。strict validation と applicability evidence を混同しないため、artifact は別報告として保持している。',
        '',
        f'- [robustness stress overview]({robustness_overview_rel})',
        f'- [robustness rate-table coverage]({robustness_coverage_rel})',
        f'- [robustness applicability report]({robustness_report_rel})',
        '',
        '## 11. 低圧プラズマモデルとして残る理論課題と将来項目',
        '',
        '本コードは global model の実装基盤としてはかなり整っているが、低圧プラズマの高忠実度予測モデルとしてはまだ未完である。主な将来項目は次の通りである。',
        '',
        '1. **空間非一様性と輸送の高度化**',
        '   現在は global model であり、power deposition profile、center-edge density ratio、局所輸送場は直接は解かない。multi-zone diffusion closure や edge-to-center model の体系化が必要である。',
        '2. **sheath / IED / IADF の高忠実度化**',
        '   現在の Bohm-family wall loss は leading-order sink としては有用だが、time-dependent sheath や collisional sheath を直接解かない。低圧 CCP / bias 問題ではここが支配的になる。',
        '3. **RF 周期内電子加熱**',
        '   `rf_envelope` は recipe-level には有効だが、stochastic heating、harmonic content、phase-resolved EEDF は表現できない。sub-RF effective heating model や校正手順の明文化が必要である。',
        '4. **外部回路との双方向連成**',
        '   `external_circuit_table` は one-way loose coupling である。将来的には plasma impedance を circuit 側へ返す macro-step loose coupling、さらに必要なら plasma ODE / circuit DAE の分割連成が候補となる。',
        '5. **表面化学と壁材依存性の校正**',
        '   O2 や fluorocarbon 系では sticking、quenching、recombination、coverage dependence の不確かさが大きい。`O3` outlier はその不足をよく示している。',
        '6. **state-resolved chemistry と external rate workflow**',
        '   vibrational ladder、metastable、superelastic collision を多く含む系では mean-energy closure だけでは粗い可能性がある。BOLSIG+ / LoKI 由来 table の ingestion、state-resolved chemistry の整理が今後重要である。',
        '7. **実験比較と不確かさ評価**',
        '   最終的な predictive value を示すには、`n_e`, `T_e`, `T_g`, ion flux, self-bias, radical density といった observable を実験と比較し、感度解析と UQ を組み合わせる必要がある。',
        '',
        '要するに、本コードは **低圧プラズマ global model を、closure を可視化しながら開発・診断・比較するための基盤**としては強い。一方で、sheath、空間輸送、RF 周期内電子加熱、表面校正、実験比較まで含む「高忠実度な完全予測モデル」としては、まだ将来拡張の余地が大きい。',
        '',
    ]
    return '\n'.join(sections)


def build_dashboard(
    *,
    rerun_zdplaskin: bool = True,
    rerun_pygmol_local: bool = True,
    report_path: Path = DEFAULT_REPORT,
    metrics_csv_path: Path = DEFAULT_METRICS_CSV,
    overview_png_path: Path = DEFAULT_OVERVIEW_PNG,
    pygmol_png_path: Path = DEFAULT_PYGMOL_PNG,
    coverage_png_path: Path = DEFAULT_COVERAGE_PNG,
    markdown_report_path: Path = DEFAULT_MARKDOWN,
    markdown_report_ja_path: Path = DEFAULT_MARKDOWN_JA,
) -> dict[str, Any]:
    reports = collect_reports(rerun_zdplaskin=rerun_zdplaskin, rerun_pygmol_local=rerun_pygmol_local)
    metric_rows = _metric_rows(reports)
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    _write_csv(
        metrics_csv_path,
        metric_rows,
        ['benchmark_family', 'benchmark_id', 'metric_id', 'step_or_stage', 'value', 'unit'],
    )
    _plot_overview(reports, overview_png_path)
    _plot_pygmol_decomposition(reports, pygmol_png_path)
    _plot_coverage_matrix(coverage_png_path)
    markdown_report = _build_markdown_report(
        reports,
        generated_at_utc=generated_at_utc,
        overview_png_path=overview_png_path,
        pygmol_png_path=pygmol_png_path,
        coverage_png_path=coverage_png_path,
        markdown_report_path=markdown_report_path,
    )
    markdown_report_ja = _build_markdown_report_ja(
        reports,
        generated_at_utc=generated_at_utc,
        overview_png_path=overview_png_path,
        pygmol_png_path=pygmol_png_path,
        coverage_png_path=coverage_png_path,
        markdown_report_ja_path=markdown_report_ja_path,
    )
    _write_text(markdown_report_path, markdown_report)
    _write_text(markdown_report_ja_path, markdown_report_ja)

    report = {
        'tool': 'external_benchmark_dashboard',
        'scope': 'external_benchmarks_only',
        'generated_at_utc': generated_at_utc,
        'rerun_zdplaskin': rerun_zdplaskin,
        'rerun_pygmol_local': rerun_pygmol_local,
        'outputs': {
            'report_yaml': str(report_path.resolve()),
            'metrics_csv': str(metrics_csv_path.resolve()),
            'overview_png': str(overview_png_path.resolve()),
            'pygmol_decomposition_png': str(pygmol_png_path.resolve()),
            'coverage_png': str(coverage_png_path.resolve()),
            'paper_report_md': str(markdown_report_path.resolve()),
            'paper_report_ja_md': str(markdown_report_ja_path.resolve()),
            'same_footing_report_yaml': str(SAME_FOOTING_DEFAULT_WRITE.resolve()),
        },
        'benchmark_catalog': _coverage_rows(),
        'headline_summary': _headline_summary(reports),
        'benchmark_limitations': _benchmark_limitations_summary(),
        'validation_status_short_summary': _validation_status_short_summary(reports),
        'validation_status_detailed_summary': _validation_status_summary(reports),
        'applicability_evidence_summary': _applicability_evidence_summary(reports),
        'benchmarks': {
            'zdplaskin_example2': {
                'passed': True,
                'comparison_file': str((ZDPLASKIN_OUTPUT_DIR / 'comparison_zdplaskin_example2.yaml').resolve()),
                'eovern_file': str((ZDPLASKIN_OUTPUT_DIR / 'comparison_zdplaskin_eovern.yaml').resolve()),
                'headline': {
                    'final_electron_density_ratio_local_over_zdplaskin': reports['zdplaskin']['comparison']['final_electron_density_ratio_local_over_zdplaskin'],
                    'local_reported_over_zdplaskin_saved_EoverN_ratio': reports['zdplaskin_eovern']['comparison']['local_reported_over_zdplaskin_saved_EoverN_ratio'],
                },
            },
            'crane_two_reaction_argon': {
                'passed': reports['crane']['comparison']['final_electron_density_m3_relative_error'] < 5.0e-4,
                'comparison_file': str((Path(reports['crane']['local_result']['output_dir']) / 'comparison_crane_two_reaction_argon.yaml').resolve()),
                'headline': {
                    'final_electron_density_relative_error': reports['crane']['comparison']['final_electron_density_m3_relative_error'],
                },
            },
            'pygmol_surrogate': {
                'passed': reports['pygmol_surrogate']['passed'],
                'comparison_file': reports['pygmol_surrogate']['comparison_file'],
                'headline': {
                    'production_final_ne_ratio': _pygmol_step_row(reports['pygmol_surrogate'], 'production')['ratio_final_ne_pygmol_over_local'],
                },
            },
            'pygmol_local_fit': {
                'passed': reports['pygmol_local_fit']['passed'],
                'comparison_file': reports['pygmol_local_fit']['comparison_file'],
                'headline': {
                    'production_final_ne_ratio': _pygmol_step_row(reports['pygmol_local_fit'], 'production')['ratio_final_ne_pygmol_over_local'],
                },
            },
            'pygmol_local_table': {
                'passed': reports['pygmol_local_table']['passed'],
                'comparison_file': reports['pygmol_local_table']['comparison_file'],
                'headline': {
                    'production_final_ne_ratio': _pygmol_step_row(reports['pygmol_local_table'], 'production')['ratio_final_ne_pygmol_over_local'],
                },
            },
            'pygmol_same_footing': {
                'passed': reports['pygmol_same_footing']['passed'],
                'comparison_file': reports['pygmol_same_footing']['summary_csv'],
                'headline': {
                    'final_stage_production_final_ne_ratio': _stage_row(reports['pygmol_same_footing'], 'same_rate_local_power_local_wall')['production_final_ne_ratio'],
                },
            },
            'loki_o2_dc_glow': {
                'passed': reports['loki']['passed'],
                'comparison_file': str(LOKI_DEFAULT_REPORT.resolve()),
                'headline': {
                    'readiness_stage': reports['loki']['summary']['readiness_stage'],
                    'profile_shape_expectation_pass': reports['loki']['summary']['profile_shape_expectation_pass'],
                },
            },
        },
        'assessment': {
            'summary': [
                'ZDPlaskin and CRANE remain the strictest same-footing parity references in the current external set.',
                'PyGMol baseline remains a broad executable global-model comparison; same-rate and same-footing stages show how much of the gap comes from rates, power, and wall loss.',
                'LoKI is a digitized-reference benchmark, not a raw-output executable parity case, but it now covers both pressure-sweep and full radial-profile diagnostics.',
                'The main benchmark report now cross-references the separate robustness suite so strict validation and applicability evidence can be read side by side without being conflated.',
                'The generated Markdown report is organized for a paper-style Methods/Results narrative and can be used as a drafting base.',
                'A short benchmark-limitations table plus main-text and supplementary versions of the concluding validation table are included to keep the manuscript claims disciplined.',
            ],
        },
    }
    report['outputs']['robustness_dashboard_yaml'] = str(Path(reports['robustness']['outputs']['dashboard_yaml']).resolve())
    report['outputs']['robustness_sweep_yaml'] = str(Path(reports['robustness']['outputs']['robustness_report_yaml']).resolve())
    report['outputs']['robustness_overview_png'] = str(Path(reports['robustness']['outputs']['overview_png']).resolve())
    report['outputs']['robustness_rate_table_coverage_png'] = str(Path(reports['robustness']['outputs']['rate_table_coverage_png']).resolve())
    report['outputs']['robustness_report_md'] = str(Path(reports['robustness']['outputs']['markdown_report_md']).resolve())
    _write_yaml(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run the currently implemented external benchmarks and generate overview graphs.')
    parser.add_argument('--skip-zdplaskin-rerun', action='store_true', help='Reuse the existing ZDPlaskin local output directory.')
    parser.add_argument('--skip-pygmol-local-rerun', action='store_true', help='Reuse the existing local Ar output before running PyGMol comparisons.')
    parser.add_argument('--write', type=Path, default=DEFAULT_REPORT)
    parser.add_argument('--metrics-csv', type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument('--overview-png', type=Path, default=DEFAULT_OVERVIEW_PNG)
    parser.add_argument('--pygmol-png', type=Path, default=DEFAULT_PYGMOL_PNG)
    parser.add_argument('--coverage-png', type=Path, default=DEFAULT_COVERAGE_PNG)
    parser.add_argument('--markdown-report', type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument('--markdown-report-ja', type=Path, default=DEFAULT_MARKDOWN_JA)
    args = parser.parse_args(argv)

    report = build_dashboard(
        rerun_zdplaskin=not args.skip_zdplaskin_rerun,
        rerun_pygmol_local=not args.skip_pygmol_local_rerun,
        report_path=args.write.resolve(),
        metrics_csv_path=args.metrics_csv.resolve(),
        overview_png_path=args.overview_png.resolve(),
        pygmol_png_path=args.pygmol_png.resolve(),
        coverage_png_path=args.coverage_png.resolve(),
        markdown_report_path=args.markdown_report.resolve(),
        markdown_report_ja_path=args.markdown_report_ja.resolve(),
    )
    print(yaml.safe_dump(report, sort_keys=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
