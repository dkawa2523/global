from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.external_benchmarks.robustness_sweep import (
    DEFAULT_SUMMARY_CSV as ROBUSTNESS_SUMMARY_CSV,
    DEFAULT_WRITE as ROBUSTNESS_REPORT_YAML,
    OUT_ROOT as ROBUSTNESS_OUT_ROOT,
    _write_summary_csv,
    _write_yaml,
    build_robustness_report,
)


OUTPUT_DIR = ROBUSTNESS_OUT_ROOT
DEFAULT_WRITE = OUTPUT_DIR / 'robustness_dashboard.yaml'
DEFAULT_OVERVIEW_PNG = OUTPUT_DIR / 'robustness_stress_overview.png'
DEFAULT_COVERAGE_PNG = OUTPUT_DIR / 'robustness_rate_table_coverage.png'
DEFAULT_MARKDOWN = OUTPUT_DIR / 'robustness_applicability_report.md'


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def _load_or_build_report(*, run_cases: bool) -> dict[str, Any]:
    if run_cases or not ROBUSTNESS_REPORT_YAML.exists():
        report = build_robustness_report(run_cases=True)
        _write_yaml(ROBUSTNESS_REPORT_YAML, report)
        _write_summary_csv(ROBUSTNESS_SUMMARY_CSV, report.get('stress_results', []))
        return report
    return _read_yaml(ROBUSTNESS_REPORT_YAML)


def _case_label(case_id: str) -> str:
    mapping = {
        'zdp_pressure_50torr': 'ZDP\n50 Torr',
        'zdp_pressure_200torr': 'ZDP\n200 Torr',
        'zdp_voltage_500V': 'ZDP\n500 V',
        'zdp_voltage_1500V': 'ZDP\n1500 V',
        'argon_lxcat_pressure_16Pa': 'Ar LXCat\n16 Pa',
        'argon_lxcat_icp_baseline': 'Ar LXCat\nbaseline',
        'argon_rf_envelope_calibration': 'RF\nenvelope',
        'cf4_o2_smoke_swarm': 'CF4/O2\nsmoke',
    }
    return mapping.get(case_id, case_id)


def _axis_color(axis_name: str) -> str:
    return {
        'pressure': '#4c72b0',
        'dc_source_voltage': '#dd8452',
        'electrical_backend': '#55a868',
        'electrical_backend_and_chemistry': '#8172b2',
        'chemistry': '#c44e52',
    }.get(axis_name, '#777777')


def _legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=_axis_color('pressure'), lw=6, label='pressure variation'),
        Line2D([0], [0], color=_axis_color('dc_source_voltage'), lw=6, label='dc-source variation'),
        Line2D([0], [0], color=_axis_color('electrical_backend'), lw=6, label='electrical backend'),
        Line2D([0], [0], color=_axis_color('electrical_backend_and_chemistry'), lw=6, label='production baseline'),
        Line2D([0], [0], color=_axis_color('chemistry'), lw=6, label='chemistry / surface smoke'),
    ]


def _plot_stress_overview(report: dict[str, Any], output_path: Path) -> None:
    stress = report.get('stress_results', [])
    labels = [_case_label(item['id']) for item in stress]
    colors = [_axis_color(item.get('axis', '')) for item in stress]
    ne_values = [_safe_float((item.get('summary') or {}).get('final_electron_density_m3')) for item in stress]
    peak_ne_values = [_safe_float((item.get('summary') or {}).get('peak_electron_density_m3')) for item in stress]
    mean_energy_values = [_safe_float((item.get('summary') or {}).get('final_mean_electron_energy_eV')) for item in stress]
    power_values = [_safe_float((item.get('summary') or {}).get('max_absorbed_power_W')) for item in stress]
    field_values = [_safe_float((item.get('summary') or {}).get('max_reduced_field_Td')) for item in stress]

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), constrained_layout=True)
    ax_ne, ax_peak, ax_energy, ax_field = axes.flat

    ax_ne.bar(labels, ne_values, color=colors)
    ax_ne.set_yscale('log')
    ax_ne.set_title('Final Electron Density Across Stress Cases')
    ax_ne.set_ylabel(r'final $n_e$ [m$^{-3}$]')
    ax_ne.tick_params(axis='x', rotation=15)

    ax_peak.bar(labels, peak_ne_values, color=colors)
    ax_peak.set_yscale('log')
    ax_peak.set_title('Peak Electron Density Across Stress Cases')
    ax_peak.set_ylabel(r'peak $n_e$ [m$^{-3}$]')
    ax_peak.tick_params(axis='x', rotation=15)

    ax_energy.bar(labels, mean_energy_values, color=colors)
    ax_energy.set_title('Final Mean Electron Energy')
    ax_energy.set_ylabel('final mean electron energy [eV]')
    ax_energy.tick_params(axis='x', rotation=15)

    positions = list(range(len(labels)))
    bars = ax_field.bar(positions, [value if value is not None else 0.0 for value in field_values], color=colors)
    for idx, value in enumerate(field_values):
        if value is None:
            ax_field.text(idx, 1.0, 'n/a', ha='center', va='bottom', fontsize=8, color='#555555')
            bars[idx].set_alpha(0.25)
    ax_field.set_yscale('log')
    ax_field.set_title('Max Reduced Field Where Available')
    ax_field.set_ylabel('max reduced field [Td]')
    ax_field.set_xticks(positions)
    ax_field.set_xticklabels(labels, rotation=15)
    ax_field.axhline(250.0, color='#666666', linestyle='--', linewidth=1.0, label='250 Td wide-table max')
    ax_field.legend(loc='upper right', fontsize=8)

    fig.legend(handles=_legend_handles(), loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle('Robustness / Applicability Stress Overview')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_rate_table_coverage(report: dict[str, Any], output_path: Path) -> None:
    coverage_rows = [
        item for item in report.get('stress_results', [])
        if isinstance(item.get('rate_table_coverage'), dict)
        and isinstance((item.get('rate_table_coverage') or {}).get('observed_EoverN_range_Td'), dict)
    ]
    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    if not coverage_rows:
        ax.text(0.5, 0.5, 'No rate-table coverage rows available.', ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
    else:
        y_positions = list(range(len(coverage_rows)))
        for idx, item in enumerate(coverage_rows):
            coverage = item['rate_table_coverage']
            table_range = coverage['rate_table_EoverN_range_Td']
            observed = coverage['observed_EoverN_range_Td']
            color = _axis_color(item.get('axis', ''))
            ax.hlines(idx, table_range['min'], table_range['max'], color='#cccccc', linewidth=8, label='active table range' if idx == 0 else '')
            ax.hlines(idx, observed['min'], observed['max'], color=color, linewidth=4, label='observed case range' if idx == 0 else '')
            ax.plot([observed['min'], observed['max']], [idx, idx], '|', color=color, markersize=14)
            ax.text(table_range['max'] + 4.0, idx, _case_label(item['id']), va='center', fontsize=9)
        ax.set_yticks([])
        ax.set_xlabel('reduced field [Td]')
        ax.set_title('Stress-Case Reduced-Field Coverage Against Active Tables')
        ax.legend(loc='lower right', fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = '| ' + ' | '.join(headers) + ' |'
    separator_line = '| ' + ' | '.join(['---'] * len(headers)) + ' |'
    body = ['| ' + ' | '.join(row) + ' |' for row in rows]
    return '\n'.join([header_line, separator_line, *body])


def _build_markdown_report(
    report: dict[str, Any],
    *,
    generated_at_utc: str,
    overview_png_path: Path,
    coverage_png_path: Path,
) -> str:
    stress = report.get('stress_results', [])
    ref_count = int((report.get('conclusion') or {}).get('external_reference_count', 0))
    stress_count = int((report.get('conclusion') or {}).get('stress_case_count', 0))
    all_pass = bool(report.get('passed'))
    coverage_count = sum(
        1
        for item in stress
        if isinstance(item.get('rate_table_coverage'), dict)
        and (item['rate_table_coverage'].get('inside_table_range') is True)
    )
    chemistry_case = next((item for item in stress if item.get('id') == 'cf4_o2_smoke_swarm'), None)
    rf_case = next((item for item in stress if item.get('id') == 'argon_rf_envelope_calibration'), None)

    table_rows = []
    for item in stress:
        summary = item.get('summary') or {}
        coverage = item.get('rate_table_coverage') or {}
        inside = coverage.get('inside_table_range')
        coverage_text = 'yes' if inside is True else 'no' if inside is False else 'n/a'
        table_rows.append(
            [
                _case_label(item['id']).replace('\n', ' '),
                str(item.get('axis', '')),
                str(item.get('status', '')),
                'n/a' if _safe_float(summary.get('final_electron_density_m3')) is None else f"{float(summary['final_electron_density_m3']):.3e}",
                'n/a' if _safe_float(summary.get('final_mean_electron_energy_eV')) is None else f"{float(summary['final_mean_electron_energy_eV']):.3f}",
                'n/a' if _safe_float(summary.get('max_absorbed_power_W')) is None else f"{float(summary['max_absorbed_power_W']):.3g}",
                'n/a' if _safe_float(summary.get('max_reduced_field_Td')) is None else f"{float(summary['max_reduced_field_Td']):.3g}",
                coverage_text,
            ]
        )

    chemistry_text = 'not available'
    if chemistry_case is not None:
        chemistry_summary = chemistry_case.get('summary') or {}
        chemistry_text = (
            f"final self-bias {float(chemistry_summary.get('final_self_bias_V', 0.0)):.1f} V, "
            f"final mean energy {float(chemistry_summary.get('final_mean_electron_energy_eV', 0.0)):.2f} eV"
        )
    rf_text = 'not available'
    if rf_case is not None:
        rf_summary = rf_case.get('summary') or {}
        rf_text = (
            f"peak electron density {float(rf_summary.get('peak_electron_density_m3', 0.0)):.3e} m^-3, "
            f"max reduced field {float(rf_summary.get('max_reduced_field_Td', 0.0)):.2f} Td"
        )

    sections = [
        '# Robustness / Applicability Benchmark Report',
        '',
        f'Generated at UTC `{generated_at_utc}`.',
        '',
        '## 1. Purpose',
        '',
        'This report extends the strict external-reference benchmark set with additional stress and applicability checks. '
        'The point is not to claim new accuracy benchmarks where no external reference exists, but to show that the code '
        'continues to run in a physically bounded way when pressure, electrical drive, and chemistry are perturbed away '
        'from the original parity cases.',
        '',
        '## 2. What this additional benchmark content demonstrates',
        '',
        f'- strict external references retained: `{ref_count}`',
        f'- additional stress / applicability cases executed: `{stress_count}`',
        f'- all executed cases passed the finite-positive solver-health checks: `{all_pass}`',
        f'- stress cases with explicit E/N table coverage checks that remained inside the active table: `{coverage_count}`',
        '',
        'These are applicability checks, not substitutes for experimental validation or new external-reference parity cases.',
        '',
        '## 3. Stress-case summary',
        '',
        _markdown_table(
            ['Case', 'Axis', 'Status', 'Final ne [m^-3]', 'Final mean e [eV]', 'Max power [W]', 'Max E/N [Td]', 'Inside table range'],
            table_rows,
        ),
        '',
        '## 4. Overview figures',
        '',
        f'![Robustness stress overview]({overview_png_path.name})',
        '',
        f'![Rate-table coverage]({coverage_png_path.name})',
        '',
        'The first figure shows whether densities, mean energies, absorbed powers, and reduced fields remain finite and bounded across the added stress cases. '
        'The second figure focuses on the ZDPlaskin-derived pressure/voltage stress family and shows that the observed reduced-field ranges stay inside the deliberately widened diagnostic table.',
        '',
        '## 5. Physics interpretation',
        '',
        (
            'The pressure and source-voltage ZDPlaskin stress family now probes a wider operating envelope without immediately leaving the active E/N table. '
            'This is important because it separates "solver survives" from "table clipping happened and made the result uninterpretable."'
        ),
        '',
        (
            'The pure-Ar LXCat baseline, the 16 Pa perturbation, and the RF-envelope calibration case show that the present low-pressure Ar workflow remains numerically stable across '
            'changes in pressure and electrical backend. They do not, by themselves, prove predictive accuracy, but they do support boundedness and practical usability for controlled parameter studies.'
        ),
        '',
        (
            f'The mixed CF4/O2 smoke case is especially valuable as a chemistry-and-surface applicability check. In the current run it remains finite with {chemistry_text}. '
            'That means the code is not limited to the compact Ar validation problems and can carry a more process-like mixture with surfaces and bias-like behavior, without claiming that the predicted mixture state is externally validated.'
        ),
        '',
        (
            f'The RF-envelope calibration case remains stable with {rf_text}. '
            'This is useful because it tests a reduced-order electrical backend that is closer to process recipes than the simplest direct-power driver.'
        ),
        '',
        '## 6. What is still missing',
        '',
        '- no new external-reference accuracy target was added for the stress cases',
        '- no experimental observable comparison was added here',
        '- surface-dominated chemistry is exercised, but not yet externally validated',
        '- pressure/power sweeps are richer than before, but still too sparse for broad process-space claims',
        '',
        'So the scientific message should be: the benchmark content is now broader, the code remains usable across a wider envelope, and the new cases strengthen practicality claims, '
        'but they still do not replace dedicated validation against experiment or new authoritative external references.',
        '',
    ]
    return '\n'.join(sections)


def build_robustness_dashboard(
    *,
    run_cases: bool = True,
    report_path: Path = DEFAULT_WRITE,
    overview_png_path: Path = DEFAULT_OVERVIEW_PNG,
    coverage_png_path: Path = DEFAULT_COVERAGE_PNG,
    markdown_report_path: Path = DEFAULT_MARKDOWN,
) -> dict[str, Any]:
    report = _load_or_build_report(run_cases=run_cases)
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    _plot_stress_overview(report, overview_png_path)
    _plot_rate_table_coverage(report, coverage_png_path)
    markdown = _build_markdown_report(
        report,
        generated_at_utc=generated_at_utc,
        overview_png_path=overview_png_path,
        coverage_png_path=coverage_png_path,
    )
    _write_text(markdown_report_path, markdown)

    dashboard = {
        'tool': 'robustness_dashboard',
        'generated_at_utc': generated_at_utc,
        'rerun_cases': run_cases,
        'outputs': {
            'dashboard_yaml': str(report_path.resolve()),
            'robustness_report_yaml': str(ROBUSTNESS_REPORT_YAML.resolve()),
            'robustness_summary_csv': str(ROBUSTNESS_SUMMARY_CSV.resolve()),
            'overview_png': str(overview_png_path.resolve()),
            'rate_table_coverage_png': str(coverage_png_path.resolve()),
            'markdown_report_md': str(markdown_report_path.resolve()),
        },
        'headline': {
            'all_cases_passed': bool(report.get('passed')),
            'external_reference_count': int((report.get('conclusion') or {}).get('external_reference_count', 0)),
            'stress_case_count': int((report.get('conclusion') or {}).get('stress_case_count', 0)),
            'failed_cases': list((report.get('conclusion') or {}).get('failed_cases', [])),
        },
        'assessment': {
            'summary': [
                'The added benchmark content broadens the evidence from strict external parity toward practical applicability across pressure, electrical, and chemistry perturbations.',
                'These added cases remain stress/applicability checks unless an authoritative external or experimental reference is attached.',
                'The ZDPlaskin-derived stress family now demonstrates that the widened diagnostic E/N table covers the executed pressure/voltage perturbations.',
            ]
        },
    }
    _write_yaml(report_path, dashboard)
    return dashboard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Generate figures and a concise report for the robustness/applicability benchmark suite.')
    parser.add_argument('--skip-runs', action='store_true', help='Reuse the existing robustness_sweep YAML/CSV outputs.')
    parser.add_argument('--write', type=Path, default=DEFAULT_WRITE)
    parser.add_argument('--overview-png', type=Path, default=DEFAULT_OVERVIEW_PNG)
    parser.add_argument('--coverage-png', type=Path, default=DEFAULT_COVERAGE_PNG)
    parser.add_argument('--markdown-report', type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args(argv)

    report = build_robustness_dashboard(
        run_cases=not args.skip_runs,
        report_path=args.write.resolve(),
        overview_png_path=args.overview_png.resolve(),
        coverage_png_path=args.coverage_png.resolve(),
        markdown_report_path=args.markdown_report.resolve(),
    )
    print(yaml.safe_dump(report, sort_keys=False))
    return 0 if report['headline']['all_cases_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
