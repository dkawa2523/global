from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / 'examples' / 'outputs' / 'external_benchmarks'
FIGURE_DIR = OUTPUT_DIR / 'agreement_by_software_figures'


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return path


def _fmt_percent(value: float) -> str:
    if abs(value) < 1.0e-4:
        return f'{value:.2e}%'
    return f'{value:.3g}%'


def _fmt_seconds(value: float) -> str:
    if value < 1.0:
        return f'{value * 1.0e3:.1f} ms'
    return f'{value:.3g} s'


def _runtime_aggregate(case_id: str) -> dict[str, Any]:
    profile = _load_yaml(OUTPUT_DIR / 'current_runtime_profile.yaml')
    for case in profile['cases']:
        if case['case_id'] == case_id:
            return case['aggregate']
    raise KeyError(case_id)


def _draw_deviation_bars(
    ax: plt.Axes,
    labels: list[str],
    deviations_percent: list[float],
    ratios: list[float],
    *,
    threshold_percent: float,
    title: str,
    threshold_label: str,
) -> None:
    values = [max(abs(value), 1.0e-12) for value in deviations_percent]
    y = np.arange(len(labels))
    colors = ['#59a14f' if value <= threshold_percent else '#e15759' for value in deviations_percent]

    bars = ax.barh(y, values, color=colors)
    ax.set_xscale('log')
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.axvspan(1.0e-12, threshold_percent, color='#59a14f', alpha=0.10, label=threshold_label)
    ax.axvline(threshold_percent, color='#555555', linestyle='--', linewidth=1.0)
    ax.set_xlabel('Absolute deviation of this code from external reference (%)')
    ax.set_title(title)
    ax.grid(axis='x', which='both', alpha=0.25)
    ax.legend(loc='lower right', frameon=False)

    upper = max(threshold_percent * 4.0, max(values) * 3.8)
    lower = min(1.0e-10, max(min(values) / 5.0, 1.0e-14))
    ax.set_xlim(lower, upper)
    for bar, deviation, ratio in zip(bars, deviations_percent, ratios, strict=True):
        x = max(abs(deviation), 1.0e-12) * 1.2
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2.0,
            f'{_fmt_percent(abs(deviation))}; ratio={ratio:.6g}',
            va='center',
            fontsize=8.4,
        )


def _draw_ratio_bars(
    ax: plt.Axes,
    labels: list[str],
    ratios: list[float],
    *,
    title: str,
    accepted_min: float,
    accepted_max: float,
    accepted_label: str,
) -> None:
    values = [max(value, 1.0e-12) for value in ratios]
    y = np.arange(len(labels))
    colors = ['#59a14f' if accepted_min <= value <= accepted_max else '#e15759' for value in ratios]

    bars = ax.barh(y, values, color=colors)
    ax.set_xscale('log')
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.axvspan(accepted_min, accepted_max, color='#59a14f', alpha=0.10, label=accepted_label)
    ax.axvline(1.0, color='#222222', linewidth=1.0, label='perfect match')
    ax.set_xlabel('External result / this code result')
    ax.set_title(title)
    ax.grid(axis='x', which='both', alpha=0.25)
    ax.legend(loc='lower right', frameon=False)
    ax.set_xlim(0.05, max(accepted_max * 2.0, max(values) * 2.2))
    for bar, value in zip(bars, ratios, strict=True):
        ax.text(
            max(value, 1.0e-12) * 1.12,
            bar.get_y() + bar.get_height() / 2.0,
            f'{value:.3g}x',
            va='center',
            fontsize=8.4,
        )


def _draw_runtime(
    ax: plt.Axes,
    labels: list[str],
    values_s: list[float],
    *,
    title: str,
    log_scale: bool = False,
) -> None:
    colors = ['#4c78a8', '#f28e2b', '#bab0ac'][: len(values_s)]
    x = np.arange(len(values_s))
    bars = ax.bar(x, values_s, color=colors)
    ax.set_xticks(x, labels)
    ax.set_ylabel('Median wall time (s)')
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.25)
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylim(min(values_s) / 2.5, max(values_s) * 3.0)
    else:
        ax.set_ylim(0.0, max(values_s) * 1.65)
    for bar, value in zip(bars, values_s, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value * (1.08 if log_scale else 1.03),
            _fmt_seconds(value),
            ha='center',
            va='bottom',
            fontsize=8.8,
        )


def _draw_decision_text(
    ax: plt.Axes,
    *,
    decision: str,
    supported: list[str],
    caveats: list[str],
) -> None:
    ax.axis('off')
    ax.text(
        0.0,
        0.96,
        decision,
        va='top',
        fontsize=11,
        fontweight='bold',
        color='#222222',
        bbox={'boxstyle': 'round,pad=0.35', 'facecolor': '#eef5ee', 'edgecolor': '#9cc49c'},
    )
    ax.text(0.0, 0.63, 'Supported by this benchmark', fontsize=9.8, fontweight='bold')
    ax.text(0.02, 0.55, '\n'.join(f'- {item}' for item in supported), fontsize=9, va='top')
    ax.text(0.52, 0.63, 'Not claimed / caveats', fontsize=9.8, fontweight='bold')
    ax.text(0.54, 0.55, '\n'.join(f'- {item}' for item in caveats), fontsize=9, va='top')


def plot_crane_decision() -> Path:
    report = _load_yaml(
        ROOT
        / 'examples'
        / 'outputs'
        / 'crane_two_reaction_argon'
        / 'comparison_crane_two_reaction_argon.yaml'
    )
    comparison = report['comparison']
    metrics = [
        ('Ar final density', 'final_Ar_density_m3'),
        ('Ar+ final density', 'final_Ar_plus_density_m3'),
        ('e final density', 'final_electron_density_m3'),
    ]
    labels = [label for label, _ in metrics]
    deviations = [float(comparison[f'{key}_relative_error']) * 100.0 for _, key in metrics]
    ratios = [float(comparison[f'{key}_ratio_local_over_crane']) for _, key in metrics]
    runtime = _runtime_aggregate('crane_two_reaction_argon_ode_parity')

    fig = plt.figure(figsize=(12.0, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[2.7, 1.25], width_ratios=[1.65, 1.0])
    ax_metrics = fig.add_subplot(grid[0, 0])
    ax_runtime = fig.add_subplot(grid[0, 1])
    ax_notes = fig.add_subplot(grid[1, :])

    _draw_deviation_bars(
        ax_metrics,
        labels,
        deviations,
        ratios,
        threshold_percent=0.1,
        threshold_label='scoped pass band: <=0.1%',
        title='CRANE TwoReactionArgon: final species agreement',
    )
    _draw_runtime(
        ax_runtime,
        ['This code\nlocal rerun'],
        [float(runtime['median_wall_time_s'])],
        title='Runtime context',
    )
    _draw_decision_text(
        ax_notes,
        decision='Decision: strong agreement for the scoped CRANE ODE benchmark.',
        supported=[
            'constant-rate reaction ODE assembly',
            'cm-based CRANE reference converted to SI',
            'final Ar, Ar+, and electron densities',
        ],
        caveats=[
            'does not validate electron-energy physics',
            'does not validate circuit or transport models',
            'external time series and runtime are not stored',
        ],
    )
    fig.suptitle('CRANE benchmark decision for this code', fontsize=14, fontweight='bold')
    return _save(fig, FIGURE_DIR / 'benchmark_decision_crane.png')


def plot_zdplaskin_decision() -> Path:
    report = _load_yaml(
        ROOT
        / 'examples'
        / 'outputs'
        / 'zdplaskin_example2_surrogate'
        / 'comparison_zdplaskin_example2.yaml'
    )
    comparison = report['comparison']
    circuit = pd.read_csv(ROOT / 'examples' / 'external' / 'zdplaskin_example2_circuit.csv')
    local_power = float(report['local_result']['final_absorbed_power_W'])
    external_power = float(circuit['absorbed_power_W'].iloc[-1])
    runtime = _runtime_aggregate('zdplaskin_example2_argon_dc_series')

    metric_specs = [
        ('final e density', 'final_electron_density_ratio_local_over_zdplaskin'),
        ('peak e density', 'peak_electron_density_ratio_local_over_zdplaskin'),
        ('final Ar* density', 'final_Ar_star_ratio_local_over_zdplaskin'),
        ('final Ar+ density', 'final_Ar_plus_ratio_local_over_zdplaskin'),
        ('final Ar2+ density', 'final_Ar2_plus_ratio_local_over_zdplaskin'),
        ('final E/N', 'final_reduced_field_ratio_local_over_zdplaskin'),
    ]
    labels = [label for label, _ in metric_specs] + ['final absorbed power']
    ratios = [float(comparison[key]) for _, key in metric_specs] + [local_power / external_power]
    deviations = [abs(ratio - 1.0) * 100.0 for ratio in ratios]

    fig = plt.figure(figsize=(12.4, 7.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[2.9, 1.25], width_ratios=[1.75, 1.0])
    ax_metrics = fig.add_subplot(grid[0, 0])
    ax_runtime = fig.add_subplot(grid[0, 1])
    ax_notes = fig.add_subplot(grid[1, :])

    _draw_deviation_bars(
        ax_metrics,
        labels,
        deviations,
        ratios,
        threshold_percent=5.0,
        threshold_label='final-value pass band: <=5%',
        title='ZDPlaskin example2: quantities comparable to stored reference',
    )
    _draw_runtime(
        ax_runtime,
        ['This code\nlocal rerun'],
        [float(runtime['median_wall_time_s'])],
        title='Runtime context',
    )
    _draw_decision_text(
        ax_notes,
        decision='Decision: strong final-state/circuit agreement; transient species parity is partial.',
        supported=[
            'final charged and excited argon species',
            'final reduced electric field on the same circuit footing',
            'final voltage-current absorbed power',
        ],
        caveats=[
            'peak electron density is outside the 5% final-value band',
            'external species time series is not stored',
            'rates come from an output-derived E/N table, not live BOLSIG+',
        ],
    )
    fig.suptitle('ZDPlaskin benchmark decision for this code', fontsize=14, fontweight='bold')
    return _save(fig, FIGURE_DIR / 'benchmark_decision_zdplaskin.png')


def plot_pygmol_decision() -> Path:
    report = _load_yaml(
        ROOT
        / 'examples'
        / 'outputs'
        / 'argon_lxcat_icp_baseline'
        / 'comparison_pygmol_report_timeseries.yaml'
    )
    external_runtime = _load_yaml(OUTPUT_DIR / 'pygmol_external_runtime_only.yaml')
    local_runtime = _runtime_aggregate('pygmol_argon_local_case')

    labels: list[str] = []
    ratios: list[float] = []
    for row in report['rows']:
        step_id = str(row['step_id'])
        labels.append(f'{step_id} final e')
        ratios.append(float(row['ratio_final_ne_pygmol_over_local']))
        labels.append(f'{step_id} energy')
        ratios.append(float(row['ratio_final_mean_energy_equiv_pygmol_over_local']))
        power_ratio = row.get('ratio_mean_absorbed_power_pygmol_over_local')
        if power_ratio not in ('', None):
            labels.append(f'{step_id} power')
            ratios.append(float(power_ratio))

    fig = plt.figure(figsize=(12.6, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[3.1, 1.3], width_ratios=[1.75, 1.0])
    ax_metrics = fig.add_subplot(grid[0, 0])
    ax_runtime = fig.add_subplot(grid[0, 1])
    ax_notes = fig.add_subplot(grid[1, :])

    _draw_ratio_bars(
        ax_metrics,
        labels,
        ratios,
        title='PyGMol argon: external result divided by this code result',
        accepted_min=0.1,
        accepted_max=10.0,
        accepted_label='sanity band: within factor 10',
    )
    _draw_runtime(
        ax_runtime,
        ['This code\nlocal case', 'External PyGMol\nmodel.run()'],
        [
            float(local_runtime['median_wall_time_s']),
            float(external_runtime['aggregate']['median_model_run_s']),
        ],
        title='Runtime context',
        log_scale=True,
    )
    _draw_decision_text(
        ax_notes,
        decision='Decision: useful powered-step sanity benchmark, not precision validation.',
        supported=[
            'powered-step electron density order of magnitude',
            'powered-step energy and absorbed-power scale',
            'external PyGMol runtime measured for context',
        ],
        caveats=[
            'not same geometry or wall model as the local two-zone case',
            'afterglow electron density diverges strongly',
            'does not support full waveform parity claims',
        ],
    )
    fig.suptitle('PyGMol benchmark decision for this code', fontsize=14, fontweight='bold')
    return _save(fig, FIGURE_DIR / 'benchmark_decision_pygmol.png')


def write_summary(paths: list[Path]) -> None:
    summary = OUTPUT_DIR / 'agreement_by_software_summary.md'
    lines = [
        '# Per-Software Benchmark Decision Graphs',
        '',
        'Generated figures:',
        '',
    ]
    for path in paths:
        lines.append(f'- `{path.relative_to(ROOT).as_posix()}`')
    lines.extend(
        [
            '',
            'Reading guide:',
            '',
            '- CRANE: read as strong agreement for the scoped constant-rate ODE and SI conversion benchmark.',
            '- ZDPlaskin: read as strong final-state and circuit-final agreement, with only partial transient species support.',
            '- PyGMol: read as broad powered-step sanity agreement only, not a precision or waveform-parity benchmark.',
            '',
        ]
    )
    summary.write_text('\n'.join(lines), encoding='utf-8')

    artifact_csv = OUTPUT_DIR / 'agreement_by_software_artifacts.csv'
    with artifact_csv.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['artifact'])
        for path in paths:
            writer.writerow([path.relative_to(ROOT).as_posix()])
        writer.writerow([summary.relative_to(ROOT).as_posix()])


def main() -> int:
    paths = [
        plot_crane_decision(),
        plot_zdplaskin_decision(),
        plot_pygmol_decision(),
    ]
    write_summary(paths)
    for path in paths:
        print(path.relative_to(ROOT))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
