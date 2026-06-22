from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / 'examples' / 'outputs' / 'external_benchmarks'
FIGURE_DIR = OUTPUT_DIR / 'figures'


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


def _as_float(value: Any) -> float:
    if value in (None, ''):
        return math.nan
    return float(value)


def _format_value(value: float) -> str:
    if not math.isfinite(value):
        return ''
    if value == 0.0:
        return '0'
    if abs(value) >= 1.0e4 or abs(value) < 1.0e-3:
        return f'{value:.2e}'
    return f'{value:.3g}'


def _finish(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_crane_accuracy(crane: dict[str, Any]) -> Path:
    labels = ['Ar density', 'Ar+ density', 'Electron density']
    keys = ['final_Ar_density_m3', 'final_Ar_plus_density_m3', 'final_electron_density_m3']
    ratios = [float(crane['comparison'][f'{key}_ratio_local_over_crane']) for key in keys]
    deviations_ppm = [(ratio - 1.0) * 1.0e6 for ratio in ratios]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bars = ax.barh(labels, deviations_ppm, color=['#4c78a8', '#59a14f', '#e15759'])
    ax.axvline(0.0, color='#222222', linewidth=1.2, label='External CRANE reference')
    ax.set_xlabel('This code minus external CRANE reference (ppm)')
    ax.set_title('CRANE TwoReactionArgon: this code vs external reference')
    ax.grid(axis='x', alpha=0.25)
    ax.legend(ncols=3, loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)
    for bar, ratio, value in zip(bars, ratios, deviations_ppm, strict=True):
        x = value / 2.0 if abs(value) > 0.2 else 0.08
        color = 'white' if abs(value) > 0.6 else '#222222'
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f'{value:+.3g} ppm ({ratio:.9f}x)',
            va='center',
            ha='center',
            fontsize=8.8,
            color=color,
        )
    ax.set_xlim(min(-3.0, min(deviations_ppm) * 1.25), max(0.5, max(deviations_ppm) * 1.25))
    fig.subplots_adjust(bottom=0.22)
    fig.text(
        0.01,
        0.01,
        'External = committed CRANE tutorial reference. This code = rerun local SI-converted ODE case.',
        fontsize=8,
        color='#555555',
    )
    path = FIGURE_DIR / 'accuracy_crane_this_code_vs_external.png'
    _finish(fig, path)
    return path


def plot_zdplaskin_accuracy(zdp: dict[str, Any]) -> Path:
    rows = [
        ('Final ne', 'final_electron_density_ratio_local_over_zdplaskin'),
        ('Peak ne', 'peak_electron_density_ratio_local_over_zdplaskin'),
        ('Final Ar*', 'final_Ar_star_ratio_local_over_zdplaskin'),
        ('Final Ar+', 'final_Ar_plus_ratio_local_over_zdplaskin'),
        ('Final Ar2+', 'final_Ar2_plus_ratio_local_over_zdplaskin'),
        ('Final E/N', 'final_reduced_field_ratio_local_over_zdplaskin'),
    ]
    labels = [row[0] for row in rows]
    values = [float(zdp['comparison'][row[1]]) for row in rows]
    deviations = [(value - 1.0) * 100.0 for value in values]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    colors = ['#4c78a8' if abs(dev) <= 5.0 else '#f28e2b' for dev in deviations]
    bars = ax.bar(labels, deviations, color=colors, label='This code deviation')
    ax.axhline(0.0, color='#222222', linewidth=1.0, label='External ZDPlaskin reference')
    ax.axhspan(-5.0, 5.0, color='#59a14f', alpha=0.12, label='within +/-5%')
    ax.set_ylabel('This code minus external ZDPlaskin reference (%)')
    ax.set_title('ZDPlaskin example2: this code vs external reference')
    ax.grid(axis='y', alpha=0.25)
    ax.legend(ncols=3, loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)
    for bar, ratio, dev in zip(bars, values, deviations, strict=True):
        va = 'bottom' if dev >= 0 else 'top'
        offset = 0.6 if dev >= 0 else -0.6
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            dev + offset,
            f'{ratio:.4f}x',
            ha='center',
            va=va,
            fontsize=8.5,
        )
    ax.set_ylim(min(-12.0, min(deviations) - 2.0), max(5.0, max(deviations) + 2.0))
    fig.subplots_adjust(bottom=0.24)
    fig.text(
        0.01,
        0.01,
        'External = committed ZDPlaskin example2 output. This code = rerun DC-series case using the output-derived E/N table.',
        fontsize=8,
        color='#555555',
    )
    path = FIGURE_DIR / 'accuracy_zdplaskin_this_code_vs_external.png'
    _finish(fig, path)
    return path


def plot_pygmol_accuracy(pygmol: dict[str, Any]) -> Path:
    powered = [
        row
        for row in pygmol['rows']
        if _as_float(row.get('local_mean_absorbed_power_W')) > 0.0
    ]
    metric_keys = [
        ('Final ne', 'ratio_final_ne_pygmol_over_local'),
        ('Mean ne', 'ratio_mean_ne_pygmol_over_local'),
        ('Mean energy', 'ratio_final_mean_energy_equiv_pygmol_over_local'),
        ('Absorbed power', 'ratio_mean_absorbed_power_pygmol_over_local'),
    ]
    labels = [row['step_id'] for row in powered]
    width = 0.18
    x_positions = list(range(len(labels)))
    palette = ['#4c78a8', '#f28e2b', '#59a14f', '#b07aa1']

    fig, ax = plt.subplots(figsize=(9.2, 5.1))
    for metric_index, (metric_label, key) in enumerate(metric_keys):
        values = [_as_float(row.get(key)) for row in powered]
        offset = (metric_index - (len(metric_keys) - 1) / 2) * width
        bars = ax.bar(
            [x + offset for x in x_positions],
            values,
            width=width,
            label=metric_label,
            color=palette[metric_index],
        )
        for bar, value in zip(bars, values, strict=True):
            if math.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value * 1.08,
                    _format_value(value),
                    ha='center',
                    va='bottom',
                    fontsize=7.8,
                    rotation=90,
                )
    ax.set_yscale('log')
    ax.axhline(1.0, color='#222222', linewidth=1.0, label='perfect match')
    ax.axhspan(1.0e-3, 1.0e3, color='#59a14f', alpha=0.08, label='sanity pass band')
    ax.set_xticks(x_positions, labels)
    ax.set_ylabel('External PyGMol / this code ratio (log scale)')
    ax.set_title('PyGMol argon: external result divided by this code result')
    ax.grid(axis='y', which='both', alpha=0.22)
    ax.legend(ncols=3, loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.set_ylim(1.0e-2, 2.0e1)
    fig.text(
        0.01,
        0.01,
        'External = PyGMol 1.1.0 compact single-cylinder surrogate. This code = rerun two-zone LXCat/table local case.',
        fontsize=8,
        color='#555555',
    )
    path = FIGURE_DIR / 'accuracy_pygmol_external_over_this_code.png'
    _finish(fig, path)
    return path


def plot_local_speed(profile: dict[str, Any]) -> Path:
    rows = [item['aggregate'] for item in profile['cases']]
    labels = ['CRANE\nODE', 'ZDPlaskin\nDC/table', 'Argon LXCat\nlocal case']
    times = [float(row['median_wall_time_s']) for row in rows]
    nfev = [float(row['nfev_median']) for row in rows]

    fig, (ax_time, ax_eval) = plt.subplots(1, 2, figsize=(10.4, 4.8))
    colors = ['#4c78a8', '#59a14f', '#f28e2b']
    bars = ax_time.bar(labels, times, color=colors)
    ax_time.set_ylabel('Median wall time (s)')
    ax_time.set_title('This code runtime')
    ax_time.grid(axis='y', alpha=0.25)
    for bar, value in zip(bars, times, strict=True):
        ax_time.text(bar.get_x() + bar.get_width() / 2, value + max(times) * 0.025, f'{value:.3f}s', ha='center', fontsize=9)
    ax_time.set_ylim(0.0, max(times) * 1.18)

    bars = ax_eval.bar(labels, nfev, color=colors)
    ax_eval.set_ylabel('Median RHS evaluations (nfev)')
    ax_eval.set_title('This code solver work')
    ax_eval.grid(axis='y', alpha=0.25)
    for bar, value in zip(bars, nfev, strict=True):
        ax_eval.text(bar.get_x() + bar.get_width() / 2, value + max(nfev) * 0.025, f'{int(value)}', ha='center', fontsize=9)
    ax_eval.set_ylim(0.0, max(nfev) * 1.18)
    fig.text(
        0.01,
        0.01,
        'All bars are this code, measured with run_from_yaml including configured outputs; 3 repetitions per case.',
        fontsize=8,
        color='#555555',
    )
    path = FIGURE_DIR / 'speed_this_code_runtime.png'
    _finish(fig, path)
    return path


def plot_speed_context(profile: dict[str, Any], pygmol_runtime: dict[str, Any]) -> Path:
    rows = {item['aggregate']['case_id']: item['aggregate'] for item in profile['cases']}
    labels = ['External PyGMol\nmodel.run()', 'This code\nlocal argon case', 'This code +\nexternal PyGMol report']
    values = [
        float(pygmol_runtime['aggregate']['median_model_run_s']),
        float(rows['pygmol_argon_local_case']['median_wall_time_s']),
        3.589512,
    ]
    colors = ['#b07aa1', '#f28e2b', '#4c78a8']

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel('Wall time (s, log scale)')
    ax.set_yscale('log')
    ax.set_title('Runtime provenance for the PyGMol comparison')
    ax.grid(axis='y', which='both', alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.12, f'{value:.3f}s', ha='center', fontsize=9)
    fig.text(
        0.01,
        0.01,
        'Only PyGMol external runtime is measured here. CRANE and ZDPlaskin live executable runtimes were not measured.',
        fontsize=8,
        color='#555555',
    )
    path = FIGURE_DIR / 'speed_external_vs_this_code_context.png'
    _finish(fig, path)
    return path


def write_summary(paths: list[Path]) -> None:
    summary_path = OUTPUT_DIR / 'graph_summary.md'
    lines = [
        '# Benchmark Graph Summary',
        '',
        'Generated figures:',
        '',
    ]
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        lines.append(f'- `{rel}`')
    lines.extend(
        [
            '',
            'Interpretation notes:',
            '',
            '- CRANE accuracy plot: the baseline/zero line is the external CRANE reference; bars are this code deviation from that reference.',
            '- ZDPlaskin accuracy plot: the baseline/zero line is the external ZDPlaskin committed output; bars are this code deviation from that output.',
            '- PyGMol accuracy plot: bars are external PyGMol result divided by this code result; 1.0 means equal.',
            '- This-code runtime plot contains only this code timings.',
            '- Runtime provenance plot contains the one measured external runtime, PyGMol model.run(), and separates it from this code and full report timing.',
            '- CRANE and ZDPlaskin live executable runtimes were not measured.',
            '',
        ]
    )
    summary_path.write_text('\n'.join(lines), encoding='utf-8')

    csv_path = OUTPUT_DIR / 'graph_artifacts.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['artifact'])
        for path in paths:
            writer.writerow([path.relative_to(ROOT).as_posix()])
        writer.writerow([summary_path.relative_to(ROOT).as_posix()])


def main() -> int:
    crane = _load_yaml(ROOT / 'examples' / 'outputs' / 'crane_two_reaction_argon' / 'comparison_crane_two_reaction_argon.yaml')
    zdp = _load_yaml(ROOT / 'examples' / 'outputs' / 'zdplaskin_example2_surrogate' / 'comparison_zdplaskin_example2.yaml')
    pygmol = _load_yaml(ROOT / 'examples' / 'outputs' / 'argon_lxcat_icp_baseline' / 'comparison_pygmol_report_rerun.yaml')
    profile = _load_yaml(OUTPUT_DIR / 'current_runtime_profile.yaml')
    pygmol_runtime = _load_yaml(OUTPUT_DIR / 'pygmol_external_runtime_only.yaml')

    paths = [
        plot_crane_accuracy(crane),
        plot_zdplaskin_accuracy(zdp),
        plot_pygmol_accuracy(pygmol),
        plot_local_speed(profile),
        plot_speed_context(profile, pygmol_runtime),
    ]
    write_summary(paths)
    for path in paths:
        print(path.relative_to(ROOT))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
