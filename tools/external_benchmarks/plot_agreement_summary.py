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
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / 'examples' / 'outputs' / 'external_benchmarks'
FIGURE_DIR = OUTPUT_DIR / 'agreement_figures'


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return path


def _fmt_percent(value: float) -> str:
    if value < 1.0e-4:
        return f'{value:.2e}%'
    return f'{value:.3g}%'


def _quantitative_agreement_rows() -> list[dict[str, Any]]:
    crane = _load_yaml(ROOT / 'examples' / 'outputs' / 'crane_two_reaction_argon' / 'comparison_crane_two_reaction_argon.yaml')
    zdp = _load_yaml(ROOT / 'examples' / 'outputs' / 'zdplaskin_example2_surrogate' / 'comparison_zdplaskin_example2.yaml')
    zdp_circuit = pd.read_csv(ROOT / 'examples' / 'external' / 'zdplaskin_example2_circuit.csv')

    rows: list[dict[str, Any]] = []
    for label, key in [
        ('CRANE final Ar', 'final_Ar_density_m3'),
        ('CRANE final Ar+', 'final_Ar_plus_density_m3'),
        ('CRANE final e', 'final_electron_density_m3'),
    ]:
        rows.append({
            'benchmark': 'CRANE',
            'metric': label,
            'deviation_percent': float(crane['comparison'][f'{key}_relative_error']) * 100.0,
            'criterion_percent': 0.1,
            'criterion_label': '<=0.1%',
        })

    zdp_metrics = [
        ('ZDPlaskin final e', 'final_electron_density_ratio_local_over_zdplaskin'),
        ('ZDPlaskin final Ar*', 'final_Ar_star_ratio_local_over_zdplaskin'),
        ('ZDPlaskin final Ar+', 'final_Ar_plus_ratio_local_over_zdplaskin'),
        ('ZDPlaskin final Ar2+', 'final_Ar2_plus_ratio_local_over_zdplaskin'),
        ('ZDPlaskin final E/N', 'final_reduced_field_ratio_local_over_zdplaskin'),
    ]
    for label, key in zdp_metrics:
        ratio = float(zdp['comparison'][key])
        rows.append({
            'benchmark': 'ZDPlaskin',
            'metric': label,
            'deviation_percent': abs(ratio - 1.0) * 100.0,
            'criterion_percent': 5.0,
            'criterion_label': '<=5%',
        })

    local_power = float(zdp['local_result']['final_absorbed_power_W'])
    external_power = float(zdp_circuit['absorbed_power_W'].iloc[-1])
    rows.append({
        'benchmark': 'ZDPlaskin',
        'metric': 'ZDPlaskin final power',
        'deviation_percent': abs(local_power / external_power - 1.0) * 100.0,
        'criterion_percent': 5.0,
        'criterion_label': '<=5%',
    })
    return rows


def plot_quantitative_agreement() -> Path:
    rows = _quantitative_agreement_rows()
    labels = [row['metric'] for row in rows]
    values = [max(float(row['deviation_percent']), 1.0e-12) for row in rows]
    colors = ['#4c78a8' if row['benchmark'] == 'CRANE' else '#59a14f' for row in rows]

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    y = np.arange(len(rows))
    bars = ax.barh(y, values, color=colors)
    ax.set_xscale('log')
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.axvspan(1.0e-12, 0.1, color='#59a14f', alpha=0.09, label='excellent: <=0.1%')
    ax.axvspan(0.1, 5.0, color='#f1ce63', alpha=0.14, label='benchmark tolerance: <=5%')
    ax.axvline(0.1, color='#777777', linestyle='--', linewidth=1.0)
    ax.axvline(5.0, color='#777777', linestyle='--', linewidth=1.0)
    ax.set_xlabel('Absolute deviation of this code from external reference (%)')
    ax.set_title('Where this code agrees quantitatively with external benchmarks')
    ax.grid(axis='x', which='both', alpha=0.25)
    ax.legend(ncols=2, loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)
    for bar, row in zip(bars, rows, strict=True):
        value = float(row['deviation_percent'])
        ax.text(
            max(value, 1.0e-12) * 1.25,
            bar.get_y() + bar.get_height() / 2,
            f'{_fmt_percent(value)} ({row["criterion_label"]})',
            va='center',
            fontsize=8.4,
        )
    ax.set_xlim(1.0e-10, 10.0)
    fig.subplots_adjust(bottom=0.22)
    fig.text(
        0.01,
        0.01,
        'CRANE uses final species errors. ZDPlaskin uses final species, final E/N, and final circuit power. Lower is better.',
        fontsize=8,
        color='#555555',
    )
    return _save(fig, FIGURE_DIR / 'agreement_quantitative_metrics_this_code_vs_external.png')


def plot_capability_matrix() -> Path:
    columns = ['CRANE', 'ZDPlaskin', 'PyGMol']
    rows = [
        'Reaction ODE / SI units',
        'Final charged density',
        'Final neutral / excited species',
        'Final E/N and circuit power',
        'Powered-step order of magnitude',
        'Full transient waveform parity',
    ]
    # 0: not claimed, 1: sanity/partial, 2: strong benchmark support
    values = np.array([
        [2, 0, 0],
        [2, 2, 1],
        [2, 2, 0],
        [0, 2, 0],
        [0, 0, 1],
        [0, 1, 0],
    ], dtype=float)
    texts = np.array([
        ['strong', '-', '-'],
        ['strong', 'strong', 'sanity'],
        ['strong', 'strong', '-'],
        ['-', 'strong', '-'],
        ['-', '-', 'sanity'],
        ['-', 'partial', '-'],
    ])
    cmap = ListedColormap(['#e8e8e8', '#f1ce63', '#59a14f'])

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.imshow(values, cmap=cmap, vmin=0, vmax=2)
    ax.set_xticks(np.arange(len(columns)), columns)
    ax.set_yticks(np.arange(len(rows)), rows)
    ax.set_title('Benchmark-supported capability map for this code')
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, texts[i, j], ha='center', va='center', fontsize=9)
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', bottom=False, left=False)
    fig.text(
        0.01,
        0.01,
        'strong = close quantitative agreement; sanity = broad consistency only; partial = useful but not full waveform parity.',
        fontsize=8,
        color='#555555',
    )
    return _save(fig, FIGURE_DIR / 'agreement_capability_map.png')


def plot_pygmol_sanity_agreement() -> Path:
    pygmol = _load_yaml(ROOT / 'examples' / 'outputs' / 'argon_lxcat_icp_baseline' / 'comparison_pygmol_report_timeseries.yaml')
    powered = [
        row
        for row in pygmol['rows']
        if float(row.get('local_mean_absorbed_power_W') or 0.0) > 0.0
    ]
    metrics = [
        ('Final e', 'ratio_final_ne_pygmol_over_local'),
        ('Energy', 'ratio_final_mean_energy_equiv_pygmol_over_local'),
        ('Power', 'ratio_mean_absorbed_power_pygmol_over_local'),
    ]
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    palette = {'Final e': '#4c78a8', 'Energy': '#59a14f', 'Power': '#f28e2b'}
    for row in powered:
        for metric_label, key in metrics:
            labels.append(f'{row["step_id"]}\n{metric_label}')
            values.append(float(row[key]))
            colors.append(palette[metric_label])

    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    x = np.arange(len(values))
    bars = ax.bar(x, values, color=colors)
    ax.set_yscale('log')
    ax.axhline(1.0, color='#222222', linewidth=1.0, label='perfect match')
    ax.axhspan(0.1, 10.0, color='#59a14f', alpha=0.10, label='within factor 10')
    ax.set_xticks(x, labels)
    ax.set_ylabel('External PyGMol / this code ratio (log)')
    ax.set_title('PyGMol powered-step metrics that pass broad sanity agreement')
    ax.grid(axis='y', which='both', alpha=0.24)
    ax.legend(frameon=False)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.08, f'{value:.3g}x', ha='center', va='bottom', fontsize=8.5)
    ax.set_ylim(0.08, 2.5)
    fig.text(
        0.01,
        0.01,
        'PyGMol is not strict parity here; this plot only highlights powered-step quantities that are within a broad factor-10 sanity band.',
        fontsize=8,
        color='#555555',
    )
    return _save(fig, FIGURE_DIR / 'agreement_pygmol_powered_sanity_metrics.png')


def write_summary(paths: list[Path]) -> None:
    summary = OUTPUT_DIR / 'agreement_graph_summary.md'
    lines = [
        '# Agreement Graph Summary',
        '',
        'Generated figures:',
        '',
    ]
    for path in paths:
        lines.append(f'- `{path.relative_to(ROOT).as_posix()}`')
    lines.extend(
        [
            '',
            'What these graphs claim:',
            '',
            '- CRANE: this code agrees very closely for final reaction-ODE species densities and SI unit conversion.',
            '- ZDPlaskin: this code agrees well for final species, final E/N, and final absorbed power in the output-derived-table benchmark.',
            '- PyGMol: this code has broad powered-step sanity agreement only; it is not a strict waveform or physics parity claim.',
            '',
        ]
    )
    summary.write_text('\n'.join(lines), encoding='utf-8')

    artifact_csv = OUTPUT_DIR / 'agreement_graph_artifacts.csv'
    with artifact_csv.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['artifact'])
        for path in paths:
            writer.writerow([path.relative_to(ROOT).as_posix()])
        writer.writerow([summary.relative_to(ROOT).as_posix()])


def main() -> int:
    paths = [
        plot_quantitative_agreement(),
        plot_capability_matrix(),
        plot_pygmol_sanity_agreement(),
    ]
    write_summary(paths)
    for path in paths:
        print(path.relative_to(ROOT))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
