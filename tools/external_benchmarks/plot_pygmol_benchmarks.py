from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SANITY_SUMMARY = ROOT / 'examples' / 'outputs' / 'argon_lxcat_icp_baseline' / 'comparison_pygmol_summary.csv'
PRECISION_DIR = ROOT / 'examples' / 'outputs' / 'external_benchmarks' / 'diagnostic_suite' / 'pygmol_precision'
PRECISION_TIMESERIES = PRECISION_DIR / 'pygmol_precision_timeseries.csv'
PRECISION_METRICS = PRECISION_DIR / 'pygmol_precision_metrics.csv'
DEFAULT_OUTPUT_DIR = ROOT / 'examples' / 'outputs' / 'external_benchmarks' / 'diagnostic_suite' / 'figures' / 'pygmol'


def _finite_float(value: Any, default: float = float('nan')) -> float:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return default
    return candidate if np.isfinite(candidate) else default


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return path


def _read_sanity_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in [
        'local_mean_absorbed_power_W',
        'ratio_final_ne_pygmol_over_local',
        'ratio_final_mean_energy_equiv_pygmol_over_local',
        'local_final_ne_m3',
        'pygmol_final_ne_m3',
        'local_final_mean_energy_eV',
        'pygmol_final_mean_energy_equiv_eV',
    ]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _read_precision_timeseries(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _read_precision_metrics(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open(encoding='utf-8', newline='') as handle:
        return {row['metric_id']: _finite_float(row['value']) for row in csv.DictReader(handle)}


def _plot_sanity_density_ratios(df: pd.DataFrame, output_dir: Path) -> Path | None:
    powered = df[df['local_mean_absorbed_power_W'] > 0.0].copy()
    powered = powered[np.isfinite(powered['ratio_final_ne_pygmol_over_local'])]
    if powered.empty:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    x = np.arange(len(powered))
    ratios = powered['ratio_final_ne_pygmol_over_local'].to_numpy(dtype=float)
    bars = ax.bar(x, ratios, color='#4e79a7', width=0.58, label='PyGMol compact / this code production')
    ax.axhline(1.0, color='#222222', lw=1.2, label='1:1 reference')
    ax.axhspan(0.1, 10.0, color='#59a14f', alpha=0.10, label='sanity band, not precision')
    ax.set_yscale('log')
    ax.set_xticks(x, powered['step_id'].astype(str))
    ax.set_ylabel('Final electron-density ratio')
    ax.set_title('PyGMol-1 Sanity: Powered-Step Density Scale')
    ax.text(
        0.01,
        0.02,
        'Not same-footing: geometry, chemistry, wall loss, and production closure differ.',
        transform=ax.transAxes,
        fontsize=8.5,
        color='#444444',
        va='bottom',
    )
    for bar, value in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f'{value:.2g}x', ha='center', va='bottom', fontsize=8)
    ax.legend(loc='upper left', fontsize=8)
    return _save(fig, output_dir / 'pygmol_sanity_density_ratio_by_powered_step.png')


def _plot_sanity_powered_final_values(df: pd.DataFrame, output_dir: Path) -> Path | None:
    powered = df[df['local_mean_absorbed_power_W'] > 0.0].copy()
    if powered.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    x = np.arange(len(powered))
    width = 0.36

    axes[0].bar(x - width / 2, powered['local_final_ne_m3'], width, label='This code production', color='#f28e2b')
    axes[0].bar(x + width / 2, powered['pygmol_final_ne_m3'], width, label='PyGMol compact', color='#4e79a7')
    axes[0].set_yscale('log')
    axes[0].set_xticks(x, powered['step_id'].astype(str))
    axes[0].set_ylabel('Final electron density [m$^{-3}$]')
    axes[0].set_title('Electron Density')
    axes[0].legend(fontsize=8)

    axes[1].bar(x - width / 2, powered['local_final_mean_energy_eV'], width, label='This code production', color='#f28e2b')
    axes[1].bar(x + width / 2, powered['pygmol_final_mean_energy_equiv_eV'], width, label='PyGMol compact, 1.5 T_e', color='#4e79a7')
    axes[1].set_xticks(x, powered['step_id'].astype(str))
    axes[1].set_ylabel('Final mean-energy-equivalent [eV]')
    axes[1].set_title('Electron Energy Scale')
    axes[1].legend(fontsize=8)

    fig.suptitle('PyGMol-1 Sanity: External Result vs This Code Production Case', y=1.02)
    return _save(fig, output_dir / 'pygmol_sanity_powered_final_values.png')


def _plot_precision_species_timeseries(df: pd.DataFrame, output_dir: Path) -> Path | None:
    if df.empty:
        return None

    time_ms = df['time_s'].to_numpy(dtype=float) * 1.0e3
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.0), sharex=True)
    for ax, label, py_col, local_col in [
        (axes[0], 'Electron density [m$^{-3}$]', 'pygmol_e', 'local_e'),
        (axes[1], 'Ar+ density [m$^{-3}$]', 'pygmol_Ar_plus', 'local_Ar_plus'),
    ]:
        ax.plot(time_ms, df[py_col], color='#4e79a7', lw=1.8, label='PyGMol')
        ax.plot(time_ms, df[local_col], color='#f28e2b', lw=1.5, ls='--', label='Local same-footing')
        ax.set_yscale('log')
        ax.set_ylabel(label)
        ax.grid(True, which='both', alpha=0.22)
    axes[0].set_title('PyGMol-Precision-1: Species Time Series Overlay')
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel('Time [ms]')
    return _save(fig, output_dir / 'pygmol_precision_species_timeseries_overlay.png')


def _plot_precision_energy_power(df: pd.DataFrame, output_dir: Path) -> Path | None:
    if df.empty:
        return None

    time_ms = df['time_s'].to_numpy(dtype=float) * 1.0e3
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.8), sharex=True)
    axes[0].plot(time_ms, df['pygmol_T_e'], color='#4e79a7', lw=1.8, label='PyGMol T_e')
    axes[0].plot(time_ms, df['local_T_e'], color='#f28e2b', lw=1.5, ls='--', label='Local same-footing T_e')
    axes[0].set_ylabel('Electron temperature [eV]')
    axes[0].set_title('PyGMol-Precision-1: Energy and Power Overlay')
    axes[0].grid(True, alpha=0.22)
    axes[0].legend(fontsize=8)

    axes[1].plot(time_ms, df['pygmol_P'], color='#4e79a7', lw=1.8, label='PyGMol absorbed power')
    axes[1].plot(time_ms, df['local_P'], color='#f28e2b', lw=1.5, ls='--', label='Local same-footing absorbed power')
    axes[1].set_ylabel('Absorbed power [W]')
    axes[1].set_xlabel('Time [ms]')
    axes[1].grid(True, alpha=0.22)
    axes[1].legend(fontsize=8)
    return _save(fig, output_dir / 'pygmol_precision_energy_power_overlay.png')


def _plot_precision_error_bars(metrics: dict[str, float], output_dir: Path) -> Path | None:
    if not metrics:
        return None
    labels = ['Ar', 'Ar+', 'e', 'T_e', 'T_n', 'p', 'P']
    metric_keys = ['Ar', 'Ar_plus', 'e', 'T_e', 'T_n', 'p', 'P']
    final = np.asarray([max(metrics.get(f'final_{key}_relative_error', 0.0), 1.0e-12) for key in metric_keys], dtype=float)
    wave = np.asarray([max(metrics.get(f'{key}_waveform_nrmse', 0.0), 1.0e-12) for key in metric_keys], dtype=float)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, final, width, color='#76b7b2', label='Final relative error')
    ax.bar(x + width / 2, wave, width, color='#e15759', label='Waveform NRMSE')
    ax.axhline(1.0e-3, color='#222222', lw=1.2, ls=':', label='precision gate 1e-3')
    ax.set_yscale('log')
    ax.set_xticks(x, labels)
    ax.set_ylabel('Error metric')
    ax.set_title('PyGMol-Precision-1: Same-Footing Error Metrics')
    ax.grid(True, axis='y', which='both', alpha=0.22)
    ax.legend(fontsize=8)
    return _save(fig, output_dir / 'pygmol_precision_error_metrics.png')


def _plot_scope_matrix(output_dir: Path) -> Path:
    rows = ['PyGMol-1 sanity\nproduction case', 'PyGMol-Precision-1\nsame-footing harness']
    cols = ['geometry', 'chemistry', 'power', 'wall return/loss', 'initial state', 'precision claim']
    values = np.array([
        [0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1],
    ], dtype=float)
    fig, ax = plt.subplots(figsize=(9.2, 3.2))
    ax.imshow(values, cmap=matplotlib.colors.ListedColormap(['#f2f2f2', '#59a14f']), vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(np.arange(len(cols)), cols, rotation=25, ha='right')
    ax.set_yticks(np.arange(len(rows)), rows)
    ax.set_title('PyGMol Benchmark Scope: What Is Actually Aligned?')
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text = 'aligned' if values[i, j] else 'not aligned'
            ax.text(j, i, text, ha='center', va='center', fontsize=8, color='#222222')
    return _save(fig, output_dir / 'pygmol_benchmark_scope_matrix.png')


def _write_artifacts(output_dir: Path, paths: list[Path]) -> Path:
    artifact_path = output_dir / 'pygmol_graph_artifacts.csv'
    rows = [{'figure': str(path.relative_to(ROOT)), 'purpose': _purpose_for(path.name)} for path in paths]
    _write_rows(artifact_path, rows, ['figure', 'purpose'])
    return artifact_path


def _write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _purpose_for(name: str) -> str:
    return {
        'pygmol_sanity_density_ratio_by_powered_step.png': 'Shows powered-step electron-density scale for the production case sanity comparison.',
        'pygmol_sanity_powered_final_values.png': 'Separates PyGMol external values from this-code production-case values.',
        'pygmol_precision_species_timeseries_overlay.png': 'Shows same-footing species time-series overlay for PyGMol and local harness.',
        'pygmol_precision_energy_power_overlay.png': 'Shows same-footing electron-temperature and absorbed-power overlay.',
        'pygmol_precision_error_metrics.png': 'Shows final relative errors and waveform NRMSE against the precision gate.',
        'pygmol_benchmark_scope_matrix.png': 'Explains which axes are aligned for sanity and precision PyGMol benchmarks.',
    }.get(name, 'PyGMol benchmark figure.')


def _write_summary(output_dir: Path, paths: list[Path], metrics: dict[str, float]) -> Path:
    summary_path = output_dir / 'pygmol_graph_summary.md'
    rel_paths = [path.relative_to(ROOT).as_posix() for path in paths if path.suffix.lower() == '.png']
    lines = [
        '# PyGMol Benchmark Graphs',
        '',
        'These figures separate the production-case PyGMol sanity comparison from the same-footing precision harness.',
        '',
        '## Interpretation',
        '',
        '- `PyGMol-1` is a sanity comparison: it checks scale only and does not claim precision because the production model and PyGMol compact model are not same-footing.',
        '- `PyGMol-Precision-1` is a tools-only compact-equation parity check: geometry, chemistry, power, wall return, initial state, and sample times are aligned.',
        f"- Current precision max final relative error: `{metrics.get('max_final_relative_error', float('nan')):.3g}`.",
        f"- Current precision max waveform NRMSE: `{metrics.get('max_waveform_nrmse', float('nan')):.3g}`.",
        '',
        '## Figures',
        '',
    ]
    for rel in rel_paths:
        lines.append(f'- `{rel}`')
    lines.append('')
    summary_path.write_text('\n'.join(lines), encoding='utf-8')
    return summary_path


def build_pygmol_graphs(
    *,
    sanity_summary: Path = SANITY_SUMMARY,
    precision_timeseries: Path = PRECISION_TIMESERIES,
    precision_metrics: Path = PRECISION_METRICS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sanity = _read_sanity_rows(sanity_summary)
    precision = _read_precision_timeseries(precision_timeseries)
    metrics = _read_precision_metrics(precision_metrics)

    paths: list[Path] = []
    for candidate in [
        _plot_sanity_density_ratios(sanity, output_dir),
        _plot_sanity_powered_final_values(sanity, output_dir),
        _plot_precision_species_timeseries(precision, output_dir),
        _plot_precision_energy_power(precision, output_dir),
        _plot_precision_error_bars(metrics, output_dir),
        _plot_scope_matrix(output_dir),
    ]:
        if candidate is not None:
            paths.append(candidate)
    paths.append(_write_artifacts(output_dir, paths))
    paths.append(_write_summary(output_dir, paths, metrics))
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Create PyGMol benchmark figures for sanity and same-footing precision comparisons.')
    parser.add_argument('--sanity-summary', type=Path, default=SANITY_SUMMARY)
    parser.add_argument('--precision-timeseries', type=Path, default=PRECISION_TIMESERIES)
    parser.add_argument('--precision-metrics', type=Path, default=PRECISION_METRICS)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    paths = build_pygmol_graphs(
        sanity_summary=args.sanity_summary,
        precision_timeseries=args.precision_timeseries,
        precision_metrics=args.precision_metrics,
        output_dir=args.output_dir,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
