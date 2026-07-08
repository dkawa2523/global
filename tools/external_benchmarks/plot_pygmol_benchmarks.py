from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_DIR = ROOT / 'examples' / 'outputs' / 'external_benchmarks' / 'diagnostic_suite' / 'figures' / 'pygmol'

BLUE = '#4e79a7'
ORANGE = '#f28e2b'
GRID_ALPHA = 0.22
MIN_LOG_VALUE = 1.0e-12
PRECISION_GATE = 1.0e-3

SANITY_RATIO = 'ratio_final_ne_pygmol_over_local'
POWERED_STEP_COLUMNS = ('local_mean_absorbed_power_W', SANITY_RATIO)

TimePanel = tuple[str, str, str, bool]
TimeSeriesFigure = tuple[str, str, tuple[TimePanel, ...]]

PRECISION_TIMESERIES_FIGURES: tuple[TimeSeriesFigure, ...] = (
    (
        'pygmol_precision_species_timeseries_overlay.png',
        'PyGMol-Precision-1: Species Time Series Overlay',
        (
            ('Electron density [m$^{-3}$]', 'pygmol_e', 'local_e', True),
            ('Ar+ density [m$^{-3}$]', 'pygmol_Ar_plus', 'local_Ar_plus', True),
        ),
    ),
    (
        'pygmol_precision_energy_power_overlay.png',
        'PyGMol-Precision-1: Energy and Power Overlay',
        (
            ('Electron temperature [eV]', 'pygmol_T_e', 'local_T_e', False),
            ('Absorbed power [W]', 'pygmol_P', 'local_P', False),
        ),
    ),
)

ERROR_METRIC_PAIRS = (
    ('Ar', 'Ar'),
    ('Ar+', 'Ar_plus'),
    ('e', 'e'),
    ('T_e', 'T_e'),
    ('T_n', 'T_n'),
    ('p', 'p'),
    ('P', 'P'),
)

PRECISION_TIMESERIES_COLUMNS = (
    'time_s',
    *(
        column
        for _filename, _title, panels in PRECISION_TIMESERIES_FIGURES
        for _ylabel, pygmol, local, _log_y in panels
        for column in (pygmol, local)
    ),
)
LOG_TIMESERIES_COLUMNS = tuple(
    column
    for _filename, _title, panels in PRECISION_TIMESERIES_FIGURES
    for _ylabel, pygmol, local, log_y in panels
    if log_y
    for column in (pygmol, local)
)
FINAL_ERROR_KEYS = tuple(f'final_{key}_relative_error' for _label, key in ERROR_METRIC_PAIRS)
WAVEFORM_ERROR_KEYS = tuple(f'{key}_waveform_nrmse' for _label, key in ERROR_METRIC_PAIRS)
SUMMARY_METRIC_KEYS = ('max_final_relative_error', 'max_waveform_nrmse')
REQUIRED_METRIC_KEYS = (*FINAL_ERROR_KEYS, *WAVEFORM_ERROR_KEYS, *SUMMARY_METRIC_KEYS)


def _finite_float(value: Any, default: float = float('nan')) -> float:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return default
    return candidate if np.isfinite(candidate) else default


def _require_columns(path: Path, fieldnames: tuple[str, ...], required: tuple[str, ...]) -> None:
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise ValueError(f'{path} is missing required CSV columns: {", ".join(missing)}')


def _read_rows(path: Path, numeric_columns: tuple[str, ...]) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f'CSV input not found: {path}')
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        if not fieldnames:
            raise ValueError(f'{path} has no CSV header')
        columns = tuple(dict.fromkeys(numeric_columns))
        _require_columns(path, fieldnames, columns)

        rows: list[dict[str, Any]] = []
        for row_index, row in enumerate(reader, start=2):
            parsed = dict(row)
            for column in columns:
                value = _finite_float(parsed.get(column))
                if not np.isfinite(value):
                    raise ValueError(f'{path}: row {row_index} column {column!r} must be a finite number')
                parsed[column] = value
            rows.append(parsed)

    if not rows:
        raise ValueError(f'{path} has no data rows')
    return rows


def _read_metrics(path: Path, required_keys: tuple[str, ...]) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f'Metrics CSV input not found: {path}')
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        _require_columns(path, tuple(reader.fieldnames or ()), ('metric_id', 'value'))
        metrics: dict[str, float] = {}
        for row_index, row in enumerate(reader, start=2):
            key = row.get('metric_id')
            if not key:
                continue
            value = _finite_float(row.get('value'))
            if not np.isfinite(value):
                raise ValueError(f'{path}: row {row_index} metric {key!r} must have a finite numeric value')
            metrics[key] = value

    missing = [key for key in required_keys if key not in metrics]
    if missing:
        raise ValueError(f'{path} is missing required metrics: {", ".join(missing)}')
    return metrics


def _require_positive(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    for row_index, row in enumerate(rows, start=2):
        for column in columns:
            if float(row[column]) <= 0.0:
                raise ValueError(f'{path}: row {row_index} column {column!r} must be positive for log-scale plotting')


def _column(rows: list[dict[str, Any]], name: str) -> np.ndarray:
    return np.asarray([float(row[name]) for row in rows], dtype=float)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return path


def _plot_sanity_density_ratio(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    x = np.arange(len(rows))
    ratios = _column(rows, SANITY_RATIO)
    labels = [str(row.get('step_id') or f'row-{i + 1}') for i, row in enumerate(rows)]

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    bars = ax.bar(x, ratios, color=BLUE, width=0.58, label='PyGMol compact / this code production')
    ax.axhline(1.0, color='#222222', lw=1.2, label='1:1 reference')
    ax.axhspan(0.1, 10.0, color='#59a14f', alpha=0.10, label='sanity band, not precision')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
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


def _plot_timeseries(rows: list[dict[str, Any]], output_dir: Path, filename: str, title: str, panels: tuple[TimePanel, ...]) -> Path:
    time_ms = _column(rows, 'time_s') * 1.0e3
    fig, axes = plt.subplots(len(panels), 1, figsize=(8.4, 3.0 * len(panels)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (ylabel, pygmol_column, local_column, log_y) in zip(axes, panels):
        pygmol = _column(rows, pygmol_column)
        local = _column(rows, local_column)
        ax.plot(time_ms, pygmol, color=BLUE, lw=1.8, label='PyGMol')
        ax.plot(time_ms, local, color=ORANGE, lw=1.5, ls='--', label='Local same-footing')
        if log_y:
            ax.set_yscale('log')
        ax.set_ylabel(ylabel)
        ax.grid(True, which='both' if log_y else 'major', alpha=GRID_ALPHA)
    axes[0].set_title(title)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel('Time [ms]')
    return _save(fig, output_dir / filename)


def _plot_precision_errors(metrics: dict[str, float], output_dir: Path) -> Path:
    labels = [label for label, _key in ERROR_METRIC_PAIRS]
    final = np.asarray([max(metrics[key], MIN_LOG_VALUE) for key in FINAL_ERROR_KEYS], dtype=float)
    waveform = np.asarray([max(metrics[key], MIN_LOG_VALUE) for key in WAVEFORM_ERROR_KEYS], dtype=float)

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.bar(x - width / 2, final, width, label='Final relative error', color=ORANGE)
    ax.bar(x + width / 2, waveform, width, label='Waveform NRMSE', color=BLUE)
    ax.axhline(PRECISION_GATE, color='#222222', lw=1.2, ls=':', label='precision gate 1e-3')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Error metric')
    ax.set_title('PyGMol-Precision-1: Same-Footing Error Metrics')
    ax.grid(True, axis='y', which='both', alpha=GRID_ALPHA)
    ax.legend(fontsize=8)
    return _save(fig, output_dir / 'pygmol_precision_error_metrics.png')


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _write_summary(output_dir: Path, figure_paths: list[Path], metrics: dict[str, float]) -> Path:
    summary_path = output_dir / 'pygmol_graph_summary.md'
    lines = [
        '# PyGMol Benchmark Graphs',
        '',
        f"- Max final relative error: `{metrics['max_final_relative_error']:.3g}`.",
        f"- Max waveform NRMSE: `{metrics['max_waveform_nrmse']:.3g}`.",
        '',
        '## Figures',
        '',
        *(f'- `{_relative_path(path)}`' for path in figure_paths),
        '',
    ]
    summary_path.write_text('\n'.join(lines), encoding='utf-8')
    return summary_path


def _plot_figures(sanity_rows: list[dict[str, Any]], precision_rows: list[dict[str, Any]], metrics: dict[str, float], output_dir: Path) -> list[Path]:
    return [
        _plot_sanity_density_ratio(sanity_rows, output_dir),
        *(
            _plot_timeseries(precision_rows, output_dir, filename, title, panels)
            for filename, title, panels in PRECISION_TIMESERIES_FIGURES
        ),
        _plot_precision_errors(metrics, output_dir),
    ]


def build_pygmol_graphs(
    *,
    sanity_summary: Path,
    precision_timeseries: Path,
    precision_metrics: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sanity_rows = _read_rows(sanity_summary, POWERED_STEP_COLUMNS)
    _require_positive(sanity_summary, sanity_rows, POWERED_STEP_COLUMNS)
    precision_rows = _read_rows(precision_timeseries, PRECISION_TIMESERIES_COLUMNS)
    _require_positive(precision_timeseries, precision_rows, LOG_TIMESERIES_COLUMNS)
    metrics = _read_metrics(precision_metrics, REQUIRED_METRIC_KEYS)

    figure_paths = _plot_figures(sanity_rows, precision_rows, metrics, output_dir)
    return [*figure_paths, _write_summary(output_dir, figure_paths, metrics)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Create PyGMol benchmark figures for sanity and precision comparisons.')
    parser.add_argument('--sanity-summary', type=Path, required=True)
    parser.add_argument('--precision-timeseries', type=Path, required=True)
    parser.add_argument('--precision-metrics', type=Path, required=True)
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
