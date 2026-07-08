from __future__ import annotations

from pathlib import Path

import pytest

from tools.external_benchmarks.plot_pygmol_benchmarks import REQUIRED_METRIC_KEYS, build_pygmol_graphs

pytestmark = pytest.mark.external_benchmark


def _write_pygmol_inputs(tmp_path: Path, *, local_e: str = '1.0e15', metric_keys: tuple[str, ...] = REQUIRED_METRIC_KEYS) -> tuple[Path, Path, Path]:
    sanity = tmp_path / 'sanity.csv'
    sanity.write_text(
        'step_id,local_mean_absorbed_power_W,ratio_final_ne_pygmol_over_local\n'
        'powered,10.0,1.2\n',
        encoding='utf-8',
    )

    timeseries = tmp_path / 'timeseries.csv'
    timeseries.write_text(
        'time_s,pygmol_e,local_e,pygmol_Ar_plus,local_Ar_plus,pygmol_T_e,local_T_e,pygmol_P,local_P\n'
        f'0.0,1.0e15,{local_e},1.0e15,1.0e15,3.0,3.0,10.0,10.0\n',
        encoding='utf-8',
    )

    metrics = tmp_path / 'metrics.csv'
    rows = ['metric_id,value', *(f'{key},1.0e-6' for key in metric_keys)]
    metrics.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    return sanity, timeseries, metrics


def test_pygmol_plotter_rejects_missing_required_metric(tmp_path: Path) -> None:
    sanity, timeseries, metrics = _write_pygmol_inputs(tmp_path, metric_keys=REQUIRED_METRIC_KEYS[:-1])

    with pytest.raises(ValueError, match='missing required metrics'):
        build_pygmol_graphs(
            sanity_summary=sanity,
            precision_timeseries=timeseries,
            precision_metrics=metrics,
            output_dir=tmp_path / 'figures',
        )


def test_pygmol_plotter_rejects_nonpositive_log_series(tmp_path: Path) -> None:
    sanity, timeseries, metrics = _write_pygmol_inputs(tmp_path, local_e='0.0')

    with pytest.raises(ValueError, match='must be positive for log-scale plotting'):
        build_pygmol_graphs(
            sanity_summary=sanity,
            precision_timeseries=timeseries,
            precision_metrics=metrics,
            output_dir=tmp_path / 'figures',
        )
