from __future__ import annotations

from pathlib import Path

from plasma_global.workflows.runner import run_from_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_smoke_case_runs_end_to_end() -> None:
    result = run_from_yaml(ROOT / 'examples' / 'configs' / 'case_smoke.yaml')
    summary = result['summary']
    assert summary['success'] is True
    assert summary['n_times'] >= 5
    assert summary['final_port_source_rf_absorbed_power_W'] > 0.0
    assert summary['final_port_source_rf_frequency_Hz'] == 13.56e6
    assert 'final_port_source_rf_E_over_H_mode_index' not in summary
    assert 'mean_port_wafer_bias_self_bias_V' in summary['step_summary']['ignition']
    assert Path(result['output_dir'], 'effective_case.yaml').exists()
    assert Path(result['output_dir'], 'resolved_paths.yaml').exists()
