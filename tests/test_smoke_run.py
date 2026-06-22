from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plasma_global.io.hdf5_writer import write_solution_h5
from plasma_global.numerics.solver_base import SolverResult
from plasma_global.observables.defaults import summarize_solution
from plasma_global.workflows.runner import run_from_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_smoke_case_runs_end_to_end() -> None:
    result = run_from_yaml(ROOT / 'examples' / 'configs' / 'case_smoke.yaml')
    summary = result['summary']
    final_obs = result['observables'][-1]
    assert summary['success'] is True
    assert summary['n_times'] >= 5
    assert summary['final_total_absorbed_power_W'] > 0.0
    assert summary['final_mean_electron_energy_eV'] > 0.0
    assert final_obs['total_absorbed_power_W'] > 0.0
    assert not any(key.startswith('final_port_') for key in summary)
    assert 'step_summary' not in summary
    assert 'projection_density_clip_count' not in summary
    assert 'final_rhs_norm_inf' not in summary
    assert summary['solver_event_count'] == 0
    assert summary['steady_state_event_count'] == 0
    assert summary['chemistry_provenance']['cross_sections_with_provenance'] >= 1
    assert result['chemistry_provenance']['cross_sections_with_provenance'] >= 1
    assert 'observables' not in result['solution'].diagnostics
    assert 'chemistry_provenance' not in result['solution'].diagnostics
    assert Path(result['output_dir'], 'effective_case.yaml').exists()
    assert Path(result['output_dir'], 'resolved_paths.yaml').exists()


def test_summary_only_promotes_allowlisted_final_observables() -> None:
    solution = SolverResult(
        t=np.array([0.0, 1.0]),
        y=np.zeros((1, 2)),
        success=True,
        status=0,
        message='ok',
        diagnostics={},
    )
    observables = [
        {
            'total_absorbed_power_W': 12.0,
            'ne_source_m3': 1.0e16,
            'port_hidden_detail': 99.0,
            'experimental_numeric_probe': 42.0,
        }
    ]

    summary = summarize_solution(solution, observables)

    assert summary['final_total_absorbed_power_W'] == 12.0
    assert summary['final_ne_source_m3'] == 1.0e16
    assert 'final_port_hidden_detail' not in summary
    assert 'final_experimental_numeric_probe' not in summary


def test_solution_h5_keeps_fixed_core_datasets(tmp_path: Path) -> None:
    h5py = pytest.importorskip('h5py')
    solution = SolverResult(
        t=np.array([0.0, 1.0]),
        y=np.array([[1.0, 2.0], [3.0, 4.0]]),
        success=True,
        status=0,
        message='ok',
        diagnostics={'nfev': 7},
    )
    path = tmp_path / 'solution.h5'

    write_solution_h5(path, solution, state_labels=['a', 'b'])

    with h5py.File(path, 'r') as h5:
        assert set(h5.keys()) == {'t_s', 'y', 'state_labels'}
