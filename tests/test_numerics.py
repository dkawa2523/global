from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from plasma_global.numerics.jacobian_check import check_jacobian
from plasma_global.numerics.scipy_backend import SciPyBDFIntegrator
from plasma_global.workflows.context import build_case, load_case_from_yaml


ROOT = Path(__file__).resolve().parents[1]
SMOKE_CASE = ROOT / 'examples' / 'configs' / 'case_smoke.yaml'


class DecaySystem:
    def rhs(self, _t: float, y: np.ndarray) -> np.ndarray:
        return -y

    def jacobian(self, _t: float, y: np.ndarray) -> csr_matrix:
        return csr_matrix([[-1.0]])

    def scipy_events(self) -> list:
        return []

    def state_labels(self) -> list[str]:
        return ['y']


class DecayEventSystem(DecaySystem):
    def scipy_events(self) -> list:
        def half_value(_t: float, y: np.ndarray) -> float:
            return float(y[0] - 0.5)

        half_value.terminal = True
        half_value.direction = -1.0
        half_value.event_name = 'half_value'
        return [half_value]


def test_scipy_bdf_allows_default_max_step() -> None:
    result = SciPyBDFIntegrator(max_step=None, first_step=None).solve(
        system=DecaySystem(),
        y0=np.array([1.0]),
        t_span=(0.0, 1.0e-6),
        t_eval=np.array([0.0, 1.0e-6]),
    )
    assert result.success is True
    assert result.y.shape == (1, 2)
    assert result.diagnostics['solver_event_count'] == 0


def test_scipy_bdf_reports_named_event_counts() -> None:
    result = SciPyBDFIntegrator(max_step=None, first_step=None).solve(
        system=DecayEventSystem(),
        y0=np.array([1.0]),
        t_span=(0.0, 2.0),
        t_eval=np.linspace(0.0, 2.0, 5),
    )

    assert result.success is True
    assert result.status == 1
    assert result.diagnostics['event_counts']['half_value'] == 1
    assert result.diagnostics['solver_event_count'] == 1
    assert np.isclose(result.t[-1], np.log(2.0), rtol=1.0e-4)


def test_jacobian_check_reports_small_error_for_decay_system() -> None:
    result = check_jacobian(DecaySystem(), 0.0, np.array([1.0]), top_n=1)
    assert result.max_relative_error < 1.0e-8
    assert result.mismatches[0].row_label == 'y'
    assert result.mismatches[0].row_group == 'other'
    assert result.mismatches[0].column_group == 'other'
    assert result.group_summaries[0].group == 'other'
    assert result.group_summaries[0].checked_entry_count == 1


def test_jacobian_check_groups_smoke_case_by_state_layout() -> None:
    built = build_case(load_case_from_yaml(SMOKE_CASE))
    system = built.system
    y0 = system.initial_state()

    result = check_jacobian(system, 0.0, y0, top_n=3)

    expected_groups = {
        name
        for name, state_slice in system.state_layout.slices.items()
        if state_slice.stop > state_slice.start
    }
    actual_groups = {summary.group for summary in result.group_summaries}

    assert expected_groups <= actual_groups
    for summary in result.group_summaries:
        assert summary.checked_entry_count > 0
        assert summary.row_label
        assert summary.column_label
        assert summary.column_group
        assert summary.max_relative_error >= 0.0
        assert np.isfinite(summary.finite_difference)
        assert np.isfinite(summary.jacobian_value)


def test_projection_diagnostics_count_density_and_energy_clips() -> None:
    built = build_case(load_case_from_yaml(SMOKE_CASE))
    system = built.system
    system.reset_numerical_diagnostics()
    y = system.initial_state()
    system.reset_numerical_diagnostics()

    gas_slice = system.state_layout.slice('gas_densities')
    y[gas_slice.start] = -5.0
    energy_idx = system.state_layout.electron_energy_index[system.zone_ids[0]]
    y[energy_idx] = 0.0

    projected = system.project_state(y, count_diagnostics=True)

    assert projected[gas_slice.start] == 0.0
    assert projected[energy_idx] == system.floor_energy
    assert system.diagnostics['projection_density_clip_count'] >= 1
    assert system.diagnostics['projection_electron_energy_clip_count'] >= 1
    assert system.diagnostics['projection_max_abs_delta'] >= 5.0


def test_steady_state_event_is_opt_in() -> None:
    built = build_case(load_case_from_yaml(SMOKE_CASE))
    system = built.system

    assert system.scipy_events() == []

    system.run_config.numerics.events = {
        'steady_state': {
            'enabled': True,
            'relative_rhs_norm_s_inv': 1.0e30,
            'min_step_time_s': 0.0,
        }
    }
    events = system.scipy_events()
    y0 = system.initial_state()

    assert len(events) == 1
    assert getattr(events[0], 'event_name') == 'steady_state'
    assert getattr(events[0], 'terminal') is True
    assert getattr(events[0], 'direction') == -1.0
    assert np.isfinite(events[0](system.recipe.steps[0].t_start_s, y0))
