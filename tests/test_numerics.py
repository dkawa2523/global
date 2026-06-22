from __future__ import annotations

from pathlib import Path

import numpy as np

from plasma_global.numerics.scipy_backend import SciPyBDFIntegrator
from plasma_global.workflows.context import build_case, load_case_from_yaml


ROOT = Path(__file__).resolve().parents[1]
SMOKE_CASE = ROOT / 'examples' / 'configs' / 'case_smoke.yaml'


class DecaySystem:
    def rhs(self, _t: float, y: np.ndarray) -> np.ndarray:
        return -y

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


def test_projection_clips_density_and_energy_to_physical_floor() -> None:
    built = build_case(load_case_from_yaml(SMOKE_CASE))
    system = built.system
    y = system.initial_state()

    gas_slice = system.state_layout.slice('gas_densities')
    y[gas_slice.start] = -5.0
    energy_idx = system.state_layout.electron_energy_index[system.zone_ids[0]]
    y[energy_idx] = 0.0

    projected = system.project_state(y)

    assert projected[gas_slice.start] == 0.0
    assert projected[energy_idx] == system.floor_energy


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
