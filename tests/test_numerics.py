from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from plasma_global.numerics.scipy_backend import SciPyBDFIntegrator
from plasma_global.numerics.solver_base import SolverResult
from plasma_global import build_case, load_case_from_yaml
from plasma_global.workflows.solve import _step_max_step, solve_built_case


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


class WorkflowSystem:
    def initial_state(self) -> np.ndarray:
        return np.array([1.0])

    def project_trajectory(self, y: np.ndarray) -> np.ndarray:
        return y

    def project_state(self, y: np.ndarray) -> np.ndarray:
        return y


class RecordingIntegrator:
    def __init__(self, max_step: float | None) -> None:
        self.max_step = max_step
        self.seen_max_steps: list[float | None] = []

    def solve(self, *, system, y0: np.ndarray, t_span: tuple[float, float], t_eval: np.ndarray | None = None) -> SolverResult:
        self.seen_max_steps.append(self.max_step)
        t = np.asarray(t_eval if t_eval is not None else t_span, dtype=float)
        y = np.repeat(np.asarray(y0, dtype=float).reshape(-1, 1), t.size, axis=1)
        return SolverResult(
            t=t,
            y=y,
            success=True,
            status=0,
            message='ok',
            diagnostics={'nfev': 0, 'njev': 0, 'nlu': 0, 'event_counts': {}},
        )


class SteadyFirstStepIntegrator:
    def __init__(self) -> None:
        self.max_step = None
        self.calls: list[tuple[float, float]] = []

    def solve(self, *, system, y0: np.ndarray, t_span: tuple[float, float], t_eval: np.ndarray | None = None) -> SolverResult:
        self.calls.append(t_span)
        if len(self.calls) == 1:
            t = np.array([t_span[0], t_span[0] + 0.25 * (t_span[1] - t_span[0])])
            diagnostics = {'nfev': 1, 'njev': 0, 'nlu': 0, 'event_counts': {'steady_state': 1}, 'steady_state_event_count': 1}
        else:
            t = np.asarray(t_eval if t_eval is not None else t_span, dtype=float)
            diagnostics = {'nfev': 1, 'njev': 0, 'nlu': 0, 'event_counts': {}, 'steady_state_event_count': 0}
        y = np.repeat(np.asarray(y0, dtype=float).reshape(-1, 1), t.size, axis=1)
        return SolverResult(t=t, y=y, success=True, status=0, message='ok', diagnostics=diagnostics)


def _built_for_steps(steps: list[SimpleNamespace], integrator: RecordingIntegrator) -> SimpleNamespace:
    return SimpleNamespace(
        system=WorkflowSystem(),
        integrator=integrator,
        loaded=SimpleNamespace(recipe=SimpleNamespace(steps=steps)),
    )


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


def test_step_max_step_limits_pulsed_square_period_without_overriding_smaller_config() -> None:
    step = SimpleNamespace(
        power_ports={
            'rf': {
                'waveform': 'pulsed_square',
                'repetition_Hz': 1000.0,
            },
        }
    )

    assert np.isclose(_step_max_step(1.0e-3, step), 5.0e-5)
    assert np.isclose(_step_max_step(1.0e-6, step), 1.0e-6)


def test_solve_built_case_limits_integrator_max_step_for_pulsed_square() -> None:
    integrator = RecordingIntegrator(max_step=1.0e-3)
    step = SimpleNamespace(
        step_id='pulse',
        t_start_s=0.0,
        t_end_s=1.0e-3,
        power_ports={
            'rf': {
                'waveform': 'pulsed_square',
                'repetition_Hz': 1000.0,
            },
        },
    )

    solve_built_case(_built_for_steps([step], integrator))

    assert np.allclose(integrator.seen_max_steps, [5.0e-5])
    assert np.isclose(integrator.max_step, 1.0e-3)


def test_solve_built_case_keeps_integrator_max_step_for_cw_and_off_steps() -> None:
    integrator = RecordingIntegrator(max_step=2.0e-6)
    steps = [
        SimpleNamespace(step_id='cw', t_start_s=0.0, t_end_s=1.0e-6, power_ports={'rf': {'waveform': 'cw'}}),
        SimpleNamespace(step_id='off', t_start_s=1.0e-6, t_end_s=2.0e-6, power_ports={'rf': {'waveform': 'off'}}),
    ]

    solve_built_case(_built_for_steps(steps, integrator))

    assert integrator.seen_max_steps == [2.0e-6, 2.0e-6]
    assert np.isclose(integrator.max_step, 2.0e-6)


def test_solve_built_case_rejects_recipe_step_gaps() -> None:
    steps = [
        SimpleNamespace(step_id='first', t_start_s=0.0, t_end_s=1.0, power_ports={}),
        SimpleNamespace(step_id='second', t_start_s=2.0, t_end_s=3.0, power_ports={}),
    ]

    with pytest.raises(ValueError, match='time gap'):
        solve_built_case(_built_for_steps(steps, RecordingIntegrator(max_step=None)))


def test_steady_state_event_holds_step_end_and_continues_recipe() -> None:
    integrator = SteadyFirstStepIntegrator()
    steps = [
        SimpleNamespace(step_id='first', t_start_s=0.0, t_end_s=1.0, power_ports={}),
        SimpleNamespace(step_id='second', t_start_s=1.0, t_end_s=2.0, power_ports={}),
    ]

    solution = solve_built_case(_built_for_steps(steps, integrator))

    assert integrator.calls == [(0.0, 1.0), (1.0, 2.0)]
    assert 1.0 in solution.t
    assert np.isclose(solution.t[-1], 2.0)
    assert solution.diagnostics['steady_state_event_count'] == 1


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


def test_solver_atol_vector_matches_state_size() -> None:
    built = build_case(load_case_from_yaml(SMOKE_CASE))
    atol = built.system.solver_atol_vector()

    assert atol.shape == (built.state_layout.size,)
    assert np.all(np.isfinite(atol))
    assert np.all(atol >= built.loaded.run_config.numerics.atol)


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
