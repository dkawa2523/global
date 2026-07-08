from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from plasma_global.numerics.solver_base import SolverResult, SolverSystem, TimeIntegrator


@dataclass
class SciPyBDFIntegrator(TimeIntegrator):
    rtol: float = 1.0e-6
    atol: float = 1.0e-12
    first_step: float | None = None
    max_step: float | None = None

    def _solver_atol(self, system: SolverSystem, y0: np.ndarray) -> float | np.ndarray:
        solver_atol_vector = getattr(system, 'solver_atol_vector', None)
        if not callable(solver_atol_vector):
            return self.atol
        atol = np.asarray(solver_atol_vector(), dtype=float)
        if atol.shape != np.asarray(y0).shape:
            raise ValueError(f'solver_atol_vector shape {atol.shape} does not match initial state shape {np.asarray(y0).shape}')
        return atol

    def solve(self, system: SolverSystem, y0: np.ndarray, t_span: tuple[float, float], t_eval: np.ndarray | None = None) -> SolverResult:
        events = system.scipy_events() or None
        event_names = []
        if events is not None:
            event_names = [str(getattr(event, 'event_name', f'event_{i}')) for i, event in enumerate(events)]
        options = {
            'fun': system.rhs,
            't_span': t_span,
            'y0': y0,
            'method': 'BDF',
            't_eval': t_eval,
            'events': events,
            'rtol': self.rtol,
            'atol': self._solver_atol(system, y0),
        }
        if self.first_step is not None:
            options['first_step'] = self.first_step
        if self.max_step is not None:
            options['max_step'] = self.max_step
        sol = solve_ivp(**options)
        t_events = getattr(sol, 't_events', None)
        if t_events is None:
            t_events = []
        y_events = getattr(sol, 'y_events', None)
        if y_events is None:
            y_events = []
        event_counts = {name: int(len(times)) for name, times in zip(event_names, t_events)}
        solver_event_count = int(sum(event_counts.values()))
        result_t = sol.t
        result_y = sol.y
        if sol.status == 1 and y_events:
            event_records = []
            for name, times, states in zip(event_names, t_events, y_events):
                for time_s, state in zip(times, states):
                    event_records.append((float(time_s), np.asarray(state, dtype=float), name))
            if event_records:
                event_time, event_state, _event_name = sorted(event_records, key=lambda item: item[0])[-1]
                already_present = bool(
                    result_t.size
                    and abs(float(result_t[-1]) - event_time) <= max(1.0e-15, 1.0e-12 * max(1.0, abs(event_time)))
                )
                if not already_present:
                    result_t = np.concatenate([result_t, np.array([event_time])])
                    result_y = np.concatenate([result_y, event_state.reshape(-1, 1)], axis=1)
        return SolverResult(
            t=result_t,
            y=result_y,
            success=sol.success,
            status=sol.status,
            message=sol.message,
            diagnostics={
                'nfev': sol.nfev,
                'njev': sol.njev,
                'nlu': getattr(sol, 'nlu', None),
                't_events': t_events,
                'event_names': event_names,
                'event_counts': event_counts,
                'solver_event_count': solver_event_count,
                'steady_state_event_count': int(event_counts.get('steady_state', 0)),
            },
        )
