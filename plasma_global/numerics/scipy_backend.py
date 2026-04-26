from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from plasma_global.numerics.solver_base import SolverResult, TimeIntegrator


@dataclass
class SciPyBDFIntegrator(TimeIntegrator):
    rtol: float = 1.0e-6
    atol: float = 1.0e-12
    first_step: float | None = None
    max_step: float | None = None

    def solve(self, system, y0: np.ndarray, t_span: tuple[float, float], t_eval: np.ndarray | None = None) -> SolverResult:
        options = {
            'fun': system.rhs,
            't_span': t_span,
            'y0': y0,
            'method': 'BDF',
            't_eval': t_eval,
            'jac': system.jacobian,
            'events': system.scipy_events() or None,
            'rtol': self.rtol,
            'atol': self.atol,
        }
        if self.first_step is not None:
            options['first_step'] = self.first_step
        if self.max_step is not None:
            options['max_step'] = self.max_step
        sol = solve_ivp(**options)
        return SolverResult(
            t=sol.t,
            y=sol.y,
            success=sol.success,
            status=sol.status,
            message=sol.message,
            diagnostics={'nfev': sol.nfev, 'njev': sol.njev, 'nlu': getattr(sol, 'nlu', None), 't_events': sol.t_events},
        )
