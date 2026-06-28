from __future__ import annotations

from typing import Any

import numpy as np

from plasma_global.numerics.solver_base import SolverResult, concatenate_solver_results


def solve_built_case(built: Any) -> SolverResult:
    system = built.system
    recipe = built.loaded.recipe
    y0 = system.initial_state()

    results: list[SolverResult] = []
    t_start = recipe.steps[0].t_start_s
    t_end = recipe.steps[-1].t_end_s
    total_duration = max(t_end - t_start, 1.0e-12)
    for step in recipe.steps:
        dt = max(step.t_end_s - step.t_start_s, 1.0e-12)
        n_eval = max(5, int(200 * dt / total_duration))
        t_eval = np.linspace(step.t_start_s, step.t_end_s, n_eval)
        seg = built.integrator.solve(system=system, y0=y0, t_span=(step.t_start_s, step.t_end_s), t_eval=t_eval)
        if not seg.success:
            step_id = getattr(step, 'step_id', f'step_{len(results)}')
            raise RuntimeError(f"Solver failed in recipe step {step_id!r}: {seg.message}")
        seg.y = system.project_trajectory(seg.y)
        results.append(seg)
        y0 = system.project_state(seg.y[:, -1].copy())
        if int(seg.diagnostics.get('steady_state_event_count', 0) or 0) > 0:
            break

    solution = concatenate_solver_results(results)
    solution.y = system.project_trajectory(solution.y)
    return solution
