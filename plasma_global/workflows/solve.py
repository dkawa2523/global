"""Recipe-step time integration.

The solver runs one recipe step at a time. Between steps, projected final state
is passed forward so piecewise recipes can change gas feeds, power commands, or
surface conditions without hiding that transition inside the integrator.
"""

from __future__ import annotations

import numpy as np

from plasma_global.numerics.solver_base import SolverResult, concatenate_solver_results
from plasma_global.reactor.models import RecipeStep
from plasma_global.workflows.types import BuiltCase


def _ensure_contiguous_recipe_steps(steps: list[RecipeStep]) -> None:
    if not steps:
        raise ValueError('Recipe must define at least one step.')
    previous_end = None
    for step in steps:
        start = float(step.t_start_s)
        end = float(step.t_end_s)
        if end <= start:
            raise ValueError(f'Recipe step {step.step_id!r} has t_end_s <= t_start_s.')
        if previous_end is not None:
            tolerance = max(1.0e-30, 1.0e-12 * max(abs(start), abs(previous_end), 1.0))
            if start < previous_end - tolerance:
                raise ValueError(f'Recipe step {step.step_id!r} overlaps the previous step.')
            if start > previous_end + tolerance:
                raise ValueError(f'Recipe step {step.step_id!r} leaves a time gap after the previous step.')
        previous_end = end


def _step_max_step(configured_max_step: float | None, step: RecipeStep) -> float | None:
    max_step = None if configured_max_step is None else float(configured_max_step)
    power_ports = getattr(step, 'power_ports', {}) or {}
    port_configs = power_ports.values() if hasattr(power_ports, 'values') else []
    for cfg in port_configs:
        if not isinstance(cfg, dict):
            continue
        if str(cfg.get('waveform', 'cw')).lower() != 'pulsed_square':
            continue
        repetition_hz = float(cfg.get('repetition_Hz', 0.0) or 0.0)
        if repetition_hz <= 0.0:
            continue
        pulse_max_step = 1.0 / repetition_hz / 20.0
        max_step = pulse_max_step if max_step is None else min(max_step, pulse_max_step)
    return max_step


def _hold_steady_segment_to_step_end(seg: SolverResult, step: RecipeStep) -> SolverResult:
    if int(seg.diagnostics.get('steady_state_event_count', 0) or 0) <= 0:
        return seg
    step_end = float(step.t_end_s)
    if not seg.t.size or float(seg.t[-1]) >= step_end:
        return seg
    seg.t = np.concatenate([seg.t, np.array([step_end])])
    seg.y = np.concatenate([seg.y, seg.y[:, -1:].copy()], axis=1)
    return seg


def solve_built_case(built: BuiltCase) -> SolverResult:
    system = built.system
    recipe = built.loaded.recipe
    _ensure_contiguous_recipe_steps(recipe.steps)
    y0 = system.initial_state()

    results: list[SolverResult] = []
    t_start = recipe.steps[0].t_start_s
    t_end = recipe.steps[-1].t_end_s
    total_duration = max(t_end - t_start, 1.0e-12)
    for step in recipe.steps:
        dt = max(step.t_end_s - step.t_start_s, 1.0e-12)
        n_eval = max(5, int(200 * dt / total_duration))
        t_eval = np.linspace(step.t_start_s, step.t_end_s, n_eval)
        original_max_step = getattr(built.integrator, 'max_step', None)
        if hasattr(built.integrator, 'max_step'):
            built.integrator.max_step = _step_max_step(original_max_step, step)
        try:
            seg = built.integrator.solve(system=system, y0=y0, t_span=(step.t_start_s, step.t_end_s), t_eval=t_eval)
        finally:
            if hasattr(built.integrator, 'max_step'):
                built.integrator.max_step = original_max_step
        if not seg.success:
            step_id = getattr(step, 'step_id', f'step_{len(results)}')
            raise RuntimeError(f"Solver failed in recipe step {step_id!r}: {seg.message}")
        seg = _hold_steady_segment_to_step_end(seg, step)
        seg.y = system.project_trajectory(seg.y)
        results.append(seg)
        y0 = system.project_state(seg.y[:, -1].copy())

    solution = concatenate_solver_results(results)
    solution.y = system.project_trajectory(solution.y)
    return solution
