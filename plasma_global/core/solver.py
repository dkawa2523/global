"""Segmented BDF integration with independent state validation and sampling."""

from __future__ import annotations

import numpy as np
from scipy.integrate import BDF

from plasma_global.core._solver_domain import canonicalize_result_state
from plasma_global.core._solver_result import _IntegrationHistory, _result_metadata
from plasma_global.core._solver_sampling import (
    _MAX_SAVED_POINTS,
    _global_sample_times,
    _segment_sample_times,
)
from plasma_global.core.compiled import BoundSegmentRHS, CompiledGlobalModel
from plasma_global.core.domain import InitialState, RecipeSegment, SolverSettings
from plasma_global.core.result import SimulationResult, SimulationStatus
from plasma_global.errors import (
    CouplingConvergenceError,
    IntegrationError,
    ModelConfigurationError,
    ModelDomainError,
)


class _ScaledSegmentRHS:
    """Continue Newton probes at physical bounds without changing BDF state."""

    def __init__(
        self,
        physical_rhs: BoundSegmentRHS,
        scale: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> None:
        self.physical_rhs = physical_rhs
        self.scale = scale
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.surface_model = physical_rhs.model.surface_model
        self.surface_coverage_slice = physical_rhs.model.layout.surface_coverage_slice

    def __call__(self, time_s: float, scaled_state: np.ndarray) -> np.ndarray:
        state = np.asarray(scaled_state, dtype=float) * self.scale
        np.clip(state, self.lower_bounds, self.upper_bounds, out=state)
        if self.surface_model is not None:
            self.surface_model._continue_solver_coverages(
                state[self.surface_coverage_slice]
            )
        return self.physical_rhs(time_s, state) / self.scale


def _prepare_initial_state(
    model: CompiledGlobalModel,
    initial: InitialState | np.ndarray,
    atol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = (
        model.initial_state(initial)
        if isinstance(initial, InitialState)
        else np.array(initial, dtype=float, copy=True)
    )
    if state.shape != (model.layout.size,):
        raise ModelConfigurationError(
            f"Initial state has shape {state.shape}, expected {(model.layout.size,)}"
        )
    if not np.all(np.isfinite(state)):
        raise ModelConfigurationError("Initial state must contain only finite values")
    for name, invalid in (
        ("lower", state < model.state_lower_bounds),
        ("upper", state > model.state_upper_bounds),
    ):
        if np.any(invalid):
            index = int(np.flatnonzero(invalid)[0])
            raise ModelConfigurationError(
                f"Initial state violates its compiled {name} bound: "
                f"state[{index}]={state[index]:.6e}"
            )
    if model.surface_model is not None:
        coverage = state[model.layout.surface_coverage_slice]
        try:
            model.surface_model._validate_solver_coverages(
                coverage, np.zeros(coverage.size)
            )
        except ModelDomainError as exc:
            raise ModelConfigurationError(
                f"Initial state violates the surface coverage domain: {exc}"
            ) from exc
    scale = np.maximum(np.abs(state), 1.0)
    domain_atol = atol * scale
    scale.setflags(write=False)
    domain_atol.setflags(write=False)
    return state, scale, domain_atol


def _integrate_segment(
    model: CompiledGlobalModel,
    segment: RecipeSegment,
    rhs: _ScaledSegmentRHS,
    initial: np.ndarray,
    controls: SolverSettings,
    domain_atol: np.ndarray,
    sample_times: np.ndarray | None,
    saved_point_budget: int,
) -> tuple[BDF, np.ndarray, np.ndarray]:
    solver = BDF(
        rhs,
        segment.start_s,
        initial / rhs.scale,
        segment.end_s,
        rtol=controls.rtol,
        atol=controls.atol,
        first_step=controls.first_step_s,
        max_step=(np.inf if controls.max_step_s is None else controls.max_step_s),
        jac_sparsity=rhs.physical_rhs.jac_sparsity,
    )
    times = [np.array([segment.start_s])]
    states = [initial.reshape(1, -1).copy()]
    next_sample = 1
    saved_points = 1
    while solver.status == "running":
        message = solver.step()
        if solver.status == "failed":
            raise IntegrationError(
                f"Integration failed in segment {segment.segment_id!r} "
                f"at t={solver.t:.16g} s: {message}"
            )
        # Validate every accepted step, even when no output is requested here.
        canonicalize_result_state(
            model,
            (solver.y * rhs.scale).reshape(1, -1),
            domain_atol,
            np.array([solver.t]),
            segment=segment,
        )
        if sample_times is None:
            step_times = np.array([solver.t])
            step_states = (solver.y * rhs.scale).reshape(1, -1)
        else:
            stop = int(np.searchsorted(sample_times, solver.t, side="right"))
            step_times = sample_times[next_sample:stop]
            if not step_times.size:
                continue
            step_states = np.asarray(solver.dense_output()(step_times)).T * rhs.scale
            next_sample = stop
        saved_points += step_times.size
        if saved_points > saved_point_budget:
            raise IntegrationError(
                "Solver output exceeded the fixed saved-point limit "
                f"({_MAX_SAVED_POINTS}); "
                "specify sample_interval_s or save_at_s"
            )
        times.append(step_times)
        states.append(step_states)
    return solver, np.concatenate(times), np.concatenate(states, axis=0)


def solve_compiled_model(
    model: CompiledGlobalModel,
    initial: InitialState | np.ndarray,
    settings: SolverSettings | None = None,
) -> SimulationResult:
    """Validate each accepted BDF step and retain only selected output points."""

    controls = settings or SolverSettings()
    if controls.first_step_s is not None and any(
        controls.first_step_s > segment.end_s - segment.start_s
        for segment in model.segments
    ):
        raise ModelConfigurationError(
            "first_step_s must not exceed any recipe segment duration"
        )
    global_times = _global_sample_times(model, controls)
    current, scale, domain_atol = _prepare_initial_state(model, initial, controls.atol)
    history = _IntegrationHistory()
    for index, segment in enumerate(model.segments):
        rhs = _ScaledSegmentRHS(
            model.bind_segment(segment, domain_atol=domain_atol),
            scale,
            model.state_lower_bounds,
            model.state_upper_bounds,
        )
        try:
            solver, times, state = _integrate_segment(
                model,
                segment,
                rhs,
                current,
                controls,
                domain_atol,
                _segment_sample_times(global_times, segment),
                _MAX_SAVED_POINTS - history.saved_points + int(index > 0),
            )
        except (ModelDomainError, CouplingConvergenceError, ValueError) as exc:
            time = rhs.physical_rhs.last_time_s
            time = segment.start_s if time is None else time
            raise IntegrationError(
                f"Integration failed in segment {segment.segment_id!r} "
                f"at t={time:.16g} s: {exc}"
            ) from exc
        current = solver.y * scale
        history.append(segment_index=index, solution=solver, time_s=times, state=state)
    time_s, raw_state = history.arrays(model.layout.size)
    state, _, _ = canonicalize_result_state(model, raw_state, domain_atol, time_s)
    return SimulationResult(
        time_s=time_s,
        state=state,
        state_labels=model.layout.labels,
        observables={},
        status=SimulationStatus.completed("All recipe segments completed"),
        solver_stats=history.solver_stats(),
        metadata=_result_metadata(controls, global_times, scale, domain_atol),
    )


__all__ = ["solve_compiled_model"]
