"""Segmented BDF integration with model-owned domains and valid partial results."""

from __future__ import annotations

import numpy as np
from scipy.integrate import BDF

from plasma_global.core._solver_domain import canonicalize_result_state
from plasma_global.core._solver_result import (
    _IntegrationHistory,
    _result_metadata,
    _SegmentProgress,
)
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
    """Continue Newton probes without changing the integrator state."""

    def __init__(self, physical_rhs: BoundSegmentRHS, scale: np.ndarray) -> None:
        self.physical_rhs = physical_rhs
        self.scale = scale

    def __call__(self, time_s: float, scaled_state: np.ndarray) -> np.ndarray:
        state = self.physical_rhs.model.continue_solver_trial(
            np.asarray(scaled_state, dtype=float) * self.scale
        )
        return self.physical_rhs(time_s, state) / self.scale


def _prepare_initial_state(
    model: CompiledGlobalModel, initial: InitialState | np.ndarray, atol: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = model.validate_initial_state(initial)
    scale = np.maximum(np.abs(state), 1.0)
    domain_atol = atol * scale
    scale.setflags(write=False)
    domain_atol.setflags(write=False)
    return state, scale, domain_atol


def _canonical_state(
    model: CompiledGlobalModel,
    state: np.ndarray,
    domain_atol: np.ndarray,
    time_s: float,
    segment: RecipeSegment,
) -> np.ndarray:
    values, _, _ = canonicalize_result_state(
        model, state.reshape(1, -1), domain_atol, np.array([time_s]), segment=segment
    )
    return values[0]


def _sample_step(
    model: CompiledGlobalModel,
    solver: BDF,
    segment: RecipeSegment,
    rhs: _ScaledSegmentRHS,
    domain_atol: np.ndarray,
    sample_times: np.ndarray | None,
    next_sample: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if sample_times is None:
        times = np.array([solver.t])
        states = (solver.y * rhs.scale).reshape(1, -1)
    else:
        stop = int(np.searchsorted(sample_times, solver.t, side="right"))
        times = sample_times[next_sample:stop]
        states = (
            np.asarray(solver.dense_output()(times)).T * rhs.scale
            if times.size
            else np.empty((0, model.layout.size))
        )
        next_sample = stop
    canonical = np.empty_like(states)
    for index, time in enumerate(times):
        canonical[index] = _canonical_state(
            model, states[index], domain_atol, float(time), segment
        )
    return times, canonical, next_sample


def _advance_segment(
    model: CompiledGlobalModel,
    segment: RecipeSegment,
    solver: BDF,
    rhs: _ScaledSegmentRHS,
    domain_atol: np.ndarray,
    sample_times: np.ndarray | None,
    saved_point_budget: int,
    progress: _SegmentProgress,
) -> None:
    next_sample = 1
    while solver.status == "running":
        message = solver.step()
        if solver.status == "failed":
            raise IntegrationError(
                f"Integration failed in segment {segment.segment_id!r} "
                f"at t={solver.t:.16g} s: {message}",
                failure_time_s=rhs.physical_rhs.last_time_s,
            )
        accepted = _canonical_state(
            model, solver.y * rhs.scale, domain_atol, float(solver.t), segment
        )
        times, states, next_sample = _sample_step(
            model, solver, segment, rhs, domain_atol, sample_times, next_sample
        )
        # Commit only after the preceding dense output has passed its own checks.
        progress.last_time_s, progress.last_state = float(solver.t), accepted
        if progress.saved_points + times.size > saved_point_budget:
            raise IntegrationError(
                "Solver output exceeded the fixed saved-point limit "
                f"({_MAX_SAVED_POINTS}); specify sample_interval_s or save_at_s",
                failure_time_s=float(solver.t),
            )
        progress.append(times, states)


def _result(
    model: CompiledGlobalModel,
    history: _IntegrationHistory,
    controls: SolverSettings,
    global_times: np.ndarray | None,
    scale: np.ndarray,
    domain_atol: np.ndarray,
    *,
    failure: IntegrationError | None = None,
    segment: RecipeSegment | None = None,
) -> SimulationResult:
    times, state = history.arrays(model.layout.size)
    metadata = _result_metadata(controls, global_times, scale, domain_atol)
    metadata.update(
        requested_end_s=model.segments[-1].end_s,
        reached_end_s=float(times[-1]),
        series_units={
            label: unit
            for block in model.state_blocks
            for label, unit in zip(block.layout.labels, block.layout.units, strict=True)
        },
    )
    if failure is not None:
        metadata.update(
            failure_time_s=failure.failure_time_s,
            failure_segment_id=None if segment is None else segment.segment_id,
        )
    return SimulationResult(
        time_s=times,
        state=state,
        state_labels=model.layout.labels,
        observables={},
        status=SimulationStatus.completed("All recipe segments completed")
        if failure is None
        else SimulationStatus.failed(str(failure)),
        solver_stats=history.solver_stats(),
        metadata=metadata,
    )


def _segment_solver(
    segment: RecipeSegment,
    rhs: _ScaledSegmentRHS,
    current: np.ndarray,
    controls: SolverSettings,
) -> BDF:
    return BDF(
        rhs,
        segment.start_s,
        current / rhs.scale,
        segment.end_s,
        rtol=controls.rtol,
        atol=controls.atol,
        first_step=controls.first_step_s,
        max_step=np.inf if controls.max_step_s is None else controls.max_step_s,
        jac_sparsity=rhs.physical_rhs.jac_sparsity,
    )


def _integration_failure(
    exc: Exception, rhs: _ScaledSegmentRHS, segment: RecipeSegment
) -> IntegrationError:
    if isinstance(exc, IntegrationError):
        failure = exc
    else:
        time = rhs.physical_rhs.last_time_s
        time = segment.start_s if time is None else time
        failure = IntegrationError(
            f"Integration failed in segment {segment.segment_id!r} "
            f"at t={time:.16g} s: {exc}",
            failure_time_s=time,
        )
    if failure.failure_time_s is None:
        last_time = rhs.physical_rhs.last_time_s
        failure.failure_time_s = segment.start_s if last_time is None else last_time
    failure.failure_segment_id = segment.segment_id
    return failure


def solve_compiled_model(
    model: CompiledGlobalModel,
    initial: InitialState | np.ndarray,
    settings: SolverSettings | None = None,
) -> SimulationResult:
    """Check each accepted step and attach only valid history to numerical errors."""

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
            model.bind_segment(segment, domain_atol=domain_atol), scale
        )
        progress = _SegmentProgress(segment.start_s, current.copy())
        solver = None
        try:
            _canonical_state(model, current, domain_atol, segment.start_s, segment)
            solver = _segment_solver(segment, rhs, current, controls)
            _advance_segment(
                model,
                segment,
                solver,
                rhs,
                domain_atol,
                _segment_sample_times(global_times, segment),
                _MAX_SAVED_POINTS - history.saved_points + int(index > 0),
                progress,
            )
        except (
            IntegrationError,
            ModelDomainError,
            CouplingConvergenceError,
            ValueError,
        ) as exc:
            failure = _integration_failure(exc, rhs, segment)
            times, state = progress.arrays(include_last=True)
            history.append(
                segment_index=index,
                solution=solver,
                time_s=times,
                state=state,
                completed=False,
            )
            failure.partial_result = _result(
                model,
                history,
                controls,
                global_times,
                scale,
                domain_atol,
                failure=failure,
                segment=segment,
            )
            if failure is exc:
                raise
            raise failure from exc
        current = solver.y * scale
        times, state = progress.arrays()
        history.append(segment_index=index, solution=solver, time_s=times, state=state)
    return _result(model, history, controls, global_times, scale, domain_atol)


__all__ = ["solve_compiled_model"]
