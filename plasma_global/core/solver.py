"""Sequential stiff integration for :class:`CompiledGlobalModel`."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.integrate import solve_ivp

from plasma_global.core.compiled import BoundSegmentRHS, CompiledGlobalModel
from plasma_global.core.domain import (
    InitialState,
    RecipeSegment,
    SolverSettings,
)
from plasma_global.core.exceptions import ModelConfigurationError
from plasma_global.errors import (
    CouplingConvergenceError,
    IntegrationError,
    ModelDomainError,
)

if TYPE_CHECKING:
    from plasma_global.core.result import SimulationResult


_MAX_SAVED_POINTS = 100_000
_EXPERIMENTAL_QUASI_STEADY_MODEL_ID = "experimental.stop_when_quasi_steady"


def _time_tolerance(start_s: float, end_s: float) -> float:
    scale = max(abs(start_s), abs(end_s), abs(end_s - start_s), 1.0)
    return 64.0 * np.finfo(float).eps * scale


def _merge_with_boundaries(requested: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Merge global sample times while giving exact boundaries precedence."""

    tolerance = _time_tolerance(float(boundaries[0]), float(boundaries[-1]))
    positions = np.searchsorted(boundaries, requested)
    left = boundaries[np.maximum(positions - 1, 0)]
    right = boundaries[np.minimum(positions, boundaries.size - 1)]
    near_boundary = np.minimum(np.abs(requested - left), np.abs(requested - right)) <= (
        tolerance
    )
    values = np.sort(np.concatenate((requested[~near_boundary], boundaries)))
    merged: list[float] = []
    for value in values:
        item = float(value)
        if not merged or item - merged[-1] > tolerance:
            merged.append(item)
    return np.asarray(merged, dtype=float)


def _global_sample_times(
    model: CompiledGlobalModel, controls: SolverSettings
) -> np.ndarray | None:
    """Build a recipe-global output grid, or select native solver points."""

    if controls.sample_interval_s is None and controls.save_at_s is None:
        return None
    start_s = float(model.segments[0].start_s)
    end_s = float(model.segments[-1].end_s)
    tolerance = _time_tolerance(start_s, end_s)
    boundaries = np.asarray(
        [model.segments[0].start_s, *(item.end_s for item in model.segments)],
        dtype=float,
    )
    if boundaries.size > _MAX_SAVED_POINTS:
        raise ModelConfigurationError(
            "Recipe forcing boundaries exceed the fixed saved-point limit "
            f"({_MAX_SAVED_POINTS})"
        )
    if controls.save_at_s is not None:
        if len(controls.save_at_s) + boundaries.size > _MAX_SAVED_POINTS:
            raise ModelConfigurationError(
                "save_at_s would exceed the fixed saved-point limit "
                f"({_MAX_SAVED_POINTS})"
            )
        requested = np.asarray(controls.save_at_s, dtype=float)
        if requested[0] < start_s - tolerance or requested[-1] > end_s + tolerance:
            raise ModelConfigurationError(
                f"save_at_s must lie within the recipe interval [{start_s:g}, {end_s:g}]"
            )
        requested = np.clip(requested, start_s, end_s)
    else:
        assert controls.sample_interval_s is not None
        interval = float(controls.sample_interval_s)
        interval_count = (end_s - start_s) / interval
        if (
            not math.isfinite(interval_count)
            or interval_count + 1 + boundaries.size > _MAX_SAVED_POINTS
        ):
            raise ModelConfigurationError(
                "sample_interval_s would exceed the fixed saved-point limit "
                f"({_MAX_SAVED_POINTS})"
            )
        count = math.floor(interval_count)
        requested = start_s + interval * np.arange(count + 1, dtype=float)
        requested = requested[requested <= end_s + tolerance]
    merged = _merge_with_boundaries(requested, boundaries)
    if merged.size > _MAX_SAVED_POINTS:
        raise ModelConfigurationError(
            "Requested output exceeds the fixed saved-point limit "
            f"({_MAX_SAVED_POINTS})"
        )
    return merged


def _segment_sample_times(
    global_times: np.ndarray | None, segment: RecipeSegment
) -> np.ndarray | None:
    if global_times is None:
        return None
    tolerance = _time_tolerance(segment.start_s, segment.end_s)
    selected = global_times[
        (global_times >= segment.start_s - tolerance)
        & (global_times <= segment.end_s + tolerance)
    ].copy()
    selected[np.abs(selected - segment.start_s) <= tolerance] = segment.start_s
    selected[np.abs(selected - segment.end_s) <= tolerance] = segment.end_s
    if (
        selected.size < 2
        or selected[0] != segment.start_s
        or selected[-1] != segment.end_s
    ):
        raise IntegrationError(
            f"Internal sampling error in segment {segment.segment_id!r}: forcing boundaries are missing"
        )
    return selected


def _domain_failure(
    segment: RecipeSegment, time_s: float | None, error: Exception
) -> IntegrationError:
    at_time = segment.start_s if time_s is None else float(time_s)
    return IntegrationError(
        f"Domain failure in segment {segment.segment_id!r} at t={at_time:.16g} s: {error}"
    )


def _state_scale(model: CompiledGlobalModel, initial_state: np.ndarray) -> np.ndarray:
    """Build one immutable reference scale for the complete integration."""

    scale = np.maximum(np.abs(initial_state), 1.0)
    for zone_id in model.layout.zone_ids:
        density_slice = model.layout.density_slices[zone_id]
        density_reference = max(
            float(np.max(np.abs(initial_state[density_slice]), initial=0.0)),
            1.0,
        )
        scale[density_slice] = density_reference
    scale.setflags(write=False)
    return scale


class _ScaledSegmentRHS:
    """Map the dimensionless BDF state to and from physical model units."""

    def __init__(self, physical_rhs: BoundSegmentRHS, scale: np.ndarray) -> None:
        self.physical_rhs = physical_rhs
        self.scale = scale

    def __call__(self, time_s: float, scaled_state: np.ndarray) -> np.ndarray:
        state = np.asarray(scaled_state, dtype=float) * self.scale
        return self.physical_rhs(time_s, state) / self.scale

    @property
    def last_time_s(self) -> float | None:
        return self.physical_rhs.last_time_s

    @property
    def jac_sparsity(self) -> Any:
        return self.physical_rhs.jac_sparsity


def _quasi_steady_event(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    segment: RecipeSegment,
    *,
    relative_rhs_norm_s_inv: float,
    min_time_s: float,
) -> Callable[[float, np.ndarray], float]:
    earliest_s = segment.start_s + min_time_s
    gate_scale_s = max(segment.end_s - segment.start_s, 1.0e-300)

    def event(time_s: float, scaled_state: np.ndarray) -> float:
        time_gate = (earliest_s - float(time_s)) / gate_scale_s
        if time_s < earliest_s:
            return time_gate
        derivative = rhs(time_s, scaled_state)
        if not np.all(np.isfinite(scaled_state)) or not np.all(np.isfinite(derivative)):
            return math.inf
        scale = np.maximum(np.abs(scaled_state), 1.0)
        relative_residual = (
            float(np.max(np.abs(derivative) / scale)) - relative_rhs_norm_s_inv
        )
        return max(time_gate, relative_residual)

    event.terminal = True  # type: ignore[attr-defined]
    event.direction = -1.0  # type: ignore[attr-defined]
    return event


def _validate_experimental_quasi_steady(model: CompiledGlobalModel) -> None:
    dynamic_rates = any(
        bool(getattr(evaluator, "time_dependent", False))
        for evaluator in model.rate_evaluators
    )
    dynamic_electron_density = model.electron_density_provider is not None and bool(
        getattr(model.electron_density_provider, "time_dependent", True)
    )
    dynamic_segments = [
        segment.segment_id
        for segment in model.segments
        if dynamic_rates
        or dynamic_electron_density
        or (
            model.power_coordinator is not None
            and model.power_coordinator.commands_are_time_dependent(
                segment.port_commands
            )
        )
    ]
    if dynamic_segments:
        raise ModelConfigurationError(
            f"{_EXPERIMENTAL_QUASI_STEADY_MODEL_ID} requires time-invariant "
            f"segments; dynamic segments: {dynamic_segments}"
        )


def _steady_event_state(solution: Any) -> tuple[float, np.ndarray] | None:
    if not solution.t_events or not len(solution.t_events[0]):
        return None
    return (
        float(solution.t_events[0][-1]),
        np.asarray(solution.y_events[0][-1], dtype=float),
    )


def _segment_output(
    solution: Any,
    requested_times: np.ndarray | None,
    segment: RecipeSegment,
    steady_event: tuple[float, np.ndarray] | None,
    state_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    solver_time = np.asarray(solution.t, dtype=float)
    solver_state = np.asarray(solution.y, dtype=float).T * state_scale
    if steady_event is None:
        return solver_time, solver_state

    event_time, scaled_event_state = steady_event
    event_state = scaled_event_state * state_scale
    if requested_times is not None:
        output_state = np.repeat(event_state[None, :], requested_times.size, axis=0)
        output_state[: solver_time.size, :] = solver_state
        return np.array(requested_times, copy=True), output_state

    tolerance = _time_tolerance(segment.start_s, segment.end_s)
    output_time = np.array(solver_time, copy=True)
    output_state = np.array(solver_state, copy=True)
    if not output_time.size or abs(output_time[-1] - event_time) > tolerance:
        output_time = np.append(output_time, event_time)
        output_state = np.vstack((output_state, event_state))
    if event_time < segment.end_s - tolerance:
        output_time = np.append(output_time, segment.end_s)
        output_state = np.vstack((output_state, event_state))
    return output_time, output_state


def _accepted_state_for_result(
    state: np.ndarray,
    domain_atol: np.ndarray,
    *,
    lower_bounds: np.ndarray | None = None,
    time_s: np.ndarray | None = None,
    segments: Sequence[RecipeSegment] = (),
) -> tuple[np.ndarray, int]:
    """Validate lower bounds and zero round-off negatives at result creation."""

    accepted = np.array(state, dtype=float, copy=True)
    tolerance = np.asarray(domain_atol, dtype=float).reshape(1, -1)
    bounds = (
        np.zeros_like(tolerance)
        if lower_bounds is None
        else np.asarray(lower_bounds, dtype=float).reshape(1, -1)
    )
    if bounds.shape != tolerance.shape or np.any(np.isnan(bounds)):
        raise ValueError(
            "result lower bounds must match domain_atol and not contain NaN"
        )
    materially_below_bound = accepted < (bounds - 10.0 * tolerance)
    if np.any(materially_below_bound):
        time_index, state_index = np.argwhere(materially_below_bound)[0]
        location = f"output index {int(time_index)}"
        if time_s is not None:
            failed_time = float(np.asarray(time_s, dtype=float)[time_index])
            segment_id = next(
                (
                    segment.segment_id
                    for segment in segments
                    if segment.start_s <= failed_time <= segment.end_s
                ),
                "unknown",
            )
            location = f"segment {segment_id!r} at t={failed_time:.16g} s"
        raise IntegrationError(
            "Accepted solver state domain failure below lower_bound-10*domain_atol "
            f"in {location}: "
            f"state[{int(time_index)},{int(state_index)}]={accepted[time_index, state_index]:.6e}"
        )
    roundoff_negative = (bounds == 0.0) & (accepted < 0.0) & ~materially_below_bound
    zeroed_count = int(np.count_nonzero(roundoff_negative))
    accepted[roundoff_negative] = 0.0
    return accepted, zeroed_count


def solve_compiled_model(
    model: CompiledGlobalModel,
    initial: InitialState | np.ndarray,
    settings: SolverSettings | None = None,
) -> SimulationResult:
    """Integrate contiguous recipe segments with a segment-bound BDF RHS.

    A solver call never performs a time-based recipe lookup.  In particular,
    the implicit endpoint evaluations of one BDF solve retain that segment's
    forcing even when ``time_s == segment.end_s``.
    """

    # Imported lazily so the physical core depends only on the small result
    # data contract and remains independent of output/serialization modules.
    from plasma_global.core.result import SimulationResult, SimulationStatus

    controls = settings or SolverSettings()
    stop_threshold = controls.experimental_quasi_steady_threshold_s_inv
    if stop_threshold is not None:
        _validate_experimental_quasi_steady(model)
    if controls.first_step_s is not None:
        too_short = [
            segment.segment_id
            for segment in model.segments
            if controls.first_step_s > segment.end_s - segment.start_s
        ]
        if too_short:
            raise ModelConfigurationError(
                "first_step_s must not exceed any recipe segment duration; "
                f"too short: {too_short}"
            )
    global_sample_times = _global_sample_times(model, controls)
    if isinstance(initial, InitialState):
        current_state = model.initial_state(initial)
    else:
        current_state = np.array(initial, dtype=float, copy=True)
        if current_state.shape != (model.layout.size,):
            raise ValueError(
                f"Initial state has shape {current_state.shape}, expected {(model.layout.size,)}"
            )
    if not np.all(np.isfinite(current_state)):
        raise ModelConfigurationError("Initial state must contain only finite values")
    state_scale = _state_scale(model, current_state)
    physical_domain_atol = controls.atol * state_scale
    physical_domain_atol.setflags(write=False)

    time_parts: list[np.ndarray] = []
    state_parts: list[np.ndarray] = []
    total_nfev = 0
    total_njev = 0
    total_nlu = 0
    completed_segments = 0
    saved_points = 0
    stop_events = 0

    for segment_index, segment in enumerate(model.segments):
        bound_rhs = model.bind_segment(
            segment,
            domain_atol=physical_domain_atol,
        )
        scaled_rhs = _ScaledSegmentRHS(bound_rhs, state_scale)
        t_eval = _segment_sample_times(global_sample_times, segment)
        options: dict[str, object] = {
            "fun": scaled_rhs,
            "t_span": (segment.start_s, segment.end_s),
            "y0": current_state / state_scale,
            "method": "BDF",
            "rtol": controls.rtol,
            "atol": controls.atol,
            "jac_sparsity": scaled_rhs.jac_sparsity,
        }
        if t_eval is not None:
            options["t_eval"] = t_eval
        if controls.first_step_s is not None:
            options["first_step"] = controls.first_step_s
        if controls.max_step_s is not None:
            options["max_step"] = controls.max_step_s
        if stop_threshold is not None:
            options["events"] = _quasi_steady_event(
                scaled_rhs,
                segment,
                relative_rhs_norm_s_inv=stop_threshold,
                min_time_s=controls.experimental_quasi_steady_min_time_s,
            )
        try:
            solution = solve_ivp(**options)
        except (ModelDomainError, CouplingConvergenceError) as exc:
            raise _domain_failure(segment, scaled_rhs.last_time_s, exc) from exc
        except ValueError as exc:
            failure_time = (
                segment.start_s
                if scaled_rhs.last_time_s is None
                else scaled_rhs.last_time_s
            )
            raise IntegrationError(
                f"Integration setup failed in segment {segment.segment_id!r} "
                f"at t={failure_time:.16g} s: {exc}"
            ) from exc
        total_nfev += int(solution.nfev)
        total_njev += int(solution.njev)
        total_nlu += int(getattr(solution, "nlu", 0) or 0)

        if not solution.success:
            failed_time = (
                float(solution.t[-1])
                if np.asarray(solution.t).size
                else segment.start_s
            )
            raise IntegrationError(
                f"Integration failed in segment {segment.segment_id!r} "
                f"at t={failed_time:.16g} s: {solution.message}"
            )

        if not solution.y.shape[1]:
            raise IntegrationError(
                f"Integration failed in segment {segment.segment_id!r} "
                f"at t={segment.start_s:.16g} s: solver returned no state"
            )
        steady_event = _steady_event_state(solution)
        if steady_event is not None:
            stop_events += 1
            current_state = steady_event[1] * state_scale
        else:
            current_state = np.asarray(solution.y[:, -1], dtype=float) * state_scale

        segment_time, segment_state = _segment_output(
            solution,
            t_eval,
            segment,
            steady_event,
            state_scale,
        )
        if segment_index > 0 and segment_time.size:
            segment_time = segment_time[1:]
            segment_state = segment_state[1:, :]
        saved_points += int(segment_time.size)
        if saved_points > _MAX_SAVED_POINTS:
            raise IntegrationError(
                "Solver output exceeded the fixed saved-point limit "
                f"({_MAX_SAVED_POINTS}); specify sample_interval_s or save_at_s"
            )
        time_parts.append(segment_time)
        state_parts.append(segment_state)
        completed_segments += 1

    time_s = np.concatenate(time_parts) if time_parts else np.empty(0, dtype=float)
    raw_state = (
        np.concatenate(state_parts, axis=0)
        if state_parts
        else np.empty((0, model.layout.size), dtype=float)
    )
    state, zeroed_negative_count = _accepted_state_for_result(
        raw_state,
        physical_domain_atol,
        lower_bounds=model.state_lower_bounds,
        time_s=time_s,
        segments=model.segments,
    )
    solver_stats = {
        "nfev": total_nfev,
        "njev": total_njev,
        "nlu": total_nlu,
        "segments_completed": completed_segments,
    }
    if stop_threshold is not None:
        solver_stats["quasi_steady_events"] = stop_events
    return SimulationResult(
        time_s=time_s,
        state=state,
        state_labels=model.layout.labels,
        observables={},
        status=SimulationStatus.completed("All recipe segments completed"),
        solver_stats=solver_stats,
        metadata={
            "electron_closure": model.electron_closure.mode,
            "jacobian_sparsity_nnz": int(model.jac_sparsity.nnz),
            "dimensionless_atol": controls.atol,
            "state_scale": state_scale.tolist(),
            "domain_atol": physical_domain_atol.tolist(),
            "result_negative_limit_multiplier": 10.0,
            "experimental_stop_policy": (
                None if stop_threshold is None else _EXPERIMENTAL_QUASI_STEADY_MODEL_ID
            ),
            "sampling_mode": (
                "native"
                if global_sample_times is None
                else (
                    "sample_interval"
                    if controls.sample_interval_s is not None
                    else "save_at"
                )
            ),
            "provenance": {
                "accepted_state_negative_policy": (
                    "At solver-to-result construction, states with zero lower bound store "
                    "-10*domain_atol <= y < 0 as zero; values materially below each finite "
                    "lower bound fail. Signed extension states remain signed. ODE state and "
                    "RHS are not clipped."
                ),
                "state_scaling_policy": (
                    "BDF integrates y/state_scale with scalar dimensionless atol; density "
                    "components share their zone's initial maximum-density scale and all "
                    "other components use max(abs(initial), 1)."
                ),
                "accepted_state_zeroed_negative_count": zeroed_negative_count,
            },
        },
    )


__all__ = ["solve_compiled_model"]
