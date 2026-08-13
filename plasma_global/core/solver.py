"""Sequential stiff integration for :class:`CompiledGlobalModel`."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
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


def _times_coincide(left_s: float, right_s: float) -> bool:
    """Treat only the nearest representable neighbours as the same time."""

    return bool(
        np.nextafter(left_s, -math.inf) <= right_s <= np.nextafter(left_s, math.inf)
    )


def _times_coincide_array(
    values_s: np.ndarray, references_s: np.ndarray | float
) -> np.ndarray:
    """Vector form of :func:`_times_coincide`."""

    return (values_s >= np.nextafter(references_s, -math.inf)) & (
        values_s <= np.nextafter(references_s, math.inf)
    )


def _merge_with_boundaries(requested: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Merge global sample times while giving exact boundaries precedence."""

    positions = np.searchsorted(boundaries, requested)
    left = boundaries[np.maximum(positions - 1, 0)]
    right = boundaries[np.minimum(positions, boundaries.size - 1)]
    near_boundary = _times_coincide_array(requested, left) | _times_coincide_array(
        requested, right
    )
    values = np.sort(np.concatenate((requested[~near_boundary], boundaries)))
    merged: list[float] = []
    for value in values:
        item = float(value)
        if not merged or not _times_coincide(merged[-1], item):
            merged.append(item)
    return np.asarray(merged, dtype=float)


def _global_sample_times(
    model: CompiledGlobalModel, controls: SolverSettings
) -> np.ndarray | None:
    """Build a recipe-global output grid, or select native solver points."""

    if controls.sample_interval_s is None and controls.save_at_s is None:
        return None
    start_s = model.segments[0].start_s
    end_s = model.segments[-1].end_s
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
        if requested[0] < np.nextafter(start_s, -math.inf) or requested[
            -1
        ] > np.nextafter(end_s, math.inf):
            raise ModelConfigurationError(
                f"save_at_s must lie within the recipe interval [{start_s:g}, "
                f"{end_s:g}]"
            )
        requested = np.clip(requested, start_s, end_s)
    else:
        interval_setting = controls.sample_interval_s
        if interval_setting is None:
            raise ModelConfigurationError(
                "sample_interval_s is required when save_at_s is not configured"
            )
        interval = interval_setting
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
        requested = requested[requested <= np.nextafter(end_s, math.inf)]
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
    selected = global_times[
        (global_times >= np.nextafter(segment.start_s, -math.inf))
        & (global_times <= np.nextafter(segment.end_s, math.inf))
    ].copy()
    selected[_times_coincide_array(selected, segment.start_s)] = segment.start_s
    selected[_times_coincide_array(selected, segment.end_s)] = segment.end_s
    if (
        selected.size < 2
        or selected[0] != segment.start_s
        or selected[-1] != segment.end_s
    ):
        raise IntegrationError(
            f"Internal sampling error in segment {segment.segment_id!r}: "
            "forcing boundaries are missing"
        )
    return selected


def _domain_failure(
    segment: RecipeSegment, time_s: float | None, error: Exception
) -> IntegrationError:
    at_time = segment.start_s if time_s is None else time_s
    return IntegrationError(
        f"Domain failure in segment {segment.segment_id!r} "
        f"at t={at_time:.16g} s: {error}"
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


@dataclass(frozen=True, slots=True)
class _QuasiSteadyObservation:
    """Observe a local quasi-steady candidate without truncating integration."""

    rhs: Callable[[float, np.ndarray], np.ndarray]
    earliest_s: float
    end_s: float
    gate_scale_s: float
    relative_rhs_norm_s_inv: float
    rtol: float
    atol: float
    terminal: bool = False
    direction: float = -1.0

    def __call__(self, time_s: float, scaled_state: np.ndarray) -> float:
        time_gate = (self.earliest_s - time_s) / self.gate_scale_s
        if time_s < self.earliest_s:
            return time_gate
        derivative = self.rhs(time_s, scaled_state)
        if not np.all(np.isfinite(scaled_state)) or not np.all(np.isfinite(derivative)):
            return math.inf
        scale = np.maximum(np.abs(scaled_state), 1.0)
        relative_residual = (
            float(np.max(np.abs(derivative) / scale)) - self.relative_rhs_norm_s_inv
        )
        remaining_s = max(self.end_s - time_s, 0.0)
        solver_error_budget = self.atol + self.rtol * np.abs(scaled_state)
        remaining_change_residual = (
            float(np.max(remaining_s * np.abs(derivative) / solver_error_budget)) - 1.0
        )
        return max(time_gate, relative_residual, remaining_change_residual)


def _quasi_steady_observation(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    segment: RecipeSegment,
    *,
    relative_rhs_norm_s_inv: float,
    min_time_s: float,
    rtol: float,
    atol: float,
) -> _QuasiSteadyObservation:
    return _QuasiSteadyObservation(
        rhs=rhs,
        earliest_s=segment.start_s + min_time_s,
        end_s=segment.end_s,
        gate_scale_s=max(segment.end_s - segment.start_s, 1.0e-300),
        relative_rhs_norm_s_inv=relative_rhs_norm_s_inv,
        rtol=rtol,
        atol=atol,
    )


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


def _observed_quasi_steady(solution: Any) -> bool:
    """Return whether the nonterminal diagnostic crossed its threshold."""

    return bool(solution.t_events and len(solution.t_events[0]))


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
            f"state[{int(time_index)},{int(state_index)}]="
            f"{accepted[time_index, state_index]:.6e}"
        )
    roundoff_negative = (bounds == 0.0) & (accepted < 0.0) & ~materially_below_bound
    zeroed_count = int(np.count_nonzero(roundoff_negative))
    accepted[roundoff_negative] = 0.0
    return accepted, zeroed_count


def _validate_solver_settings_for_model(
    model: CompiledGlobalModel, controls: SolverSettings
) -> None:
    if controls.experimental_quasi_steady_threshold_s_inv is not None:
        _validate_experimental_quasi_steady(model)

    first_step_s = controls.first_step_s
    if first_step_s is None:
        return
    too_short = [
        segment.segment_id
        for segment in model.segments
        if first_step_s > segment.end_s - segment.start_s
    ]
    if too_short:
        raise ModelConfigurationError(
            "first_step_s must not exceed any recipe segment duration; "
            f"too short: {too_short}"
        )


def _prepare_initial_state(
    model: CompiledGlobalModel,
    initial: InitialState | np.ndarray,
    dimensionless_atol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(initial, InitialState):
        current_state = model.initial_state(initial)
    else:
        current_state = np.array(initial, dtype=float, copy=True)
        if current_state.shape != (model.layout.size,):
            raise ValueError(
                f"Initial state has shape {current_state.shape}, "
                f"expected {(model.layout.size,)}"
            )
    if not np.all(np.isfinite(current_state)):
        raise ModelConfigurationError("Initial state must contain only finite values")
    state_scale = _state_scale(model, current_state)
    physical_domain_atol = dimensionless_atol * state_scale
    physical_domain_atol.setflags(write=False)
    return current_state, state_scale, physical_domain_atol


def _segment_solver_options(
    scaled_rhs: _ScaledSegmentRHS,
    segment: RecipeSegment,
    scaled_initial_state: np.ndarray,
    controls: SolverSettings,
    sample_times: np.ndarray | None,
) -> dict[str, object]:
    options: dict[str, object] = {
        "fun": scaled_rhs,
        "t_span": (segment.start_s, segment.end_s),
        "y0": scaled_initial_state,
        "method": "BDF",
        "rtol": controls.rtol,
        "atol": controls.atol,
        "jac_sparsity": scaled_rhs.jac_sparsity,
    }
    if sample_times is not None:
        options["t_eval"] = sample_times
    if controls.first_step_s is not None:
        options["first_step"] = controls.first_step_s
    if controls.max_step_s is not None:
        options["max_step"] = controls.max_step_s
    stop_threshold = controls.experimental_quasi_steady_threshold_s_inv
    if stop_threshold is not None:
        options["events"] = _quasi_steady_observation(
            scaled_rhs,
            segment,
            relative_rhs_norm_s_inv=stop_threshold,
            min_time_s=controls.experimental_quasi_steady_min_time_s,
            rtol=controls.rtol,
            atol=controls.atol,
        )
    return options


def _call_segment_solver(
    segment: RecipeSegment,
    scaled_rhs: _ScaledSegmentRHS,
    options: dict[str, object],
) -> Any:
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

    if not solution.success:
        failed_time = (
            float(solution.t[-1]) if np.asarray(solution.t).size else segment.start_s
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
    return solution


def _completed_segment_output(
    solution: Any,
    state_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    time_s = np.asarray(solution.t, dtype=float)
    state = np.asarray(solution.y, dtype=float).T * state_scale
    current_state = np.asarray(solution.y[:, -1], dtype=float) * state_scale
    return current_state, time_s, state, _observed_quasi_steady(solution)


@dataclass(slots=True)
class _IntegrationHistory:
    """Accumulate completed segment output and SciPy work counters."""

    time_parts: list[np.ndarray] = field(default_factory=list)
    state_parts: list[np.ndarray] = field(default_factory=list)
    nfev: int = 0
    njev: int = 0
    nlu: int = 0
    completed_segments: int = 0
    saved_points: int = 0
    quasi_steady_events: int = 0

    def append(
        self,
        *,
        segment_index: int,
        solution: Any,
        time_s: np.ndarray,
        state: np.ndarray,
        observed_quasi_steady: bool,
    ) -> None:
        if segment_index > 0 and time_s.size:
            time_s = time_s[1:]
            state = state[1:, :]
        saved_points = self.saved_points + time_s.size
        if saved_points > _MAX_SAVED_POINTS:
            raise IntegrationError(
                "Solver output exceeded the fixed saved-point limit "
                f"({_MAX_SAVED_POINTS}); specify sample_interval_s or save_at_s"
            )

        self.time_parts.append(time_s)
        self.state_parts.append(state)
        self.nfev += int(solution.nfev)
        self.njev += int(solution.njev)
        self.nlu += int(getattr(solution, "nlu", 0) or 0)
        self.completed_segments += 1
        self.saved_points = saved_points
        self.quasi_steady_events += int(observed_quasi_steady)

    def arrays(self, state_size: int) -> tuple[np.ndarray, np.ndarray]:
        time_s = (
            np.concatenate(self.time_parts)
            if self.time_parts
            else np.empty(0, dtype=float)
        )
        state = (
            np.concatenate(self.state_parts, axis=0)
            if self.state_parts
            else np.empty((0, state_size), dtype=float)
        )
        return time_s, state

    def solver_stats(self, *, include_quasi_steady: bool) -> dict[str, int]:
        stats = {
            "nfev": self.nfev,
            "njev": self.njev,
            "nlu": self.nlu,
            "segments_completed": self.completed_segments,
        }
        if include_quasi_steady:
            stats["quasi_steady_events"] = self.quasi_steady_events
        return stats


def _sampling_mode(
    controls: SolverSettings, global_sample_times: np.ndarray | None
) -> str:
    if global_sample_times is None:
        return "native"
    if controls.sample_interval_s is not None:
        return "sample_interval"
    return "save_at"


def _result_metadata(
    model: CompiledGlobalModel,
    controls: SolverSettings,
    global_sample_times: np.ndarray | None,
    state_scale: np.ndarray,
    physical_domain_atol: np.ndarray,
    zeroed_negative_count: int,
) -> dict[str, object]:
    return {
        "electron_closure": model.electron_closure.mode,
        "jacobian_sparsity_nnz": model.jac_sparsity.nnz,
        "dimensionless_atol": controls.atol,
        "state_scale": state_scale.tolist(),
        "domain_atol": physical_domain_atol.tolist(),
        "result_negative_limit_multiplier": 10.0,
        "experimental_stop_policy": (
            None
            if controls.experimental_quasi_steady_threshold_s_inv is None
            else _EXPERIMENTAL_QUASI_STEADY_MODEL_ID
        ),
        "experimental_quasi_steady_event_mode": (
            None
            if controls.experimental_quasi_steady_threshold_s_inv is None
            else "nonterminal_observation_v1"
        ),
        "sampling_mode": _sampling_mode(controls, global_sample_times),
        "provenance": {
            "accepted_state_negative_policy": (
                "At solver-to-result construction, states with zero lower bound store "
                "-10*domain_atol <= y < 0 as zero; values materially below each finite "
                "lower bound fail. Signed extension states remain signed. "
                "ODE state and RHS are not clipped."
            ),
            "state_scaling_policy": (
                "BDF integrates y/state_scale with scalar dimensionless atol; density "
                "components share their zone's initial maximum-density scale and all "
                "other components use max(abs(initial), 1)."
            ),
            "accepted_state_zeroed_negative_count": zeroed_negative_count,
        },
    }


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
    _validate_solver_settings_for_model(model, controls)
    stop_threshold = controls.experimental_quasi_steady_threshold_s_inv
    global_sample_times = _global_sample_times(model, controls)
    current_state, state_scale, physical_domain_atol = _prepare_initial_state(
        model,
        initial,
        controls.atol,
    )
    history = _IntegrationHistory()

    for segment_index, segment in enumerate(model.segments):
        scaled_rhs = _ScaledSegmentRHS(
            model.bind_segment(segment, domain_atol=physical_domain_atol),
            state_scale,
        )
        sample_times = _segment_sample_times(global_sample_times, segment)
        options = _segment_solver_options(
            scaled_rhs,
            segment,
            current_state / state_scale,
            controls,
            sample_times,
        )
        solution = _call_segment_solver(segment, scaled_rhs, options)
        current_state, segment_time, segment_state, observed_quasi_steady = (
            _completed_segment_output(
                solution,
                state_scale,
            )
        )
        history.append(
            segment_index=segment_index,
            solution=solution,
            time_s=segment_time,
            state=segment_state,
            observed_quasi_steady=observed_quasi_steady,
        )

    time_s, raw_state = history.arrays(model.layout.size)
    state, zeroed_negative_count = _accepted_state_for_result(
        raw_state,
        physical_domain_atol,
        lower_bounds=model.state_lower_bounds,
        time_s=time_s,
        segments=model.segments,
    )
    return SimulationResult(
        time_s=time_s,
        state=state,
        state_labels=model.layout.labels,
        observables={},
        status=SimulationStatus.completed("All recipe segments completed"),
        solver_stats=history.solver_stats(
            include_quasi_steady=stop_threshold is not None
        ),
        metadata=_result_metadata(
            model,
            controls,
            global_sample_times,
            state_scale,
            physical_domain_atol,
            zeroed_negative_count,
        ),
    )


__all__ = ["solve_compiled_model"]
