"""Sequential stiff integration for :class:`CompiledGlobalModel`."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from plasma_global.core._solver_domain import (
    _accepted_state_for_result as _accepted_state_for_result,
)
from plasma_global.core._solver_domain import canonicalize_result_state
from plasma_global.core._solver_result import (
    _EXPERIMENTAL_QUASI_STEADY_MODEL_ID,
    _IntegrationHistory,
    _result_metadata,
)
from plasma_global.core._solver_sampling import (
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


def _domain_failure(
    segment: RecipeSegment, time_s: float | None, error: Exception
) -> IntegrationError:
    at_time = segment.start_s if time_s is None else time_s
    return IntegrationError(
        f"Domain failure in segment {segment.segment_id!r} "
        f"at t={at_time:.16g} s: {error}"
    )


def _state_scale(initial_state: np.ndarray) -> np.ndarray:
    """Build one immutable, component-wise scale for the integration."""

    scale = np.maximum(np.abs(initial_state), 1.0)
    scale.setflags(write=False)
    return scale


class _ScaledSegmentRHS:
    """Evaluate a continuous lower-bound extension without projecting BDF state."""

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
        # solve_ivp does not label Newton trials separately from accepted-state
        # callbacks. Continue every RHS evaluation at the nearest component bound
        # so rejected probes do not become false integration failures. The
        # integrator's state is untouched; only this physical evaluation view
        # receives the model's continuous domain extensions.
        np.clip(state, self.lower_bounds, self.upper_bounds, out=state)
        if self.surface_model is not None:
            self.surface_model._continue_solver_coverages(
                state[self.surface_coverage_slice]
            )
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


def _initial_state_array(
    model: CompiledGlobalModel,
    initial: InitialState | np.ndarray,
) -> np.ndarray:
    """Materialize either public initial-state representation as one array."""

    if isinstance(initial, InitialState):
        return model.initial_state(initial)
    current_state = np.array(initial, dtype=float, copy=True)
    if current_state.shape != (model.layout.size,):
        raise ValueError(
            f"Initial state has shape {current_state.shape}, "
            f"expected {(model.layout.size,)}"
        )
    return current_state


def _prepare_initial_state(
    model: CompiledGlobalModel,
    initial: InitialState | np.ndarray,
    dimensionless_atol: float,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    current_state = _initial_state_array(model, initial)
    if not np.all(np.isfinite(current_state)):
        raise ModelConfigurationError("Initial state must contain only finite values")
    below_bound = current_state < model.state_lower_bounds
    if np.any(below_bound):
        index = int(np.flatnonzero(below_bound)[0])
        raise ModelConfigurationError(
            "Initial state violates its compiled lower bound: "
            f"state[{index}]={current_state[index]:.6e} < "
            f"{model.state_lower_bounds[index]:.6e}"
        )
    above_bound = current_state > model.state_upper_bounds
    if np.any(above_bound):
        index = int(np.flatnonzero(above_bound)[0])
        raise ModelConfigurationError(
            "Initial state violates its compiled upper bound: "
            f"state[{index}]={current_state[index]:.6e} > "
            f"{model.state_upper_bounds[index]:.6e}"
        )
    state_scale = _state_scale(current_state)
    solver_atol = dimensionless_atol
    physical_domain_atol = dimensionless_atol * state_scale
    physical_domain_atol.setflags(write=False)
    if model.surface_model is not None:
        coverage_slice = model.layout.surface_coverage_slice
        try:
            model.surface_model._validate_solver_coverages(
                current_state[coverage_slice], physical_domain_atol[coverage_slice]
            )
        except ModelDomainError as exc:
            raise ModelConfigurationError(
                f"Initial state violates the surface coverage domain: {exc}"
            ) from exc
    return current_state, state_scale, solver_atol, physical_domain_atol


def _segment_solver_options(
    scaled_rhs: _ScaledSegmentRHS,
    segment: RecipeSegment,
    scaled_initial_state: np.ndarray,
    solver_atol: float,
    controls: SolverSettings,
    sample_times: np.ndarray | None,
) -> dict[str, object]:
    options: dict[str, object] = {
        "fun": scaled_rhs,
        "t_span": (segment.start_s, segment.end_s),
        "y0": scaled_initial_state,
        "method": "BDF",
        "rtol": controls.rtol,
        "atol": solver_atol,
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


def solve_compiled_model(
    model: CompiledGlobalModel,
    initial: InitialState | np.ndarray,
    settings: SolverSettings | None = None,
) -> SimulationResult:
    """Integrate contiguous recipe segments with a segment-bound BDF RHS.

    A solver call never performs a time-based recipe lookup. In particular,
    the implicit endpoint evaluations of one BDF solve retain that segment's
    forcing even when ``time_s == segment.end_s``.
    """

    controls = settings or SolverSettings()
    _validate_solver_settings_for_model(model, controls)
    stop_threshold = controls.experimental_quasi_steady_threshold_s_inv
    global_sample_times = _global_sample_times(model, controls)
    current_state, state_scale, solver_atol, physical_domain_atol = (
        _prepare_initial_state(model, initial, controls.atol)
    )
    history = _IntegrationHistory()

    for segment_index, segment in enumerate(model.segments):
        scaled_rhs = _ScaledSegmentRHS(
            model.bind_segment(segment, domain_atol=physical_domain_atol),
            state_scale,
            model.state_lower_bounds,
            model.state_upper_bounds,
        )
        sample_times = _segment_sample_times(global_sample_times, segment)
        options = _segment_solver_options(
            scaled_rhs,
            segment,
            current_state / state_scale,
            solver_atol,
            controls,
            sample_times,
        )
        solution = _call_segment_solver(segment, scaled_rhs, options)
        current_state, segment_time, segment_state, observed_quasi_steady = (
            _completed_segment_output(solution, state_scale)
        )
        history.append(
            segment_index=segment_index,
            solution=solution,
            time_s=segment_time,
            state=segment_state,
            observed_quasi_steady=observed_quasi_steady,
        )

    time_s, raw_state = history.arrays(model.layout.size)
    state, zeroed_negative_count, zeroed_electron_energy_count = (
        canonicalize_result_state(
            model,
            raw_state,
            physical_domain_atol,
            time_s,
        )
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
            zeroed_electron_energy_count,
        ),
    )


__all__ = ["solve_compiled_model"]
