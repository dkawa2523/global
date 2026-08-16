"""Validation and evaluation kernels for declarative extension states."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class GenericStateLayout:
    """Immutable indexing compiled from public state and process declarations."""

    states: tuple[Any, ...]
    processes: tuple[Any, ...]
    labels: tuple[str, ...]
    index: Mapping[tuple[str, str], int]
    initial: np.ndarray
    lower_bounds: tuple[float | None, ...]
    upper_bounds: tuple[float | None, ...]


def _state_owners(values: Sequence[str]) -> tuple[str, ...]:
    owners = tuple(str(value) for value in values)
    if not owners or any(not owner for owner in owners):
        raise ValueError("state owners must be nonempty strings")
    if len(set(owners)) != len(owners):
        raise ValueError("state owners must be unique")
    return owners


def _validate_state_values(
    initial: float, lower_bound: float | None, upper_bound: float | None
) -> None:
    bounds = tuple(value for value in (lower_bound, upper_bound) if value is not None)
    if not math.isfinite(initial) or any(not math.isfinite(value) for value in bounds):
        raise ValueError("state initial value and bounds must be finite")
    if lower_bound is not None and initial < lower_bound:
        raise ValueError("state initial value is below its lower bound")
    if upper_bound is not None and initial > upper_bound:
        raise ValueError("state initial value is above its upper bound")


def _validate_bound_order(lower_bound: float | None, upper_bound: float | None) -> None:
    if (
        lower_bound is not None
        and upper_bound is not None
        and upper_bound < lower_bound
    ):
        raise ValueError("state upper bound is below its lower bound")


def validate_state_declaration(
    state_id: str,
    owners: Sequence[str],
    initial: float,
    lower_bound: float | None,
    upper_bound: float | None,
) -> tuple[str, ...]:
    if not state_id:
        raise ValueError("state_id must not be empty")
    normalized_owners = _state_owners(owners)
    _validate_state_values(initial, lower_bound, upper_bound)
    _validate_bound_order(lower_bound, upper_bound)
    return normalized_owners


def freeze_process_drivers(
    drivers: Mapping[str, Mapping[str, float]],
) -> Mapping[str, Mapping[str, float]]:
    frozen: dict[str, Mapping[str, float]] = {}
    for name, values in drivers.items():
        converted = {str(owner): float(value) for owner, value in values.items()}
        if not name or any(not math.isfinite(value) for value in converted.values()):
            raise ValueError("process drivers must have a name and finite values")
        frozen[str(name)] = MappingProxyType(converted)
    return MappingProxyType(frozen)


def validate_constant_source(target_state: str, rate_per_s: float) -> None:
    if not target_state or not math.isfinite(rate_per_s):
        raise ValueError("constant source target and rate must be valid")


def validate_relaxation(
    target_state: str, equilibrium: float, time_constant_s: float
) -> None:
    if not target_state or not math.isfinite(equilibrium):
        raise ValueError("relaxation target and equilibrium must be valid")
    if not math.isfinite(time_constant_s) or time_constant_s <= 0.0:
        raise ValueError("time_constant_s must be finite and positive")


def validate_driven_source(target_state: str, driver: str, coefficient: float) -> None:
    if not target_state or not driver or not math.isfinite(coefficient):
        raise ValueError("driven source target, driver, and coefficient must be valid")


def normalize_process_owners(owners: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(owner) for owner in owners)


def build_generic_state_layout(
    declared_states: Sequence[Any], declared_processes: Sequence[Any]
) -> GenericStateLayout:
    states = tuple(declared_states)
    processes = tuple(declared_processes)
    if not states:
        raise ValueError("at least one generic state is required")
    if len({state.state_id for state in states}) != len(states):
        raise ValueError("generic state IDs must be unique")
    owner_by_state = {state.state_id: set(state.owners) for state in states}
    labels: list[str] = []
    index: dict[tuple[str, str], int] = {}
    initial: list[float] = []
    lower_bounds: list[float | None] = []
    upper_bounds: list[float | None] = []
    for state in states:
        for owner in state.owners:
            index[state.state_id, owner] = len(labels)
            labels.append(f"{state.state_id}[{owner}]")
            initial.append(state.initial)
            lower_bounds.append(state.lower_bound)
            upper_bounds.append(state.upper_bound)
    for process in processes:
        if process.target_state not in owner_by_state:
            raise ValueError(f"process targets unknown state {process.target_state!r}")
        unknown = set(process.owners) - owner_by_state[process.target_state]
        if unknown:
            message = (
                f"process for {process.target_state!r} selects unknown owners "
                f"{sorted(unknown)}"
            )
            raise ValueError(message)
    initial_array = np.asarray(initial, dtype=float)
    initial_array.setflags(write=False)
    return GenericStateLayout(
        states=states,
        processes=processes,
        labels=tuple(labels),
        index=MappingProxyType(index),
        initial=initial_array,
        lower_bounds=tuple(lower_bounds),
        upper_bounds=tuple(upper_bounds),
    )


def _validated_generic_state(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    state = np.asarray(values, dtype=float)
    if state.shape != reference.shape or not np.all(np.isfinite(state)):
        raise ValueError(
            "generic state vector has the wrong shape or non-finite values"
        )
    return state


def _validate_declared_domains(
    state: np.ndarray,
    states: Sequence[Any],
    index: Mapping[tuple[str, str], int],
) -> None:
    for declared in states:
        for owner in declared.owners:
            value = float(state[index[declared.state_id, owner]])
            if declared.lower_bound is not None and value < declared.lower_bound:
                raise ValueError(
                    f"{declared.state_id}[{owner}] is below its declared domain"
                )
            if declared.upper_bound is not None and value > declared.upper_bound:
                raise ValueError(
                    f"{declared.state_id}[{owner}] is above its declared domain"
                )


def _add_process_rates(
    derivative: np.ndarray,
    state: np.ndarray,
    *,
    processes: Sequence[Any],
    state_by_id: Mapping[str, Any],
    index: Mapping[tuple[str, str], int],
    context: Any,
) -> None:
    for process in processes:
        owners = tuple(process.owners) or tuple(
            state_by_id[process.target_state].owners
        )
        for owner in owners:
            position = index[process.target_state, owner]
            derivative[position] += process.rate(
                owner=owner,
                current_value=float(state[position]),
                context=context,
            )


def evaluate_generic_state_rhs(
    values: np.ndarray,
    *,
    reference: np.ndarray,
    states: Sequence[Any],
    processes: Sequence[Any],
    index: Mapping[tuple[str, str], int],
    context: Any,
    context_factory: Callable[[], Any],
) -> np.ndarray:
    state = _validated_generic_state(values, reference)
    evaluation_context = context_factory() if context is None else context
    derivative = np.zeros_like(state)
    state_by_id = {item.state_id: item for item in states}
    _validate_declared_domains(state, states, index)
    _add_process_rates(
        derivative,
        state,
        processes=processes,
        state_by_id=state_by_id,
        index=index,
        context=evaluation_context,
    )
    if not np.all(np.isfinite(derivative)):
        raise FloatingPointError(
            "generic process evaluation returned a non-finite rate"
        )
    return derivative
