"""Stable, solver-independent simulation result types.

The numerical core and the output layer exchange one compact object.  State
arrays are time-major so every exported row represents one instant, while
named observables remain separate from the ODE state contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np


def _readonly_float_array(value: Any, *, name: str) -> np.ndarray:
    try:
        array = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    array.setflags(write=False)
    return array


def _freeze_metadata_value(value: Any, *, path: str) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.ndarray):
        return _freeze_metadata_value(value.tolist(), path=path)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata_value(item, path=f"{path}[]") for item in value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key).strip()
            if not key:
                raise ValueError(f"{path} contains an empty key")
            normalized[key] = _freeze_metadata_value(item, path=f"{path}.{key}")
        return MappingProxyType(normalized)
    raise TypeError(f"{path} contains unsupported value type {type(value).__name__}")


def _freeze_mapping(value: Mapping[str, Any] | None, *, name: str) -> Mapping[str, Any]:
    frozen = _freeze_metadata_value(dict(value or {}), path=name)
    assert isinstance(frozen, Mapping)
    return frozen


def to_plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a serialization-friendly copy of frozen result metadata."""

    def plain(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): plain(child) for key, child in item.items()}
        if isinstance(item, tuple):
            return [plain(child) for child in item]
        return item

    return plain(value)


@dataclass(frozen=True, slots=True)
class SimulationStatus:
    """Terminal state of a simulation independent of a solver implementation."""

    success: bool = True
    code: str = "completed"
    message: str = ""

    def __post_init__(self) -> None:
        code = str(self.code).strip()
        if not code:
            raise ValueError("SimulationStatus.code must not be empty")
        object.__setattr__(self, "success", bool(self.success))
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", str(self.message))

    @classmethod
    def completed(cls, message: str = "") -> SimulationStatus:
        return cls(success=True, code="completed", message=message)

    @classmethod
    def failed(cls, message: str, *, code: str = "failed") -> SimulationStatus:
        return cls(success=False, code=code, message=message)


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Canonical in-memory simulation result.

    ``state`` always has shape ``(n_time, n_state)``.  Observable values are
    either a scalar (0-D array) or one value per time point (1-D array).
    Input arrays are copied and made read-only so output and audit operations
    cannot accidentally mutate a completed run.
    """

    time_s: np.ndarray
    state: np.ndarray
    state_labels: tuple[str, ...]
    observables: Mapping[str, np.ndarray] = field(default_factory=dict)
    status: SimulationStatus = field(default_factory=SimulationStatus.completed)
    solver_stats: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _state_index: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        time_s = _readonly_float_array(self.time_s, name="time_s")
        if time_s.ndim != 1:
            raise ValueError(
                f"time_s must be one-dimensional, got shape {time_s.shape}"
            )
        if not np.all(np.isfinite(time_s)):
            raise ValueError("time_s must contain only finite values")
        if time_s.size > 1 and np.any(np.diff(time_s) <= 0.0):
            raise ValueError("time_s must be strictly increasing")

        labels = tuple(str(label).strip() for label in self.state_labels)
        if any(not label for label in labels):
            raise ValueError("state_labels must not contain empty labels")
        if len(set(labels)) != len(labels):
            raise ValueError("state_labels must be unique")

        state = _readonly_float_array(self.state, name="state")
        if state.ndim != 2:
            raise ValueError(
                f"state must be two-dimensional and time-major, got shape {state.shape}"
            )
        expected_shape = (time_s.size, len(labels))
        if state.shape != expected_shape:
            raise ValueError(
                f"state must have time-major shape {expected_shape}, got {state.shape}"
            )

        observables: dict[str, np.ndarray] = {}
        for raw_name, raw_values in dict(self.observables or {}).items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError("observable names must not be empty")
            if "/" in name or "\x00" in name or name in {".", ".."}:
                raise ValueError(
                    f"observable name {name!r} is not safe for the result HDF5 layout"
                )
            if name in labels:
                raise ValueError(
                    f"observable name {name!r} conflicts with a state label"
                )
            values = _readonly_float_array(raw_values, name=f"observables[{name!r}]")
            if values.ndim == 1 and values.shape != (time_s.size,):
                raise ValueError(
                    f"observable {name!r} must have shape ({time_s.size},) or be scalar, got {values.shape}"
                )
            if values.ndim not in {0, 1}:
                raise ValueError(
                    f"observable {name!r} must be scalar or one-dimensional, got {values.ndim} dimensions"
                )
            observables[name] = values

        if not isinstance(self.status, SimulationStatus):
            raise TypeError("status must be a SimulationStatus")

        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "state_labels", labels)
        object.__setattr__(self, "observables", MappingProxyType(observables))
        object.__setattr__(
            self,
            "solver_stats",
            _freeze_mapping(self.solver_stats, name="solver_stats"),
        )
        object.__setattr__(
            self, "metadata", _freeze_mapping(self.metadata, name="metadata")
        )
        object.__setattr__(
            self,
            "_state_index",
            MappingProxyType({label: i for i, label in enumerate(labels)}),
        )

    @property
    def n_times(self) -> int:
        return int(self.time_s.size)

    @property
    def n_states(self) -> int:
        return len(self.state_labels)

    def series(self, name: str) -> np.ndarray:
        """Return a state column or named observable without copying it."""

        if name in self._state_index:
            return self.state[:, self._state_index[name]]
        if name in self.observables:
            return self.observables[name]
        available = [*self.state_labels, *self.observables]
        raise KeyError(f"Unknown result series {name!r}. Available: {available}")

    def final_value(self, name: str) -> float:
        """Return the final value of an explicitly selected state or observable."""

        values = self.series(name)
        if values.ndim == 0:
            return float(values)
        if values.size == 0:
            raise ValueError(
                f"Cannot read final value of {name!r} from an empty result"
            )
        return float(values[-1])


__all__ = ["SimulationResult", "SimulationStatus", "to_plain_mapping"]
