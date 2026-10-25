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

_PUBLIC_UNITS = {
    "electron_density_m3": "m^-3",
    "charge_residual_m3": "m^-3",
    "mean_energy_eV": "eV",
    "electron_temperature_eV": "eV",
    "gas_temperature_K": "K",
    "reduced_field_Td": "Td",
    "absorbed_power_W": "W",
    "pressure_Pa": "Pa",
    "partial_pressure_Pa": "Pa",
    "neutral_density_m3": "m^-3",
    "heavy_density_m3": "m^-3",
    "mole_fraction": "1",
    "net_production_m3_s": "m^-3 s^-1",
    "incident_flux_m2_s": "m^-2 s^-1",
    "surface_temperature_K": "K",
    "wall_sheath_energy_eV": "eV",
}

_PUBLIC_UNITS.update(
    {
        f"{domain}_reaction_rate_{suffix}": unit
        for domain, suffix, unit in (
            ("gas", "m3_s", "m^-3 s^-1"),
            ("surface", "m2_s", "m^-2 s^-1"),
            ("boundary", "m2_s", "m^-2 s^-1"),
        )
    }
)

_ORIGINAL_QUANTITIES = {
    "electron_density_m3",
    "charge_residual_m3",
    "mean_energy_eV",
    "electron_temperature_eV",
    "gas_temperature_K",
    "reduced_field_Td",
    "absorbed_power_W",
}


def _legacy_unspecified_unit(label: str, unit: str) -> bool:
    quantity = label.partition("[")[0]
    added = quantity in _PUBLIC_UNITS and quantity not in _ORIGINAL_QUANTITIES
    return unit == "unspecified" and (added or quantity in {"film", "inventory"})


_PUBLIC_UNITS.update(
    {f"{domain}_event_rate_s_inv": "s^-1" for domain in ("gas", "surface", "boundary")}
)
_PUBLIC_UNITS.update(
    {f"{domain}_event_count": "1" for domain in ("gas", "surface", "boundary")}
)
_PUBLIC_UNITS.update(
    dict.fromkeys(
        (
            "electron_power_W_m3",
            "gas_power_W_m3",
            "reaction_energy_loss_W_m3",
            "wall_energy_loss_W_m3",
            "elastic_heating_W_m3",
            "gas_reaction_heating_W_m3",
            "surface_reaction_heating_W_m3",
            "transport_electron_power_W_m3",
            "transport_heavy_power_W_m3",
            "inlet_heavy_power_W_m3",
            "wall_heavy_power_W_m3",
        ),
        "W m^-3",
    )
)


_STATE_UNITS = {
    "n[": "m^-3",
    "electron_energy[": "J m^-3",
    "gas_internal_energy[": "J m^-3",
    "coverage[": "1",
    "film[": "m",
    "inventory[": "1",
}


def _canonical_unit(label: str) -> str:
    for prefix, unit in _STATE_UNITS.items():
        if label.startswith(prefix):
            return unit
    return _PUBLIC_UNITS.get(label.partition("[")[0], "unspecified")


def series_unit(label: str, declared: Mapping[str, str] | None = None) -> str:
    """Use explicit units or the public catalog; never guess a custom suffix."""

    canonical = _canonical_unit(label)
    if declared is None or label not in declared:
        return canonical
    unit = declared[label]
    if _legacy_unspecified_unit(label, unit):
        return unit
    if canonical != "unspecified" and unit != canonical:
        raise ValueError(
            f"series {label!r} requires canonical unit {canonical!r}; "
            "no unit conversion is performed"
        )
    return unit


def _readonly_float_array(value: Any, *, name: str) -> np.ndarray:
    try:
        array = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    array.setflags(write=False)
    return array


def _finite_float_array(value: Any, *, name: str) -> np.ndarray:
    array = _readonly_float_array(value, name=name)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validated_metadata_float(value: float, *, path: str) -> float:
    if not np.isfinite(value):
        raise ValueError(f"{path} must contain only finite numeric values")
    return value


def _exact_name(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip() or "\x00" in value:
        raise ValueError(
            f"{name} must be nonempty without surrounding whitespace or NUL"
        )
    return value


def _freeze_metadata_mapping(
    value: Mapping[Any, Any], *, path: str
) -> Mapping[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_key, item in value.items():
        key = _exact_name(raw_key, name=f"{path} key")
        normalized[key] = _freeze_metadata_value(item, path=f"{path}.{key}")
    return MappingProxyType(normalized)


def _freeze_metadata_value(value: Any, *, path: str) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return _validated_metadata_float(value, path=path)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.ndarray):
        return _freeze_metadata_value(value.tolist(), path=path)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata_value(item, path=f"{path}[]") for item in value)
    if isinstance(value, Mapping):
        return _freeze_metadata_mapping(value, path=path)
    raise TypeError(f"{path} contains unsupported value type {type(value).__name__}")


def _freeze_mapping(value: Mapping[str, Any] | None, *, name: str) -> Mapping[str, Any]:
    if value is not None and not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    frozen = _freeze_metadata_value({} if value is None else value, path=name)
    if not isinstance(frozen, Mapping):
        raise RuntimeError("mapping metadata did not remain a mapping")
    return frozen


def _validated_time(value: Any) -> np.ndarray:
    time_s = _finite_float_array(value, name="time_s")
    if time_s.ndim != 1:
        raise ValueError(f"time_s must be one-dimensional, got shape {time_s.shape}")
    if time_s.size > 1 and np.any(np.diff(time_s) <= 0.0):
        raise ValueError("time_s must be strictly increasing")
    return time_s


def _validated_labels(values: tuple[object, ...]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("state_labels must be a sequence of names")
    labels = tuple(_exact_name(label, name="state label") for label in values)
    if len(set(labels)) != len(labels):
        raise ValueError("state_labels must be unique")
    return labels


def _validated_state(value: Any, expected_shape: tuple[int, int]) -> np.ndarray:
    state = _finite_float_array(value, name="state")
    if state.ndim != 2:
        raise ValueError(
            f"state must be two-dimensional and time-major, got shape {state.shape}"
        )
    if state.shape != expected_shape:
        raise ValueError(
            f"state must have time-major shape {expected_shape}, got {state.shape}"
        )
    return state


def _validated_observable_name(value: Any, state_labels: tuple[str, ...]) -> str:
    name = _exact_name(value, name="observable name")
    if "/" in name or "\x00" in name or name in {".", ".."}:
        raise ValueError(
            f"observable name {name!r} is not safe for the result HDF5 layout"
        )
    if name in state_labels:
        raise ValueError(f"observable name {name!r} conflicts with a state label")
    return name


def _validated_observables(
    values: Mapping[str, np.ndarray],
    *,
    time_count: int,
    state_labels: tuple[str, ...],
) -> Mapping[str, np.ndarray]:
    if not isinstance(values, Mapping):
        raise TypeError("observables must be a mapping")
    observables: dict[str, np.ndarray] = {}
    for raw_name, raw_values in dict(values or {}).items():
        name = _validated_observable_name(raw_name, state_labels)
        observable = _finite_float_array(raw_values, name=f"observables[{name!r}]")
        if observable.ndim == 1 and observable.shape != (time_count,):
            raise ValueError(
                f"observable {name!r} must have shape ({time_count},) or be scalar, "
                f"got {observable.shape}"
            )
        if observable.ndim not in {0, 1}:
            raise ValueError(
                f"observable {name!r} must be scalar or one-dimensional, got "
                f"{observable.ndim} dimensions"
            )
        observables[name] = observable
    return MappingProxyType(observables)


def to_plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a serialization-friendly copy of frozen result metadata."""

    def plain(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): plain(child) for key, child in item.items()}
        if isinstance(item, tuple):
            return [plain(child) for child in item]
        return item

    return plain(value)


def _validate_series_units(metadata: Mapping[str, Any], names: set[str]) -> None:
    declared = metadata.get("series_units", {})
    if not isinstance(declared, Mapping):
        raise TypeError("metadata.series_units must be a mapping")
    if unknown := set(declared) - names:
        raise ValueError(f"series_units references unknown series: {sorted(unknown)}")
    for unit in declared.values():
        _exact_name(unit, name="series unit")
    for name in declared:
        series_unit(name, declared)


def _validate_summary_series(metadata: Mapping[str, Any], names: set[str]) -> None:
    selected = metadata.get("summary_series", ())
    if not isinstance(selected, tuple):
        raise TypeError("metadata.summary_series must be a sequence of names")
    for name in selected:
        _exact_name(name, name="summary series")
    if len(set(selected)) != len(selected):
        raise ValueError("metadata.summary_series must not contain duplicates")
    if unknown := set(selected) - names:
        raise ValueError(
            f"metadata.summary_series references unknown series: {sorted(unknown)}"
        )


@dataclass(frozen=True, slots=True)
class SimulationStatus:
    """Terminal state of a simulation independent of a solver implementation."""

    success: bool | None = True
    code: str = "completed"
    message: str = ""

    def __post_init__(self) -> None:
        if self.success is not None and not isinstance(self.success, bool):
            raise TypeError("SimulationStatus.success must be bool or None (unknown)")
        _exact_name(self.code, name="SimulationStatus.code")
        if not isinstance(self.message, str):
            raise TypeError("SimulationStatus.message must be a string")
        if self.code == "completed" and self.success is not True:
            raise ValueError("completed status requires success=True")
        if self.code == "failed" and self.success is not False:
            raise ValueError("failed status requires success=False")

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
        time_s = _validated_time(self.time_s)
        labels = _validated_labels(self.state_labels)
        state = _validated_state(self.state, (time_s.size, len(labels)))
        observables = _validated_observables(
            self.observables,
            time_count=time_s.size,
            state_labels=labels,
        )
        if not isinstance(self.status, SimulationStatus):
            raise TypeError("status must be a SimulationStatus")

        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "state_labels", labels)
        object.__setattr__(self, "observables", observables)
        object.__setattr__(
            self,
            "solver_stats",
            _freeze_mapping(self.solver_stats, name="solver_stats"),
        )
        metadata = _freeze_mapping(self.metadata, name="metadata")
        names = set(labels) | set(observables)
        _validate_series_units(metadata, names)
        _validate_summary_series(metadata, names)
        object.__setattr__(self, "metadata", metadata)
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


__all__ = ["SimulationResult", "SimulationStatus", "series_unit", "to_plain_mapping"]
