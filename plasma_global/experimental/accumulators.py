"""Generic state processes and extensive surface-event accumulators."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

import numpy as np


def _owner_tuple(values: Sequence[str]) -> tuple[str, ...]:
    owners = tuple(str(value) for value in values)
    if not owners or any(not owner for owner in owners):
        raise ValueError("state owners must be nonempty strings")
    if len(set(owners)) != len(owners):
        raise ValueError("state owners must be unique")
    return owners


@dataclass(frozen=True, slots=True)
class GenericState:
    """One scalar extension state repeated over explicitly named owners."""

    state_id: str
    owners: Sequence[str] = ("global",)
    initial: float = 0.0
    lower_bound: float | None = None
    upper_bound: float | None = None

    def __post_init__(self) -> None:
        if not self.state_id:
            raise ValueError("state_id must not be empty")
        owners = _owner_tuple(self.owners)
        bounds = tuple(
            value for value in (self.lower_bound, self.upper_bound) if value is not None
        )
        if not math.isfinite(self.initial) or any(
            not math.isfinite(value) for value in bounds
        ):
            raise ValueError("state initial value and bounds must be finite")
        if self.lower_bound is not None and self.initial < self.lower_bound:
            raise ValueError("state initial value is below its lower bound")
        if self.upper_bound is not None and self.initial > self.upper_bound:
            raise ValueError("state initial value is above its upper bound")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.upper_bound < self.lower_bound
        ):
            raise ValueError("state upper bound is below its lower bound")
        object.__setattr__(self, "owners", owners)


@dataclass(frozen=True, slots=True)
class ProcessContext:
    """External scalar drivers indexed by driver name and owner."""

    drivers: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frozen: dict[str, Mapping[str, float]] = {}
        for name, values in self.drivers.items():
            converted = {str(owner): float(value) for owner, value in values.items()}
            if not name or any(
                not math.isfinite(value) for value in converted.values()
            ):
                raise ValueError("process drivers must have a name and finite values")
            frozen[str(name)] = MappingProxyType(converted)
        object.__setattr__(self, "drivers", MappingProxyType(frozen))

    def value(self, driver: str, owner: str) -> float:
        values = self.drivers.get(driver)
        if values is None:
            raise KeyError(f"process driver {driver!r} is not available")
        if owner in values:
            return values[owner]
        if "*" in values:
            return values["*"]
        raise KeyError(f"process driver {driver!r} has no value for owner {owner!r}")


class StateProcess(Protocol):
    @property
    def target_state(self) -> str: ...

    @property
    def owners(self) -> Sequence[str]: ...

    def rate(
        self, *, owner: str, current_value: float, context: ProcessContext
    ) -> float: ...


@dataclass(frozen=True, slots=True)
class ConstantSource:
    target_state: str
    rate_per_s: float
    owners: Sequence[str] = ()

    def __post_init__(self) -> None:
        if not self.target_state or not math.isfinite(self.rate_per_s):
            raise ValueError("constant source target and rate must be valid")
        object.__setattr__(self, "owners", tuple(str(owner) for owner in self.owners))

    def rate(
        self, *, owner: str, current_value: float, context: ProcessContext
    ) -> float:
        del owner, current_value, context
        return self.rate_per_s


@dataclass(frozen=True, slots=True)
class LinearRelaxation:
    target_state: str
    equilibrium: float
    time_constant_s: float
    owners: Sequence[str] = ()

    def __post_init__(self) -> None:
        if not self.target_state or not math.isfinite(self.equilibrium):
            raise ValueError("relaxation target and equilibrium must be valid")
        if not math.isfinite(self.time_constant_s) or self.time_constant_s <= 0.0:
            raise ValueError("time_constant_s must be finite and positive")
        object.__setattr__(self, "owners", tuple(str(owner) for owner in self.owners))

    def rate(
        self, *, owner: str, current_value: float, context: ProcessContext
    ) -> float:
        del owner, context
        return (self.equilibrium - current_value) / self.time_constant_s


@dataclass(frozen=True, slots=True)
class DrivenSource:
    """Source proportional to a named external driver, such as ion flux."""

    target_state: str
    driver: str
    coefficient: float
    owners: Sequence[str] = ()

    def __post_init__(self) -> None:
        if (
            not self.target_state
            or not self.driver
            or not math.isfinite(self.coefficient)
        ):
            raise ValueError(
                "driven source target, driver, and coefficient must be valid"
            )
        object.__setattr__(self, "owners", tuple(str(owner) for owner in self.owners))

    def rate(
        self, *, owner: str, current_value: float, context: ProcessContext
    ) -> float:
        del current_value
        return self.coefficient * context.value(self.driver, owner)


@dataclass(frozen=True, slots=True)
class GenericStateAccumulator:
    """Compile typed state/process declarations into a small additive RHS."""

    states: Sequence[GenericState]
    processes: Sequence[StateProcess] = ()
    labels: tuple[str, ...] = field(init=False)
    index: Mapping[tuple[str, str], int] = field(init=False, repr=False)
    _initial: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        states = tuple(self.states)
        processes = tuple(self.processes)
        if not states:
            raise ValueError("at least one generic state is required")
        if len({state.state_id for state in states}) != len(states):
            raise ValueError("generic state IDs must be unique")
        owner_by_state = {state.state_id: set(state.owners) for state in states}
        labels: list[str] = []
        index: dict[tuple[str, str], int] = {}
        initial: list[float] = []
        for state in states:
            for owner in state.owners:
                index[state.state_id, owner] = len(labels)
                labels.append(f"{state.state_id}[{owner}]")
                initial.append(state.initial)
        for process in processes:
            if process.target_state not in owner_by_state:
                raise ValueError(
                    f"process targets unknown state {process.target_state!r}"
                )
            unknown = set(process.owners) - owner_by_state[process.target_state]
            if unknown:
                raise ValueError(
                    f"process for {process.target_state!r} selects unknown owners {sorted(unknown)}"
                )
        initial_array = np.asarray(initial, dtype=float)
        initial_array.setflags(write=False)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "processes", processes)
        object.__setattr__(self, "labels", tuple(labels))
        object.__setattr__(self, "index", MappingProxyType(index))
        object.__setattr__(self, "_initial", initial_array)

    def initial_state(self) -> np.ndarray:
        return self._initial.copy()

    def rhs(
        self, values: np.ndarray, context: ProcessContext | None = None
    ) -> np.ndarray:
        state = np.asarray(values, dtype=float)
        if state.shape != self._initial.shape or not np.all(np.isfinite(state)):
            raise ValueError(
                "generic state vector has the wrong shape or non-finite values"
            )
        evaluation_context = ProcessContext() if context is None else context
        derivative = np.zeros_like(state)
        state_by_id = {item.state_id: item for item in self.states}
        for declared in self.states:
            for owner in declared.owners:
                value = float(state[self.index[declared.state_id, owner]])
                if declared.lower_bound is not None and value < declared.lower_bound:
                    raise ValueError(
                        f"{declared.state_id}[{owner}] is below its declared domain"
                    )
                if declared.upper_bound is not None and value > declared.upper_bound:
                    raise ValueError(
                        f"{declared.state_id}[{owner}] is above its declared domain"
                    )
        for process in self.processes:
            owners = tuple(process.owners) or tuple(
                state_by_id[process.target_state].owners
            )
            for owner in owners:
                position = self.index[process.target_state, owner]
                derivative[position] += process.rate(
                    owner=owner,
                    current_value=float(state[position]),
                    context=evaluation_context,
                )
        if not np.all(np.isfinite(derivative)):
            raise FloatingPointError(
                "generic process evaluation returned a non-finite rate"
            )
        return derivative


@dataclass(frozen=True, slots=True)
class SurfaceEventRate:
    """One surface event expressed as an areal event rate."""

    event_id: str
    zone_id: str
    surface_id: str
    rate_m2_s: float
    area_m2: float
    site_density_m2: float
    gas_particles_per_event: Mapping[str, float] = field(default_factory=dict)
    inventory_particles_per_event: Mapping[str, float] = field(default_factory=dict)
    film_layers_per_event: float = 0.0

    def __post_init__(self) -> None:
        if not self.event_id or not self.zone_id or not self.surface_id:
            raise ValueError("surface event, zone, and surface IDs must not be empty")
        if not math.isfinite(self.rate_m2_s) or self.rate_m2_s < 0.0:
            raise ValueError("surface event rate must be finite and nonnegative")
        if not math.isfinite(self.area_m2) or self.area_m2 <= 0.0:
            raise ValueError("surface area must be finite and positive")
        if not math.isfinite(self.site_density_m2) or self.site_density_m2 <= 0.0:
            raise ValueError("surface site density must be finite and positive")
        gas = {
            str(key): float(value)
            for key, value in self.gas_particles_per_event.items()
        }
        inventory = {
            str(key): float(value)
            for key, value in self.inventory_particles_per_event.items()
        }
        if any(not key for key in (*gas, *inventory)) or any(
            not math.isfinite(value) for value in (*gas.values(), *inventory.values())
        ):
            raise ValueError("surface event stoichiometry must be named and finite")
        if not math.isfinite(self.film_layers_per_event):
            raise ValueError("film_layers_per_event must be finite")
        object.__setattr__(self, "gas_particles_per_event", MappingProxyType(gas))
        object.__setattr__(
            self, "inventory_particles_per_event", MappingProxyType(inventory)
        )


@dataclass(frozen=True, slots=True)
class SurfaceAccumulation:
    gas_particle_rate_s: Mapping[tuple[str, str], float]
    inventory_particle_rate_s: Mapping[tuple[str, str], float]
    film_growth_m_s: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "gas_particle_rate_s",
            MappingProxyType(dict(self.gas_particle_rate_s)),
        )
        object.__setattr__(
            self,
            "inventory_particle_rate_s",
            MappingProxyType(dict(self.inventory_particle_rate_s)),
        )
        object.__setattr__(
            self, "film_growth_m_s", MappingProxyType(dict(self.film_growth_m_s))
        )
        values = (
            *self.gas_particle_rate_s.values(),
            *self.inventory_particle_rate_s.values(),
            *self.film_growth_m_s.values(),
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("surface accumulation must be finite")

    def particle_balance_s(self, species_id: str) -> float:
        gas = sum(
            value
            for (_zone_id, species), value in self.gas_particle_rate_s.items()
            if species == species_id
        )
        inventory = sum(
            value
            for (_surface_id, species), value in self.inventory_particle_rate_s.items()
            if species == species_id
        )
        return gas + inventory

    def gas_density_source(
        self, volume_m3_by_zone: Mapping[str, float]
    ) -> Mapping[tuple[str, str], float]:
        sources: dict[tuple[str, str], float] = {}
        for key, particle_rate in self.gas_particle_rate_s.items():
            zone_id, _species_id = key
            volume = float(volume_m3_by_zone[zone_id])
            if not math.isfinite(volume) or volume <= 0.0:
                raise ValueError(f"zone {zone_id!r} volume must be finite and positive")
            sources[key] = particle_rate / volume
        return MappingProxyType(sources)


@dataclass(frozen=True, slots=True)
class FilmInventoryAccumulator:
    monolayer_thickness_m: float = 3.0e-10

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.monolayer_thickness_m)
            or self.monolayer_thickness_m <= 0.0
        ):
            raise ValueError("monolayer_thickness_m must be finite and positive")

    def accumulate(self, events: Sequence[SurfaceEventRate]) -> SurfaceAccumulation:
        gas: dict[tuple[str, str], float] = {}
        inventory: dict[tuple[str, str], float] = {}
        film: dict[str, float] = {}
        for event in events:
            extensive_rate_s = event.area_m2 * event.rate_m2_s
            for species_id, coefficient in event.gas_particles_per_event.items():
                key = event.zone_id, species_id
                gas[key] = gas.get(key, 0.0) + coefficient * extensive_rate_s
            for species_id, coefficient in event.inventory_particles_per_event.items():
                key = event.surface_id, species_id
                inventory[key] = (
                    inventory.get(key, 0.0) + coefficient * extensive_rate_s
                )
            growth = (
                self.monolayer_thickness_m
                * event.film_layers_per_event
                * event.rate_m2_s
                / event.site_density_m2
            )
            film[event.surface_id] = film.get(event.surface_id, 0.0) + growth
        return SurfaceAccumulation(gas, inventory, film)


__all__ = [
    "ConstantSource",
    "DrivenSource",
    "FilmInventoryAccumulator",
    "GenericState",
    "GenericStateAccumulator",
    "LinearRelaxation",
    "ProcessContext",
    "StateProcess",
    "SurfaceAccumulation",
    "SurfaceEventRate",
]
