"""Public declarations for generic states and surface-event accumulation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from plasma_global.experimental._declarative_state import (
    build_generic_state_layout,
    evaluate_generic_state_rhs,
    freeze_process_drivers,
    normalize_process_owners,
    validate_constant_source,
    validate_driven_source,
    validate_relaxation,
    validate_state_declaration,
)
from plasma_global.experimental._surface_inventory import (
    accumulate_surface_events,
    freeze_surface_accumulation,
    gas_density_sources,
    particle_balance,
    validate_monolayer_thickness,
    validate_surface_event,
)


@dataclass(frozen=True, slots=True)
class GenericState:
    """One scalar extension state repeated over explicitly named owners."""

    state_id: str
    owners: Sequence[str] = ("global",)
    initial: float = 0.0
    lower_bound: float | None = None
    upper_bound: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "owners",
            validate_state_declaration(
                self.state_id,
                self.owners,
                self.initial,
                self.lower_bound,
                self.upper_bound,
            ),
        )


@dataclass(frozen=True, slots=True)
class ProcessContext:
    """External scalar drivers indexed by driver name and owner."""

    drivers: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "drivers", freeze_process_drivers(self.drivers))

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
        validate_constant_source(self.target_state, self.rate_per_s)
        object.__setattr__(self, "owners", normalize_process_owners(self.owners))

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
        validate_relaxation(self.target_state, self.equilibrium, self.time_constant_s)
        object.__setattr__(self, "owners", normalize_process_owners(self.owners))

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
        validate_driven_source(self.target_state, self.driver, self.coefficient)
        object.__setattr__(self, "owners", normalize_process_owners(self.owners))

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
    lower_bounds: tuple[float | None, ...] = field(init=False)
    upper_bounds: tuple[float | None, ...] = field(init=False)
    index: Mapping[tuple[str, str], int] = field(init=False, repr=False)
    _initial: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        layout = build_generic_state_layout(self.states, self.processes)
        object.__setattr__(self, "states", layout.states)
        object.__setattr__(self, "processes", layout.processes)
        object.__setattr__(self, "labels", layout.labels)
        object.__setattr__(self, "lower_bounds", layout.lower_bounds)
        object.__setattr__(self, "upper_bounds", layout.upper_bounds)
        object.__setattr__(self, "index", layout.index)
        object.__setattr__(self, "_initial", layout.initial)

    def initial_state(self) -> np.ndarray:
        return self._initial.copy()

    def rhs(
        self, values: np.ndarray, context: ProcessContext | None = None
    ) -> np.ndarray:
        return evaluate_generic_state_rhs(
            values,
            reference=self._initial,
            states=self.states,
            processes=self.processes,
            index=self.index,
            context=context,
            context_factory=ProcessContext,
        )


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
        gas, inventory = validate_surface_event(
            event_id=self.event_id,
            zone_id=self.zone_id,
            surface_id=self.surface_id,
            rate_m2_s=self.rate_m2_s,
            area_m2=self.area_m2,
            site_density_m2=self.site_density_m2,
            gas_particles_per_event=self.gas_particles_per_event,
            inventory_particles_per_event=self.inventory_particles_per_event,
            film_layers_per_event=self.film_layers_per_event,
        )
        object.__setattr__(self, "gas_particles_per_event", gas)
        object.__setattr__(self, "inventory_particles_per_event", inventory)


@dataclass(frozen=True, slots=True)
class SurfaceAccumulation:
    gas_particle_rate_s: Mapping[tuple[str, str], float]
    inventory_particle_rate_s: Mapping[tuple[str, str], float]
    film_growth_m_s: Mapping[str, float]

    def __post_init__(self) -> None:
        gas, inventory, film = freeze_surface_accumulation(
            self.gas_particle_rate_s,
            self.inventory_particle_rate_s,
            self.film_growth_m_s,
        )
        object.__setattr__(self, "gas_particle_rate_s", gas)
        object.__setattr__(self, "inventory_particle_rate_s", inventory)
        object.__setattr__(self, "film_growth_m_s", film)

    def particle_balance_s(self, species_id: str) -> float:
        return particle_balance(
            species_id,
            self.gas_particle_rate_s,
            self.inventory_particle_rate_s,
        )

    def gas_density_source(
        self, volume_m3_by_zone: Mapping[str, float]
    ) -> Mapping[tuple[str, str], float]:
        return gas_density_sources(self.gas_particle_rate_s, volume_m3_by_zone)


@dataclass(frozen=True, slots=True)
class FilmInventoryAccumulator:
    monolayer_thickness_m: float = 3.0e-10

    def __post_init__(self) -> None:
        validate_monolayer_thickness(self.monolayer_thickness_m)

    def accumulate(self, events: Sequence[SurfaceEventRate]) -> SurfaceAccumulation:
        gas, inventory, film = accumulate_surface_events(
            events, self.monolayer_thickness_m
        )
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
