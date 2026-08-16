"""Film and wall-inventory validation, indexing, and accumulation kernels."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from plasma_global.experimental.accumulators import (
        GenericStateAccumulator,
        SurfaceEventRate,
    )


@dataclass(frozen=True, slots=True)
class RuntimeStateLayout:
    labels: tuple[str, ...]
    lower_bounds: tuple[float | None, ...]
    upper_bounds: tuple[float | None, ...]
    initial: np.ndarray
    film_index: Mapping[str, int]
    inventory_index: Mapping[tuple[str, str], int]
    generic_slice: slice


@dataclass(frozen=True, slots=True)
class _FilmTarget:
    surface_id: str
    index: int | None


@dataclass(frozen=True, slots=True)
class _InventoryTarget:
    key: tuple[str, str]
    index: int | None


@dataclass(frozen=True, slots=True)
class _SurfaceEventKernel:
    event_id: str
    film_target: _FilmTarget
    area_m2: float
    site_density_m2: float
    film_layers_per_event: float
    inventory_terms: tuple[tuple[_InventoryTarget, float], ...]


@dataclass(frozen=True, slots=True)
class SurfaceExecutionPlan:
    """Pre-indexed surface bookkeeping used by the experimental RHS."""

    events: tuple[_SurfaceEventKernel, ...]
    monolayer_thickness_m: float
    valid_monolayer_thickness: bool


def _require_finite_positive(value: float, message: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(message)


def _require_finite_nonnegative(value: float, message: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(message)


def _validate_event_stoichiometry(
    gas: Mapping[str, float], inventory: Mapping[str, float]
) -> None:
    if any(not key for key in (*gas, *inventory)) or any(
        not math.isfinite(value) for value in (*gas.values(), *inventory.values())
    ):
        raise ValueError("surface event stoichiometry must be named and finite")


def validate_surface_event(
    *,
    event_id: str,
    zone_id: str,
    surface_id: str,
    rate_m2_s: float,
    area_m2: float,
    site_density_m2: float,
    gas_particles_per_event: Mapping[str, float],
    inventory_particles_per_event: Mapping[str, float],
    film_layers_per_event: float,
) -> tuple[Mapping[str, float], Mapping[str, float]]:
    if not event_id or not zone_id or not surface_id:
        raise ValueError("surface event, zone, and surface IDs must not be empty")
    _require_finite_nonnegative(
        rate_m2_s, "surface event rate must be finite and nonnegative"
    )
    _require_finite_positive(area_m2, "surface area must be finite and positive")
    _require_finite_positive(
        site_density_m2, "surface site density must be finite and positive"
    )
    gas = {str(key): float(value) for key, value in gas_particles_per_event.items()}
    inventory = {
        str(key): float(value) for key, value in inventory_particles_per_event.items()
    }
    _validate_event_stoichiometry(gas, inventory)
    if not math.isfinite(film_layers_per_event):
        raise ValueError("film_layers_per_event must be finite")
    return MappingProxyType(gas), MappingProxyType(inventory)


def freeze_surface_accumulation(
    gas_particle_rate_s: Mapping[tuple[str, str], float],
    inventory_particle_rate_s: Mapping[tuple[str, str], float],
    film_growth_m_s: Mapping[str, float],
) -> tuple[
    Mapping[tuple[str, str], float],
    Mapping[tuple[str, str], float],
    Mapping[str, float],
]:
    gas = MappingProxyType(dict(gas_particle_rate_s))
    inventory = MappingProxyType(dict(inventory_particle_rate_s))
    film = MappingProxyType(dict(film_growth_m_s))
    values = (*gas.values(), *inventory.values(), *film.values())
    if any(not math.isfinite(value) for value in values):
        raise ValueError("surface accumulation must be finite")
    return gas, inventory, film


def particle_balance(
    species_id: str,
    gas_particle_rate_s: Mapping[tuple[str, str], float],
    inventory_particle_rate_s: Mapping[tuple[str, str], float],
) -> float:
    gas = sum(
        value
        for (_zone_id, species), value in gas_particle_rate_s.items()
        if species == species_id
    )
    inventory = sum(
        value
        for (_surface_id, species), value in inventory_particle_rate_s.items()
        if species == species_id
    )
    return gas + inventory


def gas_density_sources(
    gas_particle_rate_s: Mapping[tuple[str, str], float],
    volume_m3_by_zone: Mapping[str, float],
) -> Mapping[tuple[str, str], float]:
    sources: dict[tuple[str, str], float] = {}
    for key, particle_rate in gas_particle_rate_s.items():
        zone_id, _species_id = key
        volume = float(volume_m3_by_zone[zone_id])
        if not math.isfinite(volume) or volume <= 0.0:
            raise ValueError(f"zone {zone_id!r} volume must be finite and positive")
        sources[key] = particle_rate / volume
    return MappingProxyType(sources)


def validate_monolayer_thickness(value: float) -> None:
    _require_finite_positive(value, "monolayer_thickness_m must be finite and positive")


def accumulate_surface_events(
    events: Sequence[SurfaceEventRate], monolayer_thickness_m: float
) -> tuple[
    dict[tuple[str, str], float], dict[tuple[str, str], float], dict[str, float]
]:
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
            inventory[key] = inventory.get(key, 0.0) + coefficient * extensive_rate_s
        growth = (
            monolayer_thickness_m
            * event.film_layers_per_event
            * event.rate_m2_s
            / event.site_density_m2
        )
        film[event.surface_id] = film.get(event.surface_id, 0.0) + growth
    return gas, inventory, film


def normalize_film_surfaces(values: Sequence[str]) -> tuple[str, ...]:
    films = tuple(str(item) for item in values)
    if any(not item for item in films) or len(set(films)) != len(films):
        raise ValueError("film surfaces must be unique nonempty IDs")
    return films


def normalize_inventory(
    values: Mapping[str, Mapping[str, float]],
) -> dict[str, Mapping[str, float]]:
    inventory: dict[str, Mapping[str, float]] = {}
    for surface_id, entries in values.items():
        converted = {str(key): float(value) for key, value in entries.items()}
        if (
            not surface_id
            or any(not key for key in converted)
            or any(
                not math.isfinite(value) or value < 0.0 for value in converted.values()
            )
        ):
            raise ValueError("initial wall inventory must be named and nonnegative")
        inventory[str(surface_id)] = MappingProxyType(converted)
    return inventory


def build_runtime_layout(
    films: tuple[str, ...],
    inventory: Mapping[str, Mapping[str, float]],
    generic: GenericStateAccumulator | None,
) -> RuntimeStateLayout:
    labels: list[str] = []
    initial: list[float] = []
    lower_bounds: list[float | None] = []
    upper_bounds: list[float | None] = []
    film_index: dict[str, int] = {}
    inventory_index: dict[tuple[str, str], int] = {}
    for surface_id in films:
        film_index[surface_id] = len(labels)
        labels.append(f"film[{surface_id}]")
        initial.append(0.0)
        lower_bounds.append(0.0)
        upper_bounds.append(None)
    for surface_id, entries in inventory.items():
        for inventory_id, value in entries.items():
            inventory_index[surface_id, inventory_id] = len(labels)
            labels.append(f"inventory[{surface_id},{inventory_id}]")
            initial.append(value)
            lower_bounds.append(0.0)
            upper_bounds.append(None)
    generic_start = len(labels)
    if generic is not None:
        for state in generic.states:
            for owner in state.owners:
                labels.append(f"extra[{state.state_id},{owner}]")
        initial.extend(generic.initial_state().tolist())
        lower_bounds.extend(generic.lower_bounds)
        upper_bounds.extend(generic.upper_bounds)
    if len(set(labels)) != len(labels):
        raise ValueError("experimental runtime state labels must be unique")
    if not labels:
        raise ValueError("experimental runtime accumulator needs at least one state")

    initial_array = np.asarray(initial, dtype=float)
    initial_array.setflags(write=False)
    return RuntimeStateLayout(
        labels=tuple(labels),
        lower_bounds=tuple(lower_bounds),
        upper_bounds=tuple(upper_bounds),
        initial=initial_array,
        film_index=MappingProxyType(film_index),
        inventory_index=MappingProxyType(inventory_index),
        generic_slice=slice(generic_start, len(labels)),
    )


def validated_runtime_state(
    values: np.ndarray, reference: np.ndarray, physical_stop: int
) -> np.ndarray:
    state = np.asarray(values, dtype=float)
    if state.shape != reference.shape or not np.all(np.isfinite(state)):
        raise ValueError(
            "experimental runtime state has the wrong shape or non-finite values"
        )
    if np.any(state[:physical_stop] < 0.0):
        raise ValueError("film thickness and wall inventory must be nonnegative")
    return state


def compile_surface_execution_plan(
    templates: Sequence[SurfaceEventRate],
    *,
    film_index: Mapping[str, int],
    inventory_index: Mapping[tuple[str, str], int],
    monolayer_thickness_m: float,
) -> SurfaceExecutionPlan:
    """Resolve immutable event metadata and state indices once at compilation."""

    film_targets: dict[str, _FilmTarget] = {}
    inventory_targets: dict[tuple[str, str], _InventoryTarget] = {}
    events: list[_SurfaceEventKernel] = []
    for template in templates:
        film_target = film_targets.setdefault(
            template.surface_id,
            _FilmTarget(
                surface_id=template.surface_id,
                index=film_index.get(template.surface_id),
            ),
        )
        inventory_terms: list[tuple[_InventoryTarget, float]] = []
        for species_id, coefficient in template.inventory_particles_per_event.items():
            key = template.surface_id, species_id
            target = inventory_targets.setdefault(
                key,
                _InventoryTarget(key=key, index=inventory_index.get(key)),
            )
            inventory_terms.append((target, coefficient))
        events.append(
            _SurfaceEventKernel(
                event_id=template.event_id,
                film_target=film_target,
                area_m2=template.area_m2,
                site_density_m2=template.site_density_m2,
                film_layers_per_event=template.film_layers_per_event,
                inventory_terms=tuple(inventory_terms),
            )
        )
    return SurfaceExecutionPlan(
        events=tuple(events),
        monolayer_thickness_m=monolayer_thickness_m,
        valid_monolayer_thickness=(
            math.isfinite(monolayer_thickness_m) and monolayer_thickness_m > 0.0
        ),
    )


def _resolved_surface_rates(
    plan: SurfaceExecutionPlan, rates_m2_s: Mapping[str, float] | None
) -> tuple[float, ...] | None:
    rates = rates_m2_s or {}
    resolved_rates: list[float] = []
    any_nonzero = False
    for event in plan.events:
        rate = float(rates.get(event.event_id, 0.0))
        _require_finite_nonnegative(
            rate, "surface event rate must be finite and nonnegative"
        )
        resolved_rates.append(rate)
        any_nonzero = any_nonzero or rate != 0.0
    return tuple(resolved_rates) if any_nonzero else None


def _accumulated_surface_rates(
    plan: SurfaceExecutionPlan, rates: tuple[float, ...]
) -> tuple[dict[_FilmTarget, float], dict[_InventoryTarget, float]]:
    film: dict[_FilmTarget, float] = {}
    inventory: dict[_InventoryTarget, float] = {}
    for event, rate in zip(plan.events, rates, strict=True):
        extensive_rate_s = event.area_m2 * rate
        for target, coefficient in event.inventory_terms:
            inventory[target] = (
                inventory.get(target, 0.0) + coefficient * extensive_rate_s
            )
        growth = (
            plan.monolayer_thickness_m
            * event.film_layers_per_event
            * rate
            / event.site_density_m2
        )
        film[event.film_target] = film.get(event.film_target, 0.0) + growth

    if any(not math.isfinite(value) for value in (*inventory.values(), *film.values())):
        raise ValueError("surface accumulation must be finite")
    return film, inventory


def _add_film_rates(derivative: np.ndarray, film: Mapping[_FilmTarget, float]) -> None:
    for target, growth_m_s in film.items():
        if target.index is None:
            if growth_m_s != 0.0:
                raise ValueError(
                    "surface event produced undeclared film state "
                    f"{target.surface_id!r}"
                )
            continue
        derivative[target.index] += growth_m_s


def _add_inventory_rates(
    derivative: np.ndarray, inventory: Mapping[_InventoryTarget, float]
) -> None:
    for target, particle_rate_s in inventory.items():
        if target.index is None:
            if particle_rate_s != 0.0:
                raise ValueError(
                    f"surface event produced undeclared inventory state {target.key!r}"
                )
            continue
        derivative[target.index] += particle_rate_s


def add_surface_rates(
    derivative: np.ndarray,
    plan: SurfaceExecutionPlan,
    rates_m2_s: Mapping[str, float] | None,
) -> None:
    """Accumulate pre-indexed film/inventory rates without transient DTOs."""

    if not plan.events:
        return
    rates = _resolved_surface_rates(plan, rates_m2_s)
    if not plan.valid_monolayer_thickness:
        raise ValueError("monolayer_thickness_m must be finite and positive")
    if rates is None:
        return
    film, inventory = _accumulated_surface_rates(plan, rates)
    _add_film_rates(derivative, film)
    _add_inventory_rates(derivative, inventory)
