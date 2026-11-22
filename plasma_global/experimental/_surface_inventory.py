"""Film and wall-inventory validation, indexing, and accumulation kernels."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from plasma_global._identifiers import validate_identifier

if TYPE_CHECKING:
    from plasma_global.experimental.accumulators import (
        SurfaceEventRate,
    )


@dataclass(frozen=True, slots=True)
class RuntimeStateLayout:
    labels: tuple[str, ...]
    initial: np.ndarray
    film_index: Mapping[str, int]
    inventory_index: Mapping[tuple[str, str], int]


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
    event_key: tuple[str, str]
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


def validate_surface_event(
    *,
    reaction_id: str,
    zone_id: str,
    surface_id: str,
    area_m2: float,
    site_density_m2: float,
    inventory_particles_per_event: Mapping[str, float],
    film_layers_per_event: float,
) -> Mapping[str, float]:
    if not reaction_id or not zone_id or not surface_id:
        raise ValueError("surface event, zone, and surface IDs must not be empty")
    validate_identifier(reaction_id)
    validate_identifier(zone_id)
    validate_identifier(surface_id)
    _require_finite_positive(area_m2, "surface area must be finite and positive")
    _require_finite_positive(
        site_density_m2, "surface site density must be finite and positive"
    )
    inventory = {
        validate_identifier(key): float(value)
        for key, value in inventory_particles_per_event.items()
    }
    if any(not key for key in inventory) or any(
        not math.isfinite(value) for value in inventory.values()
    ):
        raise ValueError("surface event stoichiometry must be named and finite")
    if not math.isfinite(film_layers_per_event):
        raise ValueError("film_layers_per_event must be finite")
    return MappingProxyType(inventory)


def normalize_film_surfaces(values: Sequence[str]) -> tuple[str, ...]:
    films = tuple(validate_identifier(item) for item in values)
    if any(not item for item in films) or len(set(films)) != len(films):
        raise ValueError("film surfaces must be unique nonempty IDs")
    return films


def normalize_inventory(
    values: Mapping[str, Mapping[str, float]],
) -> dict[str, Mapping[str, float]]:
    inventory: dict[str, Mapping[str, float]] = {}
    for surface_id, entries in values.items():
        validate_identifier(surface_id)
        converted = {
            validate_identifier(key): float(value) for key, value in entries.items()
        }
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
) -> RuntimeStateLayout:
    labels: list[str] = []
    initial: list[float] = []
    film_index: dict[str, int] = {}
    inventory_index: dict[tuple[str, str], int] = {}
    for surface_id in films:
        film_index[surface_id] = len(labels)
        labels.append(f"film[{surface_id}]")
        initial.append(0.0)
    for surface_id, entries in inventory.items():
        for inventory_id, value in entries.items():
            inventory_index[surface_id, inventory_id] = len(labels)
            labels.append(f"inventory[{surface_id},{inventory_id}]")
            initial.append(value)
    if len(set(labels)) != len(labels):
        raise ValueError("experimental runtime state labels must be unique")
    if not labels:
        raise ValueError("experimental runtime accumulator needs at least one state")

    initial_array = np.asarray(initial, dtype=float)
    initial_array.setflags(write=False)
    return RuntimeStateLayout(
        labels=tuple(labels),
        initial=initial_array,
        film_index=MappingProxyType(film_index),
        inventory_index=MappingProxyType(inventory_index),
    )


def validated_runtime_state(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    state = np.asarray(values, dtype=float)
    if state.shape != reference.shape or not np.all(np.isfinite(state)):
        raise ValueError(
            "experimental runtime state has the wrong shape or non-finite values"
        )
    if np.any(state < 0.0):
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
                event_key=template.event_key,
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
    plan: SurfaceExecutionPlan, rates_m2_s: Mapping[tuple[str, str], float] | None
) -> tuple[float, ...] | None:
    rates = rates_m2_s or {}
    resolved_rates: list[float] = []
    any_nonzero = False
    for event in plan.events:
        rate = float(rates.get(event.event_key, 0.0))
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
    rates_m2_s: Mapping[tuple[str, str], float] | None,
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
