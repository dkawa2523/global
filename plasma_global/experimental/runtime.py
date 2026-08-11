"""Runtime adapter for opt-in generic and surface-event accumulator states."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import numpy as np

from plasma_global.experimental.accumulators import (
    ConstantSource,
    DrivenSource,
    FilmInventoryAccumulator,
    GenericState,
    GenericStateAccumulator,
    LinearRelaxation,
    ProcessContext,
    SurfaceEventRate,
)


def _mapping(value: object, where: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{where} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _only_keys(values: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"{where} contains unknown keys: {', '.join(unknown)}")


def _selected_owners(
    values: Mapping[str, Any],
    *,
    scope: str,
    zone_ids: tuple[str, ...],
    surface_ids: tuple[str, ...],
    where: str,
) -> tuple[str, ...]:
    if scope == "global":
        if values.get("zones") or values.get("surfaces"):
            raise ValueError(f"{where} global scope cannot select zones or surfaces")
        return ("global",)
    selection_key = "zones" if scope == "zone" else "surfaces"
    rejected_key = "surfaces" if scope == "zone" else "zones"
    if values.get(rejected_key):
        raise ValueError(f"{where} {scope} scope cannot select {rejected_key}")
    available = zone_ids if scope == "zone" else surface_ids
    raw = values.get(selection_key)
    selected = available if raw in (None, ()) else tuple(str(item) for item in raw)
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"{where} selects unknown owners {sorted(unknown)}")
    if not selected:
        raise ValueError(f"{where} selects no owners")
    if len(set(selected)) != len(selected):
        raise ValueError(f"{where} contains duplicate owners")
    return selected


def compile_generic_state_accumulator(
    specification: Mapping[str, Any],
    *,
    zone_ids: Sequence[str],
    surface_ids: Sequence[str],
) -> GenericStateAccumulator | None:
    """Compile strict chemistry-declared state/process mappings.

    The canonical experimental mapping uses ``state_variables`` and
    ``processes`` dictionaries. Supported processes deliberately match the
    previous feature set: constant source, linear relaxation, and a surface
    ion-flux-driven source.
    """

    raw = _mapping(specification, "chemistry.experimental")
    _only_keys(raw, {"state_variables", "processes"}, "chemistry.experimental")
    state_entries = _mapping(
        raw.get("state_variables"), "chemistry.experimental.state_variables"
    )
    process_entries = _mapping(raw.get("processes"), "chemistry.experimental.processes")
    if not state_entries:
        if process_entries:
            raise ValueError("experimental processes require state_variables")
        return None

    zones = tuple(str(item) for item in zone_ids)
    surfaces = tuple(str(item) for item in surface_ids)
    states: list[GenericState] = []
    scopes: dict[str, str] = {}
    owners_by_state: dict[str, tuple[str, ...]] = {}
    for state_id, entry in state_entries.items():
        where = f"chemistry.experimental.state_variables.{state_id}"
        values = _mapping(entry, where)
        _only_keys(
            values,
            {
                "scope",
                "initial",
                "lower_bound",
                "upper_bound",
                "zones",
                "surfaces",
            },
            where,
        )
        scope = str(values.get("scope", "")).strip().lower()
        if scope not in {"global", "zone", "surface"}:
            raise ValueError(f"{where}.scope must be global, zone, or surface")
        owners = _selected_owners(
            values,
            scope=scope,
            zone_ids=zones,
            surface_ids=surfaces,
            where=where,
        )
        lower = values.get("lower_bound", 0.0)
        upper = values.get("upper_bound")
        states.append(
            GenericState(
                state_id=state_id,
                owners=owners,
                initial=float(values.get("initial", 0.0)),
                lower_bound=None if lower is None else float(lower),
                upper_bound=None if upper is None else float(upper),
            )
        )
        scopes[state_id] = scope
        owners_by_state[state_id] = owners

    processes: list[ConstantSource | LinearRelaxation | DrivenSource] = []
    for process_id, entry in process_entries.items():
        where = f"chemistry.experimental.processes.{process_id}"
        values = _mapping(entry, where)
        kind = str(values.get("kind", "")).strip().lower()
        target = str(values.get("target", "")).strip()
        if target not in owners_by_state:
            raise ValueError(f"{where} targets unknown state {target!r}")
        common = {"kind", "target", "zones", "surfaces"}
        if kind == "source":
            _only_keys(values, common | {"value"}, where)
        elif kind == "relaxation":
            _only_keys(values, common | {"equilibrium", "tau_s"}, where)
        elif kind == "ion_flux_source":
            _only_keys(values, common | {"coefficient"}, where)
            if scopes[target] != "surface":
                raise ValueError(
                    f"{where} ion_flux_source target must use surface scope"
                )
        else:
            raise ValueError(
                f"{where}.kind must be source, relaxation, or ion_flux_source"
            )

        selected = _selected_owners(
            values,
            scope=scopes[target],
            zone_ids=(owners_by_state[target] if scopes[target] == "zone" else zones),
            surface_ids=(
                owners_by_state[target] if scopes[target] == "surface" else surfaces
            ),
            where=where,
        )
        process_owners = () if selected == owners_by_state[target] else tuple(selected)
        if kind == "source":
            if "value" not in values:
                raise ValueError(f"{where}.value is required")
            processes.append(
                ConstantSource(target, float(values["value"]), process_owners)
            )
        elif kind == "relaxation":
            if "tau_s" not in values:
                raise ValueError(f"{where}.tau_s is required")
            processes.append(
                LinearRelaxation(
                    target,
                    float(values.get("equilibrium", 0.0)),
                    float(values["tau_s"]),
                    process_owners,
                )
            )
        else:
            if "coefficient" not in values:
                raise ValueError(f"{where}.coefficient is required")
            processes.append(
                DrivenSource(
                    target,
                    "ion_flux_m2_s",
                    float(values["coefficient"]),
                    process_owners,
                )
            )
    return GenericStateAccumulator(tuple(states), tuple(processes))


@dataclass(frozen=True, slots=True)
class ExperimentalRuntimeAccumulator:
    """One independent ODE block for film, inventory, and generic state.

    Film thickness and wall inventory are extensive bookkeeping states, not
    surface site coverages. Surface chemistry remains the sole owner of the
    coverage/site balance.
    """

    generic: GenericStateAccumulator | None = None
    film_surfaces: Sequence[str] = ()
    initial_inventory_by_surface: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict
    )
    surface_event_templates: Sequence[SurfaceEventRate] = ()
    monolayer_thickness_m: float = 3.0e-10
    labels: tuple[str, ...] = field(init=False)
    lower_bounds: tuple[float | None, ...] = field(init=False)
    _initial: np.ndarray = field(init=False, repr=False)
    _film_index: Mapping[str, int] = field(init=False, repr=False)
    _inventory_index: Mapping[tuple[str, str], int] = field(init=False, repr=False)
    _generic_slice: slice = field(init=False, repr=False)

    def __post_init__(self) -> None:
        films = tuple(str(item) for item in self.film_surfaces)
        if any(not item for item in films) or len(set(films)) != len(films):
            raise ValueError("film surfaces must be unique nonempty IDs")
        inventory: dict[str, Mapping[str, float]] = {}
        for surface_id, entries in self.initial_inventory_by_surface.items():
            converted = {str(key): float(value) for key, value in entries.items()}
            if (
                not surface_id
                or any(not key for key in converted)
                or any(
                    not math.isfinite(value) or value < 0.0
                    for value in converted.values()
                )
            ):
                raise ValueError("initial wall inventory must be named and nonnegative")
            inventory[str(surface_id)] = MappingProxyType(converted)
        templates = tuple(self.surface_event_templates)
        event_ids = tuple(event.event_id for event in templates)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("surface-event template IDs must be unique")

        labels: list[str] = []
        initial: list[float] = []
        lower_bounds: list[float | None] = []
        film_index: dict[str, int] = {}
        inventory_index: dict[tuple[str, str], int] = {}
        for surface_id in films:
            film_index[surface_id] = len(labels)
            labels.append(f"film[{surface_id}]")
            initial.append(0.0)
            lower_bounds.append(0.0)
        for surface_id, entries in inventory.items():
            for inventory_id, value in entries.items():
                inventory_index[surface_id, inventory_id] = len(labels)
                labels.append(f"inventory[{surface_id},{inventory_id}]")
                initial.append(value)
                lower_bounds.append(0.0)
        generic_start = len(labels)
        if self.generic is not None:
            for state in self.generic.states:
                for owner in state.owners:
                    labels.append(f"extra[{state.state_id},{owner}]")
                    lower_bounds.append(state.lower_bound)
            initial.extend(self.generic.initial_state().tolist())
        if len(set(labels)) != len(labels):
            raise ValueError("experimental runtime state labels must be unique")
        if not labels:
            raise ValueError(
                "experimental runtime accumulator needs at least one state"
            )

        initial_array = np.asarray(initial, dtype=float)
        initial_array.setflags(write=False)
        object.__setattr__(self, "film_surfaces", films)
        object.__setattr__(
            self, "initial_inventory_by_surface", MappingProxyType(inventory)
        )
        object.__setattr__(self, "surface_event_templates", templates)
        object.__setattr__(self, "labels", tuple(labels))
        object.__setattr__(self, "lower_bounds", tuple(lower_bounds))
        object.__setattr__(self, "_initial", initial_array)
        object.__setattr__(self, "_film_index", MappingProxyType(film_index))
        object.__setattr__(self, "_inventory_index", MappingProxyType(inventory_index))
        object.__setattr__(self, "_generic_slice", slice(generic_start, len(labels)))

    def initial_state(self) -> np.ndarray:
        return self._initial.copy()

    def rhs(
        self,
        values: np.ndarray,
        *,
        drivers: Mapping[str, Mapping[str, float]] | None = None,
        surface_rates_m2_s: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        state = np.asarray(values, dtype=float)
        if state.shape != self._initial.shape or not np.all(np.isfinite(state)):
            raise ValueError(
                "experimental runtime state has the wrong shape or non-finite values"
            )
        physical_stop = self._generic_slice.start
        if np.any(state[:physical_stop] < 0.0):
            raise ValueError("film thickness and wall inventory must be nonnegative")

        derivative = np.zeros_like(state)
        if self.generic is not None:
            derivative[self._generic_slice] = self.generic.rhs(
                state[self._generic_slice], ProcessContext(drivers or {})
            )

        rates = surface_rates_m2_s or {}
        events = tuple(
            replace(template, rate_m2_s=float(rates.get(template.event_id, 0.0)))
            for template in self.surface_event_templates
        )
        if events:
            accumulation = FilmInventoryAccumulator(
                self.monolayer_thickness_m
            ).accumulate(events)
            for surface_id, growth_m_s in accumulation.film_growth_m_s.items():
                index = self._film_index.get(surface_id)
                if index is None:
                    if growth_m_s != 0.0:
                        raise ValueError(
                            f"surface event produced undeclared film state {surface_id!r}"
                        )
                    continue
                derivative[index] += growth_m_s
            for key, particle_rate_s in accumulation.inventory_particle_rate_s.items():
                index = self._inventory_index.get(key)
                if index is None:
                    if particle_rate_s != 0.0:
                        raise ValueError(
                            f"surface event produced undeclared inventory state {key!r}"
                        )
                    continue
                derivative[index] += particle_rate_s
        if not np.all(np.isfinite(derivative)):
            raise FloatingPointError(
                "experimental runtime accumulator returned a non-finite rate"
            )
        return derivative


__all__ = [
    "ExperimentalRuntimeAccumulator",
    "compile_generic_state_accumulator",
]
