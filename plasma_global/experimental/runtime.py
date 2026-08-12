"""Runtime adapter for opt-in generic and surface-event accumulator states."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, SupportsFloat

import numpy as np

from plasma_global.experimental.accumulators import (
    ConstantSource,
    DrivenSource,
    FilmInventoryAccumulator,
    GenericState,
    GenericStateAccumulator,
    LinearRelaxation,
    ProcessContext,
    SurfaceAccumulation,
    SurfaceEventRate,
)

_CompiledProcess = ConstantSource | LinearRelaxation | DrivenSource


def _python_float(value: SupportsFloat) -> float:
    return float(value)


def _string_id(value: object) -> str:
    return str(value)


@dataclass(frozen=True, slots=True)
class _CompiledStateDeclarations:
    states: tuple[GenericState, ...]
    scopes: Mapping[str, str]
    owners: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class _RuntimeLayout:
    labels: tuple[str, ...]
    lower_bounds: tuple[float | None, ...]
    initial: np.ndarray
    film_index: Mapping[str, int]
    inventory_index: Mapping[tuple[str, str], int]
    generic_slice: slice


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
    selected = (
        available if raw is None or raw == () else tuple(str(item) for item in raw)
    )
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"{where} selects unknown owners {sorted(unknown)}")
    if not selected:
        raise ValueError(f"{where} selects no owners")
    if len(set(selected)) != len(selected):
        raise ValueError(f"{where} contains duplicate owners")
    return selected


def _compile_state_declarations(
    entries: Mapping[str, Any],
    *,
    zone_ids: tuple[str, ...],
    surface_ids: tuple[str, ...],
) -> _CompiledStateDeclarations:
    states: list[GenericState] = []
    scopes: dict[str, str] = {}
    owners_by_state: dict[str, tuple[str, ...]] = {}
    for state_id, entry in entries.items():
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
            zone_ids=zone_ids,
            surface_ids=surface_ids,
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
    return _CompiledStateDeclarations(
        states=tuple(states),
        scopes=MappingProxyType(scopes),
        owners=MappingProxyType(owners_by_state),
    )


def _validate_process_declaration(
    values: Mapping[str, Any], *, kind: str, target_scope: str, where: str
) -> None:
    common = {"kind", "target", "zones", "surfaces"}
    if kind == "source":
        _only_keys(values, common | {"value"}, where)
    elif kind == "relaxation":
        _only_keys(values, common | {"equilibrium", "tau_s"}, where)
    elif kind == "ion_flux_source":
        _only_keys(values, common | {"coefficient"}, where)
        if target_scope != "surface":
            raise ValueError(f"{where} ion_flux_source target must use surface scope")
    else:
        raise ValueError(f"{where}.kind must be source, relaxation, or ion_flux_source")


def _build_process(
    kind: str,
    target: str,
    values: Mapping[str, Any],
    owners: tuple[str, ...],
    where: str,
) -> _CompiledProcess:
    if kind == "source":
        if "value" not in values:
            raise ValueError(f"{where}.value is required")
        return ConstantSource(target, float(values["value"]), owners)
    if kind == "relaxation":
        if "tau_s" not in values:
            raise ValueError(f"{where}.tau_s is required")
        return LinearRelaxation(
            target,
            float(values.get("equilibrium", 0.0)),
            float(values["tau_s"]),
            owners,
        )
    if "coefficient" not in values:
        raise ValueError(f"{where}.coefficient is required")
    return DrivenSource(
        target,
        "ion_flux_m2_s",
        float(values["coefficient"]),
        owners,
    )


def _compile_processes(
    entries: Mapping[str, Any],
    declarations: _CompiledStateDeclarations,
    *,
    zone_ids: tuple[str, ...],
    surface_ids: tuple[str, ...],
) -> tuple[_CompiledProcess, ...]:
    processes: list[_CompiledProcess] = []
    for process_id, entry in entries.items():
        where = f"chemistry.experimental.processes.{process_id}"
        values = _mapping(entry, where)
        kind = str(values.get("kind", "")).strip().lower()
        target = str(values.get("target", "")).strip()
        if target not in declarations.owners:
            raise ValueError(f"{where} targets unknown state {target!r}")
        target_scope = declarations.scopes[target]
        _validate_process_declaration(
            values,
            kind=kind,
            target_scope=target_scope,
            where=where,
        )
        selected = _selected_owners(
            values,
            scope=target_scope,
            zone_ids=(
                declarations.owners[target] if target_scope == "zone" else zone_ids
            ),
            surface_ids=(
                declarations.owners[target]
                if target_scope == "surface"
                else surface_ids
            ),
            where=where,
        )
        owners = () if selected == declarations.owners[target] else selected
        processes.append(_build_process(kind, target, values, owners, where))
    return tuple(processes)


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

    zones = tuple(_string_id(item) for item in zone_ids)
    surfaces = tuple(_string_id(item) for item in surface_ids)
    declarations = _compile_state_declarations(
        state_entries,
        zone_ids=zones,
        surface_ids=surfaces,
    )
    processes = _compile_processes(
        process_entries,
        declarations,
        zone_ids=zones,
        surface_ids=surfaces,
    )
    return GenericStateAccumulator(declarations.states, processes)


def _normalize_film_surfaces(values: Sequence[str]) -> tuple[str, ...]:
    films = tuple(_string_id(item) for item in values)
    if any(not item for item in films) or len(set(films)) != len(films):
        raise ValueError("film surfaces must be unique nonempty IDs")
    return films


def _normalize_inventory(
    values: Mapping[str, Mapping[str, float]],
) -> dict[str, Mapping[str, float]]:
    inventory: dict[str, Mapping[str, float]] = {}
    for surface_id, entries in values.items():
        converted = {
            _string_id(key): _python_float(value) for key, value in entries.items()
        }
        if (
            not surface_id
            or any(not key for key in converted)
            or any(
                not math.isfinite(value) or value < 0.0 for value in converted.values()
            )
        ):
            raise ValueError("initial wall inventory must be named and nonnegative")
        inventory[_string_id(surface_id)] = MappingProxyType(converted)
    return inventory


def _build_runtime_layout(
    films: tuple[str, ...],
    inventory: Mapping[str, Mapping[str, float]],
    generic: GenericStateAccumulator | None,
) -> _RuntimeLayout:
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
    if generic is not None:
        for state in generic.states:
            for owner in state.owners:
                labels.append(f"extra[{state.state_id},{owner}]")
                lower_bounds.append(state.lower_bound)
        initial.extend(generic.initial_state().tolist())
    if len(set(labels)) != len(labels):
        raise ValueError("experimental runtime state labels must be unique")
    if not labels:
        raise ValueError("experimental runtime accumulator needs at least one state")

    initial_array = np.asarray(initial, dtype=float)
    initial_array.setflags(write=False)
    return _RuntimeLayout(
        labels=tuple(labels),
        lower_bounds=tuple(lower_bounds),
        initial=initial_array,
        film_index=MappingProxyType(film_index),
        inventory_index=MappingProxyType(inventory_index),
        generic_slice=slice(generic_start, len(labels)),
    )


def _validated_runtime_state(
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


def _surface_accumulation(
    templates: Sequence[SurfaceEventRate],
    monolayer_thickness_m: float,
    rates_m2_s: Mapping[str, float] | None,
) -> SurfaceAccumulation | None:
    rates = rates_m2_s or {}
    events = tuple(
        replace(
            template,
            rate_m2_s=_python_float(rates.get(template.event_id, 0.0)),
        )
        for template in templates
    )
    if not events:
        return None
    return FilmInventoryAccumulator(monolayer_thickness_m).accumulate(events)


def _add_surface_accumulation(
    derivative: np.ndarray,
    accumulation: SurfaceAccumulation,
    film_index: Mapping[str, int],
    inventory_index: Mapping[tuple[str, str], int],
) -> None:
    for surface_id, growth_m_s in accumulation.film_growth_m_s.items():
        index = film_index.get(surface_id)
        if index is None:
            if growth_m_s != 0.0:
                raise ValueError(
                    f"surface event produced undeclared film state {surface_id!r}"
                )
            continue
        derivative[index] += growth_m_s
    for key, particle_rate_s in accumulation.inventory_particle_rate_s.items():
        index = inventory_index.get(key)
        if index is None:
            if particle_rate_s != 0.0:
                raise ValueError(
                    f"surface event produced undeclared inventory state {key!r}"
                )
            continue
        derivative[index] += particle_rate_s


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
        films = _normalize_film_surfaces(self.film_surfaces)
        inventory = _normalize_inventory(self.initial_inventory_by_surface)
        templates = tuple(self.surface_event_templates)
        event_ids = tuple(event.event_id for event in templates)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("surface-event template IDs must be unique")
        layout = _build_runtime_layout(films, inventory, self.generic)
        object.__setattr__(self, "film_surfaces", films)
        object.__setattr__(
            self, "initial_inventory_by_surface", MappingProxyType(inventory)
        )
        object.__setattr__(self, "surface_event_templates", templates)
        object.__setattr__(self, "labels", layout.labels)
        object.__setattr__(self, "lower_bounds", layout.lower_bounds)
        object.__setattr__(self, "_initial", layout.initial)
        object.__setattr__(self, "_film_index", layout.film_index)
        object.__setattr__(self, "_inventory_index", layout.inventory_index)
        object.__setattr__(self, "_generic_slice", layout.generic_slice)

    def initial_state(self) -> np.ndarray:
        return self._initial.copy()

    def rhs(
        self,
        values: np.ndarray,
        *,
        drivers: Mapping[str, Mapping[str, float]] | None = None,
        surface_rates_m2_s: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        physical_stop = self._generic_slice.start
        state = _validated_runtime_state(values, self._initial, physical_stop)

        derivative = np.zeros_like(state)
        if self.generic is not None:
            derivative[self._generic_slice] = self.generic.rhs(
                state[self._generic_slice], ProcessContext(drivers or {})
            )

        accumulation = _surface_accumulation(
            self.surface_event_templates,
            self.monolayer_thickness_m,
            surface_rates_m2_s,
        )
        if accumulation is not None:
            _add_surface_accumulation(
                derivative,
                accumulation,
                self._film_index,
                self._inventory_index,
            )
        if not np.all(np.isfinite(derivative)):
            raise FloatingPointError(
                "experimental runtime accumulator returned a non-finite rate"
            )
        return derivative


__all__ = [
    "ExperimentalRuntimeAccumulator",
    "compile_generic_state_accumulator",
]
