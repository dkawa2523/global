"""Compiler for chemistry-declared experimental state and process mappings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from plasma_global.experimental.accumulators import (
    ConstantSource,
    DrivenSource,
    GenericState,
    GenericStateAccumulator,
    LinearRelaxation,
)

_CompiledProcess = ConstantSource | LinearRelaxation | DrivenSource


@dataclass(frozen=True, slots=True)
class _CompiledStateDeclarations:
    states: tuple[GenericState, ...]
    scopes: Mapping[str, str]
    owners: Mapping[str, tuple[str, ...]]


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
        return _global_owner(values, where)
    selection_key = "zones" if scope == "zone" else "surfaces"
    rejected_key = "surfaces" if scope == "zone" else "zones"
    if values.get(rejected_key):
        raise ValueError(f"{where} {scope} scope cannot select {rejected_key}")
    available = zone_ids if scope == "zone" else surface_ids
    selected = _owner_selection(values.get(selection_key), available)
    _validate_owner_selection(selected, available, where)
    return selected


def _global_owner(values: Mapping[str, Any], where: str) -> tuple[str, ...]:
    if values.get("zones") or values.get("surfaces"):
        raise ValueError(f"{where} global scope cannot select zones or surfaces")
    return ("global",)


def _owner_selection(raw: Any, available: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None or raw == ():
        return available
    return tuple(str(item) for item in raw)


def _validate_owner_selection(
    selected: tuple[str, ...], available: tuple[str, ...], where: str
) -> None:
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"{where} selects unknown owners {sorted(unknown)}")
    if not selected:
        raise ValueError(f"{where} selects no owners")
    if len(set(selected)) != len(selected):
        raise ValueError(f"{where} contains duplicate owners")


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
            {"scope", "initial", "lower_bound", "upper_bound", "zones", "surfaces"},
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
    return DrivenSource(target, "ion_flux_m2_s", float(values["coefficient"]), owners)


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
            values, kind=kind, target_scope=target_scope, where=where
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


def compile_generic_state(
    specification: Mapping[str, Any],
    *,
    zone_ids: Sequence[str],
    surface_ids: Sequence[str],
) -> GenericStateAccumulator | None:
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
    declarations = _compile_state_declarations(
        state_entries, zone_ids=zones, surface_ids=surfaces
    )
    processes = _compile_processes(
        process_entries,
        declarations,
        zone_ids=zones,
        surface_ids=surfaces,
    )
    return GenericStateAccumulator(declarations.states, processes)
