"""Explicit derived-series evaluation for completed compiled simulations."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping

import numpy as np

from plasma_global.core.compiled import CompiledGlobalModel, ModelEvaluation
from plasma_global.core.domain import RecipeSegment
from plasma_global.errors import CaseValidationError
from plasma_global.models.electrons import ElectronState


def _has_reduced_field_series(model: CompiledGlobalModel, zone_id: str) -> bool:
    coordinator = model.power_coordinator
    return all(
        zone_id in segment.reduced_field_Td_by_zone
        or (
            coordinator is not None
            and coordinator.has_reduced_field_source(zone_id, segment.port_commands)
        )
        for segment in model.segments
    )


def _zone_observable_names(model: CompiledGlobalModel, zone_id: str) -> tuple[str, ...]:
    names = [
        f"electron_density_m3[{zone_id}]",
        f"mean_energy_eV[{zone_id}]",
        f"electron_temperature_eV[{zone_id}]",
    ]
    if _has_reduced_field_series(model, zone_id):
        names.append(f"reduced_field_Td[{zone_id}]")
    names.extend(
        (
            f"gas_temperature_K[{zone_id}]",
            f"charge_residual_m3[{zone_id}]",
            f"absorbed_power_W[{zone_id}]",
        )
    )
    return tuple(names)


def available_observables(model: CompiledGlobalModel) -> tuple[str, ...]:
    """Return the finite catalog determined by one compiled model."""

    names: list[str] = []
    for zone in model.zones:
        names.extend(_zone_observable_names(model, zone.zone_id))
    if model.surface_model is not None:
        for surface in model.surface_model.surfaces:
            free_species = model.surface_model.layout.free_species_by_surface[
                surface.surface_id
            ]
            names.append(f"coverage[{surface.surface_id},{free_species}]")
    return tuple(names)


def validate_observable_selection(
    model: CompiledGlobalModel, names: Iterable[str]
) -> tuple[str, ...]:
    """Validate requested names at compile time and preserve their order."""

    selected = tuple(names)
    available = set(available_observables(model))
    unknown = sorted(set(selected) - available)
    if unknown:
        raise CaseValidationError(
            f"unknown output observables {unknown}; available: {sorted(available)}"
        )
    return selected


def validate_summary_selection(
    model: CompiledGlobalModel,
    observable_names: Iterable[str],
    summary_names: Iterable[str],
) -> tuple[str, ...]:
    """Validate summary names against stored states and selected derived series."""

    selected_observables = validate_observable_selection(model, observable_names)
    selected_summary = tuple(summary_names)
    available = set(model.layout.labels) | set(selected_observables)
    unknown = sorted(set(selected_summary) - available)
    if unknown:
        raise CaseValidationError(
            f"unknown output summary series {unknown}; available: {sorted(available)}"
        )
    return selected_summary


def derive_observables(
    model: CompiledGlobalModel,
    time_s: np.ndarray,
    state: np.ndarray,
    names: Iterable[str],
) -> Mapping[str, np.ndarray]:
    """Evaluate only selected derived series at saved solution points."""

    observables, _ = derive_observables_and_diagnostics(
        model,
        time_s,
        state,
        names,
    )
    return observables


def _segment_for_saved_time(
    model: CompiledGlobalModel, time_s: float, start_index: int
) -> tuple[RecipeSegment, int]:
    index = start_index
    while index + 1 < len(model.segments) and time_s > model.segments[index].end_s:
        index += 1
    segment = model.segments[index]
    if time_s < segment.start_s or time_s > segment.end_s:
        raise ValueError(f"saved time {time_s:g} is outside compiled recipe segments")
    return segment, index


def _zone_observable_values(
    model: CompiledGlobalModel,
    evaluated: ModelEvaluation,
    zone_id: str,
    electrons: ElectronState,
    state: np.ndarray,
    *,
    needs_power_ledger: bool,
    volume_m3: float,
) -> tuple[dict[str, float], float]:
    density = state[model.layout.density_slices[zone_id]]
    charge_scale = max(
        electrons.density_m3
        + float(np.dot(np.abs(model.charges), np.maximum(density, 0.0))),
        1.0,
    )
    normalized_residual = (
        abs(evaluated.charge_residual_m3_by_zone[zone_id]) / charge_scale
    )
    values = {
        f"electron_density_m3[{zone_id}]": electrons.density_m3,
        f"mean_energy_eV[{zone_id}]": electrons.mean_energy_eV,
        f"electron_temperature_eV[{zone_id}]": electrons.temperature_eV,
        f"gas_temperature_K[{zone_id}]": evaluated.gas_temperature_K_by_zone[zone_id],
        f"charge_residual_m3[{zone_id}]": (
            evaluated.charge_residual_m3_by_zone[zone_id]
        ),
    }
    if electrons.reduced_field_Td is not None:
        values[f"reduced_field_Td[{zone_id}]"] = electrons.reduced_field_Td
    if needs_power_ledger:
        ledger = evaluated.ledger_by_zone[zone_id]
        values[f"absorbed_power_W[{zone_id}]"] = (
            ledger.absorbed_power_J_m3_s + ledger.gas_power_J_m3_s
        ) * volume_m3
    return values, normalized_residual


def _surface_observable_values(
    model: CompiledGlobalModel,
    state: np.ndarray,
    selected: Collection[str],
) -> dict[str, float]:
    surface_model = model.surface_model
    if surface_model is None:
        return {}
    coverage_state = state[model.layout.surface_coverage_slice]
    values: dict[str, float] = {}
    for surface in surface_model.surfaces:
        free_species = surface_model.layout.free_species_by_surface[surface.surface_id]
        name = f"coverage[{surface.surface_id},{free_species}]"
        if name in selected:
            values[name] = surface_model.coverage(
                coverage_state, surface.surface_id, free_species
            )
    return values


def _saved_point_observables(
    model: CompiledGlobalModel,
    time_s: float,
    state: np.ndarray,
    segment_index: int,
    *,
    needs_power_ledger: bool,
    volume_by_zone: Mapping[str, float],
    selected: Collection[str],
) -> tuple[dict[str, float], float, int]:
    segment, segment_index = _segment_for_saved_time(model, time_s, segment_index)
    evaluated = model.evaluate(
        time_s, state, segment, collect_ledger=needs_power_ledger
    )
    observable_values: dict[str, float] = {}
    maximum_charge_residual = 0.0
    for zone_id, electrons in evaluated.electron_states.items():
        zone_values, normalized_residual = _zone_observable_values(
            model,
            evaluated,
            zone_id,
            electrons,
            state,
            needs_power_ledger=needs_power_ledger,
            volume_m3=volume_by_zone[zone_id],
        )
        observable_values.update(zone_values)
        maximum_charge_residual = max(maximum_charge_residual, normalized_residual)
    observable_values.update(_surface_observable_values(model, state, selected))
    return observable_values, maximum_charge_residual, segment_index


def _requires_postprocessing(
    model: CompiledGlobalModel, selected: tuple[str, ...]
) -> bool:
    if selected or model.electron_density_provider is not None:
        return True
    return any(
        segment.prescribed_electron_density_m3_by_zone for segment in model.segments
    )


def derive_observables_and_diagnostics(
    model: CompiledGlobalModel,
    time_s: np.ndarray,
    state: np.ndarray,
    names: Iterable[str],
) -> tuple[Mapping[str, np.ndarray], dict[str, float]]:
    """Create selected series and normal diagnostics in one saved-point pass."""

    selected = validate_observable_selection(model, names)
    diagnostics = {"charge_closure_normalized": 0.0}
    if not _requires_postprocessing(model, selected):
        # Quasineutrality is algebraic, so its residual is identically zero.  Do
        # not rerun the complete physical model after integration to prove it.
        return {}, diagnostics
    times = np.asarray(time_s, dtype=float)
    states = np.asarray(state, dtype=float)
    if times.ndim != 1 or states.shape != (times.size, model.layout.size):
        raise ValueError(
            "postprocess time/state shape does not match the compiled model"
        )

    output = {name: np.empty(times.size, dtype=float) for name in selected}
    needs_power_ledger = any(name.startswith("absorbed_power_W[") for name in selected)
    volume_by_zone = {zone.zone_id: zone.volume_m3 for zone in model.zones}
    segment_index = 0
    for time_index, (time, state_values) in enumerate(zip(times, states, strict=True)):
        values_by_name, normalized_residual, segment_index = _saved_point_observables(
            model,
            float(time),
            state_values,
            segment_index,
            needs_power_ledger=needs_power_ledger,
            volume_by_zone=volume_by_zone,
            selected=output,
        )
        diagnostics["charge_closure_normalized"] = max(
            diagnostics["charge_closure_normalized"], normalized_residual
        )
        for name, values in output.items():
            if name not in values_by_name:
                raise ValueError(
                    f"observable {name!r} is undefined at saved time {float(time):g}"
                )
            values[time_index] = values_by_name[name]
    return output, diagnostics


__all__ = [
    "available_observables",
    "derive_observables",
    "derive_observables_and_diagnostics",
    "validate_observable_selection",
    "validate_summary_selection",
]
