"""Explicit derived-series evaluation for completed compiled simulations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from plasma_global.core.compiled import CompiledGlobalModel
from plasma_global.errors import CaseValidationError


def available_observables(model: CompiledGlobalModel) -> tuple[str, ...]:
    """Return the finite catalog determined by one compiled model."""

    names: list[str] = []
    for zone in model.zones:
        zone_id = zone.zone_id
        names.extend(
            (
                f"electron_density_m3[{zone_id}]",
                f"mean_energy_eV[{zone_id}]",
                f"electron_temperature_eV[{zone_id}]",
                f"reduced_field_Td[{zone_id}]",
                f"gas_temperature_K[{zone_id}]",
                f"charge_residual_m3[{zone_id}]",
                f"absorbed_power_W[{zone_id}]",
            )
        )
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


def derive_observables_and_diagnostics(
    model: CompiledGlobalModel,
    time_s: np.ndarray,
    state: np.ndarray,
    names: Iterable[str],
) -> tuple[Mapping[str, np.ndarray], dict[str, float]]:
    """Create selected series and normal diagnostics in one saved-point pass."""

    selected = validate_observable_selection(model, names)
    uses_prescribed_electrons = model.electron_density_provider is not None or any(
        segment.prescribed_electron_density_m3_by_zone for segment in model.segments
    )
    diagnostics = {"charge_closure_normalized": 0.0}
    if not selected and not uses_prescribed_electrons:
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
    zone_by_id = {zone.zone_id: zone for zone in model.zones}
    segment_index = 0
    for time_index, (time, values) in enumerate(zip(times, states)):
        while (
            segment_index + 1 < len(model.segments)
            and time > model.segments[segment_index].end_s
        ):
            segment_index += 1
        segment = model.segments[segment_index]
        if time < segment.start_s or time > segment.end_s:
            raise ValueError(f"saved time {time:g} is outside compiled recipe segments")
        evaluated = model.evaluate(
            float(time), values, segment, collect_ledger=needs_power_ledger
        )
        for zone_id, electrons in evaluated.electron_states.items():
            density = values[model.layout.density_slices[zone_id]]
            charge_scale = max(
                electrons.density_m3
                + float(np.dot(np.abs(model.charges), np.maximum(density, 0.0))),
                1.0,
            )
            diagnostics["charge_closure_normalized"] = max(
                diagnostics["charge_closure_normalized"],
                abs(evaluated.charge_residual_m3_by_zone[zone_id]) / charge_scale,
            )
            values_by_name = {
                f"electron_density_m3[{zone_id}]": electrons.density_m3,
                f"mean_energy_eV[{zone_id}]": electrons.mean_energy_eV,
                f"electron_temperature_eV[{zone_id}]": electrons.temperature_eV,
                f"reduced_field_Td[{zone_id}]": (
                    np.nan
                    if electrons.reduced_field_Td is None
                    else electrons.reduced_field_Td
                ),
                f"gas_temperature_K[{zone_id}]": (
                    evaluated.gas_temperature_K_by_zone[zone_id]
                ),
                f"charge_residual_m3[{zone_id}]": (
                    evaluated.charge_residual_m3_by_zone[zone_id]
                ),
            }
            if needs_power_ledger:
                ledger = evaluated.ledger_by_zone[zone_id]
                values_by_name[f"absorbed_power_W[{zone_id}]"] = (
                    ledger.absorbed_power_J_m3_s + ledger.gas_power_J_m3_s
                ) * zone_by_id[zone_id].volume_m3
            for name, value in values_by_name.items():
                if name in output:
                    output[name][time_index] = value
        if model.surface_model is not None:
            coverage_state = values[model.layout.surface_coverage_slice]
            for surface in model.surface_model.surfaces:
                free_species = model.surface_model.layout.free_species_by_surface[
                    surface.surface_id
                ]
                name = f"coverage[{surface.surface_id},{free_species}]"
                if name in output:
                    output[name][time_index] = model.surface_model.coverage(
                        coverage_state, surface.surface_id, free_species
                    )
    return output, diagnostics


__all__ = [
    "available_observables",
    "derive_observables",
    "derive_observables_and_diagnostics",
    "validate_observable_selection",
    "validate_summary_selection",
]
