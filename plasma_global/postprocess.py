"""Explicit derived-series evaluation for completed compiled simulations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from plasma_global._output_channels import EventChannel, channel_rates, event_channels
from plasma_global._output_values import (
    LEDGER_QUANTITIES,
    port_values,
    surface_values,
    zone_values,
)
from plasma_global.core.compiled import CompiledGlobalModel
from plasma_global.core.domain import RecipeSegment
from plasma_global.errors import CaseValidationError


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
    names.extend(
        f"{quantity}[{zone_id}]"
        for quantity in (
            "pressure_Pa",
            "heavy_density_m3",
            "neutral_density_m3",
            *LEDGER_QUANTITIES,
        )
    )
    names.extend(
        f"{quantity}[{zone_id},{species}]"
        for quantity in ("mole_fraction", "partial_pressure_Pa", "net_production_m3_s")
        for species in model.species_ids
    )
    return tuple(names)


def _surface_observable_names(model: CompiledGlobalModel) -> tuple[str, ...]:
    names: list[str] = []
    for wall in model.wall_boundaries:
        if wall.surface_id is not None:
            names.append(f"wall_sheath_energy_eV[{wall.surface_id}]")
            names.extend(
                f"incident_flux_m2_s[{wall.surface_id},{species}]"
                for species, charge in zip(
                    model.species_ids, model.charges, strict=True
                )
                if charge > 0
            )
    if model.surface_model is not None:
        names.extend(
            f"surface_temperature_K[{surface.surface_id}]"
            for surface in model.surface_model.surfaces
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
    names.extend(_surface_observable_names(model))
    for channel in event_channels(model):
        names.extend((channel.rate_name, channel.frequency_name, channel.count_name))
    return tuple(dict.fromkeys(names))


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


def _point_values(
    model: CompiledGlobalModel,
    time_s: float,
    state: np.ndarray,
    segment: RecipeSegment,
    channels: tuple[EventChannel, ...],
    needs_ledger: bool,
) -> dict[str, float]:
    evaluated = model.evaluate(time_s, state, segment, collect_ledger=needs_ledger)
    values = zone_values(model, state, evaluated, collect_ledger=needs_ledger)
    if needs_ledger:
        values.update(surface_values(model, state, evaluated, segment))
        values.update(channel_rates(model, evaluated, channels))
    values.update(port_values(evaluated))
    return values


def _accumulate_events(
    model: CompiledGlobalModel,
    times: np.ndarray,
    states: np.ndarray,
    index: int,
    segment: RecipeSegment,
    previous_segment: RecipeSegment,
    values: Mapping[str, float],
    previous_values: Mapping[str, float],
    channels: tuple[EventChannel, ...],
    output: Mapping[str, np.ndarray],
) -> None:
    if index == 0:
        return
    if segment is not previous_segment:
        if previous_segment.end_s != times[index - 1]:
            raise ValueError(
                "event integration requires every forcing boundary to be saved"
            )
        previous_values = _point_values(
            model, float(times[index - 1]), states[index - 1], segment, channels, True
        )
    duration = float(times[index] - times[index - 1])
    for channel in channels:
        if channel.count_name in output:
            left = previous_values[channel.frequency_name]
            right = values[channel.frequency_name]
            output[channel.count_name][index] = output[channel.count_name][
                index - 1
            ] + duration * 0.5 * (left + right)


def _evaluation_requirements(
    model: CompiledGlobalModel, selected: tuple[str, ...]
) -> tuple[tuple[EventChannel, ...], bool]:
    channels = tuple(
        channel
        for channel in event_channels(model)
        if any(
            name in selected
            for name in (channel.rate_name, channel.frequency_name, channel.count_name)
        )
    )
    needs_ledger = bool(channels) or any(
        name.startswith(
            (
                "absorbed_power_W[",
                "incident_flux_m2_s[",
                "wall_sheath_energy_eV[",
                "surface_temperature_K[",
                "coverage[",
                *tuple(name + "[" for name in LEDGER_QUANTITIES),
            )
        )
        for name in selected
    )
    return channels, needs_ledger


def _store_point(
    output: Mapping[str, np.ndarray],
    point: Mapping[str, float],
    counts: set[str],
    index: int,
    time: float,
) -> None:
    for name in output:
        if name in counts:
            continue
        if name not in point:
            raise ValueError(
                f"observable {name!r} is undefined at saved time {time:g}; "
                "no substitute value is used"
            )
        output[name][index] = point[name]


def _initialize_counts(
    model: CompiledGlobalModel,
    times: np.ndarray,
    channels: tuple[EventChannel, ...],
    output: Mapping[str, np.ndarray],
) -> None:
    names = [channel.count_name for channel in channels if channel.count_name in output]
    if not names or not times.size:
        return
    if times[0] != model.segments[0].start_s:
        raise ValueError("cumulative events require a saved sample at the run start")
    for name in names:
        output[name][0] = 0.0


def derive_observables(
    model: CompiledGlobalModel,
    time_s: np.ndarray,
    state: np.ndarray,
    names: Iterable[str],
    *,
    extra_names: Iterable[str] = (),
) -> Mapping[str, np.ndarray]:
    """Evaluate selected physical values and segmented trapezoidal event counts."""
    selected = tuple(names)
    unknown = set(selected) - set(available_observables(model)) - set(extra_names)
    if unknown:
        raise CaseValidationError(f"unknown output observables {sorted(unknown)}")
    if not selected:
        return {}
    channels, needs_ledger = _evaluation_requirements(model, selected)
    output = {name: np.empty(len(time_s), dtype=float) for name in selected}
    _initialize_counts(model, time_s, channels, output)
    segment_index = 0
    previous_values: dict[str, float] = {}
    previous_segment = model.segments[0]
    for index, (time, values) in enumerate(zip(time_s, state, strict=True)):
        segment, segment_index = _segment_for_saved_time(
            model, float(time), segment_index
        )
        point = _point_values(
            model, float(time), values, segment, channels, needs_ledger
        )
        _accumulate_events(
            model,
            time_s,
            state,
            index,
            segment,
            previous_segment,
            point,
            previous_values,
            channels,
            output,
        )
        _store_point(
            output,
            point,
            {channel.count_name for channel in channels},
            index,
            float(time),
        )
        previous_values, previous_segment = point, segment
    return output


__all__ = [
    "available_observables",
    "derive_observables",
]
