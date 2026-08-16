"""Canonicalize accepted BDF states for the strict public result domain."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from plasma_global.core.compiled import CompiledGlobalModel
from plasma_global.core.domain import RecipeSegment
from plasma_global.errors import IntegrationError, ModelDomainError
from plasma_global.models.electrons import (
    ELEMENTARY_CHARGE_C,
    resolved_electron_density,
)

_DOMAIN_LIMIT = 10.0


def _accepted_location(
    time_index: int,
    time_s: np.ndarray | None,
    segments: Sequence[RecipeSegment],
) -> str:
    if time_s is None:
        return f"output index {time_index}"
    failed_time = float(np.asarray(time_s)[time_index])
    segment_id = next(
        (
            item.segment_id
            for item in segments
            if item.start_s <= failed_time <= item.end_s
        ),
        "unknown",
    )
    return f"segment {segment_id!r} at t={failed_time:.16g} s"


def _raise_material_bound_failure(
    mask: np.ndarray,
    accepted: np.ndarray,
    limit: str,
    time_s: np.ndarray | None,
    segments: Sequence[RecipeSegment],
) -> None:
    positions = np.argwhere(mask)
    if not positions.size:
        return
    time_index, state_index = positions[0]
    location = _accepted_location(int(time_index), time_s, segments)
    raise IntegrationError(
        f"Accepted solver state domain failure {limit} in {location}: "
        f"state[{int(time_index)},{int(state_index)}]="
        f"{accepted[time_index, state_index]:.6e}"
    )


def _accepted_state_for_result(
    state: np.ndarray,
    domain_atol: np.ndarray,
    *,
    lower_bounds: np.ndarray | None = None,
    upper_bounds: np.ndarray | None = None,
    time_s: np.ndarray | None = None,
    segments: Sequence[RecipeSegment] = (),
) -> tuple[np.ndarray, int]:
    """Validate component bounds and snap tolerance-scale excursions."""

    accepted = np.array(state, dtype=float, copy=True)
    tolerance = np.asarray(domain_atol, dtype=float).reshape(1, -1)
    lower = (
        np.zeros_like(tolerance)
        if lower_bounds is None
        else np.asarray(lower_bounds, dtype=float).reshape(1, -1)
    )
    upper = (
        np.full_like(tolerance, float("inf"))
        if upper_bounds is None
        else np.asarray(upper_bounds, dtype=float).reshape(1, -1)
    )
    if (
        lower.shape != tolerance.shape
        or upper.shape != tolerance.shape
        or np.any(np.isnan(lower))
        or np.any(np.isnan(upper))
        or np.any(lower > upper)
    ):
        raise ValueError(
            "result bounds must match domain_atol and form valid intervals"
        )
    _raise_material_bound_failure(
        accepted < (lower - _DOMAIN_LIMIT * tolerance),
        accepted,
        "below lower_bound-10*domain_atol",
        time_s,
        segments,
    )
    _raise_material_bound_failure(
        accepted > (upper + _DOMAIN_LIMIT * tolerance),
        accepted,
        "above upper_bound+10*domain_atol",
        time_s,
        segments,
    )
    below_bound = accepted < lower
    roundoff_negative = (lower == 0.0) & below_bound
    zeroed_count = int(np.count_nonzero(roundoff_negative))
    np.clip(accepted, lower, upper, out=accepted)
    return accepted, zeroed_count


def _saved_segments(
    times: np.ndarray, segments: Sequence[RecipeSegment]
) -> tuple[RecipeSegment, ...]:
    if times.ndim != 1 or not segments:
        raise ValueError("result times need at least one recipe segment")
    ends = np.fromiter((item.end_s for item in segments), dtype=float)
    starts = np.fromiter((item.start_s for item in segments), dtype=float)
    indices = np.searchsorted(ends, times, side="left")
    safe_indices = np.minimum(indices, len(segments) - 1)
    valid = np.isfinite(times) & (indices < len(segments))
    valid &= times >= starts[safe_indices]
    if times.size > 1:
        valid[1:] &= np.diff(times) >= 0.0
    if not np.all(valid):
        index = int(np.flatnonzero(~valid)[0])
        raise IntegrationError(
            f"Accepted solver time {times[index]:.16g} s is outside ordered recipe "
            "segments"
        )
    return tuple(segments[int(index)] for index in indices)


def _failure(message: str, time: float, segment: RecipeSegment) -> IntegrationError:
    return IntegrationError(
        "Accepted solver state coupled-domain failure in "
        f"segment {segment.segment_id!r} at t={time:.16g} s: {message}"
    )


def _electron_density(
    model: CompiledGlobalModel,
    segment: RecipeSegment,
    time: float,
    zone_id: str,
    density: np.ndarray,
) -> float:
    prescribed = segment.prescribed_electron_density_m3_by_zone.get(zone_id)
    if prescribed is None and model.electron_density_provider is not None:
        prescribed = model.electron_density_provider(time, zone_id)
    return resolved_electron_density(float(model.charges @ density), prescribed)


def _validate_electron_mean_energy(
    model: CompiledGlobalModel,
    energy: float,
    electron_density: float,
    segment: RecipeSegment,
    time: float,
    zone_id: str,
) -> None:
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        mean_energy = float(np.divide(energy, electron_density * ELEMENTARY_CHARGE_C))
    if not math.isfinite(mean_energy):
        raise _failure(
            f"electron mean energy is not finite in zone {zone_id!r}", time, segment
        )
    table = model.electron_kinetics_by_zone.get(zone_id)
    if table is not None and table.lookup == "mean_energy":
        try:
            table._query(mean_energy)
        except ModelDomainError as exc:
            raise _failure(str(exc), time, segment) from exc


def _canonicalize_electron_energy(
    model: CompiledGlobalModel,
    row: np.ndarray,
    tolerance: np.ndarray,
    segment: RecipeSegment,
    time: float,
    zone_id: str,
    density: np.ndarray,
) -> bool:
    if not model.layout.evolves_electron_energy:
        return False
    try:
        electron_density = _electron_density(model, segment, time, zone_id, density)
    except ModelDomainError as exc:
        raise _failure(str(exc), time, segment) from exc
    index = model.layout.electron_energy_indices[zone_id]
    energy = float(row[index])
    if electron_density != 0.0:
        _validate_electron_mean_energy(
            model, energy, electron_density, segment, time, zone_id
        )
        return False
    if energy == 0.0:
        return False
    if energy > _DOMAIN_LIMIT * tolerance[index]:
        raise _failure(
            "positive electron energy at zero electron density exceeds "
            f"{_DOMAIN_LIMIT:g}*domain_atol in zone {zone_id!r}",
            time,
            segment,
        )
    row[index] = 0.0
    return True


def _validate_heavy_energy(
    model: CompiledGlobalModel,
    row: np.ndarray,
    segment: RecipeSegment,
    time: float,
    zone_id: str,
    density: np.ndarray,
) -> None:
    if not model.layout.evolves_heavy_energy:
        return
    closure = model.heavy_energy_closure
    if closure is None:
        raise RuntimeError("Evolved gas energy needs its closure")
    energy = float(row[model.layout.heavy_energy_indices[zone_id]])
    capacity = float(closure.heat_capacity_J_m3_K(density.reshape(1, -1))[0])
    if not math.isfinite(capacity) or capacity <= 0.0:
        raise _failure(
            f"gas temperature is undefined for an empty mixture in zone {zone_id!r}",
            time,
            segment,
        )
    if energy <= 0.0:
        raise _failure(
            f"gas internal energy is nonpositive in zone {zone_id!r}", time, segment
        )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        temperature = energy / capacity
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise _failure(
            f"gas temperature is not finite and positive in zone {zone_id!r}",
            time,
            segment,
        )


def _map_surface_state(
    model: CompiledGlobalModel,
    state: np.ndarray,
    tolerance: np.ndarray,
    times: np.ndarray,
    segments: tuple[RecipeSegment, ...],
) -> None:
    surface = model.surface_model
    if surface is None:
        return
    coverage_slice = model.layout.surface_coverage_slice
    coverage = state[:, coverage_slice]
    coverage_atol = tolerance[coverage_slice]
    for row, time, segment in zip(coverage, times, segments, strict=True):
        try:
            surface._validate_solver_coverages(row, coverage_atol)
        except ModelDomainError as exc:
            raise _failure(str(exc), float(time), segment) from exc
    surface._continue_solver_coverages(coverage)
    strict_atol = np.zeros(coverage.shape[1])
    for row, time, segment in zip(coverage, times, segments, strict=True):
        try:
            surface._validate_solver_coverages(row, strict_atol)
        except ModelDomainError as exc:
            raise _failure(str(exc), float(time), segment) from exc


def canonicalize_result_state(
    model: CompiledGlobalModel,
    state: np.ndarray,
    domain_atol: np.ndarray,
    time_s: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Return a strict-domain copy without re-evaluating the physical RHS."""

    times = np.asarray(time_s, dtype=float)
    values = np.asarray(state, dtype=float)
    tolerance = np.asarray(domain_atol, dtype=float)
    if values.shape != (times.size, model.layout.size):
        raise ValueError("result state shape does not match times and model layout")
    if tolerance.shape != (model.layout.size,):
        raise ValueError("result domain_atol does not match the model layout")
    segments = _saved_segments(times, model.segments)
    if not np.all(np.isfinite(values)):
        row, column = np.argwhere(~np.isfinite(values))[0]
        raise _failure(
            f"state[{int(column)}] is not finite",
            float(times[row]),
            segments[int(row)],
        )
    accepted, negative_count = _accepted_state_for_result(
        values,
        tolerance,
        lower_bounds=model.state_lower_bounds,
        upper_bounds=model.state_upper_bounds,
        time_s=times,
        segments=model.segments,
    )
    # Normalize component bounds, then the simplex, then coupled zone domains.
    _map_surface_state(model, accepted, tolerance, times, segments)
    electron_count = 0
    for row, time, segment in zip(accepted, times, segments, strict=True):
        for zone_id in model.layout.zone_ids:
            density = row[model.layout.density_slices[zone_id]]
            electron_count += _canonicalize_electron_energy(
                model,
                row,
                tolerance,
                segment,
                float(time),
                zone_id,
                density,
            )
            _validate_heavy_energy(model, row, segment, float(time), zone_id, density)
    return accepted, negative_count, electron_count


__all__ = ["_accepted_state_for_result", "canonicalize_result_state"]
