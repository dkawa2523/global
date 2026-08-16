"""Build deterministic global and per-segment solver output grids."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from plasma_global.core.domain import RecipeSegment, SolverSettings
from plasma_global.errors import IntegrationError, ModelConfigurationError

if TYPE_CHECKING:
    from plasma_global.core.compiled import CompiledGlobalModel


_MAX_SAVED_POINTS = 100_000


def _times_coincide(left_s: float, right_s: float) -> bool:
    """Treat only the nearest representable neighbours as the same time."""

    return bool(
        np.nextafter(left_s, -math.inf) <= right_s <= np.nextafter(left_s, math.inf)
    )


def _times_coincide_array(
    values_s: np.ndarray, references_s: np.ndarray | float
) -> np.ndarray:
    """Vector form of :func:`_times_coincide`."""

    return (values_s >= np.nextafter(references_s, -math.inf)) & (
        values_s <= np.nextafter(references_s, math.inf)
    )


def _merge_with_boundaries(requested: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Merge global sample times while giving exact boundaries precedence."""

    positions = np.searchsorted(boundaries, requested)
    left = boundaries[np.maximum(positions - 1, 0)]
    right = boundaries[np.minimum(positions, boundaries.size - 1)]
    near_boundary = _times_coincide_array(requested, left) | _times_coincide_array(
        requested, right
    )
    values = np.sort(np.concatenate((requested[~near_boundary], boundaries)))
    merged: list[float] = []
    for value in values:
        item = float(value)
        if not merged or not _times_coincide(merged[-1], item):
            merged.append(item)
    return np.asarray(merged, dtype=float)


def _explicit_sample_times(
    save_at_s: tuple[float, ...],
    *,
    start_s: float,
    end_s: float,
    boundary_count: int,
) -> np.ndarray:
    if len(save_at_s) + boundary_count > _MAX_SAVED_POINTS:
        raise ModelConfigurationError(
            f"save_at_s would exceed the fixed saved-point limit ({_MAX_SAVED_POINTS})"
        )
    requested = np.asarray(save_at_s, dtype=float)
    if requested[0] < np.nextafter(start_s, -math.inf) or requested[-1] > np.nextafter(
        end_s, math.inf
    ):
        raise ModelConfigurationError(
            f"save_at_s must lie within the recipe interval [{start_s:g}, {end_s:g}]"
        )
    return np.clip(requested, start_s, end_s)


def _interval_sample_times(
    interval: float,
    *,
    start_s: float,
    end_s: float,
    boundary_count: int,
) -> np.ndarray:
    interval_count = (end_s - start_s) / interval
    if (
        not math.isfinite(interval_count)
        or interval_count + 1 + boundary_count > _MAX_SAVED_POINTS
    ):
        raise ModelConfigurationError(
            "sample_interval_s would exceed the fixed saved-point limit "
            f"({_MAX_SAVED_POINTS})"
        )
    requested = start_s + interval * np.arange(
        math.floor(interval_count) + 1, dtype=float
    )
    return requested[requested <= np.nextafter(end_s, math.inf)]


def _global_sample_times(
    model: CompiledGlobalModel, controls: SolverSettings
) -> np.ndarray | None:
    """Build a recipe-global output grid, or select native solver points."""

    if controls.sample_interval_s is None and controls.save_at_s is None:
        return None
    start_s = model.segments[0].start_s
    end_s = model.segments[-1].end_s
    boundaries = np.asarray(
        [model.segments[0].start_s, *(item.end_s for item in model.segments)],
        dtype=float,
    )
    if boundaries.size > _MAX_SAVED_POINTS:
        raise ModelConfigurationError(
            "Recipe forcing boundaries exceed the fixed saved-point limit "
            f"({_MAX_SAVED_POINTS})"
        )
    if controls.save_at_s is not None:
        requested = _explicit_sample_times(
            controls.save_at_s,
            start_s=start_s,
            end_s=end_s,
            boundary_count=boundaries.size,
        )
    else:
        interval = controls.sample_interval_s
        if interval is None:
            raise ModelConfigurationError(
                "sample_interval_s is required when save_at_s is not configured"
            )
        requested = _interval_sample_times(
            interval,
            start_s=start_s,
            end_s=end_s,
            boundary_count=boundaries.size,
        )
    merged = _merge_with_boundaries(requested, boundaries)
    if merged.size > _MAX_SAVED_POINTS:
        raise ModelConfigurationError(
            "Requested output exceeds the fixed saved-point limit "
            f"({_MAX_SAVED_POINTS})"
        )
    return merged


def _segment_sample_times(
    global_times: np.ndarray | None, segment: RecipeSegment
) -> np.ndarray | None:
    if global_times is None:
        return None
    selected = global_times[
        (global_times >= np.nextafter(segment.start_s, -math.inf))
        & (global_times <= np.nextafter(segment.end_s, math.inf))
    ].copy()
    selected[_times_coincide_array(selected, segment.start_s)] = segment.start_s
    selected[_times_coincide_array(selected, segment.end_s)] = segment.end_s
    if (
        selected.size < 2
        or selected[0] != segment.start_s
        or selected[-1] != segment.end_s
    ):
        raise IntegrationError(
            f"Internal sampling error in segment {segment.segment_id!r}: "
            "forcing boundaries are missing"
        )
    return selected


__all__ = ["_global_sample_times", "_segment_sample_times"]
