"""Canonicalize accepted BDF states for the strict public result domain."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from plasma_global.core.compiled import CompiledGlobalModel
from plasma_global.core.domain import RecipeSegment
from plasma_global.errors import IntegrationError, ModelDomainError

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
        f"{accepted[time_index, state_index]:.6e}",
        failure_time_s=None if time_s is None else float(time_s[time_index]),
    )


def _accepted_state_for_result(
    state: np.ndarray,
    domain_atol: np.ndarray,
    *,
    time_s: np.ndarray | None = None,
    segments: Sequence[RecipeSegment] = (),
    nonnegative_indices: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, int]:
    """Reject material negative values and zero tolerance-scale excursions."""

    accepted = np.array(state, dtype=float, copy=True)
    tolerance = np.asarray(domain_atol, dtype=float).reshape(1, -1)
    nonnegative = (
        np.ones(accepted.shape[1], dtype=bool)
        if nonnegative_indices is None
        else np.zeros(accepted.shape[1], dtype=bool)
    )
    if nonnegative_indices is not None:
        nonnegative[list(nonnegative_indices)] = True
    negative = (accepted < 0.0) & nonnegative
    _raise_material_bound_failure(
        (accepted < -_DOMAIN_LIMIT * tolerance) & nonnegative,
        accepted,
        "below -10*domain_atol",
        time_s,
        segments,
    )
    zeroed_count = int(np.count_nonzero(negative))
    accepted[negative] = 0.0
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
        f"segment {segment.segment_id!r} at t={time:.16g} s: {message}",
        failure_time_s=time,
        failure_segment_id=segment.segment_id,
    )


def _canonicalize_coupled_blocks(
    model: CompiledGlobalModel,
    accepted: np.ndarray,
    tolerance: np.ndarray,
    times: np.ndarray,
    segments: tuple[RecipeSegment, ...],
) -> int:
    electron_count = 0
    for row, time, owner in zip(accepted, times, segments, strict=True):
        for block in model.state_blocks:
            try:
                count = block.domain.canonicalize(row, tolerance, owner, float(time))
            except ModelDomainError as exc:
                raise _failure(str(exc), float(time), owner) from exc
            if block.layout.kind == "electron_energy":
                electron_count += count
    return electron_count


def canonicalize_result_state(
    model: CompiledGlobalModel,
    state: np.ndarray,
    domain_atol: np.ndarray,
    time_s: np.ndarray,
    *,
    segment: RecipeSegment | None = None,
) -> tuple[np.ndarray, int, int]:
    """Return a strict-domain copy without re-evaluating the physical RHS."""

    times = np.asarray(time_s, dtype=float)
    values = np.asarray(state, dtype=float)
    tolerance = np.asarray(domain_atol, dtype=float)
    if values.shape != (times.size, model.layout.size):
        raise ValueError("result state shape does not match times and model layout")
    if tolerance.shape != (model.layout.size,):
        raise ValueError("result domain_atol does not match the model layout")
    segments = (
        _saved_segments(times, model.segments)
        if segment is None
        else (segment,) * times.size
    )
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
        time_s=times,
        segments=model.segments if segment is None else (segment,),
        nonnegative_indices=tuple(
            index
            for block in model.state_blocks
            if block.domain.nonnegative
            for index in block.layout.indices
        ),
    )
    electron_count = _canonicalize_coupled_blocks(
        model, accepted, tolerance, times, segments
    )
    return accepted, negative_count, electron_count


__all__ = ["_accepted_state_for_result", "canonicalize_result_state"]
