from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

import numpy as np

STATE_GROUP_ORDER = (
    'gas_densities',
    'electron_energy',
    'gas_temperature',
    'surface_coverages',
    'wall_inventory',
    'film_thickness',
)


@dataclass
class JacobianMismatch:
    relative_error: float
    row: int
    column: int
    row_label: str
    column_label: str
    finite_difference: float
    jacobian_value: float
    row_group: str = 'other'
    column_group: str = 'other'


@dataclass
class JacobianGroupSummary:
    group: str
    max_relative_error: float
    row: int
    column: int
    row_label: str
    column_label: str
    column_group: str
    finite_difference: float
    jacobian_value: float
    checked_entry_count: int = 0


@dataclass
class JacobianCheckResult:
    max_relative_error: float
    mismatches: list[JacobianMismatch]
    group_summaries: list[JacobianGroupSummary] = field(default_factory=list)


def _labels(system, size: int) -> list[str]:
    labels = list(system.state_labels())
    if len(labels) < size:
        labels.extend(f'y[{i}]' for i in range(len(labels), size))
    return labels[:size]


def _state_groups(system, size: int) -> list[tuple[str, np.ndarray]]:
    layout = getattr(system, 'state_layout', None)
    slices = getattr(layout, 'slices', {}) or {}
    groups: list[tuple[str, np.ndarray]] = []
    covered = np.zeros(size, dtype=bool)

    for name in STATE_GROUP_ORDER:
        state_slice = slices.get(name)
        if state_slice is None:
            continue
        start = max(0, min(size, int(state_slice.start)))
        stop = max(start, min(size, int(state_slice.stop)))
        if stop <= start:
            continue
        indices = np.arange(start, stop, dtype=int)
        groups.append((name, indices))
        covered[start:stop] = True

    other = np.flatnonzero(~covered)
    if other.size or not groups:
        groups.append(('other', other if other.size else np.arange(size, dtype=int)))
    return groups


def _index_groups(groups: list[tuple[str, np.ndarray]], size: int) -> list[str]:
    out = ['other'] * size
    for name, indices in groups:
        for idx in indices:
            if 0 <= int(idx) < size:
                out[int(idx)] = name
    return out


def check_jacobian(
    system,
    time_s: float,
    state: np.ndarray,
    *,
    top_n: int = 10,
    relative_step: float = 1.0e-6,
    absolute_step: float = 1.0e-8,
    skip_label_prefixes: Iterable[str] = ('theta',),
) -> JacobianCheckResult:
    labels = _labels(system, int(state.size))
    groups = _state_groups(system, int(state.size))
    index_groups = _index_groups(groups, int(state.size))
    skip_prefixes = tuple(skip_label_prefixes)
    jac = system.jacobian(time_s, state)
    jac_dense = jac.toarray() if hasattr(jac, 'toarray') else np.asarray(jac, dtype=float)
    mismatches: list[JacobianMismatch] = []
    group_best: dict[str, JacobianGroupSummary | None] = {name: None for name, _ in groups}
    group_counts: dict[str, int] = {name: 0 for name, _ in groups}

    for col, label in enumerate(labels):
        if skip_prefixes and label.startswith(skip_prefixes):
            continue
        step = max(abs(float(state[col])) * relative_step, absolute_step)
        y_plus = np.array(state, dtype=float, copy=True)
        y_minus = np.array(state, dtype=float, copy=True)
        y_plus[col] += step
        y_minus[col] -= step
        fd_col = (system.rhs(time_s, y_plus) - system.rhs(time_s, y_minus)) / (2.0 * step)
        jac_col = jac_dense[:, col]
        denom = np.maximum.reduce([np.ones_like(fd_col), np.abs(fd_col), np.abs(jac_col)])
        rel = np.abs(fd_col - jac_col) / denom
        row = int(np.argmax(rel))
        mismatches.append(
            JacobianMismatch(
                relative_error=float(rel[row]),
                row=row,
                column=col,
                row_label=labels[row],
                column_label=label,
                finite_difference=float(fd_col[row]),
                jacobian_value=float(jac_col[row]),
                row_group=index_groups[row],
                column_group=index_groups[col],
            )
        )
        for group_name, indices in groups:
            if indices.size == 0:
                continue
            group_counts[group_name] += int(indices.size)
            local_row = int(indices[int(np.argmax(rel[indices]))])
            candidate = JacobianGroupSummary(
                group=group_name,
                max_relative_error=float(rel[local_row]),
                row=local_row,
                column=col,
                row_label=labels[local_row],
                column_label=label,
                column_group=index_groups[col],
                finite_difference=float(fd_col[local_row]),
                jacobian_value=float(jac_col[local_row]),
            )
            best = group_best[group_name]
            if best is None or candidate.max_relative_error > best.max_relative_error:
                group_best[group_name] = candidate

    mismatches.sort(key=lambda item: item.relative_error, reverse=True)
    group_summaries = [
        replace(summary, checked_entry_count=group_counts[summary.group])
        for summary in group_best.values()
        if summary is not None
    ]
    group_summaries.sort(key=lambda item: item.max_relative_error, reverse=True)
    top = mismatches[:max(int(top_n), 0)]
    max_rel = mismatches[0].relative_error if mismatches else 0.0
    return JacobianCheckResult(max_relative_error=max_rel, mismatches=top, group_summaries=group_summaries)
