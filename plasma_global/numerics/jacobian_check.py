from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class JacobianMismatch:
    relative_error: float
    row: int
    column: int
    row_label: str
    column_label: str
    finite_difference: float
    jacobian_value: float


@dataclass
class JacobianCheckResult:
    max_relative_error: float
    mismatches: list[JacobianMismatch]


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
    labels = system.state_labels()
    skip_prefixes = tuple(skip_label_prefixes)
    jac = system.jacobian(time_s, state)
    jac_dense = jac.toarray() if hasattr(jac, 'toarray') else np.asarray(jac, dtype=float)
    mismatches: list[JacobianMismatch] = []

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
            )
        )

    mismatches.sort(key=lambda item: item.relative_error, reverse=True)
    top = mismatches[:max(int(top_n), 0)]
    max_rel = mismatches[0].relative_error if mismatches else 0.0
    return JacobianCheckResult(max_relative_error=max_rel, mismatches=top)
