from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SolverResult:
    t: np.ndarray
    y: np.ndarray
    success: bool
    status: int
    message: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


class TimeIntegrator:
    def solve(self, system: Any, y0: np.ndarray, t_span: tuple[float, float], t_eval: np.ndarray | None = None) -> SolverResult:  # pragma: no cover
        raise NotImplementedError


def concatenate_solver_results(results: list[SolverResult]) -> SolverResult:
    if len(results) == 1:
        return results[0]
    t_parts = []
    y_parts = []
    for i, res in enumerate(results):
        if i == 0:
            t_parts.append(res.t)
            y_parts.append(res.y)
        else:
            t_parts.append(res.t[1:])
            y_parts.append(res.y[:, 1:])
    t = np.concatenate(t_parts)
    y = np.concatenate(y_parts, axis=1)
    success = all(r.success for r in results)
    status = 0 if success else next(r.status for r in results if not r.success)
    message = '; '.join(r.message for r in results)
    diagnostics = {'segments': len(results)}
    for key in ['nfev', 'njev', 'nlu']:
        diagnostics[key] = sum((r.diagnostics.get(key) or 0) for r in results)
    event_counts: dict[str, int] = {}
    for res in results:
        for name, count in (res.diagnostics.get('event_counts') or {}).items():
            event_counts[str(name)] = event_counts.get(str(name), 0) + int(count)
    diagnostics['event_counts'] = event_counts
    diagnostics['solver_event_count'] = int(sum(event_counts.values()))
    diagnostics['steady_state_event_count'] = int(event_counts.get('steady_state', 0))
    return SolverResult(t=t, y=y, success=success, status=status, message=message, diagnostics=diagnostics)
