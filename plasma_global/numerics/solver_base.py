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
