"""Collect sampled history and the numerical settings needed to reproduce it."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import BDF

from plasma_global.core.domain import SolverSettings


@dataclass(slots=True)
class _SegmentProgress:
    """Sampled history plus one valid accepted state; never trial states."""

    last_time_s: float
    last_state: np.ndarray
    time_parts: list[np.ndarray] = field(default_factory=list)
    state_parts: list[np.ndarray] = field(default_factory=list)
    saved_points: int = 0

    def __post_init__(self) -> None:
        self.append(np.array([self.last_time_s]), self.last_state.reshape(1, -1))

    def append(self, time_s: np.ndarray, state: np.ndarray) -> None:
        if time_s.size:
            self.time_parts.append(time_s.copy())
            self.state_parts.append(state.copy())
            self.saved_points += time_s.size

    def arrays(self, *, include_last: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if include_last and self.time_parts[-1][-1] < self.last_time_s:
            self.append(np.array([self.last_time_s]), self.last_state.reshape(1, -1))
        elif include_last:
            self.state_parts[-1][-1] = self.last_state
        return np.concatenate(self.time_parts), np.concatenate(self.state_parts, axis=0)


@dataclass(slots=True)
class _IntegrationHistory:
    time_parts: list[np.ndarray] = field(default_factory=list)
    state_parts: list[np.ndarray] = field(default_factory=list)
    nfev: int = 0
    njev: int = 0
    nlu: int = 0
    completed_segments: int = 0
    saved_points: int = 0

    def append(
        self,
        *,
        segment_index: int,
        solution: BDF | None,
        time_s: np.ndarray,
        state: np.ndarray,
        completed: bool = True,
    ) -> None:
        if segment_index > 0:
            time_s, state = time_s[1:], state[1:, :]
        self.time_parts.append(time_s)
        self.state_parts.append(state)
        if solution is not None:
            self.nfev += int(solution.nfev)
            self.njev += int(solution.njev)
            self.nlu += int(solution.nlu)
        self.completed_segments += int(completed)
        self.saved_points += time_s.size

    def arrays(self, state_size: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.concatenate(self.time_parts),
            np.concatenate(self.state_parts, axis=0).reshape(-1, state_size),
        )

    def solver_stats(self) -> dict[str, int]:
        return {
            "nfev": self.nfev,
            "njev": self.njev,
            "nlu": self.nlu,
            "segments_completed": self.completed_segments,
        }


def _result_metadata(
    controls: SolverSettings,
    global_times: np.ndarray | None,
    state_scale: np.ndarray,
    domain_atol: np.ndarray,
) -> dict[str, object]:
    sampling = (
        "native"
        if global_times is None
        else (
            "sample_interval" if controls.sample_interval_s is not None else "save_at"
        )
    )
    return {
        "solver_settings": {
            "method": "BDF",
            "rtol": controls.rtol,
            "atol": controls.atol,
            "first_step_s": controls.first_step_s,
            "max_step_s": controls.max_step_s,
            "sample_interval_s": controls.sample_interval_s,
            "save_at_s": controls.save_at_s,
        },
        "state_scale": state_scale.tolist(),
        "domain_atol": domain_atol.tolist(),
        "sampling_mode": sampling,
    }
