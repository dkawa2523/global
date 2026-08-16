"""Validate and assemble solver history for the public result contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from plasma_global.core._solver_sampling import _MAX_SAVED_POINTS
from plasma_global.core.domain import SolverSettings
from plasma_global.errors import IntegrationError

if TYPE_CHECKING:
    from plasma_global.core.compiled import CompiledGlobalModel


_EXPERIMENTAL_QUASI_STEADY_MODEL_ID = "experimental.stop_when_quasi_steady"


@dataclass(slots=True)
class _IntegrationHistory:
    """Accumulate completed segment output and SciPy work counters."""

    time_parts: list[np.ndarray] = field(default_factory=list)
    state_parts: list[np.ndarray] = field(default_factory=list)
    nfev: int = 0
    njev: int = 0
    nlu: int = 0
    completed_segments: int = 0
    saved_points: int = 0
    quasi_steady_events: int = 0

    def append(
        self,
        *,
        segment_index: int,
        solution: Any,
        time_s: np.ndarray,
        state: np.ndarray,
        observed_quasi_steady: bool,
    ) -> None:
        if segment_index > 0 and time_s.size:
            time_s = time_s[1:]
            state = state[1:, :]
        saved_points = self.saved_points + time_s.size
        if saved_points > _MAX_SAVED_POINTS:
            raise IntegrationError(
                "Solver output exceeded the fixed saved-point limit "
                f"({_MAX_SAVED_POINTS}); specify sample_interval_s or save_at_s"
            )

        self.time_parts.append(time_s)
        self.state_parts.append(state)
        self.nfev += int(solution.nfev)
        self.njev += int(solution.njev)
        self.nlu += int(getattr(solution, "nlu", 0) or 0)
        self.completed_segments += 1
        self.saved_points = saved_points
        self.quasi_steady_events += int(observed_quasi_steady)

    def arrays(self, state_size: int) -> tuple[np.ndarray, np.ndarray]:
        time_s = (
            np.concatenate(self.time_parts)
            if self.time_parts
            else np.empty(0, dtype=float)
        )
        state = (
            np.concatenate(self.state_parts, axis=0)
            if self.state_parts
            else np.empty((0, state_size), dtype=float)
        )
        return time_s, state

    def solver_stats(self, *, include_quasi_steady: bool) -> dict[str, int]:
        stats = {
            "nfev": self.nfev,
            "njev": self.njev,
            "nlu": self.nlu,
            "segments_completed": self.completed_segments,
        }
        if include_quasi_steady:
            stats["quasi_steady_events"] = self.quasi_steady_events
        return stats


def _sampling_mode(
    controls: SolverSettings, global_sample_times: np.ndarray | None
) -> str:
    if global_sample_times is None:
        return "native"
    if controls.sample_interval_s is not None:
        return "sample_interval"
    return "save_at"


def _result_metadata(
    model: CompiledGlobalModel,
    controls: SolverSettings,
    global_sample_times: np.ndarray | None,
    state_scale: np.ndarray,
    physical_domain_atol: np.ndarray,
    zeroed_negative_count: int,
    zeroed_electron_energy_count: int,
) -> dict[str, object]:
    return {
        "electron_closure": model.electron_closure.mode,
        "jacobian_sparsity_nnz": model.jac_sparsity.nnz,
        "dimensionless_atol": controls.atol,
        "state_scale": state_scale.tolist(),
        "domain_atol": physical_domain_atol.tolist(),
        "result_negative_limit_multiplier": 10.0,
        "experimental_stop_policy": (
            None
            if controls.experimental_quasi_steady_threshold_s_inv is None
            else _EXPERIMENTAL_QUASI_STEADY_MODEL_ID
        ),
        "experimental_quasi_steady_event_mode": (
            None
            if controls.experimental_quasi_steady_threshold_s_inv is None
            else "nonterminal_observation_v1"
        ),
        "sampling_mode": _sampling_mode(controls, global_sample_times),
        "provenance": {
            "accepted_state_negative_policy": (
                "At solver-to-result construction, states within 10*domain_atol below "
                "a finite lower bound are snapped to that bound; materially lower "
                "values fail. States with an unbounded lower domain remain signed. "
                "Every BDF "
                "RHS callback evaluates continuous lower-bound and surface-simplex "
                "extensions. Saved surface coverages are mapped to their closed "
                "site-occupancy simplex. Coupled electron and gas-energy domains are "
                "validated after those maps; cold round-off electron energy is stored "
                "as exact zero."
            ),
            "state_scaling_policy": (
                "BDF integrates y/state_scale with a component-wise dimensionless atol "
                "vector; every component independently uses max(abs(initial), 1) as "
                "its immutable scale."
            ),
            "accepted_state_zeroed_negative_count": zeroed_negative_count,
            "accepted_state_zeroed_electron_energy_count": (
                zeroed_electron_energy_count
            ),
        },
    }


__all__ = [
    "_EXPERIMENTAL_QUASI_STEADY_MODEL_ID",
    "_IntegrationHistory",
    "_result_metadata",
]
