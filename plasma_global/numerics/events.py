"""SciPy event construction for recipe-step integration."""

from __future__ import annotations

from typing import Any

import numpy as np

from plasma_global.numerics.tolerances import relative_rhs_norm_s_inv


def scipy_events(system: Any) -> list[Any]:
    events_cfg = system.run_config.numerics.events or {}
    steady_cfg = events_cfg.get('steady_state') if isinstance(events_cfg, dict) else None
    if not isinstance(steady_cfg, dict) or not bool(steady_cfg.get('enabled', False)):
        return []

    threshold = max(float(steady_cfg.get('relative_rhs_norm_s_inv', 1.0e-3)), 0.0)
    min_step_time_s = max(float(steady_cfg.get('min_step_time_s', 0.0)), 0.0)

    def steady_state_event(time_s: float, y: np.ndarray) -> float:
        step = system.current_step(float(time_s))
        step_start = float(getattr(step, 't_start_s', time_s))
        if float(time_s) < step_start + min_step_time_s:
            return 1.0
        dydt = system.rhs(float(time_s), y)
        return relative_rhs_norm_s_inv(system, y, dydt) - threshold

    steady_state_event.terminal = True
    steady_state_event.direction = -1.0
    steady_state_event.event_name = 'steady_state'
    return [steady_state_event]


__all__ = ['scipy_events']
