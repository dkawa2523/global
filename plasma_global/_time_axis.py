"""Closed run-clock intervals with adjacent-float endpoint equivalence."""

from __future__ import annotations

import math


def inside_time_interval(time_s: float, lower_s: float, upper_s: float) -> bool:
    return math.isfinite(time_s) and math.nextafter(
        lower_s, -math.inf
    ) <= time_s <= math.nextafter(upper_s, math.inf)
