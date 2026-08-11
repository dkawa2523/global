"""Shared deterministic profiles for property-based quality tests."""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

settings.register_profile(
    "ci",
    max_examples=100,
    deadline=None,
    suppress_health_check=(HealthCheck.too_slow,),
)
settings.register_profile(
    "nightly",
    max_examples=1_000,
    deadline=None,
    suppress_health_check=(HealthCheck.too_slow,),
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))
