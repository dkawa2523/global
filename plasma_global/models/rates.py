"""Typed rate-law boundary for the compiled chemistry model."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from plasma_global.core.exceptions import ModelConfigurationError, StateDomainError


@dataclass(frozen=True, slots=True)
class DensityView(Mapping[str, float]):
    """Read-only species mapping backed directly by the current state row."""

    species_ids: tuple[str, ...]
    index_by_species: Mapping[str, int]
    values: np.ndarray

    def __getitem__(self, species_id: str) -> float:
        return float(self.values[self.index_by_species[species_id]])

    def __iter__(self):
        return iter(self.species_ids)

    def __len__(self) -> int:
        return len(self.species_ids)


@dataclass(frozen=True, slots=True)
class RateContext:
    """Local plasma quantities available to one rate-coefficient evaluator.

    ``mean_energy_eV`` and ``electron_temperature_eV`` are deliberately
    separate.  A rate model must choose which quantity its calibration uses.
    """

    time_s: float
    zone_id: str
    gas_temperature_K: float
    pressure_Pa: float
    electron_density_m3: float
    mean_energy_eV: float
    electron_temperature_eV: float
    reduced_field_Td: float | None
    electron_mobility_m2_V_s: float | None
    rate_coefficients: Mapping[str, float]
    densities_m3: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.electron_mobility_m2_V_s is not None and (
            not math.isfinite(self.electron_mobility_m2_V_s)
            or self.electron_mobility_m2_V_s <= 0.0
        ):
            raise StateDomainError("Electron mobility must be finite and positive")


class RateEvaluator(Protocol):
    def __call__(self, context: RateContext) -> float:
        """Return a non-negative SI rate coefficient."""


@dataclass(frozen=True)
class ConstantRate:
    coefficient: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.coefficient) or self.coefficient < 0.0:
            raise ModelConfigurationError(
                "Constant rate coefficient must be finite and non-negative"
            )

    def __call__(self, _context: RateContext) -> float:
        return self.coefficient


__all__ = ["ConstantRate", "DensityView", "RateContext", "RateEvaluator"]
