"""Typed rate-law boundary for the compiled chemistry model."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from typing_extensions import override

from plasma_global.errors import ModelConfigurationError, StateDomainError


def _density_value(
    values: np.ndarray,
    index_by_species: Mapping[str, int],
    species_id: str,
) -> float:
    return float(values[index_by_species[species_id]])


def _validate_electron_mobility(mobility_m2_V_s: float | None) -> None:
    if mobility_m2_V_s is not None and (
        not math.isfinite(mobility_m2_V_s) or mobility_m2_V_s <= 0.0
    ):
        raise StateDomainError("Electron mobility must be finite and positive")


def _validate_constant_coefficient(coefficient: float) -> None:
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ModelConfigurationError(
            "Constant rate coefficient must be finite and non-negative"
        )


@dataclass(frozen=True, slots=True)
class DensityView(Mapping[str, float]):
    """Read-only species mapping backed directly by the current state row."""

    species_ids: tuple[str, ...]
    index_by_species: Mapping[str, int]
    density_values: np.ndarray

    @override
    def __getitem__(self, species_id: str) -> float:
        return _density_value(
            self.density_values,
            self.index_by_species,
            species_id,
        )

    @override
    def __iter__(self):
        return iter(self.species_ids)

    @override
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
        _validate_electron_mobility(self.electron_mobility_m2_V_s)


class RateEvaluator(Protocol):
    def __call__(self, context: RateContext) -> float:
        """Return a non-negative SI rate coefficient."""


@dataclass(frozen=True)
class ConstantRate:
    coefficient: float

    def __post_init__(self) -> None:
        _validate_constant_coefficient(self.coefficient)

    def __call__(self, context: RateContext) -> float:
        del context
        return self.coefficient


__all__ = ["ConstantRate", "DensityView", "RateContext", "RateEvaluator"]
