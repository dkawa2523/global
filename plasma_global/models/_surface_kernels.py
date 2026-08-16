"""Private numeric kernels used by compiled surface chemistry.

This module owns rate-law evaluation and rate-model parameter lowering.  It
deliberately has no knowledge of the public ``CompiledSurfaceModel`` facade or
of reactor state layout.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from plasma_global.chemistry.data import RateModelData, ReactionData
from plasma_global.errors import CaseValidationError

BOLTZMANN_J_K = 1.380649e-23
E_CHARGE = 1.602176634e-19


@dataclass(frozen=True, slots=True)
class _CoveragePower:
    """One already-resolved coverage raised to a fixed power."""

    state_index: int | None
    exponent: float

    def evaluate(self, state: np.ndarray, free_coverage: float) -> float:
        value = (
            free_coverage
            if self.state_index is None
            else float(state[self.state_index])
        )
        value = max(value, 0.0)
        if value == 0.0 and self.exponent > 0.0:
            return 0.0
        return value**self.exponent


class _NumericRateKernel(Protocol):
    def evaluate(
        self,
        gas: np.ndarray,
        gas_temperature_K: float,
        ion_fluxes: Mapping[tuple[str, str], float],
        ion_energy_eV: float,
        surface_temperature_K: float,
    ) -> float: ...


@dataclass(frozen=True, slots=True)
class _StickingRate:
    gas_index: int
    thermal_prefactor_m_s_K_half: float

    def evaluate(
        self,
        gas: np.ndarray,
        gas_temperature_K: float,
        ion_fluxes: Mapping[tuple[str, str], float],
        ion_energy_eV: float,
        surface_temperature_K: float,
    ) -> float:
        del ion_fluxes, ion_energy_eV, surface_temperature_K
        return (
            self.thermal_prefactor_m_s_K_half
            * math.sqrt(gas_temperature_K)
            * float(gas[self.gas_index])
        )


@dataclass(frozen=True, slots=True)
class _IonAssistedRate:
    flux_key: tuple[str, str]
    yield_factor: float
    threshold_eV: float
    inverse_energy_span_eV_inv: float
    exponent: float

    def evaluate(
        self,
        gas: np.ndarray,
        gas_temperature_K: float,
        ion_fluxes: Mapping[tuple[str, str], float],
        ion_energy_eV: float,
        surface_temperature_K: float,
    ) -> float:
        del gas, gas_temperature_K, surface_temperature_K
        if ion_energy_eV <= self.threshold_eV:
            return 0.0
        energy_factor = (
            (ion_energy_eV - self.threshold_eV) * self.inverse_energy_span_eV_inv
        ) ** self.exponent
        return float(
            ion_fluxes.get(self.flux_key, 0.0) * self.yield_factor * energy_factor
        )


def _arrhenius_factor(activation_temperature_K: float, temperature_K: float) -> float:
    return math.exp(-activation_temperature_K / temperature_K)


@dataclass(frozen=True, slots=True)
class _DesorptionRate:
    frequency_s_inv: float
    site_density_m2: float
    activation_temperature_K: float

    def evaluate(
        self,
        gas: np.ndarray,
        gas_temperature_K: float,
        ion_fluxes: Mapping[tuple[str, str], float],
        ion_energy_eV: float,
        surface_temperature_K: float,
    ) -> float:
        del gas, gas_temperature_K, ion_fluxes, ion_energy_eV
        return (
            self.frequency_s_inv
            * self.site_density_m2
            * _arrhenius_factor(self.activation_temperature_K, surface_temperature_K)
        )


@dataclass(frozen=True, slots=True)
class _LangmuirHinshelwoodRate:
    coefficient_m2_s: float
    site_density_m2: float
    activation_temperature_K: float

    def evaluate(
        self,
        gas: np.ndarray,
        gas_temperature_K: float,
        ion_fluxes: Mapping[tuple[str, str], float],
        ion_energy_eV: float,
        surface_temperature_K: float,
    ) -> float:
        del gas, gas_temperature_K, ion_fluxes, ion_energy_eV
        return (
            self.coefficient_m2_s
            * self.site_density_m2**2
            * _arrhenius_factor(self.activation_temperature_K, surface_temperature_K)
        )


@dataclass(frozen=True, slots=True)
class _Kernel:
    reaction: ReactionData
    surface_index: int
    zone_index: int
    rate_id: str
    rate_kernel: _NumericRateKernel
    coverage_factor: _CoveragePower | None
    surface_mass_action: tuple[_CoveragePower, ...]
    gas_delta: tuple[tuple[int, float], ...]
    coverage_delta: tuple[tuple[int, float], ...]
    area_over_volume_m_inv: float
    inverse_site_density_m2: float
    gas_heating_J_m3_per_event_m2: float


def _coverage_multiplier(
    kernel: _Kernel, state: np.ndarray, free_coverage: float
) -> float:
    mass_action = 1.0
    for coverage_power in kernel.surface_mass_action:
        mass_action *= coverage_power.evaluate(state, free_coverage)
        if mass_action == 0.0:
            break
    coverage_factor = (
        1.0
        if kernel.coverage_factor is None
        else kernel.coverage_factor.evaluate(state, free_coverage)
    )
    return mass_action * coverage_factor


def _numeric_parameter(model: RateModelData, name: str) -> float:
    try:
        value = float(model.parameters[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise CaseValidationError(
            f"surface rate model {model.id!r} requires numeric parameter {name!r}"
        ) from exc
    if not math.isfinite(value):
        raise CaseValidationError(
            f"surface rate model {model.id!r} parameter {name!r} must be finite"
        )
    return value


def _thermal_rate(
    model: RateModelData,
    *,
    site_density_m2: float,
) -> _NumericRateKernel:
    activation_eV = _numeric_parameter(model, "activation_eV")
    activation_temperature_K = activation_eV * E_CHARGE / BOLTZMANN_J_K
    if model.kind == "desorption":
        return _DesorptionRate(
            frequency_s_inv=_numeric_parameter(model, "frequency_s_inv"),
            site_density_m2=site_density_m2,
            activation_temperature_K=activation_temperature_K,
        )
    if model.kind == "langmuir_hinshelwood":
        return _LangmuirHinshelwoodRate(
            coefficient_m2_s=_numeric_parameter(model, "A_m2_s_inv"),
            site_density_m2=site_density_m2,
            activation_temperature_K=activation_temperature_K,
        )
    raise CaseValidationError(f"unsupported surface rate model {model.kind!r}")


def compile_numeric_rate(
    model: RateModelData,
    *,
    surface_id: str,
    site_density_m2: float,
    gas_reactants: Sequence[tuple[int, float, str]],
    gas_masses_kg: np.ndarray,
) -> _NumericRateKernel:
    """Lower one validated surface rate model to its hot-path kernel."""

    if model.kind == "sticking":
        gas_index, _order, _species_id = gas_reactants[0]
        value = _numeric_parameter(model, "value")
        return _StickingRate(
            gas_index=gas_index,
            thermal_prefactor_m_s_K_half=(
                0.25
                * value
                * math.sqrt(8.0 * BOLTZMANN_J_K / (math.pi * gas_masses_kg[gas_index]))
            ),
        )
    if model.kind == "ion_assisted":
        _gas_index, _order, ion_species_id = gas_reactants[0]
        threshold = _numeric_parameter(model, "threshold_eV")
        reference = _numeric_parameter(model, "reference_energy_eV")
        if reference <= threshold:
            raise CaseValidationError(
                f"surface rate model {model.id!r} reference_energy_eV must "
                "exceed threshold_eV"
            )
        return _IonAssistedRate(
            flux_key=(surface_id, ion_species_id),
            yield_factor=_numeric_parameter(model, "yield"),
            threshold_eV=threshold,
            inverse_energy_span_eV_inv=1.0 / (reference - threshold),
            exponent=_numeric_parameter(model, "exponent"),
        )
    return _thermal_rate(model, site_density_m2=site_density_m2)
