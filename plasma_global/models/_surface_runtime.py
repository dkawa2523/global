"""Private hot-path evaluator for compiled surface chemistry."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from plasma_global.errors import ModelDomainError
from plasma_global.models._surface_kernels import _coverage_multiplier, _Kernel

_EMPTY_RATES: Mapping[str, float] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class _SurfaceRuntimeResult:
    gas_derivative_m3_s: np.ndarray
    coverage_derivative_s_inv: np.ndarray
    gas_heating_J_m3_s: np.ndarray
    rates_m2_s: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class _RateContext:
    gas_densities_m3: np.ndarray
    gas_temperature_K: np.ndarray
    ion_flux_m2_s: Mapping[tuple[str, str], float]
    ion_energy_eV_by_surface: np.ndarray
    temperature_K_by_surface: np.ndarray


@dataclass(frozen=True, slots=True)
class _SurfaceRuntime:
    surface_ids: tuple[str, ...]
    surface_temperatures_K: tuple[float, ...]
    zone_count: int
    gas_species_count: int
    coverage_count: int
    domain_atol: float
    free_state_indices: tuple[np.ndarray, ...]
    free_state_occupancies: tuple[np.ndarray, ...]
    kernels: tuple[_Kernel, ...]

    def continue_coverages(self, state: np.ndarray) -> None:
        """Continue solver probes radially at each site-occupancy ceiling."""

        values = np.asarray(state, dtype=float)
        if values.shape[-1:] != (self.coverage_count,):
            raise ValueError("surface coverage state has the wrong shape")
        for indices, occupancies in zip(
            self.free_state_indices, self.free_state_occupancies, strict=True
        ):
            occupied = values[..., indices] @ occupancies
            if not np.any(occupied > 1.0):
                continue
            scale = np.maximum(occupied, 1.0)
            values[..., indices] = values[..., indices] / np.expand_dims(scale, -1)

    def free_coverages(self, state: np.ndarray) -> np.ndarray:
        values = np.empty(len(self.surface_ids), dtype=float)
        for surface_index, (indices, occupancies) in enumerate(
            zip(self.free_state_indices, self.free_state_occupancies, strict=True)
        ):
            values[surface_index] = 1.0 - float(occupancies @ state[indices])
        return values

    def validate_coverages(
        self, coverage_state: np.ndarray, domain_atol: np.ndarray
    ) -> None:
        coverage, tolerance = self._validated_coverage(coverage_state, domain_atol)
        self._reactive_coverages(coverage, tolerance)

    def _validated_coverage(
        self,
        coverage_state: np.ndarray,
        domain_atol: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        coverage = np.asarray(coverage_state, dtype=float)
        if coverage.shape != (self.coverage_count,):
            raise ValueError("surface coverage state has the wrong shape")
        tolerance = (
            np.full(coverage.shape, self.domain_atol, dtype=float)
            if domain_atol is None
            else np.asarray(domain_atol, dtype=float)
        )
        if tolerance.shape != coverage.shape or np.any(tolerance < 0.0):
            raise ValueError("surface domain_atol has the wrong shape")
        if np.any(coverage < -10.0 * tolerance):
            raise ModelDomainError("surface coverage left the nonnegative domain")
        return coverage, tolerance

    def _validated_arrays(
        self,
        coverage_state: np.ndarray,
        gas_densities_m3: np.ndarray,
        gas_temperature_K: np.ndarray,
        domain_atol: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        coverage, tolerance = self._validated_coverage(coverage_state, domain_atol)
        gas = np.asarray(gas_densities_m3, dtype=float)
        temperatures = np.asarray(gas_temperature_K, dtype=float)
        if gas.shape != (
            self.zone_count,
            self.gas_species_count,
        ) or temperatures.shape != (self.zone_count,):
            raise ValueError("surface gas arrays have the wrong shape")
        return coverage, gas, temperatures, tolerance

    def _reactive_coverages(
        self, coverage_state: np.ndarray, tolerance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        reactive_state = np.where(coverage_state < 0.0, 0.0, coverage_state)
        free_coverages = self.free_coverages(reactive_state)
        for surface_index, (surface_id, indices, occupancies) in enumerate(
            zip(
                self.surface_ids,
                self.free_state_indices,
                self.free_state_occupancies,
                strict=True,
            )
        ):
            free_site_tolerance = 10.0 * float(occupancies @ tolerance[indices])
            if free_coverages[surface_index] < -free_site_tolerance:
                raise ModelDomainError(
                    f"surface {surface_id!r} has negative algebraic free-site coverage"
                )
        return reactive_state, free_coverages

    def _rate_context(
        self,
        gas: np.ndarray,
        gas_temperatures: np.ndarray,
        ion_flux_m2_s: Mapping[tuple[str, str], float] | None,
        ion_energy_eV: Mapping[str, float] | None,
        surface_temperature_K: Mapping[str, float] | None,
    ) -> _RateContext:
        fluxes = ion_flux_m2_s or {}
        energies = ion_energy_eV or {}
        surface_temperatures = surface_temperature_K or {}
        ion_energies = np.asarray(
            [energies.get(surface_id, 0.0) for surface_id in self.surface_ids],
            dtype=float,
        )
        resolved_temperatures = np.asarray(
            [
                surface_temperatures.get(surface_id, default_temperature)
                for surface_id, default_temperature in zip(
                    self.surface_ids,
                    self.surface_temperatures_K,
                    strict=True,
                )
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(resolved_temperatures)) or np.any(
            resolved_temperatures <= 0.0
        ):
            raise ModelDomainError("surface temperature must be finite and positive")
        return _RateContext(
            gas_densities_m3=gas,
            gas_temperature_K=gas_temperatures,
            ion_flux_m2_s=fluxes,
            ion_energy_eV_by_surface=ion_energies,
            temperature_K_by_surface=resolved_temperatures,
        )

    @staticmethod
    def _evaluate_kernel(
        kernel: _Kernel,
        reactive_state: np.ndarray,
        free_coverages: np.ndarray,
        context: _RateContext,
    ) -> float:
        surface_index = kernel.surface_index
        coverage_multiplier = _coverage_multiplier(
            kernel, reactive_state, float(free_coverages[surface_index])
        )
        rate = kernel.rate_kernel.evaluate(
            context.gas_densities_m3[kernel.zone_index],
            float(context.gas_temperature_K[kernel.zone_index]),
            context.ion_flux_m2_s,
            float(context.ion_energy_eV_by_surface[surface_index]),
            float(context.temperature_K_by_surface[surface_index]),
        )
        rate *= coverage_multiplier
        if not math.isfinite(rate) or rate < 0.0:
            raise ModelDomainError(
                f"surface reaction {kernel.reaction.id!r} returned invalid rate "
                f"{rate!r}"
            )
        return rate

    @staticmethod
    def _accumulate_kernel(
        kernel: _Kernel,
        rate: float,
        gas_rhs: np.ndarray,
        coverage_rhs: np.ndarray,
        gas_heating: np.ndarray,
    ) -> None:
        gas_heating[kernel.zone_index] += kernel.gas_heating_J_m3_per_event_m2 * rate
        for index, delta in kernel.gas_delta:
            gas_rhs[kernel.zone_index, index] += (
                kernel.area_over_volume_m_inv * delta * rate
            )
        for index, delta in kernel.coverage_delta:
            coverage_rhs[index] += delta * rate * kernel.inverse_site_density_m2

    def evaluate(
        self,
        coverage_state: np.ndarray,
        gas_densities_m3: np.ndarray,
        gas_temperature_K: np.ndarray,
        *,
        ion_flux_m2_s: Mapping[tuple[str, str], float] | None,
        ion_energy_eV: Mapping[str, float] | None,
        surface_temperature_K: Mapping[str, float] | None,
        domain_atol: np.ndarray | None,
        collect_rates: bool,
    ) -> _SurfaceRuntimeResult:
        coverage, gas, temperatures, tolerance = self._validated_arrays(
            coverage_state, gas_densities_m3, gas_temperature_K, domain_atol
        )
        reactive_state, free_coverages = self._reactive_coverages(coverage, tolerance)
        context = self._rate_context(
            gas,
            temperatures,
            ion_flux_m2_s,
            ion_energy_eV,
            surface_temperature_K,
        )
        gas_rhs = np.zeros_like(gas)
        coverage_rhs = np.zeros_like(coverage)
        gas_heating = np.zeros(self.zone_count, dtype=float)
        rates: dict[str, float] | None = {} if collect_rates else None
        for kernel in self.kernels:
            rate = self._evaluate_kernel(
                kernel, reactive_state, free_coverages, context
            )
            if rates is not None:
                rates[kernel.rate_id] = rate
            self._accumulate_kernel(kernel, rate, gas_rhs, coverage_rhs, gas_heating)
        return _SurfaceRuntimeResult(
            gas_rhs,
            coverage_rhs,
            gas_heating,
            _EMPTY_RATES if rates is None else MappingProxyType(rates),
        )
