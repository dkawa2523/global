"""Heavy-particle internal-energy closure and electron elastic heating."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from plasma_global.errors import CaseValidationError, ModelDomainError

BOLTZMANN_J_K = 1.380649e-23
E_CHARGE = 1.602176634e-19
ELECTRON_MASS_KG = 9.1093837139e-31


def _validate_heat_capacities(cv: np.ndarray) -> None:
    if cv.ndim != 1 or cv.size == 0 or np.any(cv <= 0.0) or not np.all(np.isfinite(cv)):
        raise CaseValidationError("cv_over_kb must be a finite positive species array")


def _validate_wall_energy_arrays(
    wall_temperature: np.ndarray, relaxation: np.ndarray
) -> None:
    if wall_temperature.ndim != 1 or relaxation.shape != wall_temperature.shape:
        raise CaseValidationError(
            "wall temperature and relaxation must be equal zone arrays"
        )
    if (
        np.any(wall_temperature <= 0.0)
        or np.any(relaxation < 0.0)
        or not np.all(np.isfinite([*wall_temperature, *relaxation]))
    ):
        raise CaseValidationError("wall-energy parameters are invalid")


@dataclass(frozen=True, slots=True)
class HeavyParticleThermodynamics:
    """Constant species heat capacities and the ideal-mixture energy law."""

    cv_over_kb: np.ndarray

    def __post_init__(self) -> None:
        cv = np.asarray(self.cv_over_kb, dtype=float)
        _validate_heat_capacities(cv)
        frozen = np.array(cv, copy=True)
        frozen.setflags(write=False)
        object.__setattr__(self, "cv_over_kb", frozen)

    def heat_capacity_J_m3_K(self, densities_m3: np.ndarray) -> np.ndarray:
        densities = np.asarray(densities_m3, dtype=float)
        if densities.ndim != 2 or densities.shape[1] != self.cv_over_kb.size:
            raise ValueError(
                "density array shape is incompatible with species heat capacities"
            )
        return BOLTZMANN_J_K * (densities @ self.cv_over_kb)

    def energy_J_m3(
        self, densities_m3: np.ndarray, temperature_K: np.ndarray
    ) -> np.ndarray:
        temperature = np.asarray(temperature_K, dtype=float)
        capacity = self.heat_capacity_J_m3_K(densities_m3)
        if temperature.shape != capacity.shape or np.any(temperature <= 0.0):
            raise ModelDomainError("gas temperature must be positive for every zone")
        return capacity * temperature

    def temperature_K(
        self, densities_m3: np.ndarray, energy_J_m3: np.ndarray
    ) -> np.ndarray:
        energy = np.asarray(energy_J_m3, dtype=float)
        capacity = self.heat_capacity_J_m3_K(densities_m3)
        if (
            energy.shape != capacity.shape
            or np.any(energy < 0.0)
            or not np.all(np.isfinite(energy))
        ):
            raise ModelDomainError(
                "heavy-particle energy must be finite and nonnegative"
            )
        if np.any(capacity <= 0.0):
            raise ModelDomainError("gas temperature is undefined for an empty mixture")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            return energy / capacity

    def enthalpy_J_m3(
        self, densities_m3: np.ndarray, energy_J_m3: np.ndarray
    ) -> np.ndarray:
        """Return U+p, the energy carried by an ideal-mixture volume flow."""

        density = np.asarray(densities_m3, dtype=float)
        if np.any(density < 0.0):
            raise ModelDomainError(
                "heavy-particle enthalpy requires nonnegative densities"
            )
        self.heat_capacity_J_m3_K(density)
        capacity_over_kb = density @ self.cv_over_kb
        if np.any(capacity_over_kb <= 0.0):
            raise ModelDomainError(
                "heavy-particle enthalpy is undefined for an empty mixture"
            )
        energy = np.asarray(energy_J_m3, dtype=float)
        return energy + energy * np.sum(density, axis=1) / capacity_over_kb

    def species_enthalpy_J(self, temperature_K: float) -> np.ndarray:
        if not math.isfinite(temperature_K) or temperature_K <= 0.0:
            raise ModelDomainError("inlet gas temperature must be finite and positive")
        return BOLTZMANN_J_K * temperature_K * (self.cv_over_kb + 1.0)

    def internal_energy_source_J_m3_s(
        self, species_rhs_m3_s: np.ndarray, temperature_K: np.ndarray
    ) -> np.ndarray:
        """Energy carried by species entering or leaving at mixture temperature."""

        return BOLTZMANN_J_K * temperature_K * (species_rhs_m3_s @ self.cv_over_kb)


@dataclass(frozen=True, slots=True)
class HeavyEnergyClosure:
    """Evolved gas energy with a mixture EOS and wall relaxation law."""

    thermodynamics: HeavyParticleThermodynamics
    wall_temperature_K: np.ndarray
    wall_relaxation_s_inv: np.ndarray

    def __post_init__(self) -> None:
        wall = np.asarray(self.wall_temperature_K, dtype=float)
        relaxation = np.asarray(self.wall_relaxation_s_inv, dtype=float)
        _validate_wall_energy_arrays(wall, relaxation)
        for name, array in (
            ("wall_temperature_K", wall),
            ("wall_relaxation_s_inv", relaxation),
        ):
            frozen = np.array(array, copy=True)
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)

    @property
    def cv_over_kb(self) -> np.ndarray:
        return self.thermodynamics.cv_over_kb

    def heat_capacity_J_m3_K(self, densities_m3: np.ndarray) -> np.ndarray:
        return self.thermodynamics.heat_capacity_J_m3_K(densities_m3)

    def energy_J_m3(
        self, densities_m3: np.ndarray, temperature_K: np.ndarray
    ) -> np.ndarray:
        return self.thermodynamics.energy_J_m3(densities_m3, temperature_K)

    def temperature_K(
        self, densities_m3: np.ndarray, energy_J_m3: np.ndarray
    ) -> np.ndarray:
        return self.thermodynamics.temperature_K(densities_m3, energy_J_m3)

    def validate_energy(
        self, densities_m3: np.ndarray, energy_J_m3: np.ndarray
    ) -> None:
        temperature = self.temperature_K(densities_m3, energy_J_m3)
        if np.any(np.asarray(energy_J_m3) <= 0.0):
            raise ModelDomainError("gas internal energy is nonpositive")
        if not np.all(np.isfinite(temperature)) or np.any(temperature <= 0.0):
            raise ModelDomainError("gas temperature is not finite and positive")

    def wall_exchange_J_m3_s(
        self,
        densities_m3: np.ndarray,
        energy_J_m3: np.ndarray,
        *,
        wall_temperature_K: np.ndarray | None = None,
    ) -> np.ndarray:
        wall = (
            self.wall_temperature_K
            if wall_temperature_K is None
            else wall_temperature_K
        )
        equilibrium = self.energy_J_m3(densities_m3, wall)
        return -self.wall_relaxation_s_inv * (np.asarray(energy_J_m3) - equilibrium)


def elastic_electron_heating_J_m3_s(
    *,
    electron_density_m3: float,
    electron_temperature_eV: float,
    gas_temperature_K: float,
    neutral_densities_m3: np.ndarray,
    neutral_masses_kg: np.ndarray,
    momentum_rate_coefficients_m3_s: np.ndarray,
) -> float:
    """Electron-to-heavy elastic transfer using momentum-transfer rates.

    A positive value heats the gas and the identical value must be removed
    from the electron-energy ledger.
    """

    density = np.asarray(neutral_densities_m3, dtype=float)
    masses = np.asarray(neutral_masses_kg, dtype=float)
    rates = np.asarray(momentum_rate_coefficients_m3_s, dtype=float)
    if density.shape != masses.shape or rates.shape != density.shape:
        raise ValueError("elastic-transfer arrays must have equal shape")
    if np.any(density < 0.0) or np.any(masses <= 0.0) or np.any(rates < 0.0):
        raise ModelDomainError(
            "elastic-transfer inputs are outside their physical domain"
        )
    gas_temperature_eV = BOLTZMANN_J_K * float(gas_temperature_K) / E_CHARGE
    temperature_difference_eV = float(electron_temperature_eV) - gas_temperature_eV
    transfer = np.sum(3.0 * ELECTRON_MASS_KG / masses * density * rates)
    value = (
        float(electron_density_m3)
        * E_CHARGE
        * temperature_difference_eV
        * float(transfer)
    )
    if not math.isfinite(value):
        raise ModelDomainError("elastic electron heating is non-finite")
    return value


__all__ = [
    "BOLTZMANN_J_K",
    "HeavyEnergyClosure",
    "HeavyParticleThermodynamics",
    "elastic_electron_heating_J_m3_s",
]
