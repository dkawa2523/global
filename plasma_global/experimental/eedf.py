"""Small cross-section-driven EEDF closure for exploratory calculations.

This is an approximate two-term-like energy-space balance.  It is useful for
trend studies and for building local-field tables, but is not a replacement
for a validated Boltzmann solver.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

import numpy as np

ELECTRON_CHARGE_C = 1.602176634e-19
ELECTRON_MASS_KG = 9.1093837015e-31
TD_TO_V_M2 = 1.0e-21


@dataclass(frozen=True, slots=True)
class ElectronCollision:
    """One tabulated electron-neutral collision cross section."""

    collision_id: str
    target_species: str
    kind: Literal["momentum", "inelastic"]
    energy_eV: np.ndarray
    cross_section_m2: np.ndarray
    energy_loss_eV: float = 0.0

    def __post_init__(self) -> None:
        energy = np.array(self.energy_eV, dtype=float, copy=True)
        sigma = np.array(self.cross_section_m2, dtype=float, copy=True)
        if not self.collision_id or not self.target_species:
            raise ValueError("collision_id and target_species must not be empty")
        if self.kind not in {"momentum", "inelastic"}:
            raise ValueError("collision kind must be 'momentum' or 'inelastic'")
        if energy.ndim != 1 or energy.size < 2 or sigma.shape != energy.shape:
            raise ValueError(
                "collision tables must be equal 1D arrays with at least two rows"
            )
        if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(sigma)):
            raise ValueError("collision tables must contain only finite values")
        if np.any(energy < 0.0) or np.any(np.diff(energy) <= 0.0):
            raise ValueError(
                "collision energy must be nonnegative and strictly increasing"
            )
        if np.any(sigma < 0.0):
            raise ValueError("cross sections must be nonnegative")
        if not math.isfinite(self.energy_loss_eV) or self.energy_loss_eV < 0.0:
            raise ValueError("energy_loss_eV must be finite and nonnegative")
        energy.setflags(write=False)
        sigma.setflags(write=False)
        object.__setattr__(self, "energy_eV", energy)
        object.__setattr__(self, "cross_section_m2", sigma)

    def interpolate(self, energy_eV: np.ndarray) -> np.ndarray:
        return np.asarray(
            np.interp(
                energy_eV,
                self.energy_eV,
                self.cross_section_m2,
                left=0.0,
                right=float(self.cross_section_m2[-1]),
            ),
            dtype=float,
        )


@dataclass(frozen=True, slots=True)
class TwoTermResult:
    """Finite transport and rate quantities from one EEDF evaluation."""

    reduced_field_Td: float
    energy_eV: np.ndarray
    distribution_eV_inv: np.ndarray
    mean_energy_eV: float
    mobility_m2_V_s: float
    diffusion_m2_s: float
    rate_coefficients_m3_s: Mapping[str, float]
    collisional_loss_W_per_electron: float

    def __post_init__(self) -> None:
        energy = np.array(self.energy_eV, dtype=float, copy=True)
        distribution = np.array(self.distribution_eV_inv, dtype=float, copy=True)
        rates = {
            str(key): float(value) for key, value in self.rate_coefficients_m3_s.items()
        }
        scalars = (
            self.reduced_field_Td,
            self.mean_energy_eV,
            self.mobility_m2_V_s,
            self.diffusion_m2_s,
            self.collisional_loss_W_per_electron,
            *rates.values(),
        )
        if energy.ndim != 1 or distribution.shape != energy.shape:
            raise ValueError(
                "EEDF energy and distribution arrays must have the same 1D shape"
            )
        if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(distribution)):
            raise ValueError("EEDF arrays must be finite")
        if not np.all(np.isfinite(scalars)) or min(scalars) < 0.0:
            raise ValueError(
                "EEDF transport and rate quantities must be finite and nonnegative"
            )
        energy.setflags(write=False)
        distribution.setflags(write=False)
        object.__setattr__(self, "energy_eV", energy)
        object.__setattr__(self, "distribution_eV_inv", distribution)
        object.__setattr__(self, "rate_coefficients_m3_s", MappingProxyType(rates))


@dataclass(frozen=True, slots=True)
class ApproximateTwoTermEEDF:
    """Approximate steady EEDF obtained from field heating and collision loss."""

    collisions: Sequence[ElectronCollision]
    energy_min_eV: float = 1.0e-3
    energy_max_eV: float = 160.0
    energy_points: int = 240
    elastic_loss_eV: float = 0.03
    max_iterations: int = 48
    energy_eV: np.ndarray = field(init=False, repr=False)
    _speed_m_s: np.ndarray = field(init=False, repr=False)
    _sigma_by_id: Mapping[str, np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        collisions = tuple(self.collisions)
        if not collisions:
            raise ValueError("at least one collision cross section is required")
        ids = tuple(collision.collision_id for collision in collisions)
        if len(set(ids)) != len(ids):
            raise ValueError("collision IDs must be unique")
        if (
            not math.isfinite(self.energy_min_eV)
            or not math.isfinite(self.energy_max_eV)
            or self.energy_min_eV <= 0.0
            or self.energy_max_eV <= self.energy_min_eV
        ):
            raise ValueError(
                "energy grid bounds must be finite, positive, and increasing"
            )
        if self.energy_points < 32:
            raise ValueError("energy_points must be at least 32")
        if not math.isfinite(self.elastic_loss_eV) or self.elastic_loss_eV < 0.0:
            raise ValueError("elastic_loss_eV must be finite and nonnegative")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")

        energy = np.geomspace(
            self.energy_min_eV, self.energy_max_eV, self.energy_points
        )
        speed = np.sqrt(2.0 * energy * ELECTRON_CHARGE_C / ELECTRON_MASS_KG)
        sigma = {
            collision.collision_id: collision.interpolate(energy)
            for collision in collisions
        }
        energy.setflags(write=False)
        speed.setflags(write=False)
        for values in sigma.values():
            values.setflags(write=False)
        object.__setattr__(self, "collisions", collisions)
        object.__setattr__(self, "energy_eV", energy)
        object.__setattr__(self, "_speed_m_s", speed)
        object.__setattr__(self, "_sigma_by_id", MappingProxyType(sigma))

    def _mixture(
        self, target_densities_m3: Mapping[str, float]
    ) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
        target_ids = {collision.target_species for collision in self.collisions}
        densities = {
            target: float(target_densities_m3.get(target, 0.0)) for target in target_ids
        }
        if any(not math.isfinite(value) or value < 0.0 for value in densities.values()):
            raise ValueError("target densities must be finite and nonnegative")
        total_density = sum(densities.values())
        if total_density <= 0.0:
            raise ValueError("at least one collision target must have positive density")
        fractions = {
            target: value / total_density for target, value in densities.items()
        }

        momentum = np.zeros_like(self.energy_eV)
        loss_sigma_eV = np.zeros_like(self.energy_eV)
        for target, fraction in fractions.items():
            if fraction == 0.0:
                continue
            target_momentum = [
                collision
                for collision in self.collisions
                if collision.target_species == target and collision.kind == "momentum"
            ]
            if not target_momentum:
                raise ValueError(
                    f"positive-density target {target!r} needs a momentum cross section"
                )
            for collision in target_momentum:
                momentum += fraction * self._sigma_by_id[collision.collision_id]
            for collision in self.collisions:
                if collision.target_species == target and collision.kind == "inelastic":
                    loss_sigma_eV += (
                        fraction
                        * collision.energy_loss_eV
                        * self._sigma_by_id[collision.collision_id]
                    )
        if not np.any(momentum > 0.0):
            raise ValueError("mixture momentum cross section is zero")
        return total_density, densities, momentum, loss_sigma_eV

    def _distribution(
        self, shape_energy_eV: float, momentum: np.ndarray, loss_sigma_eV: np.ndarray
    ) -> np.ndarray:
        energy = self.energy_eV
        delta_energy = np.gradient(energy)
        momentum_norm = momentum / max(float(np.mean(momentum)), 1.0e-40)
        positive_loss = loss_sigma_eV[loss_sigma_eV > 0.0]
        loss_scale = float(np.mean(positive_loss)) if positive_loss.size else 1.0
        loss_norm = loss_sigma_eV / max(loss_scale, 1.0e-40)
        diffusion = (shape_energy_eV + 0.02) * np.sqrt(energy + 0.03)
        diffusion /= 1.0 + 0.35 * momentum_norm
        drift = 0.8 + 0.6 * momentum_norm
        drift += 1.4 * loss_norm / max(shape_energy_eV, 1.0e-8)
        step = np.clip(
            drift * delta_energy / np.maximum(diffusion, 1.0e-30), 1.0e-10, 90.0
        )
        log_distribution = np.zeros_like(energy)
        log_distribution[1:] = -np.cumsum(step[:-1])
        distribution = np.exp(log_distribution - np.max(log_distribution))
        distribution /= np.sqrt(energy + 0.02)
        distribution /= max(float(np.trapezoid(distribution, energy)), 1.0e-300)
        return distribution

    def _rate(self, sigma_m2: np.ndarray, distribution: np.ndarray) -> float:
        return float(
            np.trapezoid(sigma_m2 * self._speed_m_s * distribution, self.energy_eV)
        )

    def _transport(
        self, distribution: np.ndarray, momentum: np.ndarray, total_density_m3: float
    ) -> tuple[float, float, float]:
        momentum_frequency_s = total_density_m3 * self._rate(momentum, distribution)
        mobility = ELECTRON_CHARGE_C / max(
            ELECTRON_MASS_KG * momentum_frequency_s, 1.0e-300
        )
        mean_energy = float(np.trapezoid(self.energy_eV * distribution, self.energy_eV))
        diffusion = mobility * (2.0 / 3.0) * mean_energy
        return mobility, diffusion, mean_energy

    def _power_balance(
        self,
        shape_energy_eV: float,
        reduced_field_Td: float,
        total_density_m3: float,
        momentum: np.ndarray,
        loss_sigma_eV: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        distribution = self._distribution(shape_energy_eV, momentum, loss_sigma_eV)
        mobility, _, _ = self._transport(distribution, momentum, total_density_m3)
        electric_field_V_m = reduced_field_Td * TD_TO_V_M2 * total_density_m3
        heating_W = ELECTRON_CHARGE_C * mobility * electric_field_V_m**2
        inelastic_loss_W = (
            total_density_m3
            * self._rate(loss_sigma_eV, distribution)
            * ELECTRON_CHARGE_C
        )
        elastic_loss_W = (
            total_density_m3
            * self._rate(momentum, distribution)
            * self.elastic_loss_eV
            * ELECTRON_CHARGE_C
        )
        total_loss_W = inelastic_loss_W + elastic_loss_W
        mismatch = math.log(max(heating_W, 1.0e-300) / max(total_loss_W, 1.0e-300))
        return mismatch, distribution

    def evaluate(
        self,
        reduced_field_Td: float,
        target_densities_m3: Mapping[str, float],
    ) -> TwoTermResult:
        """Evaluate one local-field state for a collision-target mixture."""

        field_value = float(reduced_field_Td)
        if not math.isfinite(field_value) or field_value < 0.0:
            raise ValueError("reduced_field_Td must be finite and nonnegative")
        total_density, densities, momentum, loss_sigma_eV = self._mixture(
            target_densities_m3
        )

        lower, upper = 0.02, 220.0
        lower_mismatch, lower_distribution = self._power_balance(
            lower, field_value, total_density, momentum, loss_sigma_eV
        )
        upper_mismatch, upper_distribution = self._power_balance(
            upper, field_value, total_density, momentum, loss_sigma_eV
        )
        if lower_mismatch * upper_mismatch > 0.0:
            distribution = (
                lower_distribution
                if abs(lower_mismatch) <= abs(upper_mismatch)
                else upper_distribution
            )
        else:
            distribution = upper_distribution
            for _ in range(self.max_iterations):
                middle = math.sqrt(lower * upper)
                mismatch, distribution = self._power_balance(
                    middle, field_value, total_density, momentum, loss_sigma_eV
                )
                if abs(mismatch) < 5.0e-4:
                    break
                if lower_mismatch * mismatch <= 0.0:
                    upper = middle
                else:
                    lower = middle
                    lower_mismatch = mismatch

        mobility, diffusion, mean_energy = self._transport(
            distribution, momentum, total_density
        )
        rates = {
            collision.collision_id: self._rate(
                self._sigma_by_id[collision.collision_id], distribution
            )
            for collision in self.collisions
        }
        collisional_loss = 0.0
        for collision in self.collisions:
            if collision.kind == "inelastic":
                collisional_loss += (
                    densities[collision.target_species]
                    * rates[collision.collision_id]
                    * collision.energy_loss_eV
                    * ELECTRON_CHARGE_C
                )
        collisional_loss += (
            total_density
            * self._rate(momentum, distribution)
            * self.elastic_loss_eV
            * ELECTRON_CHARGE_C
        )
        return TwoTermResult(
            reduced_field_Td=field_value,
            energy_eV=self.energy_eV,
            distribution_eV_inv=distribution,
            mean_energy_eV=mean_energy,
            mobility_m2_V_s=mobility,
            diffusion_m2_s=diffusion,
            rate_coefficients_m3_s=rates,
            collisional_loss_W_per_electron=collisional_loss,
        )


__all__ = ["ApproximateTwoTermEEDF", "ElectronCollision", "TwoTermResult"]
