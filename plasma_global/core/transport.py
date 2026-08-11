"""Compiled inlet, pump, and directed inter-zone transport terms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plasma_global.errors import CaseValidationError

SCCM_TO_PARTICLES_PER_S = 4.477962e17


def _readonly(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise CaseValidationError(f"{name} has shape {array.shape}, expected {shape}")
    if not np.all(np.isfinite(array)):
        raise CaseValidationError(f"{name} must contain finite values")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class SegmentTransport:
    """Forcing that is constant within one compiled recipe segment."""

    particle_source_m3_s: np.ndarray
    inlet_heavy_energy_J_m3_s: np.ndarray

    @classmethod
    def zeros(cls, n_zones: int, n_species: int) -> SegmentTransport:
        """Return a no-inlet forcing with the compiled array dimensions."""

        return cls(np.zeros((n_zones, n_species)), np.zeros(n_zones))

    def __post_init__(self) -> None:
        particle = np.asarray(self.particle_source_m3_s, dtype=float)
        if (
            particle.ndim != 2
            or np.any(particle < 0.0)
            or not np.all(np.isfinite(particle))
        ):
            raise CaseValidationError(
                "particle_source_m3_s must be a finite nonnegative 2-D array"
            )
        energy = _readonly(
            self.inlet_heavy_energy_J_m3_s,
            (particle.shape[0],),
            "inlet_heavy_energy_J_m3_s",
        )
        if np.any(energy < 0.0):
            raise CaseValidationError("inlet heavy-energy source must be nonnegative")
        frozen_particle = np.array(particle, copy=True)
        frozen_particle.setflags(write=False)
        object.__setattr__(self, "particle_source_m3_s", frozen_particle)
        object.__setattr__(self, "inlet_heavy_energy_J_m3_s", energy)


@dataclass(frozen=True, slots=True)
class CompiledTransport:
    """Zone-indexed static transport coefficients; no ID lookup occurs in RHS."""

    volumes_m3: np.ndarray
    pump_frequency_s_inv: np.ndarray
    edge_from: np.ndarray
    edge_to: np.ndarray
    edge_conductance_m3_s: np.ndarray
    n_species: int

    def __post_init__(self) -> None:
        volumes = np.asarray(self.volumes_m3, dtype=float)
        if volumes.ndim != 1 or volumes.size == 0 or np.any(volumes <= 0.0):
            raise CaseValidationError(
                "volumes_m3 must be a positive one-dimensional array"
            )
        n_zones = volumes.size
        pump = _readonly(self.pump_frequency_s_inv, (n_zones,), "pump_frequency_s_inv")
        if np.any(pump < 0.0):
            raise CaseValidationError("pump_frequency_s_inv must be nonnegative")
        edge_from = np.asarray(self.edge_from, dtype=int)
        edge_to = np.asarray(self.edge_to, dtype=int)
        conductance = np.asarray(self.edge_conductance_m3_s, dtype=float)
        if (
            edge_from.ndim != 1
            or edge_to.shape != edge_from.shape
            or conductance.shape != edge_from.shape
        ):
            raise CaseValidationError(
                "edge arrays must be equal one-dimensional arrays"
            )
        if (
            np.any(edge_from < 0)
            or np.any(edge_from >= n_zones)
            or np.any(edge_to < 0)
            or np.any(edge_to >= n_zones)
        ):
            raise CaseValidationError("edge indexes are outside the zone array")
        if (
            np.any(edge_from == edge_to)
            or np.any(conductance < 0.0)
            or not np.all(np.isfinite(conductance))
        ):
            raise CaseValidationError(
                "edges must be directed between distinct zones with finite nonnegative conductance"
            )
        if self.n_species <= 0:
            raise CaseValidationError("n_species must be positive")
        frozen_volumes = np.array(volumes, copy=True)
        frozen_volumes.setflags(write=False)
        for name, value in (
            ("edge_from", edge_from),
            ("edge_to", edge_to),
            ("edge_conductance_m3_s", conductance),
        ):
            frozen = np.array(value, copy=True)
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)
        object.__setattr__(self, "volumes_m3", frozen_volumes)
        object.__setattr__(self, "pump_frequency_s_inv", pump)

    @property
    def n_zones(self) -> int:
        return int(self.volumes_m3.size)

    def evaluate(
        self,
        densities_m3: np.ndarray,
        forcing: SegmentTransport,
        *,
        electron_energy_J_m3: np.ndarray | None = None,
        heavy_energy_J_m3: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        density = np.asarray(densities_m3, dtype=float)
        expected = (self.n_zones, self.n_species)
        if density.shape != expected or forcing.particle_source_m3_s.shape != expected:
            raise ValueError(
                f"transport density/source arrays must have shape {expected}"
            )
        if not np.all(np.isfinite(density)):
            raise ValueError("transport density array must contain finite values")
        density_rhs = np.array(forcing.particle_source_m3_s, copy=True)
        density_rhs -= self.pump_frequency_s_inv[:, None] * density

        electron_rhs = None
        if electron_energy_J_m3 is not None:
            electron = np.asarray(electron_energy_J_m3, dtype=float)
            if electron.shape != (self.n_zones,):
                raise ValueError("electron-energy array has wrong shape")
            if not np.all(np.isfinite(electron)):
                raise ValueError("electron-energy array must contain finite values")
            electron_rhs = -self.pump_frequency_s_inv * electron

        heavy_rhs = None
        if heavy_energy_J_m3 is not None:
            heavy = np.asarray(heavy_energy_J_m3, dtype=float)
            if heavy.shape != (self.n_zones,):
                raise ValueError("heavy-energy array has wrong shape")
            if not np.all(np.isfinite(heavy)):
                raise ValueError("heavy-energy array must contain finite values")
            heavy_rhs = (
                forcing.inlet_heavy_energy_J_m3_s - self.pump_frequency_s_inv * heavy
            )

        for source, target, conductance in zip(
            self.edge_from, self.edge_to, self.edge_conductance_m3_s
        ):
            source_frequency = conductance / self.volumes_m3[source]
            target_frequency = conductance / self.volumes_m3[target]
            density_rhs[source] -= source_frequency * density[source]
            density_rhs[target] += target_frequency * density[source]
            if electron_rhs is not None:
                electron_rhs[source] -= source_frequency * electron_energy_J_m3[source]
                electron_rhs[target] += target_frequency * electron_energy_J_m3[source]
            if heavy_rhs is not None:
                heavy_rhs[source] -= source_frequency * heavy_energy_J_m3[source]
                heavy_rhs[target] += target_frequency * heavy_energy_J_m3[source]
        return density_rhs, electron_rhs, heavy_rhs


__all__ = ["SCCM_TO_PARTICLES_PER_S", "CompiledTransport", "SegmentTransport"]
