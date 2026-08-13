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


def _optional_heavy_cv(value: object | None, n_species: int) -> np.ndarray | None:
    if value is None:
        return None
    heavy_cv = _readonly(value, (n_species,), "heavy_cv_over_kb")
    if np.any(heavy_cv <= 0.0):
        raise CaseValidationError("heavy_cv_over_kb must be positive")
    return heavy_cv


def _validated_flow_energy(
    value: np.ndarray,
    *,
    n_zones: int,
    flow_energy_J_m3: np.ndarray | None,
    name: str,
) -> np.ndarray:
    energy = np.asarray(value, dtype=float)
    if energy.shape != (n_zones,):
        raise ValueError(f"{name} array has wrong shape")
    if not np.all(np.isfinite(energy)):
        raise ValueError(f"{name} array must contain finite values")
    flow_energy = energy if flow_energy_J_m3 is None else flow_energy_J_m3
    if flow_energy.shape != (n_zones,) or not np.all(np.isfinite(flow_energy)):
        raise ValueError(f"{name} flow-energy array must be finite with zone shape")
    return flow_energy


def _energy_transport_rhs(
    value: np.ndarray | None,
    *,
    n_zones: int,
    pump_frequency_s_inv: np.ndarray,
    inlet_J_m3_s: np.ndarray | None,
    flow_energy_J_m3: np.ndarray | None = None,
    name: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if value is None:
        return None, None
    flow_energy = _validated_flow_energy(
        value,
        n_zones=n_zones,
        flow_energy_J_m3=flow_energy_J_m3,
        name=name,
    )
    rhs = -pump_frequency_s_inv * flow_energy
    if inlet_J_m3_s is not None:
        rhs = inlet_J_m3_s + rhs
    return flow_energy, rhs


@dataclass(frozen=True, slots=True)
class SegmentTransport:
    """Constant particle and inlet-enthalpy forcing for one recipe segment."""

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
    """Zone-indexed particle and open-control-volume energy transport."""

    volumes_m3: np.ndarray
    pump_frequency_s_inv: np.ndarray
    edge_from: np.ndarray
    edge_to: np.ndarray
    edge_conductance_m3_s: np.ndarray
    n_species: int
    heavy_cv_over_kb: np.ndarray | None = None

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
                "edges must be directed between distinct zones with finite "
                "nonnegative conductance"
            )
        if self.n_species <= 0:
            raise CaseValidationError("n_species must be positive")
        heavy_cv = _optional_heavy_cv(self.heavy_cv_over_kb, self.n_species)
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
        object.__setattr__(self, "heavy_cv_over_kb", heavy_cv)

    @property
    def n_zones(self) -> int:
        return int(self.volumes_m3.size)

    def _heavy_enthalpy_J_m3(
        self, density_m3: np.ndarray, internal_energy_J_m3: np.ndarray
    ) -> np.ndarray:
        cv = self.heavy_cv_over_kb
        if cv is None:
            raise ValueError(
                "heavy-energy transport requires heavy_cv_over_kb so flow carries "
                "enthalpy rather than internal energy"
            )
        if np.any(density_m3 < 0.0):
            raise ValueError("heavy-particle enthalpy requires nonnegative densities")
        heat_capacity_over_kb = density_m3 @ cv
        if np.any(heat_capacity_over_kb <= 0.0):
            raise ValueError(
                "heavy-particle enthalpy is undefined for an empty mixture"
            )
        pressure_J_m3 = (
            internal_energy_J_m3 * np.sum(density_m3, axis=1) / heat_capacity_over_kb
        )
        return internal_energy_J_m3 + pressure_J_m3

    def _heavy_flow_energy_J_m3(
        self,
        density_m3: np.ndarray,
        internal_energy_J_m3: np.ndarray | None,
        flow_density_m3: np.ndarray | None,
    ) -> np.ndarray | None:
        if internal_energy_J_m3 is None:
            return None
        expected = (self.n_zones, self.n_species)
        density = density_m3 if flow_density_m3 is None else np.asarray(flow_density_m3)
        if density.shape != expected or not np.all(np.isfinite(density)):
            raise ValueError(
                f"heavy-flow density array must be finite with shape {expected}"
            )
        energy = np.asarray(internal_energy_J_m3, dtype=float)
        if energy.shape != (self.n_zones,) or not np.all(np.isfinite(energy)):
            raise ValueError("heavy-energy array must be finite with zone shape")
        return self._heavy_enthalpy_J_m3(density, energy)

    def evaluate(
        self,
        densities_m3: np.ndarray,
        forcing: SegmentTransport,
        *,
        electron_energy_J_m3: np.ndarray | None = None,
        heavy_energy_J_m3: np.ndarray | None = None,
        heavy_flow_densities_m3: np.ndarray | None = None,
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

        electron, electron_rhs = _energy_transport_rhs(
            electron_energy_J_m3,
            n_zones=self.n_zones,
            pump_frequency_s_inv=self.pump_frequency_s_inv,
            inlet_J_m3_s=None,
            name="electron-energy",
        )
        heavy, heavy_rhs = _energy_transport_rhs(
            heavy_energy_J_m3,
            n_zones=self.n_zones,
            pump_frequency_s_inv=self.pump_frequency_s_inv,
            inlet_J_m3_s=forcing.inlet_heavy_energy_J_m3_s,
            flow_energy_J_m3=self._heavy_flow_energy_J_m3(
                density,
                heavy_energy_J_m3,
                heavy_flow_densities_m3,
            ),
            name="heavy-energy",
        )

        for source, target, conductance in zip(
            self.edge_from,
            self.edge_to,
            self.edge_conductance_m3_s,
            strict=True,
        ):
            source_frequency = conductance / self.volumes_m3[source]
            target_frequency = conductance / self.volumes_m3[target]
            density_rhs[source] -= source_frequency * density[source]
            density_rhs[target] += target_frequency * density[source]
            if electron_rhs is not None and electron is not None:
                electron_rhs[source] -= source_frequency * electron[source]
                electron_rhs[target] += target_frequency * electron[source]
            if heavy_rhs is not None and heavy is not None:
                heavy_rhs[source] -= source_frequency * heavy[source]
                heavy_rhs[target] += target_frequency * heavy[source]
        return density_rhs, electron_rhs, heavy_rhs


__all__ = ["SCCM_TO_PARTICLES_PER_S", "CompiledTransport", "SegmentTransport"]
