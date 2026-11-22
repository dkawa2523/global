"""Bind model-owned physical domains to their compiled state blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from typing_extensions import override

from plasma_global.core._runtime_state import RuntimeStatePreparer
from plasma_global.core._state_blocks import StateBlockLayout, StateDomain
from plasma_global.core.domain import RecipeSegment
from plasma_global.models.electrons import ElectronEnergyClosure

if TYPE_CHECKING:
    from plasma_global.core.compiled import CompiledGlobalModel
    from plasma_global.models.surface import CompiledSurfaceModel


@dataclass(frozen=True, slots=True)
class NonnegativeDomain:
    nonnegative: ClassVar[bool] = True

    def continue_trial(self, values: np.ndarray) -> None:
        pass

    def canonicalize(
        self,
        state: np.ndarray,
        tolerance: np.ndarray,
        segment: RecipeSegment,
        time_s: float,
    ) -> int:
        return 0


@dataclass(frozen=True, slots=True)
class ElectronEnergyDomain(NonnegativeDomain):
    model: CompiledGlobalModel
    zone_id: str
    energy_index: int

    @override
    def canonicalize(
        self,
        state: np.ndarray,
        tolerance: np.ndarray,
        segment: RecipeSegment,
        time_s: float,
    ) -> int:
        density = state[self.model.layout.density_slices[self.zone_id]]
        charge = float(self.model.charges @ density)
        electron_density = RuntimeStatePreparer(self.model).electron_density(
            time_s, self.zone_id, charge, segment
        )
        closure = self.model.electron_closure
        if not isinstance(closure, ElectronEnergyClosure):
            raise RuntimeError("An electron-energy state requires its closure")
        energy = float(state[self.energy_index])
        canonical = closure.canonical_energy_density(
            net_heavy_charge_density_m3=charge,
            energy_density_J_m3=energy,
            energy_atol_J_m3=float(tolerance[self.energy_index]),
            electron_density_m3=electron_density,
        )
        table = self.model.electron_kinetics_by_zone.get(self.zone_id)
        if (
            electron_density > 0.0
            and table is not None
            and table.lookup == "mean_energy"
        ):
            electron = closure.evaluate(
                net_heavy_charge_density_m3=charge,
                energy_density_J_m3=canonical,
                electron_density_m3=electron_density,
                reduced_field_Td=None,
            )
            table.query_value(electron.mean_energy_eV)
        state[self.energy_index] = canonical
        return int(canonical != energy)


@dataclass(frozen=True, slots=True)
class HeavyEnergyDomain(NonnegativeDomain):
    model: CompiledGlobalModel
    zone_id: str
    energy_index: int

    @override
    def canonicalize(
        self,
        state: np.ndarray,
        tolerance: np.ndarray,
        segment: RecipeSegment,
        time_s: float,
    ) -> int:
        closure = self.model.heavy_energy_closure
        if closure is None:
            raise RuntimeError("A heavy-energy state requires its closure")
        density = state[self.model.layout.density_slices[self.zone_id]]
        closure.validate_energy(density.reshape(1, -1), state[[self.energy_index]])
        return 0


@dataclass(frozen=True, slots=True)
class SurfaceCoverageDomain(NonnegativeDomain):
    model: CompiledSurfaceModel
    indices: tuple[int, ...]

    @override
    def continue_trial(self, values: np.ndarray) -> None:
        self.model.continue_solver_coverages(values)

    @override
    def canonicalize(
        self,
        state: np.ndarray,
        tolerance: np.ndarray,
        segment: RecipeSegment,
        time_s: float,
    ) -> int:
        indices = list(self.indices)
        values = state[indices]
        self.model.validate_solver_coverages(values, tolerance[indices])
        self.model.continue_solver_coverages(values)
        self.model.validate_solver_coverages(values, np.zeros(values.size))
        state[indices] = values
        return 0


def block_domain(model: CompiledGlobalModel, layout: StateBlockLayout) -> StateDomain:
    if layout.kind == "electron_energy":
        return ElectronEnergyDomain(model, layout.owner_id, layout.indices[0])
    if layout.kind == "heavy_energy":
        return HeavyEnergyDomain(model, layout.owner_id, layout.indices[0])
    if layout.kind == "coverage":
        if model.surface_model is None:
            raise RuntimeError("A coverage state requires its surface model")
        return SurfaceCoverageDomain(model.surface_model, layout.indices)
    return NonnegativeDomain()
