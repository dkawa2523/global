"""Prepared electron-heavy elastic energy transfer from momentum cross sections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plasma_global.chemistry.compile import CompiledChemistry, maxwell_rate_table
from plasma_global.errors import CaseValidationError, ModelDomainError
from plasma_global.models.electrons import ElectronState
from plasma_global.models.gas_energy import elastic_electron_heating_J_m3_s
from plasma_global.models.kinetics import ElectronKineticsResult


@dataclass(frozen=True, slots=True)
class _MomentumTable:
    cross_section_id: str
    neutral_index: int
    mean_energy_eV: np.ndarray
    rate_coefficient_m3_s: np.ndarray

    def coefficient(
        self,
        mean_energy_eV: float,
        kinetics: ElectronKineticsResult | None,
    ) -> float:
        if kinetics is not None and self.cross_section_id in kinetics.rate_coefficients:
            return float(kinetics.rate_coefficients[self.cross_section_id])
        value = float(mean_energy_eV)
        if value < self.mean_energy_eV[0] or value > self.mean_energy_eV[-1]:
            raise ModelDomainError(
                f"mean_energy_eV={value:g} is outside momentum-rate table "
                f"[{self.mean_energy_eV[0]:g}, {self.mean_energy_eV[-1]:g}]"
            )
        return float(np.interp(value, self.mean_energy_eV, self.rate_coefficient_m3_s))


@dataclass(frozen=True, slots=True)
class CompiledElasticHeating:
    """RHS-ready momentum-transfer evaluator; no quadrature occurs per call."""

    neutral_indices: np.ndarray
    neutral_masses_kg: np.ndarray
    tables: tuple[_MomentumTable, ...]

    def __post_init__(self) -> None:
        indices = np.asarray(self.neutral_indices, dtype=int)
        masses = np.asarray(self.neutral_masses_kg, dtype=float)
        if indices.ndim != 1 or masses.shape != indices.shape:
            raise CaseValidationError("elastic-heating neutral arrays are invalid")
        if np.any(masses <= 0.0):
            raise CaseValidationError("elastic-heating neutral masses must be positive")
        for name, values in (
            ("neutral_indices", indices),
            ("neutral_masses_kg", masses),
        ):
            frozen = np.array(values, copy=True)
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)

    def __call__(
        self,
        zone_id: str,
        electrons: ElectronState,
        reaction_densities_m3: np.ndarray,
        gas_temperature_K: float,
        kinetics: ElectronKineticsResult | None,
    ) -> float:
        del zone_id
        if electrons.density_m3 == 0.0:
            return 0.0
        coefficients = np.zeros(self.neutral_indices.size)
        for table in self.tables:
            coefficients[table.neutral_index] += table.coefficient(
                electrons.mean_energy_eV, kinetics
            )
        return elastic_electron_heating_J_m3_s(
            electron_density_m3=electrons.density_m3,
            electron_temperature_eV=electrons.temperature_eV,
            gas_temperature_K=gas_temperature_K,
            neutral_densities_m3=np.asarray(reaction_densities_m3)[
                self.neutral_indices
            ],
            neutral_masses_kg=self.neutral_masses_kg,
            momentum_rate_coefficients_m3_s=coefficients,
        )


def compile_elastic_heating(
    chemistry: CompiledChemistry,
) -> tuple[CompiledElasticHeating | None, tuple[str, ...]]:
    """Compile canonical momentum cross sections and report uncovered neutrals."""

    neutral_indices = np.flatnonzero(chemistry.charges == 0.0)
    neutral_ids = tuple(chemistry.species_ids[index] for index in neutral_indices)
    neutral_local_index = {
        chemistry.species_ids[index]: local_index
        for local_index, index in enumerate(neutral_indices)
    }
    tables: list[_MomentumTable] = []
    covered: set[str] = set()
    for cross_section in chemistry.cross_sections.values():
        if cross_section.kind != "momentum_transfer":
            continue
        if cross_section.target not in neutral_local_index:
            raise CaseValidationError(
                f"momentum cross section {cross_section.id!r} targets unknown or "
                f"non-neutral species {cross_section.target!r}"
            )
        axis, rate = maxwell_rate_table(cross_section)
        tables.append(
            _MomentumTable(
                cross_section_id=cross_section.id,
                neutral_index=neutral_local_index[cross_section.target],
                mean_energy_eV=axis,
                rate_coefficient_m3_s=rate,
            )
        )
        covered.add(cross_section.target)
    missing = tuple(sorted(set(neutral_ids) - covered))
    if not tables:
        return None, missing
    return (
        CompiledElasticHeating(
            neutral_indices=neutral_indices,
            neutral_masses_kg=chemistry.masses_kg[neutral_indices],
            tables=tuple(tables),
        ),
        missing,
    )


def momentum_cross_section_ids(chemistry: CompiledChemistry) -> tuple[str, ...]:
    return tuple(
        sorted(
            cross_section.id
            for cross_section in chemistry.cross_sections.values()
            if cross_section.kind == "momentum_transfer"
        )
    )


__all__ = [
    "CompiledElasticHeating",
    "compile_elastic_heating",
    "momentum_cross_section_ids",
]
