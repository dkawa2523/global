"""Compile recipe-level gas transport and thermal boundary commands."""

from __future__ import annotations

from typing import assert_never

import numpy as np

from plasma_global.chemistry.compile import CompiledChemistry
from plasma_global.chemistry.data import ChemistryData
from plasma_global.core.transport import SCCM_TO_PARTICLES_PER_S, SegmentTransport
from plasma_global.errors import CaseValidationError
from plasma_global.input.schema import (
    CaseSpec,
    EvolvedGasEnergy,
    FixedGasEnergy,
    RecipeStepConfig,
)

_BOLTZMANN_J_K = 1.380649e-23


def _evolves_gas_energy(case: CaseSpec) -> bool:
    gas_energy = case.models.gas_energy
    if isinstance(gas_energy, EvolvedGasEnergy):
        return True
    if isinstance(gas_energy, FixedGasEnergy):
        return False
    assert_never(gas_energy)


def _add_inlet_species(
    *,
    inlet_id: str,
    species_id: str,
    flow_sccm: float,
    zone_row: int,
    volume_m3: float,
    temperature_K: float,
    species_index: dict[str, int],
    cv_by_species: dict[str, float],
    evolves_gas_energy: bool,
    sources: np.ndarray,
    heavy_energy: np.ndarray,
) -> None:
    if species_id not in species_index:
        raise CaseValidationError(
            f"gas inlet {inlet_id!r} references unknown/non-heavy "
            f"species {species_id!r}"
        )
    rate_m3_s = SCCM_TO_PARTICLES_PER_S * flow_sccm / volume_m3
    sources[zone_row, species_index[species_id]] += rate_m3_s
    if not evolves_gas_energy:
        return
    if species_id not in cv_by_species:
        raise CaseValidationError(
            f"evolved gas energy requires cv_over_kb for inlet species {species_id!r}"
        )
    heavy_energy[zone_row] += (
        rate_m3_s * (cv_by_species[species_id] + 1.0) * _BOLTZMANN_J_K * temperature_K
    )


def compile_segment_transport(
    case: CaseSpec,
    chemistry_data: ChemistryData,
    chemistry: CompiledChemistry,
    step: RecipeStepConfig,
) -> SegmentTransport:
    """Compile constant inlet source terms for one declared recipe step."""

    zones = tuple(case.reactor.zones)
    zone_index = {zone.zone_id: index for index, zone in enumerate(zones)}
    species_index = {
        species_id: index for index, species_id in enumerate(chemistry.species_ids)
    }
    cv_by_species = {
        item.id: item.cv_over_kb
        for item in chemistry_data.species
        if item.id in species_index and item.cv_over_kb is not None
    }
    sources = np.zeros((len(zones), len(chemistry.species_ids)))
    heavy_energy = np.zeros(len(zones))
    evolves_gas_energy = _evolves_gas_energy(case)
    commands = step.commands.gas_inlets
    for inlet in case.reactor.gas_inlets:
        flow = (
            commands[inlet.inlet_id].flow_sccm
            if inlet.inlet_id in commands
            else inlet.flow_sccm
        )
        zone_row = zone_index[inlet.zone_id]
        volume = zones[zone_row].volume_m3
        for species_id, flow_sccm in flow.items():
            _add_inlet_species(
                inlet_id=inlet.inlet_id,
                species_id=species_id,
                flow_sccm=flow_sccm,
                zone_row=zone_row,
                volume_m3=volume,
                temperature_K=inlet.temperature_K,
                species_index=species_index,
                cv_by_species=cv_by_species,
                evolves_gas_energy=evolves_gas_energy,
                sources=sources,
                heavy_energy=heavy_energy,
            )
    return SegmentTransport(sources, heavy_energy)


def step_temperatures(
    case: CaseSpec, step: RecipeStepConfig
) -> tuple[dict[str, float], dict[str, float]]:
    """Resolve explicit surface temperatures and zone wall averages."""

    surface_temperatures = {
        surface_id: command.temperature_K
        for surface_id, command in step.commands.surfaces.items()
        if command.temperature_K is not None
    }
    return surface_temperatures, _wall_temperatures_by_zone(
        case,
        surface_temperatures,
    )


def _wall_temperatures_by_zone(
    case: CaseSpec, overrides: dict[str, float]
) -> dict[str, float]:
    values: dict[str, float] = {}
    for zone in case.reactor.zones:
        surfaces = [
            surface
            for surface in case.reactor.surfaces
            if surface.zone_id == zone.zone_id
        ]
        total_area = sum(surface.area_m2 for surface in surfaces)
        values[zone.zone_id] = (
            sum(
                surface.area_m2
                * overrides.get(surface.surface_id, surface.temperature_K)
                for surface in surfaces
            )
            / total_area
            if total_area > 0.0
            else zone.gas_temperature_K
        )
    return values
