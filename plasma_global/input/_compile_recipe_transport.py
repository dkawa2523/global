"""Compile recipe-level gas transport and thermal boundary commands."""

from __future__ import annotations

import numpy as np

from plasma_global.chemistry.compile import CompiledChemistry
from plasma_global.chemistry.data import ChemistryData
from plasma_global.core.transport import SCCM_TO_PARTICLES_PER_S, SegmentTransport
from plasma_global.errors import CaseValidationError
from plasma_global.input._compile_thermal import heavy_thermodynamics
from plasma_global.input.schema import CaseSpec, RunStep
from plasma_global.models.gas_energy import HeavyParticleThermodynamics


def _add_inlet_species(
    *,
    inlet_id: str,
    species_id: str,
    flow_sccm: float,
    zone_row: int,
    volume_m3: float,
    temperature_K: float,
    species_index: dict[str, int],
    thermodynamics: HeavyParticleThermodynamics | None,
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
    if thermodynamics is None:
        return
    heavy_energy[zone_row] += (
        rate_m3_s
        * thermodynamics.species_enthalpy_J(temperature_K)[species_index[species_id]]
    )


def compile_segment_transport(
    case: CaseSpec,
    chemistry_data: ChemistryData,
    chemistry: CompiledChemistry,
    step: RunStep,
) -> SegmentTransport:
    """Compile constant inlet source terms for one declared recipe step."""

    zones = tuple(case.reactor.zones.values())
    zone_index = {zone_id: index for index, zone_id in enumerate(case.reactor.zones)}
    species_index = {
        species_id: index for index, species_id in enumerate(chemistry.species_ids)
    }
    sources = np.zeros((len(zones), len(chemistry.species_ids)))
    heavy_energy = np.zeros(len(zones))
    thermodynamics = (
        heavy_thermodynamics(chemistry_data, chemistry)
        if case.models.gas_energy.kind == "evolved"
        else None
    )
    commands = step.commands.gas_inlets
    for inlet_id, inlet in case.reactor.gas_inlets.items():
        flow = commands[inlet_id].flow_sccm if inlet_id in commands else inlet.flow_sccm
        zone_row = zone_index[inlet.zone_id]
        volume = zones[zone_row].volume_m3
        for species_id, flow_sccm in flow.items():
            _add_inlet_species(
                inlet_id=inlet_id,
                species_id=species_id,
                flow_sccm=flow_sccm,
                zone_row=zone_row,
                volume_m3=volume,
                temperature_K=inlet.temperature_K,
                species_index=species_index,
                thermodynamics=thermodynamics,
                sources=sources,
                heavy_energy=heavy_energy,
            )
    return SegmentTransport(sources, heavy_energy)


def step_temperatures(
    case: CaseSpec, step: RunStep
) -> tuple[dict[str, float], dict[str, float]]:
    """Resolve explicit surface temperatures and zone wall averages."""

    surface_temperatures = {
        surface_id: command.temperature_K
        for surface_id, command in step.commands.surfaces.items()
        if command.temperature_K is not None
    }
    return surface_temperatures, wall_temperatures(
        case,
        surface_temperatures,
    )


def wall_temperatures(case: CaseSpec, overrides: dict[str, float]) -> dict[str, float]:
    values: dict[str, float] = {}
    for zone_id, zone in case.reactor.zones.items():
        surfaces = [
            (surface_id, surface)
            for surface_id, surface in case.reactor.surfaces.items()
            if surface.zone_id == zone_id
        ]
        total_area = sum(surface.area_m2 for _, surface in surfaces)
        values[zone_id] = (
            sum(
                surface.area_m2 * overrides.get(surface_id, surface.temperature_K)
                for surface_id, surface in surfaces
            )
            / total_area
            if total_area > 0.0
            else zone.initial.gas_temperature_K
        )
    return values
