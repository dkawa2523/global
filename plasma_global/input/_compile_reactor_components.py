"""Compile the static physical components of a schema-v3 reactor."""

from __future__ import annotations

from typing import Literal, assert_never

import numpy as np

from plasma_global.chemistry.compile import CompiledChemistry
from plasma_global.chemistry.data import ChemistryData
from plasma_global.core.domain import Zone
from plasma_global.core.transport import CompiledTransport
from plasma_global.errors import CaseValidationError
from plasma_global.input.schema import (
    AmbipolarWallTransport,
    BohmWallTransport,
    CaseSpec,
    EvolvedGasEnergy,
    FixedGasEnergy,
    OffWallTransport,
    PrescribedFrequencyWallTransport,
    SurfaceConfig,
)
from plasma_global.models.gas_energy import HeavyEnergyClosure
from plasma_global.models.surface import CompiledSurfaceModel, SurfaceGeometry
from plasma_global.models.walls import AutoBohmHFactor, BoundaryReaction, WallBoundary


def compile_zones(case: CaseSpec) -> tuple[Zone, ...]:
    """Compile the immutable zone geometry used by the runtime."""

    return tuple(
        Zone(zone.zone_id, zone.volume_m3, zone.gas_temperature_K)
        for zone in case.reactor.zones
    )


def compile_static_transport(
    case: CaseSpec,
    chemistry: CompiledChemistry,
    heavy_energy_closure: HeavyEnergyClosure | None,
) -> CompiledTransport:
    """Compile pumps, inter-zone edges, and fixed transport geometry."""

    zones = tuple(case.reactor.zones)
    zone_index = {zone.zone_id: index for index, zone in enumerate(zones)}
    pump_frequency = np.zeros(len(zones))
    for pump in case.reactor.pumps:
        index = zone_index[pump.zone_id]
        pump_frequency[index] += pump.speed_m3_s / zones[index].volume_m3
    return CompiledTransport(
        volumes_m3=np.asarray([zone.volume_m3 for zone in zones]),
        pump_frequency_s_inv=pump_frequency,
        edge_from=np.asarray(
            [zone_index[edge.from_zone] for edge in case.reactor.edges], dtype=int
        ),
        edge_to=np.asarray(
            [zone_index[edge.to_zone] for edge in case.reactor.edges], dtype=int
        ),
        edge_conductance_m3_s=np.asarray(
            [edge.conductance_m3_s for edge in case.reactor.edges]
        ),
        n_species=len(chemistry.species_ids),
        heavy_cv_over_kb=(
            None if heavy_energy_closure is None else heavy_energy_closure.cv_over_kb
        ),
    )


def compile_heavy_energy(
    case: CaseSpec,
    chemistry_data: ChemistryData,
    chemistry: CompiledChemistry,
) -> HeavyEnergyClosure | None:
    """Compile the optional evolved heavy-particle energy closure."""

    config = case.models.gas_energy
    if isinstance(config, FixedGasEnergy):
        return None
    if not isinstance(config, EvolvedGasEnergy):
        assert_never(config)
    return HeavyEnergyClosure(
        cv_over_kb=_heavy_heat_capacities(chemistry_data, chemistry),
        wall_temperature_K=_zone_wall_temperatures(case),
        wall_relaxation_s_inv=np.asarray(
            [
                config.wall_energy_relaxation_s_inv_by_zone.get(zone.zone_id, 0.0)
                for zone in case.reactor.zones
            ]
        ),
    )


def compile_surface_model(
    case: CaseSpec,
    chemistry_data: ChemistryData,
    chemistry: CompiledChemistry,
) -> CompiledSurfaceModel | None:
    """Compile optional surface kinetics against reactor geometry."""

    if case.models.surface_kinetics is None:
        return None
    surfaces = tuple(
        SurfaceGeometry(
            surface_id=surface.surface_id,
            zone_id=surface.zone_id,
            area_m2=surface.area_m2,
            site_density_m2=surface.site_density_m2,
            temperature_K=surface.temperature_K,
            initial_coverages=surface.initial_coverages,
        )
        for surface in case.reactor.surfaces
    )
    if not surfaces:
        raise CaseValidationError(
            "models.surface_kinetics requires at least one reactor surface"
        )
    return CompiledSurfaceModel(
        chemistry=chemistry_data,
        gas_species_ids=chemistry.species_ids,
        gas_masses_kg=chemistry.masses_kg,
        zone_ids=tuple(zone.zone_id for zone in case.reactor.zones),
        zone_volumes_m3=np.asarray([zone.volume_m3 for zone in case.reactor.zones]),
        surfaces=surfaces,
        domain_atol=case.solver.atol,
    )


def compile_walls(
    case: CaseSpec,
    chemistry: CompiledChemistry,
) -> tuple[WallBoundary, ...]:
    """Compile wall transport and chemistry for every enabled surface."""

    walls: list[WallBoundary] = []
    for surface in case.reactor.surfaces:
        transport = surface.wall_transport
        if isinstance(transport, OffWallTransport):
            continue
        if isinstance(transport, BohmWallTransport):
            bohm_factor, auto_bohm_h_factor = _bohm_h_factor(case, surface)
            boundary = _compile_wall_boundary(
                chemistry,
                zone_id=surface.zone_id,
                area_m2=surface.area_m2,
                surface_id=surface.surface_id,
                sheath_energy_eV=surface.ion_impact_energy_eV,
                transport_kind="bohm",
                bohm_factor=bohm_factor,
                auto_bohm_h_factor=auto_bohm_h_factor,
            )
        elif isinstance(transport, PrescribedFrequencyWallTransport):
            boundary = _compile_wall_boundary(
                chemistry,
                zone_id=surface.zone_id,
                area_m2=surface.area_m2,
                surface_id=surface.surface_id,
                sheath_energy_eV=surface.ion_impact_energy_eV,
                transport_kind="prescribed_frequency",
                prescribed_frequency_s_inv=transport.frequency_s_inv,
            )
        elif isinstance(transport, AmbipolarWallTransport):
            zone = next(
                item for item in case.reactor.zones if item.zone_id == surface.zone_id
            )
            boundary = _compile_wall_boundary(
                chemistry,
                zone_id=surface.zone_id,
                area_m2=surface.area_m2,
                surface_id=surface.surface_id,
                sheath_energy_eV=surface.ion_impact_energy_eV,
                transport_kind="ambipolar",
                diffusion_coefficient_m2_s=transport.diffusion_coefficient_m2_s,
                diffusion_length_m=(
                    transport.diffusion_length_m or zone.volume_m3 / surface.area_m2
                ),
            )
        else:
            assert_never(transport)
        walls.append(boundary)
    return tuple(walls)


def _heavy_heat_capacities(
    chemistry_data: ChemistryData,
    chemistry: CompiledChemistry,
) -> np.ndarray:
    """Return heat capacities in compiled species order or report all omissions."""

    species = {item.id: item for item in chemistry_data.species}
    missing_cv = [
        species_id
        for species_id in chemistry.species_ids
        if species[species_id].cv_over_kb is None
    ]
    if missing_cv:
        raise CaseValidationError(
            "evolved gas energy requires cv_over_kb for every heavy gas species; "
            f"missing {missing_cv}"
        )
    return np.asarray(
        [species[species_id].cv_over_kb for species_id in chemistry.species_ids],
        dtype=float,
    )


def _zone_wall_temperatures(case: CaseSpec) -> np.ndarray:
    """Area-average each zone's surfaces, falling back to its gas temperature."""

    wall_temperatures: list[float] = []
    for zone in case.reactor.zones:
        surfaces = [
            item for item in case.reactor.surfaces if item.zone_id == zone.zone_id
        ]
        total_area = sum(item.area_m2 for item in surfaces)
        wall_temperatures.append(
            sum(item.area_m2 * item.temperature_K for item in surfaces) / total_area
            if total_area > 0.0
            else zone.gas_temperature_K
        )
    return np.asarray(wall_temperatures)


def _bohm_h_factor(
    case: CaseSpec,
    surface: SurfaceConfig,
) -> tuple[float, AutoBohmHFactor | None]:
    config = surface.wall_transport
    if not isinstance(config, BohmWallTransport):
        raise TypeError("_bohm_h_factor requires Bohm wall transport")
    if config.h_factor != "auto":
        return config.h_factor, None
    zone = next(item for item in case.reactor.zones if item.zone_id == surface.zone_id)
    length = config.characteristic_length_m or zone.volume_m3 / surface.area_m2
    closure = AutoBohmHFactor(
        characteristic_length_m=length,
        ion_neutral_cross_section_m2=config.ion_neutral_cross_section_m2,
        min_h_factor=config.min_h_factor,
        max_h_factor=config.max_h_factor,
    )
    return 1.0, closure


def _compile_wall_boundary(
    chemistry: CompiledChemistry,
    *,
    zone_id: str,
    surface_id: str,
    area_m2: float,
    transport_kind: Literal["bohm", "prescribed_frequency", "ambipolar", "off"],
    bohm_factor: float = 0.61,
    sheath_energy_eV: float = 0.0,
    prescribed_frequency_s_inv: float | None = None,
    diffusion_coefficient_m2_s: float | None = None,
    diffusion_length_m: float | None = None,
    auto_bohm_h_factor: AutoBohmHFactor | None = None,
) -> WallBoundary:
    """Bind typed compiled boundary reactions to one reactor surface."""

    reactions = tuple(
        BoundaryReaction(
            reaction_id=reaction.id,
            incident_species=reaction.incident_species,
            products=reaction.products,
            probability=reaction.probability,
        )
        for reaction in chemistry.boundary_reactions
        if (not reaction.zones or zone_id in reaction.zones)
        and (not reaction.surfaces or surface_id in reaction.surfaces)
    )
    return WallBoundary(
        zone_id=zone_id,
        surface_id=surface_id,
        area_m2=area_m2,
        reactions=reactions,
        transport_kind=transport_kind,
        bohm_factor=bohm_factor,
        sheath_energy_eV=sheath_energy_eV,
        prescribed_frequency_s_inv=prescribed_frequency_s_inv,
        diffusion_coefficient_m2_s=diffusion_coefficient_m2_s,
        diffusion_length_m=diffusion_length_m,
        auto_bohm_h_factor=auto_bohm_h_factor,
    )
