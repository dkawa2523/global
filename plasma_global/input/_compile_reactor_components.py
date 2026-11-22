"""Compile the static physical components of a schema-v5 reactor."""

from __future__ import annotations

from typing import Literal, assert_never

import numpy as np

from plasma_global.chemistry.compile import CompiledChemistry
from plasma_global.chemistry.data import ChemistryData
from plasma_global.core.domain import Zone
from plasma_global.core.transport import CompiledTransport
from plasma_global.errors import CaseValidationError
from plasma_global.input._compile_recipe_transport import wall_temperatures
from plasma_global.input._compile_thermal import heavy_thermodynamics
from plasma_global.input._schema_models import EvolvedGasEnergy, FixedGasEnergy
from plasma_global.input._schema_reactor import (
    AmbipolarWallTransport,
    BohmWallTransport,
    OffWallTransport,
    PrescribedFrequencyWallTransport,
)
from plasma_global.input.schema import CaseSpec, SurfaceDefinition
from plasma_global.models.gas_energy import HeavyEnergyClosure
from plasma_global.models.surface import CompiledSurfaceModel, SurfaceGeometry
from plasma_global.models.walls import AutoBohmHFactor, BoundaryReaction, WallBoundary


def compile_zones(case: CaseSpec) -> tuple[Zone, ...]:
    """Compile the immutable zone geometry used by the runtime."""

    return tuple(
        Zone(zone_id, zone.volume_m3, zone.initial.gas_temperature_K)
        for zone_id, zone in case.reactor.zones.items()
    )


def compile_static_transport(
    case: CaseSpec,
    chemistry: CompiledChemistry,
    heavy_energy_closure: HeavyEnergyClosure | None,
) -> CompiledTransport:
    """Compile pumps, inter-zone edges, and fixed transport geometry."""

    zones = tuple(case.reactor.zones.values())
    zone_index = {zone_id: index for index, zone_id in enumerate(case.reactor.zones)}
    pump_frequency = np.zeros(len(zones))
    for pump in case.reactor.pumps.values():
        index = zone_index[pump.zone_id]
        pump_frequency[index] += pump.speed_m3_s / zones[index].volume_m3
    return CompiledTransport(
        volumes_m3=np.asarray([zone.volume_m3 for zone in zones]),
        pump_frequency_s_inv=pump_frequency,
        edge_from=np.asarray(
            [zone_index[edge.from_zone] for edge in case.reactor.edges.values()],
            dtype=int,
        ),
        edge_to=np.asarray(
            [zone_index[edge.to_zone] for edge in case.reactor.edges.values()],
            dtype=int,
        ),
        edge_conductance_m3_s=np.asarray(
            [edge.conductance_m3_s for edge in case.reactor.edges.values()]
        ),
        n_species=len(chemistry.species_ids),
        thermodynamics=(
            None
            if heavy_energy_closure is None
            else heavy_energy_closure.thermodynamics
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
        thermodynamics=heavy_thermodynamics(chemistry_data, chemistry),
        wall_temperature_K=_zone_wall_temperatures(case),
        wall_relaxation_s_inv=np.asarray(
            [
                config.wall_energy_relaxation_s_inv_by_zone.get(zone_id, 0.0)
                for zone_id in case.reactor.zones
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
            surface_id=surface_id,
            zone_id=surface.zone_id,
            area_m2=surface.area_m2,
            site_density_m2=surface.site_density_m2,
            temperature_K=surface.temperature_K,
            initial_coverages=surface.initial_coverages,
        )
        for surface_id, surface in case.reactor.surfaces.items()
    )
    if not surfaces:
        raise CaseValidationError(
            "models.surface_kinetics requires at least one reactor surface"
        )
    return CompiledSurfaceModel(
        chemistry=chemistry_data,
        gas_species_ids=chemistry.species_ids,
        gas_masses_kg=chemistry.masses_kg,
        zone_ids=tuple(case.reactor.zones),
        zone_volumes_m3=np.asarray(
            [zone.volume_m3 for zone in case.reactor.zones.values()]
        ),
        surfaces=surfaces,
        domain_atol=case.solver.atol,
    )


def compile_walls(
    case: CaseSpec,
    chemistry: CompiledChemistry,
) -> tuple[WallBoundary, ...]:
    """Compile wall transport and chemistry for every enabled surface."""

    walls: list[WallBoundary] = []
    for surface_id, surface in case.reactor.surfaces.items():
        transport = surface.wall_transport
        if isinstance(transport, OffWallTransport):
            continue
        if isinstance(transport, BohmWallTransport):
            bohm_factor, auto_bohm_h_factor = _bohm_h_factor(case, surface)
            boundary = _compile_wall_boundary(
                chemistry,
                zone_id=surface.zone_id,
                area_m2=surface.area_m2,
                surface_id=surface_id,
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
                surface_id=surface_id,
                sheath_energy_eV=surface.ion_impact_energy_eV,
                transport_kind="prescribed_frequency",
                prescribed_frequency_s_inv=transport.frequency_s_inv,
            )
        elif isinstance(transport, AmbipolarWallTransport):
            boundary = _compile_wall_boundary(
                chemistry,
                zone_id=surface.zone_id,
                area_m2=surface.area_m2,
                surface_id=surface_id,
                sheath_energy_eV=surface.ion_impact_energy_eV,
                transport_kind="ambipolar",
                diffusion_coefficient_m2_s=transport.diffusion_coefficient_m2_s,
                diffusion_length_m=transport.diffusion_length_m,
                electron_temperature_reference_eV=transport.electron_temperature_reference_eV,
            )
        else:
            assert_never(transport)
        walls.append(boundary)
    return tuple(walls)


def _zone_wall_temperatures(case: CaseSpec) -> np.ndarray:
    """Area-average each zone's surfaces, falling back to its gas temperature."""

    values = wall_temperatures(case, {})
    return np.asarray([values[zone_id] for zone_id in case.reactor.zones])


def _bohm_h_factor(
    case: CaseSpec,
    surface: SurfaceDefinition,
) -> tuple[float, AutoBohmHFactor | None]:
    config = surface.wall_transport
    if not isinstance(config, BohmWallTransport):
        raise TypeError("_bohm_h_factor requires Bohm wall transport")
    if config.h_factor != "auto":
        return config.h_factor, None
    zone = case.reactor.zones[surface.zone_id]
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
    electron_temperature_reference_eV: float | None = None,
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
        electron_temperature_reference_eV=electron_temperature_reference_eV,
    )
