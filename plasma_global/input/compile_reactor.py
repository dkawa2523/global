"""Adapt schema-v3 reactor topology to static compiled components."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, assert_never

import numpy as np

from plasma_global.chemistry.compile import CompiledChemistry
from plasma_global.chemistry.data import ChemistryData
from plasma_global.chemistry.surface_roles import (
    SurfaceSpeciesRoles,
    surface_species_roles,
)
from plasma_global.core.domain import Zone
from plasma_global.core.transport import CompiledTransport
from plasma_global.errors import CaseValidationError
from plasma_global.input._compile_power import (
    compile_external_binding,
    compile_power_coordinator,
)
from plasma_global.input.schema import (
    AmbipolarWallTransport,
    BohmWallTransport,
    CaseSpec,
    EvolvedGasEnergy,
    FixedGasEnergy,
    OffWallTransport,
    PrescribedFrequencyWallTransport,
    SurfaceConfig,
    ZoneConfig,
)
from plasma_global.models.external_table import ExternalTableStore
from plasma_global.models.gas_energy import HeavyEnergyClosure
from plasma_global.models.power import PowerCoordinator
from plasma_global.models.surface import CompiledSurfaceModel, SurfaceGeometry
from plasma_global.models.walls import AutoBohmHFactor, BoundaryReaction, WallBoundary

_BOLTZMANN_J_K = 1.380649e-23


@dataclass(frozen=True, slots=True)
class CompiledReactor:
    zones: tuple[Zone, ...]
    transport: CompiledTransport
    wall_boundaries: tuple[WallBoundary, ...]
    power_coordinator: PowerCoordinator | None
    heavy_energy_closure: HeavyEnergyClosure | None
    surface_model: CompiledSurfaceModel | None


def _compile_explicit_densities(
    zone: ZoneConfig,
    initial_densities_m3: Mapping[str, float],
    known_species: set[str],
) -> dict[str, float]:
    """Validate an explicit density state without changing its declared values."""

    densities = dict(initial_densities_m3)
    unknown = set(densities) - known_species
    if unknown:
        raise CaseValidationError(
            f"zone {zone.zone_id!r} initializes unknown species {sorted(unknown)}"
        )
    pressure_from_state = (
        sum(densities.values()) * _BOLTZMANN_J_K * zone.gas_temperature_K
    )
    if not math.isclose(
        pressure_from_state,
        zone.pressure_Pa,
        rel_tol=1.0e-6,
        abs_tol=1.0e-12,
    ):
        raise CaseValidationError(
            f"zone {zone.zone_id!r} pressure_Pa={zone.pressure_Pa:g} is "
            "inconsistent with sum(initial_densities_m3)*kB*T="
            f"{pressure_from_state:g}"
        )
    return densities


def _compile_mole_fraction_composition(
    zone: ZoneConfig,
    initial_mole_fractions: Mapping[str, float],
    chemistry: CompiledChemistry,
    species_index: dict[str, int],
) -> dict[str, float]:
    """Convert neutral mole fractions plus charged seeds to number densities."""

    fractions = dict(initial_mole_fractions)
    seeds = dict(zone.initial_seed_densities_m3)
    unknown = (set(fractions) | set(seeds)) - set(species_index)
    if unknown:
        raise CaseValidationError(
            f"zone {zone.zone_id!r} initializes unknown species {sorted(unknown)}"
        )
    nonneutral_fractions = [
        species_id
        for species_id in fractions
        if chemistry.charges[species_index[species_id]] != 0.0
    ]
    neutral_seeds = [
        species_id
        for species_id in seeds
        if chemistry.charges[species_index[species_id]] == 0.0
    ]
    if nonneutral_fractions or neutral_seeds:
        raise CaseValidationError(
            f"zone {zone.zone_id!r} mole fractions must be neutral and seed "
            "densities charged; invalid "
            f"fractions={sorted(nonneutral_fractions)}, "
            f"seeds={sorted(neutral_seeds)}"
        )
    total_density = zone.pressure_Pa / (_BOLTZMANN_J_K * zone.gas_temperature_K)
    seed_density = sum(seeds.values())
    if seed_density >= total_density:
        raise CaseValidationError(
            f"zone {zone.zone_id!r} seed density must be below p/(kB*T)"
        )
    neutral_density = total_density - seed_density
    densities = {
        species_id: fraction * neutral_density
        for species_id, fraction in fractions.items()
    }
    densities.update(seeds)
    return densities


def compile_initial_densities(
    case: CaseSpec, chemistry: CompiledChemistry
) -> Mapping[str, Mapping[str, float]]:
    """Compile every zone's selected initial-composition representation."""

    species_index = {
        species_id: index for index, species_id in enumerate(chemistry.species_ids)
    }
    known_species = set(species_index)
    resolved: dict[str, Mapping[str, float]] = {}
    for zone in case.reactor.zones:
        if zone.initial_densities_m3 is not None:
            densities = _compile_explicit_densities(
                zone,
                zone.initial_densities_m3,
                known_species,
            )
        elif zone.initial_mole_fractions is not None:
            densities = _compile_mole_fraction_composition(
                zone,
                zone.initial_mole_fractions,
                chemistry,
                species_index,
            )
        else:
            raise CaseValidationError(
                f"zone {zone.zone_id!r} needs densities or mole fractions"
            )
        positive_charge_density = sum(
            chemistry.charges[species_index[species_id]] * density
            for species_id, density in densities.items()
            if chemistry.charges[species_index[species_id]] > 0.0
        )
        if positive_charge_density <= 0.0:
            raise CaseValidationError(
                f"zone {zone.zone_id!r} requires an explicit positive-ion seed"
            )
        resolved[zone.zone_id] = MappingProxyType(densities)
    return MappingProxyType(resolved)


def _validate_surface_coverages(
    surface: SurfaceConfig,
    roles: SurfaceSpeciesRoles,
) -> None:
    """Validate that supplied coverages are independent and fit one site layer."""

    supplied = set(surface.initial_coverages)
    if explicit_free := supplied & set(roles.free_site_ids):
        raise CaseValidationError(
            f"surface {surface.surface_id!r} must not initialize algebraic "
            f"free site(s) {sorted(explicit_free)}"
        )
    if unknown := supplied - set(roles.independent_ids):
        raise CaseValidationError(
            f"surface {surface.surface_id!r} initializes unknown or dependent "
            f"coverage species {sorted(unknown)}"
        )
    occupied = sum(
        roles.occupancy_by_species[species_id] * coverage
        for species_id, coverage in surface.initial_coverages.items()
    )
    if occupied > 1.0 + 1.0e-12:
        raise CaseValidationError(
            f"surface {surface.surface_id!r} initial site occupancy exceeds one"
        )


def validate_surface_initial_conditions(
    case: CaseSpec, chemistry_data: ChemistryData
) -> None:
    """Validate independent adsorbates after chemistry IDs are available."""

    for surface in case.reactor.surfaces:
        roles = surface_species_roles(
            chemistry_data.species,
            surface.surface_id,
        )
        _validate_surface_coverages(surface, roles)


def _compile_zones(case: CaseSpec) -> tuple[Zone, ...]:
    return tuple(
        Zone(zone.zone_id, zone.volume_m3, zone.gas_temperature_K)
        for zone in case.reactor.zones
    )


def _transport_heat_capacities(
    heavy_energy_closure: HeavyEnergyClosure | None,
) -> np.ndarray | None:
    if heavy_energy_closure is None:
        return None
    return heavy_energy_closure.cv_over_kb


def _compile_static_transport(
    case: CaseSpec,
    chemistry: CompiledChemistry,
    heavy_energy_closure: HeavyEnergyClosure | None,
) -> CompiledTransport:
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
        heavy_cv_over_kb=_transport_heat_capacities(heavy_energy_closure),
    )


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


def _compile_heavy_energy(
    case: CaseSpec,
    chemistry_data: ChemistryData,
    chemistry: CompiledChemistry,
) -> HeavyEnergyClosure | None:
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


def _compile_surface_model(
    case: CaseSpec,
    chemistry_data: ChemistryData,
    chemistry: CompiledChemistry,
) -> CompiledSurfaceModel | None:
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


def _compile_walls(
    case: CaseSpec,
    chemistry: CompiledChemistry,
) -> tuple[WallBoundary, ...]:
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


def compile_reactor(
    case: CaseSpec,
    chemistry_data: ChemistryData,
    chemistry: CompiledChemistry,
    external_tables: ExternalTableStore,
    initial_densities: Mapping[str, Mapping[str, float]],
) -> CompiledReactor:
    """Compile topology, transport, walls, and typed power ports.

    ``initial_densities`` remains in this adapter API for compatibility. Dynamic
    wall closures deliberately read the evolving state instead of this snapshot.
    """

    validate_surface_initial_conditions(case, chemistry_data)
    heavy_energy_closure = _compile_heavy_energy(case, chemistry_data, chemistry)
    return CompiledReactor(
        zones=_compile_zones(case),
        transport=_compile_static_transport(case, chemistry, heavy_energy_closure),
        wall_boundaries=_compile_walls(case, chemistry),
        power_coordinator=compile_power_coordinator(case, chemistry, external_tables),
        heavy_energy_closure=heavy_energy_closure,
        surface_model=_compile_surface_model(case, chemistry_data, chemistry),
    )


__all__ = [
    "CompiledReactor",
    "compile_external_binding",
    "compile_initial_densities",
    "compile_reactor",
    "validate_surface_initial_conditions",
]
