"""Validate reactor initial conditions and assemble compiled components."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from plasma_global.chemistry.compile import CompiledChemistry
from plasma_global.chemistry.data import ChemistryData
from plasma_global.chemistry.surface_roles import (
    _surface_initial_state_error,
    surface_species_roles,
)
from plasma_global.core.domain import Zone
from plasma_global.core.transport import CompiledTransport
from plasma_global.errors import CaseValidationError
from plasma_global.input._compile_power import (
    compile_external_binding as compile_external_binding,
)
from plasma_global.input._compile_power import compile_power_coordinator
from plasma_global.input._compile_reactor_components import (
    compile_heavy_energy,
    compile_static_transport,
    compile_surface_model,
    compile_walls,
    compile_zones,
)
from plasma_global.input.schema import CaseSpec, ZoneConfig
from plasma_global.models.external_table import ExternalTableStore
from plasma_global.models.gas_energy import HeavyEnergyClosure
from plasma_global.models.power import PowerCoordinator
from plasma_global.models.surface import CompiledSurfaceModel
from plasma_global.models.walls import WallBoundary

_BOLTZMANN_J_K = 1.380649e-23


@dataclass(frozen=True, slots=True)
class CompiledReactor:
    """Static reactor components consumed by the compiled runtime."""

    zones: tuple[Zone, ...]
    transport: CompiledTransport
    wall_boundaries: tuple[WallBoundary, ...]
    power_coordinator: PowerCoordinator | None
    heavy_energy_closure: HeavyEnergyClosure | None
    surface_model: CompiledSurfaceModel | None


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


def validate_surface_initial_conditions(
    case: CaseSpec, chemistry_data: ChemistryData
) -> None:
    """Validate the public input adapter through the shared surface contract."""

    for surface in case.reactor.surfaces:
        roles = surface_species_roles(chemistry_data.species, surface.surface_id)
        if error := _surface_initial_state_error(
            surface.surface_id, surface.initial_coverages, roles
        ):
            raise CaseValidationError(error)


def compile_reactor(
    case: CaseSpec,
    chemistry_data: ChemistryData,
    chemistry: CompiledChemistry,
    external_tables: ExternalTableStore,
) -> CompiledReactor:
    """Compile topology, transport, walls, and typed power ports."""

    heavy_energy_closure = compile_heavy_energy(case, chemistry_data, chemistry)
    return CompiledReactor(
        zones=compile_zones(case),
        transport=compile_static_transport(case, chemistry, heavy_energy_closure),
        wall_boundaries=compile_walls(case, chemistry),
        power_coordinator=compile_power_coordinator(case, chemistry, external_tables),
        heavy_energy_closure=heavy_energy_closure,
        surface_model=compile_surface_model(case, chemistry_data, chemistry),
    )


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


__all__ = [
    "CompiledReactor",
    "compile_external_binding",
    "compile_initial_densities",
    "compile_reactor",
    "validate_surface_initial_conditions",
]
