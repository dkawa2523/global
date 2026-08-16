"""Electron-model compilation used by the schema-v3 composition root."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, assert_never

from plasma_global.chemistry.data import ChemistryData
from plasma_global.errors import CaseValidationError
from plasma_global.input.schema import (
    ApproximateTwoTermElectronModel,
    CaseSpec,
    MaxwellianElectronModel,
    PrescribedElectronDensity,
    QuasiNeutralElectronDensity,
    TableElectronModel,
)
from plasma_global.input.schema import (
    ElectronEnergyClosure as ElectronEnergyClosureConfig,
)
from plasma_global.input.schema import LocalFieldClosure as LocalFieldClosureConfig
from plasma_global.models.electrons import (
    ElectronClosure,
    ElectronEnergyClosure,
    LocalFieldClosure,
)
from plasma_global.models.kinetics import TabulatedElectronKinetics


def _required_table_rate_ids(chemistry_data: ChemistryData) -> tuple[str, ...]:
    values = {
        str(model.parameters["cross_section"])
        for reaction in chemistry_data.gas_reactions
        if (model := chemistry_data.rate_models[str(reaction.rate_model)]).kind
        == "electron_impact"
    }
    return tuple(sorted(values))


def _compile_approximate_two_term(
    case: CaseSpec,
    chemistry_data: ChemistryData,
    initial_densities: Mapping[str, Mapping[str, float]],
) -> tuple[ElectronClosure, Mapping[str, TabulatedElectronKinetics], Mapping[str, Any]]:
    config = case.models.electrons
    if not isinstance(config, ApproximateTwoTermElectronModel):
        raise TypeError("expected approximate-two-term electron configuration")
    if not isinstance(case.models.electron_closure, LocalFieldClosureConfig):
        raise CaseValidationError(
            "experimental.approximate_two_term requires the local_field closure"
        )
    from plasma_global.experimental.prepared_kinetics import (
        prepare_approximate_two_term_kinetics,
    )

    prepared = prepare_approximate_two_term_kinetics(
        chemistry_data,
        initial_densities,
        cache_max_entries=config.cache.max_entries,
        energy_min_eV=config.energy_grid.min_eV,
        energy_max_eV=config.energy_grid.max_eV,
        energy_points=config.energy_grid.n,
        field_min_Td=config.reduced_field_grid.min_Td,
        field_max_Td=config.reduced_field_grid.max_Td,
        field_points=config.reduced_field_grid.n,
        max_iterations=config.max_shape_iterations,
    )
    representative = next(iter(prepared.by_zone.values()))
    return (
        LocalFieldClosure(representative.mean_energy_from_field),
        prepared.by_zone,
        prepared.provenance,
    )


def _initial_neutral_densities(
    chemistry_data: ChemistryData,
    initial_densities: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    neutral_ids = tuple(
        species.id
        for species in chemistry_data.species
        if species.phase == "gas" and species.charge == 0
    )
    return {
        zone_id: sum(
            float(densities.get(species_id, 0.0)) for species_id in neutral_ids
        )
        for zone_id, densities in initial_densities.items()
    }


def _table_mobility_reference(
    config: TableElectronModel,
    neutral_density_by_zone: Mapping[str, float],
) -> tuple[float, str]:
    explicit = config.mobility_reference_neutral_density_m3
    if explicit is not None:
        return float(explicit), "input"
    values = tuple(neutral_density_by_zone.values())
    if not values or values[0] <= 0.0:
        raise CaseValidationError(
            "table electron mobility requires positive initial neutral density"
        )
    reference = values[0]
    if any(
        not math.isclose(value, reference, rel_tol=1.0e-12, abs_tol=0.0)
        for value in values[1:]
    ):
        raise CaseValidationError(
            "table electron mobility shared by zones with different initial neutral "
            "densities requires mobility_reference_neutral_density_m3"
        )
    return reference, "uniform_initial_neutral_density"


def _compile_table(
    case: CaseSpec,
    chemistry_data: ChemistryData,
    initial_densities: Mapping[str, Mapping[str, float]],
) -> tuple[ElectronClosure, Mapping[str, TabulatedElectronKinetics], Mapping[str, Any]]:
    config = case.models.electrons
    if not isinstance(config, TableElectronModel):
        raise TypeError("expected tabulated electron configuration")
    reference_density, reference_source = _table_mobility_reference(
        config,
        _initial_neutral_densities(chemistry_data, initial_densities),
    )
    table = TabulatedElectronKinetics.from_hdf5(
        config.file,
        lookup=config.lookup,
        bounds=config.bounds_policy,
        required_rate_ids=_required_table_rate_ids(chemistry_data),
        optional_rate_ids=tuple(
            sorted(
                cross_section.id
                for cross_section in chemistry_data.cross_sections.values()
                if cross_section.kind == "momentum_transfer"
            )
        ),
        mobility_reference_neutral_density_m3=reference_density,
    )
    by_zone = MappingProxyType({zone.zone_id: table for zone in case.reactor.zones})
    provenance = MappingProxyType(
        {
            "mobility_scaling": "inverse_neutral_density",
            "mobility_reference_neutral_density_m3": reference_density,
            "mobility_reference_source": reference_source,
        }
    )
    closure = case.models.electron_closure
    if isinstance(closure, ElectronEnergyClosureConfig):
        return ElectronEnergyClosure(), by_zone, provenance
    if isinstance(closure, LocalFieldClosureConfig):
        return (
            LocalFieldClosure(table.mean_energy_from_field),
            by_zone,
            provenance,
        )
    assert_never(closure)


def compile_electrons(
    case: CaseSpec,
    chemistry_data: ChemistryData,
    initial_densities: Mapping[str, Mapping[str, float]],
) -> tuple[ElectronClosure, Mapping[str, TabulatedElectronKinetics], Mapping[str, Any]]:
    """Compile the configured electron closure and optional kinetic tables."""

    config = case.models.electrons
    if isinstance(config, MaxwellianElectronModel):
        if not isinstance(case.models.electron_closure, ElectronEnergyClosureConfig):
            raise CaseValidationError(
                "maxwellian electrons require the electron_energy closure"
            )
        return ElectronEnergyClosure(), MappingProxyType({}), MappingProxyType({})
    if isinstance(config, ApproximateTwoTermElectronModel):
        return _compile_approximate_two_term(case, chemistry_data, initial_densities)
    if isinstance(config, TableElectronModel):
        return _compile_table(case, chemistry_data, initial_densities)
    assert_never(config)


def compile_electron_density(case: CaseSpec) -> Any | None:
    """Compile the optional prescribed electron-density provider."""

    config = case.models.electron_density
    if isinstance(config, QuasiNeutralElectronDensity):
        return None
    if isinstance(config, PrescribedElectronDensity):
        from plasma_global.experimental.profile import PrescribedElectronProfile

        profile = PrescribedElectronProfile.from_csv(
            config.file,
            zone_columns=config.zone_columns or None,
            interpolation=config.interpolation,
            bounds="hold" if config.hold == "edge" else "error",
        )
        for zone in case.reactor.zones:
            profile.density(case.recipe.start_time_s, zone.zone_id)
            profile.density(case.recipe.end_time_s, zone.zone_id)
        return profile
    assert_never(config)
