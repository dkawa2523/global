"""Electron-model compilation used by the schema-v4 composition root."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Any, assert_never

from plasma_global.chemistry.data import ChemistryData
from plasma_global.errors import CaseValidationError
from plasma_global.input._resolved_case import (
    ApproximateTwoTermElectronModel,
    MaxwellianElectronModel,
    PrescribedElectronDensity,
    QuasiNeutralElectronDensity,
    ResolvedCase,
    TableElectronModel,
)
from plasma_global.input._resolved_case import (
    ElectronEnergyClosure as ElectronEnergyClosureConfig,
)
from plasma_global.input._resolved_case import (
    LocalFieldClosure as LocalFieldClosureConfig,
)
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
    values.update(
        str(model.parameters["rate_id"])
        for reaction in chemistry_data.gas_reactions
        if (model := chemistry_data.rate_models[str(reaction.rate_model)]).kind
        == "electron_table"
    )
    values.update(
        cross_section.id
        for cross_section in chemistry_data.cross_sections.values()
        if cross_section.kind == "momentum_transfer"
    )
    return tuple(sorted(values))


def _compile_approximate_two_term(
    case: ResolvedCase,
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


def _needs_table_mobility(case: ResolvedCase) -> bool:
    return any(
        port.model.kind == "dc_series"
        and (
            isinstance(case.models.electron_closure, LocalFieldClosureConfig)
            or port.model.electron_mobility_m2_V_s is None
        )
        for port in case.reactor.power_ports
    )


def _compile_table(
    case: ResolvedCase,
    chemistry_data: ChemistryData,
    initial_densities: Mapping[str, Mapping[str, float]],
) -> tuple[ElectronClosure, Mapping[str, TabulatedElectronKinetics], Mapping[str, Any]]:
    config = case.models.electrons
    if not isinstance(config, TableElectronModel):
        raise TypeError("expected tabulated electron configuration")
    del initial_densities
    table = TabulatedElectronKinetics.from_csv(
        config.file,
        columns=config.columns,
        lookup=config.lookup,
        bounds=config.bounds_policy,
        required_rate_ids=_required_table_rate_ids(chemistry_data),
        mobility_reference_neutral_density_m3=config.mobility_reference_neutral_density_m3,
        require_mobility=_needs_table_mobility(case),
    )
    _validate_mobility_source(case, table)
    by_zone = MappingProxyType({zone.zone_id: table for zone in case.reactor.zones})
    provenance = MappingProxyType(
        {
            "mobility_scaling": (
                "inverse_neutral_density"
                if table.mobility_quantity is not None
                else None
            ),
            "mobility_quantity": table.mobility_quantity,
            "mobility_reference_neutral_density_m3": (
                table.mobility_reference_neutral_density_m3
            ),
            "mobility_reference_source": "table_input",
            "source": config.provenance,
            "momentum_rate_ids": tuple(
                name for name in table.rate_tables if name.startswith("momentum:")
            ),
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


def _validate_mobility_source(
    case: ResolvedCase, table: TabulatedElectronKinetics
) -> None:
    for port in case.reactor.power_ports:
        if (
            port.model.kind == "dc_series"
            and port.model.electron_mobility_m2_V_s is not None
            and table.mobility_quantity is not None
        ):
            raise CaseValidationError(
                f"port {port.port_id!r}: choose table mobility or configured mobility"
            )


def compile_electrons(
    case: ResolvedCase,
    chemistry_data: ChemistryData,
    initial_densities: Mapping[str, Mapping[str, float]],
) -> tuple[ElectronClosure, Mapping[str, TabulatedElectronKinetics], Mapping[str, Any]]:
    """Compile the configured electron closure and optional kinetic tables."""

    config = case.models.electrons
    if not isinstance(config, TableElectronModel) and any(
        chemistry_data.rate_models[str(reaction.rate_model)].kind == "electron_table"
        for reaction in chemistry_data.gas_reactions
    ):
        raise CaseValidationError("electron_table reactions require a coefficient CSV")
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


def compile_electron_density(case: ResolvedCase) -> Any | None:
    """Compile the optional prescribed electron-density provider."""

    config = case.models.electron_density
    if isinstance(config, QuasiNeutralElectronDensity):
        return None
    if isinstance(config, PrescribedElectronDensity):
        from plasma_global.experimental.profile import PrescribedElectronProfile

        profile = PrescribedElectronProfile.from_csv(
            config.file,
            columns=config.columns,
            zone_columns=config.zone_columns or None,
            interpolation=config.interpolation,
            bounds="hold" if config.hold == "edge" else "error",
        )
        if config.time_offset_s != 0.0:
            profile = replace(profile, time_s=profile.time_s + config.time_offset_s)
        zone_ids = {zone.zone_id for zone in case.reactor.zones}
        if unknown := set(profile.density_m3_by_zone) - zone_ids - {"*"}:
            raise CaseValidationError(
                f"electron profile selects unknown zones: {sorted(unknown)}"
            )
        for zone in case.reactor.zones:
            profile.density(case.recipe.start_time_s, zone.zone_id)
            profile.density(case.recipe.end_time_s, zone.zone_id)
        return profile
    assert_never(config)
