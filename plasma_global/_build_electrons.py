"""Electron-model compilation used by the schema-v5 composition root."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Any, assert_never

from plasma_global._input_sources import InputSourceStore
from plasma_global.chemistry.data import ChemistryData
from plasma_global.errors import CaseValidationError
from plasma_global.input._schema_models import QuasiNeutralElectronDensity
from plasma_global.input.schema import (
    CaseSpec,
    DensityProfile,
    MaxwellianCoefficients,
    TableCoefficients,
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


def _needs_table_mobility(case: CaseSpec) -> bool:
    return any(
        port.model.kind == "dc_series"
        and (
            case.models.electrons.kind == "local_field"
            or port.model.electron_mobility_m2_V_s is None
        )
        for port in case.reactor.power_ports.values()
    )


def _compile_table(
    case: CaseSpec,
    chemistry_data: ChemistryData,
    sources: InputSourceStore,
) -> tuple[ElectronClosure, Mapping[str, TabulatedElectronKinetics], Mapping[str, Any]]:
    config = case.models.electrons.coefficients
    if not isinstance(config, TableCoefficients):
        raise TypeError("expected tabulated electron configuration")
    table = TabulatedElectronKinetics.from_csv(
        config.file,
        columns=config.columns,
        lookup="mean_energy"
        if case.models.electrons.kind == "energy_balance"
        else "local_field",
        bounds=config.bounds,
        sources=sources,
        required_rate_ids=_required_table_rate_ids(chemistry_data),
        mobility_reference_neutral_density_m3=config.mobility_reference_neutral_density_m3,
        require_mobility=_needs_table_mobility(case),
    )
    _validate_mobility_source(case, table)
    by_zone = MappingProxyType(dict.fromkeys(case.reactor.zones, table))
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
            **(
                {
                    "zero_field_policy": (
                        "explicit_zero_field_row"
                        if table.axis[0] == 0.0
                        else "cold_boundary_at_zero_with_first_resolved_mobility"
                    )
                }
                if table.lookup == "local_field"
                else {}
            ),
        }
    )
    closure: ElectronClosure = (
        ElectronEnergyClosure()
        if case.models.electrons.kind == "energy_balance"
        else LocalFieldClosure()
    )
    return closure, by_zone, provenance


def _validate_mobility_source(case: CaseSpec, table: TabulatedElectronKinetics) -> None:
    for port_id, port in case.reactor.power_ports.items():
        if (
            port.model.kind == "dc_series"
            and port.model.electron_mobility_m2_V_s is not None
            and table.mobility_quantity is not None
        ):
            raise CaseValidationError(
                f"port {port_id!r}: choose table mobility or configured mobility"
            )


def compile_electrons(
    case: CaseSpec,
    chemistry_data: ChemistryData,
    sources: InputSourceStore,
) -> tuple[ElectronClosure, Mapping[str, TabulatedElectronKinetics], Mapping[str, Any]]:
    """Compile the configured electron closure and optional kinetic tables."""

    config = case.models.electrons.coefficients
    if not isinstance(config, TableCoefficients) and any(
        chemistry_data.rate_models[str(reaction.rate_model)].kind == "electron_table"
        for reaction in chemistry_data.gas_reactions
    ):
        raise CaseValidationError("electron_table reactions require a coefficient CSV")
    if isinstance(config, MaxwellianCoefficients):
        return ElectronEnergyClosure(), MappingProxyType({}), MappingProxyType({})
    if isinstance(config, TableCoefficients):
        return _compile_table(case, chemistry_data, sources)
    assert_never(config)


def compile_electron_density(case: CaseSpec, sources: InputSourceStore) -> Any | None:
    """Compile the optional prescribed electron-density provider."""

    config = case.models.electron_density
    if isinstance(config, QuasiNeutralElectronDensity):
        return None
    if isinstance(config, DensityProfile):
        from plasma_global.experimental.profile import PrescribedElectronProfile

        profile = PrescribedElectronProfile.from_csv(
            config.file,
            columns=config.columns,
            zone_columns=config.zone_columns or None,
            interpolation=config.interpolation,
            bounds=config.bounds,
            sources=sources,
        )
        offset = config.time_offset_s + case.run.start_time_s
        if offset != 0.0:
            profile = replace(profile, time_s=profile.time_s + offset)
        zone_ids = set(case.reactor.zones)
        if unknown := set(profile.density_m3_by_zone) - zone_ids - {"*"}:
            raise CaseValidationError(
                f"electron profile selects unknown zones: {sorted(unknown)}"
            )
        for zone_id in case.reactor.zones:
            profile.density(case.run.start_time_s, zone_id)
            profile.density(case.run.end_time_s, zone_id)
        return profile
    assert_never(config)
