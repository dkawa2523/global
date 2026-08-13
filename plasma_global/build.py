"""Composition root from strict schema-v3 input to the compiled solver."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, assert_never

import yaml
from pydantic import ValidationError

from plasma_global._audit_runtime import runtime_diagnostic_maxima
from plasma_global.chemistry.compile import CompiledChemistry, compile_chemistry
from plasma_global.chemistry.data import ChemistryData, load_chemistry
from plasma_global.core.compiled import CompiledGlobalModel
from plasma_global.core.domain import (
    InitialState,
    RecipeSegment,
    SolverSettings,
)
from plasma_global.core.result import SimulationResult
from plasma_global.core.solver import solve_compiled_model
from plasma_global.errors import CaseValidationError
from plasma_global.input.compile_experimental import compile_experimental_accumulator
from plasma_global.input.compile_reactor import (
    compile_initial_densities,
    compile_reactor,
)
from plasma_global.input.compile_recipe import compile_recipe
from plasma_global.input.schema import (
    ApproximateTwoTermElectronModel,
    CaseSpec,
    EvolvedGasEnergy,
    FixedGasEnergy,
    MaxwellianElectronModel,
    PrescribedElectronDensity,
    QuasiNeutralElectronDensity,
    TableElectronModel,
)
from plasma_global.input.schema import (
    ElectronEnergyClosure as ElectronEnergyClosureConfig,
)
from plasma_global.input.schema import (
    LocalFieldClosure as LocalFieldClosureConfig,
)
from plasma_global.models.elastic import (
    compile_elastic_heating,
    momentum_cross_section_ids,
)
from plasma_global.models.electrons import (
    ElectronClosure,
    ElectronEnergyClosure,
    LocalFieldClosure,
)
from plasma_global.models.external_table import ExternalTableStore
from plasma_global.models.kinetics import TabulatedElectronKinetics
from plasma_global.models.walls import BOHM_H_FACTOR_CLOSURE_VERSION
from plasma_global.postprocess import (
    derive_observables_and_diagnostics,
    validate_summary_selection,
)


@dataclass(frozen=True, slots=True)
class CompiledCase:
    """Complete immutable inputs for one deterministic simulation."""

    case: CaseSpec
    chemistry_data: ChemistryData
    chemistry: CompiledChemistry
    model: CompiledGlobalModel
    initial_state: InitialState
    solver_settings: SolverSettings
    metadata: Mapping[str, Any]

    @property
    def segments(self) -> tuple[RecipeSegment, ...]:
        return self.model.segments


def _compile_initial_state(
    case: CaseSpec, initial_densities: Mapping[str, Mapping[str, float]]
) -> InitialState:
    gas_energy = case.models.gas_energy
    if isinstance(gas_energy, EvolvedGasEnergy):
        gas_temperatures = {
            zone.zone_id: zone.gas_temperature_K for zone in case.reactor.zones
        }
    elif isinstance(gas_energy, FixedGasEnergy):
        gas_temperatures: dict[str, float] = {}
    else:
        assert_never(gas_energy)
    return InitialState(
        densities_m3_by_zone={
            zone.zone_id: dict(initial_densities[zone.zone_id])
            for zone in case.reactor.zones
        },
        mean_energy_eV_by_zone={
            zone.zone_id: float(zone.initial_mean_energy_eV)
            for zone in case.reactor.zones
            if zone.initial_mean_energy_eV is not None
        },
        gas_temperature_K_by_zone=gas_temperatures,
        surface_coverages=(
            {
                surface.surface_id: dict(surface.initial_coverages)
                for surface in case.reactor.surfaces
            }
            if case.models.surface_kinetics is not None
            else {}
        ),
    )


def _required_table_rate_ids(chemistry_data: ChemistryData) -> tuple[str, ...]:
    values: set[str] = set()
    for reaction in chemistry_data.gas_reactions:
        model = chemistry_data.rate_models[str(reaction.rate_model)]
        if model.kind == "electron_impact":
            values.add(str(model.parameters["cross_section"]))
    return tuple(sorted(values))


def _compile_electrons(
    case: CaseSpec,
    chemistry_data: ChemistryData,
    initial_densities: Mapping[str, Mapping[str, float]],
) -> tuple[
    ElectronClosure,
    Mapping[str, TabulatedElectronKinetics],
    Mapping[str, Any],
]:
    config = case.models.electrons
    closure = case.models.electron_closure
    if isinstance(config, MaxwellianElectronModel):
        if not isinstance(closure, ElectronEnergyClosureConfig):
            raise CaseValidationError(
                "maxwellian electrons require the electron_energy closure"
            )
        return ElectronEnergyClosure(), MappingProxyType({}), MappingProxyType({})
    if isinstance(config, ApproximateTwoTermElectronModel):
        if not isinstance(closure, LocalFieldClosureConfig):
            raise CaseValidationError(
                "experimental.approximate_two_term requires the local_field closure"
            )
        from plasma_global.experimental.prepared_kinetics import (
            prepare_approximate_two_term_kinetics,
        )

        prepared = prepare_approximate_two_term_kinetics(
            chemistry_data,
            initial_densities,
            mixture_key_species=config.mixture_key_species,
            cache_max_entries=config.cache.max_entries,
            fraction_decimals=config.cache.fraction_decimals,
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
    if isinstance(config, TableElectronModel):
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
        )
        by_zone = MappingProxyType({zone.zone_id: table for zone in case.reactor.zones})
        if isinstance(closure, ElectronEnergyClosureConfig):
            return ElectronEnergyClosure(), by_zone, MappingProxyType({})
        if isinstance(closure, LocalFieldClosureConfig):
            return (
                LocalFieldClosure(table.mean_energy_from_field),
                by_zone,
                MappingProxyType({}),
            )
        assert_never(closure)
    assert_never(config)


def _compile_electron_density(case: CaseSpec) -> Any | None:
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


def _validated_case_snapshot(case: CaseSpec) -> CaseSpec:
    """Revalidate and detach mutable containers before compilation starts."""

    if not isinstance(case, CaseSpec):
        raise TypeError("compile_case expects a CaseSpec")
    try:
        snapshot = CaseSpec.model_validate(
            case.model_dump(mode="python", round_trip=True, warnings=False)
        )
    except (TypeError, ValidationError, ValueError) as exc:
        raise CaseValidationError(
            "CaseSpec contains invalid changes made after validation"
        ) from exc
    if case.source_path is not None:
        snapshot._with_source(case.source_path, case.included_files)
    return snapshot


def _effective_case_yaml(case: CaseSpec) -> str:
    return yaml.safe_dump(
        case.model_dump(mode="json", exclude_none=True),
        sort_keys=False,
        allow_unicode=True,
    )


def _wall_transport_closure_provenance(case: CaseSpec) -> dict[str, Any] | None:
    surface_modes = {
        surface.surface_id: (
            "auto" if surface.wall_transport.h_factor == "auto" else "numeric"
        )
        for surface in case.reactor.surfaces
        if surface.wall_transport.kind == "bohm"
    }
    if not surface_modes:
        return None
    return {
        "bohm_h_factor": {
            "version": BOHM_H_FACTOR_CLOSURE_VERSION,
            "surface_modes": surface_modes,
        }
    }


def _model_ids(
    case: CaseSpec,
    chemistry: CompiledChemistry,
    experimental_features: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "electrons": case.models.electrons.kind,
        "electron_closure": case.models.electron_closure.kind,
        "electron_density": case.models.electron_density.kind,
        "gas_energy": case.models.gas_energy.kind,
        "surface_kinetics": (
            "compiled" if case.models.surface_kinetics is not None else None
        ),
        "power_ports": {
            port.port_id: port.model.kind for port in case.reactor.power_ports
        },
        "wall_transport": {
            surface.surface_id: surface.wall_transport.kind
            for surface in case.reactor.surfaces
        },
        "elastic_heating": (
            "momentum_cross_sections" if momentum_cross_section_ids(chemistry) else None
        ),
        "experimental_accumulator": experimental_features or None,
        "experimental_stop_policy": (
            "experimental.stop_when_quasi_steady"
            if case.experimental is not None
            and case.experimental.stop_when_quasi_steady is not None
            else None
        ),
    }


def _metadata(
    case: CaseSpec,
    chemistry: CompiledChemistry,
    missing_momentum_targets: tuple[str, ...],
    electron_kinetics_provenance: Mapping[str, Any],
    experimental_features: tuple[str, ...],
) -> Mapping[str, Any]:
    provenance: dict[str, Any] = {
        "case_source": None if case.source_path is None else str(case.source_path),
        "included_files": [str(path) for path in case.included_files],
        "chemistry_manifest": str(case.chemistry.manifest),
        "chemistry": dict(chemistry.provenance),
        "missing_momentum_cross_section_targets": missing_momentum_targets,
    }
    wall_transport_closure = _wall_transport_closure_provenance(case)
    if wall_transport_closure is not None:
        provenance["wall_transport_closure"] = wall_transport_closure
    if electron_kinetics_provenance:
        provenance["electron_kinetics"] = dict(electron_kinetics_provenance)
    return MappingProxyType(
        {
            "effective_case_yaml": _effective_case_yaml(case),
            "model_ids": _model_ids(case, chemistry, experimental_features),
            "provenance": provenance,
            "summary_series": tuple(case.output.summary_series),
        }
    )


def _solver_settings(case: CaseSpec) -> SolverSettings:
    stop = (
        None if case.experimental is None else case.experimental.stop_when_quasi_steady
    )
    return SolverSettings(
        rtol=case.solver.rtol,
        atol=case.solver.atol,
        first_step_s=case.solver.first_step_s,
        max_step_s=case.solver.max_step_s,
        sample_interval_s=case.solver.sample_interval_s,
        save_at_s=(
            None if case.solver.save_at_s is None else tuple(case.solver.save_at_s)
        ),
        experimental_quasi_steady_threshold_s_inv=(
            None if stop is None else stop.relative_rhs_norm_s_inv
        ),
        experimental_quasi_steady_min_time_s=(0.0 if stop is None else stop.min_time_s),
    )


def compile_case(case: CaseSpec) -> CompiledCase:
    """Compile a validated v3 case without creating outputs or running an ODE."""

    case = _validated_case_snapshot(case)
    chemistry_data = load_chemistry(case.chemistry.manifest)
    chemistry = compile_chemistry(chemistry_data)
    initial_densities = compile_initial_densities(case, chemistry)
    external_tables = ExternalTableStore()
    reactor = compile_reactor(
        case,
        chemistry_data,
        chemistry,
        external_tables,
        initial_densities,
    )
    electron_density_profile = _compile_electron_density(case)
    segments = compile_recipe(
        case,
        chemistry_data,
        chemistry,
        external_tables,
        prescribed_electron_profile=electron_density_profile,
    )
    initial_state = _compile_initial_state(case, initial_densities)
    electron_closure, kinetics, electron_kinetics_provenance = _compile_electrons(
        case, chemistry_data, initial_densities
    )
    electron_density_provider = (
        None
        if electron_density_profile is None
        or electron_density_profile.interpolation == "previous"
        else electron_density_profile.density
    )
    elastic_heating, missing_momentum_targets = compile_elastic_heating(chemistry)
    extension_accumulator, experimental_features = compile_experimental_accumulator(
        case, chemistry_data, reactor.surface_model
    )
    model = CompiledGlobalModel(
        chemistry=chemistry,
        zones=reactor.zones,
        segments=segments,
        electron_closure=electron_closure,
        wall_boundaries=reactor.wall_boundaries,
        transport=reactor.transport,
        power_coordinator=reactor.power_coordinator,
        electron_kinetics_by_zone=kinetics,
        heavy_energy_closure=reactor.heavy_energy_closure,
        surface_model=reactor.surface_model,
        extension_accumulator=extension_accumulator,
        elastic_heating_evaluator=elastic_heating,
        electron_density_provider=electron_density_provider,
        domain_atol=case.solver.atol,
    )
    validate_summary_selection(
        model,
        case.output.observables,
        case.output.summary_series,
    )
    # Fail at compilation, not after output creation or the first RHS call.
    model.initial_state(initial_state)
    solver_settings = _solver_settings(case)
    return CompiledCase(
        case=case,
        chemistry_data=chemistry_data,
        chemistry=chemistry,
        model=model,
        initial_state=initial_state,
        solver_settings=solver_settings,
        metadata=_metadata(
            case,
            chemistry,
            missing_momentum_targets,
            electron_kinetics_provenance,
            experimental_features,
        ),
    )


def _simulate_compiled_case(
    compiled: CompiledCase, *, detailed_audit: bool = False
) -> SimulationResult:
    """Simulate one compiled case, optionally evaluating detailed ledgers."""

    result = solve_compiled_model(
        compiled.model,
        compiled.initial_state,
        compiled.solver_settings,
    )
    observables, normal_diagnostics = derive_observables_and_diagnostics(
        compiled.model,
        result.time_s,
        result.state,
        compiled.case.output.observables,
    )
    metadata = {**result.metadata, **compiled.metadata}
    metadata["provenance"] = {
        **dict(result.metadata.get("provenance", {})),
        **dict(compiled.metadata["provenance"]),
    }
    metadata["provenance"]["runtime_diagnostics"] = {
        "simulation_status": {
            "success": result.status.success,
            "code": result.status.code,
            "message": result.status.message,
        },
        "conservation_max_abs_residual": (
            runtime_diagnostic_maxima(
                compiled,
                result,
                include_detailed_ledgers=True,
            )
            if detailed_audit
            else normal_diagnostics
        ),
    }
    return replace(result, observables=observables, metadata=metadata)


def simulate_case(case: CaseSpec) -> SimulationResult:
    """Compile and simulate one v3 case, returning the canonical result type."""

    return _simulate_compiled_case(compile_case(case))


__all__ = ["CompiledCase", "compile_case", "simulate_case"]
