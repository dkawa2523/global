"""Composition root from strict schema-v3 input to the compiled solver."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, assert_never

from pydantic import ValidationError

from plasma_global._audit_runtime import _derive_observables_and_runtime_diagnostics
from plasma_global._build_electrons import (
    compile_electron_density,
    compile_electrons,
)
from plasma_global._build_metadata import build_metadata, solver_settings
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
    CaseSpec,
    EvolvedGasEnergy,
    FixedGasEnergy,
)
from plasma_global.models.elastic import (
    compile_elastic_heating,
)
from plasma_global.models.external_table import ExternalTableStore
from plasma_global.models.kinetics import (
    TabulatedElectronKinetics as TabulatedElectronKinetics,
)
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
    """Lower validated schema values to the solver's initial-state contract."""

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
            zone.zone_id: zone.initial_mean_energy_eV
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
    )
    electron_density_profile = compile_electron_density(case)
    segments = compile_recipe(
        case,
        chemistry_data,
        chemistry,
        external_tables,
        prescribed_electron_profile=electron_density_profile,
    )
    initial_state = _compile_initial_state(case, initial_densities)
    electron_closure, kinetics, electron_kinetics_provenance = compile_electrons(
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
    settings = solver_settings(case)
    return CompiledCase(
        case=case,
        chemistry_data=chemistry_data,
        chemistry=chemistry,
        model=model,
        initial_state=initial_state,
        solver_settings=settings,
        metadata=build_metadata(
            case,
            chemistry_data,
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
    if detailed_audit:
        observables, runtime_diagnostics = _derive_observables_and_runtime_diagnostics(
            compiled,
            result,
            compiled.case.output.observables,
        )
    else:
        observables, runtime_diagnostics = derive_observables_and_diagnostics(
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
        "conservation_max_abs_residual": runtime_diagnostics,
    }
    return replace(result, observables=observables, metadata=metadata)


def simulate_case(case: CaseSpec) -> SimulationResult:
    """Compile and simulate one v3 case, returning the canonical result type."""

    return _simulate_compiled_case(compile_case(case))


__all__ = ["CompiledCase", "compile_case", "simulate_case"]
