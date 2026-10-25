"""Composition root from strict schema-v4 input to the compiled solver."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, assert_never

from pydantic import ValidationError

from plasma_global._audit_classification import assess_case
from plasma_global._audit_types import CaseAssessment
from plasma_global._build_electrons import (
    compile_electron_density,
    compile_electrons,
)
from plasma_global._build_metadata import build_metadata, solver_settings
from plasma_global._output_plan import OutputPlan, compile_output
from plasma_global.chemistry._data_yaml import load_definition
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
from plasma_global.input._compile_targets import validate_chemistry_targets
from plasma_global.input._resolved_case import (
    EvolvedGasEnergy,
    FixedGasEnergy,
    ResolvedCase,
)
from plasma_global.input.compile_experimental import compile_experimental_accumulator
from plasma_global.input.compile_reactor import (
    compile_initial_densities,
    compile_reactor,
)
from plasma_global.input.compile_recipe import compile_recipe
from plasma_global.models.elastic import (
    compile_elastic_heating,
)
from plasma_global.models.external_table import ExternalTableStore
from plasma_global.models.kinetics import (
    TabulatedElectronKinetics as TabulatedElectronKinetics,
)
from plasma_global.postprocess import derive_observables


@dataclass(frozen=True, slots=True)
class CompiledCase:
    """Complete immutable inputs for one deterministic simulation."""

    case: ResolvedCase
    chemistry_data: ChemistryData
    chemistry: CompiledChemistry
    model: CompiledGlobalModel
    initial_state: InitialState
    solver_settings: SolverSettings
    assessment: CaseAssessment
    metadata: Mapping[str, Any]
    output_plan: OutputPlan

    @property
    def segments(self) -> tuple[RecipeSegment, ...]:
        return self.model.segments


def _compile_initial_state(
    case: ResolvedCase, initial_densities: Mapping[str, Mapping[str, float]]
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


def _validated_case_snapshot(case: ResolvedCase) -> ResolvedCase:
    """Revalidate and detach mutable containers before compilation starts."""

    if not isinstance(case, ResolvedCase):
        raise TypeError("compile_case expects a ResolvedCase")
    try:
        snapshot = ResolvedCase.model_validate(
            case.model_dump(mode="python", round_trip=True, warnings=False)
        )
    except (TypeError, ValidationError, ValueError) as exc:
        raise CaseValidationError(
            "ResolvedCase contains invalid changes made after validation"
        ) from exc
    if case.source_path is not None:
        snapshot._with_source(case.source_path, case.included_files)
    return snapshot


def _chemistry_data(case: ResolvedCase) -> ChemistryData:
    chemistry_input = case.chemistry
    if chemistry_input.manifest is not None:
        chemistry_data = load_chemistry(chemistry_input.manifest)
    else:
        chemistry_data = load_definition(
            chemistry_input.definition or {},
            case.source_path or chemistry_input.base_directory / "case.yaml",
            chemistry_input.base_directory,
        )
    return chemistry_data


def _table_momentum(
    case: ResolvedCase,
    chemistry: CompiledChemistry,
    kinetics: Mapping[str, TabulatedElectronKinetics],
) -> dict[str, str] | None:
    table_momentum = None
    if case.models.electrons.kind == "table":
        representative = next(iter(kinetics.values()))
        table_momentum = {
            name.removeprefix("momentum:"): name
            for name in representative.rate_tables
            if name.startswith("momentum:")
        }
        for curve in chemistry.cross_sections.values():
            if (
                curve.kind == "momentum_transfer"
                and curve.id in representative.rate_tables
            ):
                if curve.target in table_momentum:
                    raise CaseValidationError(
                        f"duplicate momentum rates for {curve.target}"
                    )
                table_momentum[curve.target] = curve.id
    return table_momentum


def compile_case(case: ResolvedCase) -> CompiledCase:
    """Compile a resolved case without creating outputs or running an ODE."""

    case = _validated_case_snapshot(case)
    chemistry_data = _chemistry_data(case)
    chemistry = compile_chemistry(chemistry_data)
    validate_chemistry_targets(case, chemistry_data)
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
    table_momentum = _table_momentum(case, chemistry, kinetics)
    elastic_heating, missing_momentum_targets = compile_elastic_heating(
        chemistry, table_momentum
    )
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
    # Fail at compilation, not after output creation or the first RHS call.
    initial_values = model.initial_state(initial_state)
    output_plan = compile_output(case, model, chemistry_data, initial_values)
    settings = solver_settings(case)
    assessment = assess_case(case, chemistry_data, missing_momentum_targets)
    return CompiledCase(
        case=case,
        chemistry_data=chemistry_data,
        chemistry=chemistry,
        model=model,
        initial_state=initial_state,
        solver_settings=settings,
        assessment=assessment,
        output_plan=output_plan,
        metadata=build_metadata(
            case,
            chemistry_data,
            chemistry,
            missing_momentum_targets,
            electron_kinetics_provenance,
            experimental_features,
            assessment,
        ),
    )


def _simulate_compiled_case(compiled: CompiledCase) -> SimulationResult:
    """Integrate once and evaluate only the derived series the user selected."""

    result = solve_compiled_model(
        compiled.model,
        compiled.initial_state,
        compiled.solver_settings,
    )
    observables = derive_observables(
        compiled.model,
        result.time_s,
        result.state,
        compiled.output_plan.evaluated,
        extra_names=compiled.output_plan.port_names,
    )
    stored = {name: observables[name] for name in compiled.output_plan.stored}
    for alias, source in compiled.output_plan.aliases.items():
        stored[alias] = (
            result.series(source)
            if source in result.state_labels
            else observables[source]
        )
    return replace(
        result,
        observables=stored,
        metadata={
            **result.metadata,
            **compiled.metadata,
            **compiled.output_plan.metadata,
        },
    )


def simulate_case(case: ResolvedCase) -> SimulationResult:
    """Compile and simulate one resolved case, returning the canonical result."""

    return _simulate_compiled_case(compile_case(case))


__all__ = ["CompiledCase", "compile_case", "simulate_case"]
