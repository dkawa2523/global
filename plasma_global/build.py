"""Composition root from strict schema-v5 input to the compiled solver."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, assert_never

from pydantic import ValidationError

from plasma_global._audit_types import CaseAssessment
from plasma_global._build_electrons import (
    compile_electron_density,
    compile_electrons,
)
from plasma_global._build_metadata import build_metadata, solver_settings
from plasma_global._input_sources import InputSourceStore
from plasma_global._model_assumptions import assess_case
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
from plasma_global.errors import CaseValidationError, IntegrationError
from plasma_global.input._compile_targets import validate_chemistry_targets
from plasma_global.input._paths import absolute_input_paths, map_input_paths
from plasma_global.input._schema_models import EvolvedGasEnergy, FixedGasEnergy
from plasma_global.input.compile_experimental import compile_experimental_accumulator
from plasma_global.input.compile_reactor import (
    compile_initial_densities,
    compile_reactor,
)
from plasma_global.input.compile_recipe import compile_recipe
from plasma_global.input.schema import CaseSpec, TableCoefficients
from plasma_global.models.elastic import (
    compile_elastic_heating,
)
from plasma_global.models.external_table import ExternalTableStore
from plasma_global.models.kinetics import (
    TabulatedElectronKinetics as TabulatedElectronKinetics,
)
from plasma_global.postprocess import derive_observables


@dataclass(frozen=True, slots=True)
class PreparedPhysics:
    """Complete immutable inputs for one deterministic simulation."""

    case: CaseSpec
    chemistry_data: ChemistryData
    chemistry: CompiledChemistry
    model: CompiledGlobalModel
    initial_state: InitialState
    solver_settings: SolverSettings
    assessment: CaseAssessment
    metadata: Mapping[str, Any]
    input_sources: Mapping[Path, bytes]

    @property
    def segments(self) -> tuple[RecipeSegment, ...]:
        return self.model.segments

    @property
    def warnings(self):
        return self.assessment.issues


@dataclass(frozen=True, slots=True)
class PreparedCase(PreparedPhysics):
    """A detached physical problem with its validated output selection."""

    output_plan: OutputPlan


def _compile_initial_state(
    case: CaseSpec, initial_densities: Mapping[str, Mapping[str, float]]
) -> InitialState:
    """Lower validated schema values to the solver's initial-state contract."""

    gas_energy = case.models.gas_energy
    if isinstance(gas_energy, EvolvedGasEnergy):
        gas_temperatures = {
            zone_id: zone.initial.gas_temperature_K
            for zone_id, zone in case.reactor.zones.items()
        }
    elif isinstance(gas_energy, FixedGasEnergy):
        gas_temperatures: dict[str, float] = {}
    else:
        assert_never(gas_energy)
    return InitialState(
        densities_m3_by_zone={
            zone_id: dict(initial_densities[zone_id]) for zone_id in case.reactor.zones
        },
        mean_energy_eV_by_zone={
            zone_id: zone.initial.mean_energy_eV
            for zone_id, zone in case.reactor.zones.items()
            if zone.initial.mean_energy_eV is not None
        },
        gas_temperature_K_by_zone=gas_temperatures,
        surface_coverages=(
            {
                surface_id: dict(surface.initial_coverages)
                for surface_id, surface in case.reactor.surfaces.items()
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
        data = case.model_dump(mode="python", round_trip=True, warnings=False)
        absolute_input_paths(data, case.base_directory)
        snapshot = CaseSpec.model_validate(data)
    except (TypeError, ValidationError, ValueError) as exc:
        raise CaseValidationError(
            "CaseSpec contains invalid changes made after validation"
        ) from exc
    return snapshot._with_origin(
        case.base_directory, case.source_path, case.source_bytes
    )


def _chemistry_data(case: CaseSpec, sources: InputSourceStore) -> ChemistryData:
    if "file" in case.chemistry:
        if len(case.chemistry) != 1:
            raise CaseValidationError(
                "chemistry.file cannot be combined with inline definitions"
            )
        return load_chemistry(case.chemistry["file"], sources=sources)
    return load_definition(
        case.chemistry,
        case.source_path or case.base_directory / "case.yaml",
        case.base_directory,
        sources=sources,
    )


def _capture_declared_references(case: CaseSpec, sources: InputSourceStore) -> None:
    """Keep declared fallback inputs portable even when a step overrides them."""

    def capture(reference: str) -> str:
        sources.read_bytes(reference, "declared_case_reference")
        return reference

    map_input_paths(case.to_dict(), capture)


def _table_momentum(
    case: CaseSpec,
    chemistry: CompiledChemistry,
    kinetics: Mapping[str, TabulatedElectronKinetics],
) -> dict[str, str] | None:
    table_momentum = None
    if isinstance(case.models.electrons.coefficients, TableCoefficients):
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


def compile_physics(case: CaseSpec) -> PreparedPhysics:
    """Compile a resolved case without creating outputs or running an ODE."""

    case = _validated_case_snapshot(case)
    sources = InputSourceStore()
    if case.source_path is not None and case.source_bytes is not None:
        sources.import_record(case.source_path, case.source_bytes, "case")
    _capture_declared_references(case, sources)
    chemistry_data = _chemistry_data(case, sources)
    chemistry = compile_chemistry(chemistry_data, sources=sources)
    validate_chemistry_targets(case, chemistry_data)
    initial_densities = compile_initial_densities(case, chemistry)
    external_tables = ExternalTableStore(sources)
    electron_density_profile = compile_electron_density(case, sources)
    segments = compile_recipe(
        case,
        chemistry_data,
        chemistry,
        external_tables,
        prescribed_electron_profile=electron_density_profile,
    )
    reactor = compile_reactor(
        case,
        chemistry_data,
        chemistry,
        external_tables,
        initial_commands=segments[0].port_commands,
    )
    initial_state = _compile_initial_state(case, initial_densities)
    electron_closure, kinetics, electron_kinetics_provenance = compile_electrons(
        case, chemistry_data, sources
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
    model.validate_initial_state(initial_state)
    settings = solver_settings(case)
    assessment = assess_case(case, chemistry_data, missing_momentum_targets)
    return PreparedPhysics(
        case=case,
        chemistry_data=chemistry_data,
        chemistry=chemistry,
        model=model,
        initial_state=initial_state,
        solver_settings=settings,
        assessment=assessment,
        input_sources=sources.contents(),
        metadata=build_metadata(
            case,
            chemistry_data,
            chemistry,
            missing_momentum_targets,
            electron_kinetics_provenance,
            experimental_features,
            assessment,
            sources,
        ),
    )


def compile_case(case: CaseSpec) -> PreparedCase:
    """Prepare physics and bind the requested output without integration."""
    physics = compile_physics(case)
    return PreparedCase(
        case=physics.case,
        chemistry_data=physics.chemistry_data,
        chemistry=physics.chemistry,
        model=physics.model,
        initial_state=physics.initial_state,
        solver_settings=physics.solver_settings,
        assessment=physics.assessment,
        metadata=physics.metadata,
        input_sources=physics.input_sources,
        output_plan=compile_output(physics.case, physics.model, physics.chemistry_data),
    )


def _complete_result(
    compiled: PreparedCase, result: SimulationResult
) -> SimulationResult:
    observables = derive_observables(
        compiled.model,
        result.time_s,
        result.state,
        compiled.output_plan.evaluated,
        extra_names=compiled.output_plan.available_units,
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


def simulate_case(case: CaseSpec | PreparedCase) -> SimulationResult:
    """Run a public declaration or a read-once prepared problem."""
    compiled = case if isinstance(case, PreparedCase) else compile_case(case)
    try:
        result = solve_compiled_model(
            compiled.model, compiled.initial_state, compiled.solver_settings
        )
    except IntegrationError as exc:
        if exc.partial_result is not None:
            exc.partial_result = _complete_result(compiled, exc.partial_result)
        raise
    return _complete_result(compiled, result)


__all__ = ["PreparedCase", "compile_case", "compile_physics", "simulate_case"]
