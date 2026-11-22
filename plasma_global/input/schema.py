"""The single public declaration used by YAML, Python, and case preparation."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import Field, PrivateAttr, model_validator

from plasma_global.input._schema_base import (
    Identifier,
    InputPath,
    NonemptyText,
    NonNegativeFloat,
    PositiveFloat,
    StrictModel,
)
from plasma_global.input._schema_models import (
    FixedGasEnergy,
    GasEnergyModel,
    QuasiNeutralElectronDensity,
)
from plasma_global.input._schema_power import PowerModel
from plasma_global.input._schema_reactor import WallTransport
from plasma_global.input._schema_recipe import GasInletCommand, SurfaceCommand
from plasma_global.input._schema_run import (
    ExperimentalConfig,
    OutputConfig,
    SolverConfig,
)


class CaseMetadata(StrictModel):
    name: NonemptyText
    description: str = ""


class InitialConditions(StrictModel):
    pressure_Pa: PositiveFloat
    gas_temperature_K: PositiveFloat
    densities_m3: dict[str, NonNegativeFloat] | None = None
    mole_fractions: dict[str, NonNegativeFloat] | None = None
    seed_densities_m3: dict[str, NonNegativeFloat] = Field(default_factory=dict)
    mean_energy_eV: PositiveFloat | None = None

    @model_validator(mode="after")
    def valid_composition(self) -> Self:
        from plasma_global.input._validation import validate_composition

        validate_composition(self)
        return self


class ZoneDefinition(StrictModel):
    volume_m3: PositiveFloat
    description: str = ""
    initial: InitialConditions


class SurfaceDefinition(StrictModel):
    zone_id: Identifier
    area_m2: PositiveFloat
    temperature_K: PositiveFloat
    wall_transport: WallTransport
    ion_impact_energy_eV: NonNegativeFloat = 0.0
    site_density_m2: NonNegativeFloat = 0.0
    initial_coverages: dict[str, NonNegativeFloat] = Field(default_factory=dict)


class EdgeDefinition(StrictModel):
    from_zone: Identifier
    to_zone: Identifier
    conductance_m3_s: NonNegativeFloat

    @model_validator(mode="after")
    def distinct_zones(self) -> Self:
        if self.from_zone == self.to_zone:
            raise ValueError("an edge must connect two different zones")
        return self


class InletDefinition(StrictModel):
    zone_id: Identifier
    temperature_K: PositiveFloat
    flow_sccm: dict[str, NonNegativeFloat] = Field(default_factory=dict)


class PumpDefinition(StrictModel):
    zone_id: Identifier
    speed_m3_s: NonNegativeFloat


class PowerDefinition(StrictModel):
    zone_id: Identifier
    model: PowerModel


class ReactorDefinition(StrictModel):
    chamber_id: NonemptyText = "reactor"
    description: str = ""
    zones: dict[Identifier, ZoneDefinition]
    surfaces: dict[Identifier, SurfaceDefinition] = Field(default_factory=dict)
    edges: dict[Identifier, EdgeDefinition] = Field(default_factory=dict)
    gas_inlets: dict[Identifier, InletDefinition] = Field(default_factory=dict)
    pumps: dict[Identifier, PumpDefinition] = Field(default_factory=dict)
    power_ports: dict[Identifier, PowerDefinition] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_topology(self) -> Self:
        from plasma_global.input._validation import validate_topology

        validate_topology(self)
        return self


class TableCoefficients(StrictModel):
    file: InputPath
    columns: dict[str, str] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    bounds: Literal["error", "hold"] = "error"
    mobility_reference_neutral_density_m3: PositiveFloat | None = None


class MaxwellianCoefficients(StrictModel):
    kind: Literal["maxwellian"] = "maxwellian"


class EnergyBalanceElectrons(StrictModel):
    kind: Literal["energy_balance"]
    coefficients: MaxwellianCoefficients | TableCoefficients = Field(
        default_factory=MaxwellianCoefficients
    )


class LocalFieldElectrons(StrictModel):
    kind: Literal["local_field"]
    coefficients: TableCoefficients


ElectronModel = Annotated[
    EnergyBalanceElectrons | LocalFieldElectrons, Field(discriminator="kind")
]


class SurfaceKinetics(StrictModel):
    kind: Literal["mass_action"]


class DensityProfile(StrictModel):
    kind: Literal["experimental.prescribed_profile"]
    file: InputPath
    columns: dict[str, str] = Field(default_factory=dict)
    zone_columns: dict[Identifier, NonemptyText] = Field(default_factory=dict)
    interpolation: Literal["linear", "previous"] = "linear"
    bounds: Literal["error", "hold"] = "error"
    time_offset_s: float = 0.0


class ModelsDefinition(StrictModel):
    electrons: ElectronModel
    electron_density: Annotated[
        QuasiNeutralElectronDensity | DensityProfile, Field(discriminator="kind")
    ] = Field(default_factory=lambda: QuasiNeutralElectronDensity(kind="quasineutral"))
    gas_energy: GasEnergyModel = Field(
        default_factory=lambda: FixedGasEnergy(kind="fixed")
    )
    surface_kinetics: SurfaceKinetics | None = None


class RunCommands(StrictModel):
    power_ports: dict[Identifier, dict[str, Any]] = Field(default_factory=dict)
    gas_inlets: dict[Identifier, GasInletCommand] = Field(default_factory=dict)
    surfaces: dict[Identifier, SurfaceCommand] = Field(default_factory=dict)


class RunStep(StrictModel):
    label: str | None = None
    duration_s: PositiveFloat
    commands: RunCommands = Field(default_factory=RunCommands)


class RunDefinition(StrictModel):
    start_time_s: NonNegativeFloat = 0.0
    duration_s: PositiveFloat | None = None
    steps: list[RunStep] | None = None

    @model_validator(mode="after")
    def one_duration(self) -> Self:
        if (self.duration_s is None) == (self.steps is None):
            raise ValueError("run requires exactly one of duration_s or steps")
        if self.steps is not None and not self.steps:
            raise ValueError("run.steps must not be empty")
        return self

    def normalized_steps(self) -> tuple[RunStep, ...]:
        if self.steps is not None:
            return tuple(self.steps)
        if self.duration_s is None:
            raise ValueError("run requires duration_s or steps")
        return (RunStep(duration_s=self.duration_s),)

    @property
    def total_duration_s(self) -> float:
        return math.fsum(step.duration_s for step in self.normalized_steps())

    @property
    def end_time_s(self) -> float:
        return self.step_bounds()[-1][1]

    def step_bounds(self) -> tuple[tuple[float, float], ...]:
        start = self.start_time_s
        elapsed = 0.0
        correction = 0.0
        bounds = []
        for step in self.normalized_steps():
            increment = step.duration_s - correction
            next_elapsed = elapsed + increment
            correction = (next_elapsed - elapsed) - increment
            elapsed = next_elapsed
            end = self.start_time_s + elapsed
            bounds.append((start, end))
            start = end
        return tuple(bounds)


class OutputSelection(OutputConfig):
    profile: Literal["basic", "minimal"] = "basic"
    quantities: list[Identifier] = Field(default_factory=list)
    zones: list[Identifier] = Field(default_factory=list)

    @model_validator(mode="after")
    def one_selection_form(self) -> Self:
        if self.zones and not self.quantities:
            raise ValueError("output.zones requires output.quantities")
        return self


class CaseSpec(StrictModel):
    schema_version: Literal[5]
    case: CaseMetadata
    chemistry: dict[str, Any]
    reactor: ReactorDefinition
    run: RunDefinition
    models: ModelsDefinition
    solver: SolverConfig = Field(default_factory=SolverConfig)
    output: OutputSelection = Field(default_factory=OutputSelection)
    experimental: ExperimentalConfig | None = None

    _base_directory: Path | None = PrivateAttr(default=None)
    _source_path: Path | None = PrivateAttr(default=None)
    _source_bytes: bytes | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def valid_references(self) -> Self:
        from plasma_global.input._validation import validate_case

        validate_case(self)
        return self

    @property
    def base_directory(self) -> Path:
        if self._base_directory is None:
            raise ValueError("use load_case or case_from_mapping with base_directory")
        return self._base_directory

    @property
    def source_path(self) -> Path | None:
        return self._source_path

    @property
    def source_bytes(self) -> bytes | None:
        return self._source_bytes

    def _with_origin(
        self,
        base_directory: Path,
        source: Path | None = None,
        content: bytes | None = None,
    ) -> Self:
        private = self.__pydantic_private__
        if private is None:
            raise RuntimeError("case origin storage was not initialized")
        private.update(
            _base_directory=base_directory.resolve(),
            _source_path=source,
            _source_bytes=content,
        )
        return self

    def to_dict(self, *, relative_to: str | Path | None = None) -> dict[str, Any]:
        """Return public keys; optionally rebase declared file references."""
        from plasma_global.input._paths import map_input_paths

        data = self.model_dump(mode="json")
        if relative_to is not None:
            target = Path(relative_to).resolve()
            base = self.base_directory
            map_input_paths(
                data, lambda value: os.path.relpath((base / value).resolve(), target)
            )
        return data

    def to_yaml(self, *, relative_to: str | Path | None = None) -> str:
        import yaml

        return yaml.safe_dump(
            self.to_dict(relative_to=relative_to), sort_keys=False, allow_unicode=True
        )


__all__ = ["CaseSpec"]
