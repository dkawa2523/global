"""The single public YAML input schema (version 4)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from plasma_global.input._resolved_case import ResolvedCase
from plasma_global.input._schema_base import (
    Identifier,
    InputPath,
    NonNegativeFloat,
    PositiveFloat,
    StrictModel,
)
from plasma_global.input._schema_models import (
    ApproximateTwoTermElectronModel,
    FixedGasEnergy,
    GasEnergyModel,
    QuasiNeutralElectronDensity,
)
from plasma_global.input._schema_power import (
    DCSeriesModel,
    ExperimentalCCPModel,
    ExperimentalICPModel,
    ExperimentalRFEnvelopeModel,
    ExternalTableModel,
    PrescribedPowerModel,
)
from plasma_global.input._schema_reactor import WallTransport
from plasma_global.input._schema_recipe import GasInletCommand, SurfaceCommand
from plasma_global.input._schema_run import (
    ExperimentalConfig,
    OutputConfig,
    SolverConfig,
)


class CaseMetadata(StrictModel):
    name: Identifier
    description: str = ""


class InitialConditions(StrictModel):
    pressure_Pa: PositiveFloat
    gas_temperature_K: PositiveFloat
    densities_m3: dict[str, NonNegativeFloat] | None = None
    mole_fractions: dict[str, NonNegativeFloat] | None = None
    seed_densities_m3: dict[str, NonNegativeFloat] = Field(default_factory=dict)
    mean_energy_eV: PositiveFloat | None = None


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


class InletDefinition(StrictModel):
    zone_id: Identifier
    temperature_K: PositiveFloat
    flow_sccm: dict[str, NonNegativeFloat] = Field(default_factory=dict)


class PumpDefinition(StrictModel):
    zone_id: Identifier
    speed_m3_s: NonNegativeFloat


class AbsorbedPower(PrescribedPowerModel):
    default_absorbed_power_W: NonNegativeFloat | None = Field(
        default=None, alias="absorbed_power_W"
    )


class RFPower(ExperimentalRFEnvelopeModel):
    default_absorbed_power_W: NonNegativeFloat | None = Field(
        default=None, alias="absorbed_power_W"
    )
    default_voltage_rms_V: NonNegativeFloat | None = Field(
        default=None, alias="voltage_rms_V"
    )


class CCPPower(ExperimentalCCPModel):
    default_absorbed_power_W: NonNegativeFloat | None = Field(
        default=None, alias="absorbed_power_W"
    )
    default_voltage_rms_V: NonNegativeFloat | None = Field(
        default=None, alias="voltage_rms_V"
    )


class ICPPower(ExperimentalICPModel):
    default_delivered_power_W: NonNegativeFloat | None = Field(
        default=None, alias="delivered_power_W"
    )


class ExternalPower(ExternalTableModel):
    bounds_policy: Literal["error", "hold"] = Field(default="error", alias="bounds")
    time_reference: Literal["run_start", "step_start"] = "run_start"


PowerModel = Annotated[
    AbsorbedPower | DCSeriesModel | ExternalPower | RFPower | CCPPower | ICPPower,
    Field(discriminator="kind"),
]


class PowerDefinition(StrictModel):
    zone_id: Identifier
    coupling_target: str = ""
    model: PowerModel


class ReactorDefinition(StrictModel):
    chamber_id: Identifier = "reactor"
    description: str = ""
    zones: dict[Identifier, ZoneDefinition]
    surfaces: dict[Identifier, SurfaceDefinition] = Field(default_factory=dict)
    edges: dict[Identifier, EdgeDefinition] = Field(default_factory=dict)
    gas_inlets: dict[Identifier, InletDefinition] = Field(default_factory=dict)
    pumps: dict[Identifier, PumpDefinition] = Field(default_factory=dict)
    power_ports: dict[Identifier, PowerDefinition] = Field(default_factory=dict)


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
    coefficients: TableCoefficients | ApproximateTwoTermElectronModel


ElectronModel = Annotated[
    EnergyBalanceElectrons | LocalFieldElectrons, Field(discriminator="kind")
]


class SurfaceKinetics(StrictModel):
    kind: Literal["mass_action"]


class DensityProfile(StrictModel):
    kind: Literal["experimental.prescribed_profile"]
    file: InputPath
    columns: dict[str, str] = Field(default_factory=dict)
    zone_columns: dict[str, Identifier] = Field(default_factory=dict)
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
    label: Identifier | None = None
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


class OutputSelection(OutputConfig):
    quantities: list[
        Literal[
            "electron_density_m3",
            "electron_temperature_eV",
            "mean_energy_eV",
            "gas_temperature_K",
            "absorbed_power_W",
            "reduced_field_Td",
            "charge_residual_m3",
            "pressure_Pa",
            "neutral_density_m3",
            "heavy_density_m3",
        ]
    ] = Field(default_factory=list)
    zones: list[Identifier] = Field(default_factory=list)

    @model_validator(mode="after")
    def one_selection_form(self) -> Self:
        if self.zones and not self.quantities:
            raise ValueError("output.zones requires output.quantities")
        if self.quantities and self.observables:
            raise ValueError("choose quantities/zones or explicit observables")
        return self


class CaseSpec(StrictModel):
    schema_version: Literal[4]
    case: CaseMetadata
    chemistry: dict[str, Any]
    reactor: ReactorDefinition
    run: RunDefinition
    models: ModelsDefinition
    solver: SolverConfig = Field(default_factory=SolverConfig)
    output: OutputSelection = Field(default_factory=OutputSelection)
    experimental: ExperimentalConfig | None = None

    def resolve(self, base_directory: Path) -> ResolvedCase:
        from plasma_global.input._resolve import resolve_case

        return resolve_case(self, base_directory)


__all__ = ["CaseSpec"]
