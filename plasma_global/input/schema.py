"""Strict, declarative schema for a schema-v3 plasma-global case.

This module describes input only.  It deliberately does not load chemistry,
compile reactions, construct numerical state, or create output directories.
Those are separate stages of the application.
"""

from __future__ import annotations

import math
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Any, Literal, Self, assert_never

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    model_validator,
)


def _path_value(value: Any) -> Path:
    """Accept YAML's string representation without enabling other coercions."""

    if isinstance(value, Path):
        return value
    if not isinstance(value, str):
        raise TypeError("path values must be strings")
    if not value.strip():
        raise ValueError("path values must not be empty")
    return Path(value)


InputPath = Annotated[Path, BeforeValidator(_path_value)]
Identifier = Annotated[str, Field(min_length=1, pattern=r".*\S.*")]
PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
PositiveInt = Annotated[int, Field(gt=0)]


class StrictModel(BaseModel):
    """Common contract for all public v3 input objects."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        validate_default=True,
        allow_inf_nan=False,
    )


def _require_unique(items: list[Any], attribute: str, label: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in items:
        value = str(getattr(item, attribute))
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        values = ", ".join(sorted(duplicates))
        raise ValueError(f"duplicate {label}: {values}")


class CaseMetadata(StrictModel):
    name: Identifier
    description: str = ""


class ChemistryConfig(StrictModel):
    manifest: InputPath


class ZoneConfig(StrictModel):
    zone_id: Identifier
    description: str = ""
    volume_m3: PositiveFloat
    pressure_Pa: PositiveFloat
    gas_temperature_K: PositiveFloat
    initial_densities_m3: dict[str, NonNegativeFloat] | None = None
    initial_mole_fractions: dict[str, NonNegativeFloat] | None = None
    initial_seed_densities_m3: dict[str, NonNegativeFloat] = Field(default_factory=dict)
    initial_mean_energy_eV: PositiveFloat | None = None

    @model_validator(mode="after")
    def has_initial_composition(self) -> Self:
        densities = self.initial_densities_m3
        fractions = self.initial_mole_fractions
        if (densities is None) == (fractions is None):
            raise ValueError(
                "set exactly one of initial_densities_m3 or initial_mole_fractions"
            )
        if densities is not None:
            if self.initial_seed_densities_m3:
                raise ValueError(
                    "initial_seed_densities_m3 is only valid with "
                    "initial_mole_fractions"
                )
            if not densities or not any(value > 0.0 for value in densities.values()):
                raise ValueError("initial_densities_m3 must contain a positive density")
        elif fractions is not None:
            if not fractions or not any(value > 0.0 for value in fractions.values()):
                raise ValueError(
                    "initial_mole_fractions must contain a positive fraction"
                )
            total = sum(fractions.values())
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError(
                    f"initial_mole_fractions must sum to one, got {total:g}"
                )
        return self


class EdgeConfig(StrictModel):
    edge_id: Identifier
    from_zone: Identifier
    to_zone: Identifier
    conductance_m3_s: NonNegativeFloat

    @model_validator(mode="after")
    def distinct_zones(self) -> Self:
        if self.from_zone == self.to_zone:
            raise ValueError("an edge must connect two different zones")
        return self


class BohmWallTransport(StrictModel):
    kind: Literal["bohm"]
    characteristic_length_m: PositiveFloat | None = None
    h_factor: Literal["auto"] | Annotated[float, Field(gt=0.0, le=1.0)] = 1.0
    min_h_factor: Annotated[float, Field(gt=0.0, le=1.0)] = 0.02
    max_h_factor: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0
    ion_neutral_cross_section_m2: PositiveFloat = 1.0e-18

    @model_validator(mode="after")
    def ordered_h_factors(self) -> Self:
        if self.max_h_factor < self.min_h_factor:
            raise ValueError("max_h_factor must be at least min_h_factor")
        return self


class PrescribedFrequencyWallTransport(StrictModel):
    kind: Literal["prescribed_frequency"]
    frequency_s_inv: NonNegativeFloat


class AmbipolarWallTransport(StrictModel):
    kind: Literal["ambipolar"]
    diffusion_coefficient_m2_s: PositiveFloat
    diffusion_length_m: PositiveFloat | None = None


class OffWallTransport(StrictModel):
    kind: Literal["off"]


WallTransport = Annotated[
    BohmWallTransport
    | PrescribedFrequencyWallTransport
    | AmbipolarWallTransport
    | OffWallTransport,
    Field(discriminator="kind"),
]


class SurfaceConfig(StrictModel):
    surface_id: Identifier
    zone_id: Identifier
    area_m2: PositiveFloat
    temperature_K: PositiveFloat
    ion_impact_energy_eV: NonNegativeFloat = 0.0
    site_density_m2: NonNegativeFloat = 0.0
    initial_coverages: dict[str, NonNegativeFloat] = Field(default_factory=dict)
    wall_transport: WallTransport


class GasInletConfig(StrictModel):
    inlet_id: Identifier
    zone_id: Identifier
    flow_sccm: dict[str, NonNegativeFloat] = Field(default_factory=dict)
    temperature_K: PositiveFloat


class PumpConfig(StrictModel):
    pump_id: Identifier
    zone_id: Identifier
    speed_m3_s: NonNegativeFloat


class PrescribedPowerModel(StrictModel):
    kind: Literal["prescribed_power"]
    electron_fraction: Annotated[float, Field(ge=0.0, le=1.0)]
    gas_fraction: Annotated[float, Field(ge=0.0, le=1.0)]
    default_absorbed_power_W: NonNegativeFloat | None = None

    @model_validator(mode="after")
    def complete_partition(self) -> Self:
        if abs(self.electron_fraction + self.gas_fraction - 1.0) > 1.0e-12:
            raise ValueError("electron_fraction + gas_fraction must equal 1")
        return self


class DCSeriesModel(StrictModel):
    kind: Literal["dc_series"]
    ballast_resistance_ohm: PositiveFloat
    gap_m: PositiveFloat
    electrode_area_m2: PositiveFloat
    source_voltage_V: float | None = None
    power_absorption_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    electron_mobility_m2_V_s: NonNegativeFloat | None = None


class ExternalTableModel(StrictModel):
    kind: Literal["external_table"]
    file: InputPath
    interpolation: Literal["linear", "previous"] = "linear"
    bounds_policy: Literal["error", "hold"] = "error"
    power_scale: NonNegativeFloat = 1.0
    voltage_scale: NonNegativeFloat = 1.0
    current_scale: NonNegativeFloat = 1.0
    gap_m: PositiveFloat | None = None
    total_density_m3: PositiveFloat | None = None
    plasma_potential_V: float = 0.0


class ExperimentalRFEnvelopeModel(StrictModel):
    kind: Literal["experimental.rf_envelope"]
    role: Literal["source", "bias"] = "source"
    frequency_Hz: PositiveFloat = 13.56e6
    control: Literal["absorbed_power", "voltage"] = "absorbed_power"
    default_absorbed_power_W: NonNegativeFloat | None = None
    default_voltage_rms_V: NonNegativeFloat | None = None
    effective_impedance_ohm: PositiveFloat = 50.0
    coupling_efficiency: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    base_reduced_field_Td: NonNegativeFloat = 0.0
    reduced_field_per_sqrt_W_Td: NonNegativeFloat = 0.0
    self_bias_fraction: NonNegativeFloat = 0.35
    plasma_potential_offset_V: float = 0.0
    plasma_potential_per_sqrt_W: NonNegativeFloat = 0.0


class ExperimentalCCPModel(StrictModel):
    kind: Literal["experimental.ccp"]
    frequency_Hz: PositiveFloat = 2.0e6
    control: Literal["absorbed_power", "voltage"] = "absorbed_power"
    default_absorbed_power_W: NonNegativeFloat | None = None
    default_voltage_rms_V: NonNegativeFloat | None = None


class ExperimentalICPModel(StrictModel):
    kind: Literal["experimental.icp"]
    frequency_Hz: PositiveFloat = 13.56e6
    default_delivered_power_W: NonNegativeFloat | None = None


PowerModel = Annotated[
    PrescribedPowerModel
    | DCSeriesModel
    | ExternalTableModel
    | ExperimentalRFEnvelopeModel
    | ExperimentalCCPModel
    | ExperimentalICPModel,
    Field(discriminator="kind"),
]


class PowerPortConfig(StrictModel):
    port_id: Identifier
    zone_id: Identifier
    coupling_target: str = ""
    model: PowerModel


class ReactorConfig(StrictModel):
    chamber_id: Identifier
    description: str = ""
    zones: list[ZoneConfig]
    edges: list[EdgeConfig] = Field(default_factory=list)
    surfaces: list[SurfaceConfig] = Field(default_factory=list)
    gas_inlets: list[GasInletConfig] = Field(default_factory=list)
    pumps: list[PumpConfig] = Field(default_factory=list)
    power_ports: list[PowerPortConfig] = Field(default_factory=list)

    def _validate_unique_ids(self) -> None:
        for items, attribute, label in (
            (self.zones, "zone_id", "zone_id"),
            (self.edges, "edge_id", "edge_id"),
            (self.surfaces, "surface_id", "surface_id"),
            (self.gas_inlets, "inlet_id", "inlet_id"),
            (self.pumps, "pump_id", "pump_id"),
            (self.power_ports, "port_id", "port_id"),
        ):
            _require_unique(items, attribute, label)

    def _validate_zone_references(self, zone_ids: set[str]) -> None:
        for edge in self.edges:
            for field_name, zone_id in (
                ("from_zone", edge.from_zone),
                ("to_zone", edge.to_zone),
            ):
                if zone_id not in zone_ids:
                    raise ValueError(
                        f"edge {edge.edge_id!r} {field_name} references unknown "
                        f"zone {zone_id!r}"
                    )
        for label, entities, id_field in (
            ("surface", self.surfaces, "surface_id"),
            ("gas inlet", self.gas_inlets, "inlet_id"),
            ("pump", self.pumps, "pump_id"),
            ("power port", self.power_ports, "port_id"),
        ):
            for entity in entities:
                if entity.zone_id not in zone_ids:
                    raise ValueError(
                        f"{label} {getattr(entity, id_field)!r} "
                        f"references unknown zone {entity.zone_id!r}"
                    )

    def _validate_coupling_targets(
        self, zone_ids: set[str], surface_ids: set[str]
    ) -> None:
        valid_targets = zone_ids | surface_ids
        for port in self.power_ports:
            if port.coupling_target and port.coupling_target not in valid_targets:
                raise ValueError(
                    f"power port {port.port_id!r} references unknown coupling target "
                    f"{port.coupling_target!r}"
                )

    @model_validator(mode="after")
    def valid_topology(self) -> Self:
        if not self.zones:
            raise ValueError("reactor.zones must contain at least one zone")
        self._validate_unique_ids()
        zone_ids = {zone.zone_id for zone in self.zones}
        self._validate_zone_references(zone_ids)
        self._validate_coupling_targets(
            zone_ids,
            {surface.surface_id for surface in self.surfaces},
        )
        return self


class ContinuousWaveform(StrictModel):
    kind: Literal["continuous"] = "continuous"


class SquarePulseWaveform(StrictModel):
    kind: Literal["square_pulse"]
    duty_cycle: Annotated[float, Field(ge=0.0, le=1.0)]
    repetition_Hz: PositiveFloat
    phase_s: NonNegativeFloat = 0.0


Waveform = Annotated[
    ContinuousWaveform | SquarePulseWaveform,
    Field(discriminator="kind"),
]


class GasInletCommand(StrictModel):
    flow_sccm: dict[str, NonNegativeFloat]


class SurfaceCommand(StrictModel):
    temperature_K: PositiveFloat | None = None


class PrescribedPowerCommand(StrictModel):
    kind: Literal["prescribed_power"]
    absorbed_power_W: NonNegativeFloat | None = None
    waveform: Waveform = Field(
        default_factory=lambda: ContinuousWaveform(kind="continuous")
    )


class DCSeriesCommand(StrictModel):
    kind: Literal["dc_series"]
    source_voltage_V: float | None = None
    off_voltage_V: float = 0.0
    waveform: Waveform = Field(
        default_factory=lambda: ContinuousWaveform(kind="continuous")
    )


class ExternalTableCommand(StrictModel):
    kind: Literal["external_table"]
    file: InputPath | None = None
    power_scale: NonNegativeFloat | None = None
    voltage_scale: NonNegativeFloat | None = None
    current_scale: NonNegativeFloat | None = None
    time_offset_s: float = 0.0
    interpolation: Literal["linear", "previous"] | None = None
    bounds_policy: Literal["error", "hold"] | None = None
    gap_m: PositiveFloat | None = None
    total_density_m3: PositiveFloat | None = None
    plasma_potential_V: float | None = None


class ExperimentalRFEnvelopeCommand(StrictModel):
    kind: Literal["experimental.rf_envelope"]
    absorbed_power_W: NonNegativeFloat | None = None
    voltage_rms_V: NonNegativeFloat | None = None
    waveform: Waveform = Field(
        default_factory=lambda: ContinuousWaveform(kind="continuous")
    )


class ExperimentalCCPCommand(StrictModel):
    kind: Literal["experimental.ccp"]
    absorbed_power_W: NonNegativeFloat | None = None
    voltage_rms_V: NonNegativeFloat | None = None
    waveform: Waveform = Field(
        default_factory=lambda: ContinuousWaveform(kind="continuous")
    )


class ExperimentalICPCommand(StrictModel):
    kind: Literal["experimental.icp"]
    delivered_power_W: NonNegativeFloat | None = None
    waveform: Waveform = Field(
        default_factory=lambda: ContinuousWaveform(kind="continuous")
    )


PowerPortCommand = Annotated[
    PrescribedPowerCommand
    | DCSeriesCommand
    | ExternalTableCommand
    | ExperimentalRFEnvelopeCommand
    | ExperimentalCCPCommand
    | ExperimentalICPCommand,
    Field(discriminator="kind"),
]


def _power_command_matches_model(model: PowerModel, command: PowerPortCommand) -> bool:
    if isinstance(model, PrescribedPowerModel):
        return isinstance(command, PrescribedPowerCommand)
    if isinstance(model, DCSeriesModel):
        return isinstance(command, DCSeriesCommand)
    if isinstance(model, ExternalTableModel):
        return isinstance(command, ExternalTableCommand)
    if isinstance(model, ExperimentalRFEnvelopeModel):
        return isinstance(command, ExperimentalRFEnvelopeCommand)
    if isinstance(model, ExperimentalCCPModel):
        return isinstance(command, ExperimentalCCPCommand)
    if isinstance(model, ExperimentalICPModel):
        return isinstance(command, ExperimentalICPCommand)
    assert_never(model)


class RecipeCommands(StrictModel):
    gas_inlets: dict[str, GasInletCommand] = Field(default_factory=dict)
    power_ports: dict[str, PowerPortCommand] = Field(default_factory=dict)
    surfaces: dict[str, SurfaceCommand] = Field(default_factory=dict)


class RecipeStepConfig(StrictModel):
    step_id: Identifier
    duration_s: PositiveFloat
    commands: RecipeCommands = Field(default_factory=RecipeCommands)


class RecipeConfig(StrictModel):
    recipe_id: Identifier
    description: str = ""
    start_time_s: NonNegativeFloat = 0.0
    steps: list[RecipeStepConfig]

    @model_validator(mode="after")
    def valid_steps(self) -> Self:
        if not self.steps:
            raise ValueError("recipe.steps must contain at least one step")
        _require_unique(self.steps, "step_id", "step_id")
        return self

    @property
    def duration_s(self) -> float:
        return sum(step.duration_s for step in self.steps)

    @property
    def end_time_s(self) -> float:
        return self.start_time_s + self.duration_s

    def step_bounds(self) -> list[tuple[float, float]]:
        start = self.start_time_s
        bounds: list[tuple[float, float]] = []
        for step in self.steps:
            end = start + step.duration_s
            bounds.append((start, end))
            start = end
        return bounds


class MaxwellianElectronModel(StrictModel):
    kind: Literal["maxwellian"]


class TableElectronModel(StrictModel):
    kind: Literal["table"]
    file: InputPath
    lookup: Literal["mean_energy", "local_field"]
    bounds_policy: Literal["error", "clip"] = "error"


class CacheConfig(StrictModel):
    max_entries: PositiveInt = 12
    fraction_decimals: Annotated[int, Field(ge=0)] = 3


class EnergyGridConfig(StrictModel):
    min_eV: PositiveFloat = 1.0e-3
    max_eV: PositiveFloat = 160.0
    n: Annotated[int, Field(ge=32)] = 360

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.max_eV <= self.min_eV:
            raise ValueError("energy_grid.max_eV must exceed min_eV")
        return self


class ReducedFieldGridConfig(StrictModel):
    # The approximate closure only publishes states whose power-balance root
    # is bracketed.  These conservative defaults cover the validated argon
    # examples; users may narrow or extend them when their own collision set
    # demonstrates a root at every requested point.
    min_Td: PositiveFloat = 1.0
    max_Td: PositiveFloat = 100.0
    n: Annotated[int, Field(ge=2)] = 48

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.max_Td <= self.min_Td:
            raise ValueError("reduced_field_grid.max_Td must exceed min_Td")
        return self


class ApproximateTwoTermElectronModel(StrictModel):
    kind: Literal["experimental.approximate_two_term"]
    mixture_key_species: list[str] = Field(default_factory=list)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    energy_grid: EnergyGridConfig = Field(default_factory=EnergyGridConfig)
    reduced_field_grid: ReducedFieldGridConfig = Field(
        default_factory=ReducedFieldGridConfig
    )
    max_shape_iterations: PositiveInt = 48


ElectronModel = Annotated[
    MaxwellianElectronModel | TableElectronModel | ApproximateTwoTermElectronModel,
    Field(discriminator="kind"),
]


class ElectronEnergyClosure(StrictModel):
    kind: Literal["electron_energy"]


class LocalFieldClosure(StrictModel):
    kind: Literal["local_field"]


ElectronClosure = Annotated[
    ElectronEnergyClosure | LocalFieldClosure,
    Field(discriminator="kind"),
]


class QuasiNeutralElectronDensity(StrictModel):
    kind: Literal["quasineutral"]


class PrescribedElectronDensity(StrictModel):
    kind: Literal["experimental.prescribed_profile"]
    file: InputPath
    zone_columns: dict[str, Identifier] = Field(default_factory=dict)
    interpolation: Literal["linear", "previous"] = "linear"
    hold: Literal["edge", "error"] = "error"


ElectronDensityModel = Annotated[
    QuasiNeutralElectronDensity | PrescribedElectronDensity,
    Field(discriminator="kind"),
]


class FixedGasEnergy(StrictModel):
    kind: Literal["fixed"]


class EvolvedGasEnergy(StrictModel):
    kind: Literal["evolved"]
    wall_energy_relaxation_s_inv_by_zone: dict[str, NonNegativeFloat] = Field(
        default_factory=dict
    )


GasEnergyModel = Annotated[
    FixedGasEnergy | EvolvedGasEnergy,
    Field(discriminator="kind"),
]


class SurfaceKineticsConfig(StrictModel):
    """Presence enables surface-coverage state and surface reactions."""


class ModelsConfig(StrictModel):
    electrons: ElectronModel
    electron_closure: ElectronClosure
    electron_density: ElectronDensityModel = Field(
        default_factory=lambda: QuasiNeutralElectronDensity(kind="quasineutral")
    )
    gas_energy: GasEnergyModel = Field(
        default_factory=lambda: FixedGasEnergy(kind="fixed")
    )
    surface_kinetics: SurfaceKineticsConfig | None = None


class SolverConfig(StrictModel):
    method: Literal["BDF"] = "BDF"
    rtol: float = Field(
        default=1.0e-6,
        ge=100.0 * math.ulp(1.0),
        lt=1.0,
    )
    atol: PositiveFloat = Field(
        default=1.0e-14,
        description="Dimensionless absolute tolerance applied after state scaling.",
    )
    first_step_s: PositiveFloat | None = None
    max_step_s: PositiveFloat | None = None
    sample_interval_s: PositiveFloat | None = None
    save_at_s: list[NonNegativeFloat] | None = None

    @model_validator(mode="after")
    def one_sampling_mode(self) -> Self:
        if self.sample_interval_s is not None and self.save_at_s is not None:
            raise ValueError(
                "set at most one of solver.sample_interval_s or solver.save_at_s"
            )
        if self.save_at_s is not None:
            if not self.save_at_s:
                raise ValueError("solver.save_at_s must not be empty")
            if any(right <= left for left, right in pairwise(self.save_at_s)):
                raise ValueError("solver.save_at_s must be strictly increasing")
        if (
            self.first_step_s is not None
            and self.max_step_s is not None
            and self.first_step_s > self.max_step_s
        ):
            raise ValueError("solver.first_step_s must not exceed solver.max_step_s")
        return self


class OutputConfig(StrictModel):
    observables: list[str] = Field(default_factory=list)
    summary_series: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_series_selection(self) -> Self:
        for field_name, values in (
            ("observables", self.observables),
            ("summary_series", self.summary_series),
        ):
            if any(not value.strip() for value in values):
                raise ValueError(f"output.{field_name} must not contain empty names")
            if len(set(values)) != len(values):
                raise ValueError(f"output.{field_name} must not contain duplicates")
        return self


class ExperimentalFilmConfig(StrictModel):
    """Presence enables chemistry-declared film state."""

    monolayer_thickness_m: PositiveFloat = 3.0e-10


class ExperimentalInventoryEventConfig(StrictModel):
    reaction_id: Identifier
    surface_id: Identifier
    inventory_particles_per_event: dict[Identifier, float]

    @model_validator(mode="after")
    def nonzero_named_yields(self) -> Self:
        if not self.inventory_particles_per_event:
            raise ValueError("inventory event yield mapping must not be empty")
        if any(value == 0.0 for value in self.inventory_particles_per_event.values()):
            raise ValueError("inventory event yields must be explicitly nonzero")
        return self


class ExperimentalWallInventoryConfig(StrictModel):
    """Presence enables chemistry-declared wall inventories."""

    initial_by_surface: dict[str, dict[str, NonNegativeFloat]] = Field(
        default_factory=dict
    )
    events: list[ExperimentalInventoryEventConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_events(self) -> Self:
        pairs = [(item.reaction_id, item.surface_id) for item in self.events]
        if len(set(pairs)) != len(pairs):
            raise ValueError("wall inventory reaction/surface events must be unique")
        return self


class ExperimentalExtensionsConfig(StrictModel):
    """Presence enables experimental chemistry extension state and terms."""


class ExperimentalStopWhenQuasiSteadyConfig(StrictModel):
    """Legacy-named nonterminal quasi-steady observation, not a root solve."""

    relative_rhs_norm_s_inv: NonNegativeFloat = 1.0e-3
    min_time_s: NonNegativeFloat = 0.0


class ExperimentalConfig(StrictModel):
    film: ExperimentalFilmConfig | None = None
    wall_inventory: ExperimentalWallInventoryConfig | None = None
    extensions: ExperimentalExtensionsConfig | None = None
    stop_when_quasi_steady: ExperimentalStopWhenQuasiSteadyConfig | None = None

    @model_validator(mode="after")
    def enables_at_least_one_feature(self) -> Self:
        if (
            self.film is None
            and self.wall_inventory is None
            and self.extensions is None
            and self.stop_when_quasi_steady is None
        ):
            raise ValueError("experimental must enable at least one explicit feature")
        return self


class CaseSpec(StrictModel):
    schema_version: Literal[3]
    case: CaseMetadata
    chemistry: ChemistryConfig
    reactor: ReactorConfig
    recipe: RecipeConfig
    models: ModelsConfig
    solver: SolverConfig
    output: OutputConfig
    experimental: ExperimentalConfig | None = None

    _source_path: Path | None = PrivateAttr(default=None)
    _included_files: tuple[Path, ...] = PrivateAttr(default=())

    def _validate_recipe_commands(self) -> None:
        inlet_ids = {item.inlet_id for item in self.reactor.gas_inlets}
        ports = {item.port_id: item for item in self.reactor.power_ports}
        surface_ids = {item.surface_id for item in self.reactor.surfaces}
        for step in self.recipe.steps:
            commands = step.commands
            for label, references, available in (
                ("gas inlet", set(commands.gas_inlets), inlet_ids),
                ("power port", set(commands.power_ports), set(ports)),
                ("surface", set(commands.surfaces), surface_ids),
            ):
                unknown = references - available
                if unknown:
                    values = ", ".join(sorted(unknown))
                    raise ValueError(
                        f"recipe step {step.step_id!r} references unknown {label}(s): "
                        f"{values}"
                    )
            for port_id, command in commands.power_ports.items():
                port = ports[port_id]
                if not _power_command_matches_model(port.model, command):
                    raise ValueError(
                        f"recipe step {step.step_id!r} command kind "
                        f"{command.kind!r} does not match power port {port_id!r} "
                        f"model kind {port.model.kind!r}"
                    )

    def _validate_initial_electron_state(
        self, closure: Literal["electron_energy", "local_field"]
    ) -> None:
        for zone in self.reactor.zones:
            if closure == "electron_energy" and zone.initial_mean_energy_eV is None:
                raise ValueError(
                    f"zone {zone.zone_id!r} requires initial_mean_energy_eV for "
                    "the electron_energy closure"
                )
            if closure == "local_field" and zone.initial_mean_energy_eV is not None:
                raise ValueError(
                    f"zone {zone.zone_id!r} initial_mean_energy_eV is incompatible "
                    "with the local_field closure"
                )

    def _validate_electron_model(
        self, closure: Literal["electron_energy", "local_field"]
    ) -> None:
        electron_model = self.models.electrons
        if isinstance(electron_model, MaxwellianElectronModel):
            if closure != "electron_energy":
                raise ValueError(
                    "maxwellian electrons require the electron_energy closure"
                )
        elif isinstance(electron_model, ApproximateTwoTermElectronModel):
            if closure != "local_field":
                raise ValueError(
                    "experimental.approximate_two_term electrons require the "
                    "local_field closure"
                )
        elif isinstance(electron_model, TableElectronModel):
            expected_lookup = (
                "mean_energy" if closure == "electron_energy" else "local_field"
            )
            if electron_model.lookup != expected_lookup:
                raise ValueError(
                    f"table lookup {electron_model.lookup!r} must match electron "
                    f"closure {closure!r}"
                )
        else:
            assert_never(electron_model)

    def _validate_gas_energy_references(self, zone_ids: set[str]) -> None:
        gas_energy = self.models.gas_energy
        if isinstance(gas_energy, EvolvedGasEnergy):
            unknown = set(gas_energy.wall_energy_relaxation_s_inv_by_zone) - zone_ids
            if unknown:
                values = ", ".join(sorted(unknown))
                raise ValueError(
                    "models.gas_energy.wall_energy_relaxation_s_inv_by_zone "
                    f"references unknown zones: {values}"
                )
        elif not isinstance(gas_energy, FixedGasEnergy):
            assert_never(gas_energy)

    def _validate_wall_inventory_references(self, surface_ids: set[str]) -> None:
        wall_inventory = (
            None if self.experimental is None else self.experimental.wall_inventory
        )
        if wall_inventory is None:
            return
        unknown = set(wall_inventory.initial_by_surface) - surface_ids
        if unknown:
            values = ", ".join(sorted(unknown))
            raise ValueError(
                "experimental.wall_inventory.initial_by_surface references "
                f"unknown surfaces: {values}"
            )
        unknown_event_surfaces = {
            event.surface_id for event in wall_inventory.events
        } - surface_ids
        if unknown_event_surfaces:
            raise ValueError(
                "experimental.wall_inventory.events references unknown "
                f"surfaces: {', '.join(sorted(unknown_event_surfaces))}"
            )
        undeclared_inventory = {
            (event.surface_id, inventory_id)
            for event in wall_inventory.events
            for inventory_id in event.inventory_particles_per_event
            if inventory_id
            not in wall_inventory.initial_by_surface.get(event.surface_id, {})
        }
        if undeclared_inventory:
            raise ValueError(
                "experimental.wall_inventory.events writes undeclared "
                f"inventory states: {sorted(undeclared_inventory)}"
            )

    def _validate_save_times(self) -> None:
        save_at = self.solver.save_at_s
        if save_at is not None and (
            save_at[0] < self.recipe.start_time_s
            or save_at[-1] > self.recipe.end_time_s
        ):
            raise ValueError(
                "solver.save_at_s must lie within the recipe time interval"
            )

    @model_validator(mode="after")
    def valid_commands(self) -> Self:
        zone_ids = {item.zone_id for item in self.reactor.zones}
        surface_ids = {item.surface_id for item in self.reactor.surfaces}
        closure: Literal["electron_energy", "local_field"] = (
            self.models.electron_closure.kind
        )
        self._validate_recipe_commands()
        self._validate_initial_electron_state(closure)
        self._validate_electron_model(closure)
        self._validate_gas_energy_references(zone_ids)
        self._validate_wall_inventory_references(surface_ids)
        self._validate_save_times()
        return self

    @property
    def source_path(self) -> Path | None:
        return self._source_path

    @property
    def included_files(self) -> tuple[Path, ...]:
        return self._included_files

    def _with_source(self, source_path: Path, included_files: tuple[Path, ...]) -> Self:
        object.__setattr__(self, "_source_path", source_path)
        object.__setattr__(self, "_included_files", included_files)
        return self


__all__ = [
    "AmbipolarWallTransport",
    "ApproximateTwoTermElectronModel",
    "BohmWallTransport",
    "CaseMetadata",
    "CaseSpec",
    "ChemistryConfig",
    "ContinuousWaveform",
    "DCSeriesCommand",
    "DCSeriesModel",
    "EdgeConfig",
    "ElectronClosure",
    "ElectronDensityModel",
    "ElectronEnergyClosure",
    "ElectronModel",
    "EvolvedGasEnergy",
    "ExperimentalCCPCommand",
    "ExperimentalCCPModel",
    "ExperimentalConfig",
    "ExperimentalExtensionsConfig",
    "ExperimentalFilmConfig",
    "ExperimentalICPCommand",
    "ExperimentalICPModel",
    "ExperimentalInventoryEventConfig",
    "ExperimentalRFEnvelopeCommand",
    "ExperimentalRFEnvelopeModel",
    "ExperimentalWallInventoryConfig",
    "ExternalTableCommand",
    "ExternalTableModel",
    "FixedGasEnergy",
    "GasEnergyModel",
    "GasInletCommand",
    "GasInletConfig",
    "InputPath",
    "LocalFieldClosure",
    "MaxwellianElectronModel",
    "ModelsConfig",
    "OffWallTransport",
    "OutputConfig",
    "PowerModel",
    "PowerPortCommand",
    "PowerPortConfig",
    "PrescribedElectronDensity",
    "PrescribedFrequencyWallTransport",
    "PrescribedPowerCommand",
    "PrescribedPowerModel",
    "PumpConfig",
    "QuasiNeutralElectronDensity",
    "ReactorConfig",
    "RecipeCommands",
    "RecipeConfig",
    "RecipeStepConfig",
    "SolverConfig",
    "SquarePulseWaveform",
    "SurfaceCommand",
    "SurfaceConfig",
    "SurfaceKineticsConfig",
    "TableElectronModel",
    "WallTransport",
    "Waveform",
    "ZoneConfig",
]
