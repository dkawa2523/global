"""Power-source models and their recipe command variants."""

from __future__ import annotations

from typing import Annotated, Literal, Self, assert_never

from pydantic import Field, model_validator

from plasma_global.input._schema_base import (
    Identifier,
    InputPath,
    NonNegativeFloat,
    PositiveFloat,
    StrictModel,
    _publish_schema_types,
)


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


_publish_schema_types(globals())
