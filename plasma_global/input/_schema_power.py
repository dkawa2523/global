"""Power-source models and their recipe command variants."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal, Self

from pydantic import Field, TypeAdapter, model_validator

from plasma_global.input._schema_base import (
    Identifier,
    InputPath,
    NonNegativeFloat,
    PositiveFloat,
    StrictModel,
)


class PrescribedPowerModel(StrictModel):
    kind: Literal["prescribed_power"]
    electron_fraction: Annotated[float, Field(ge=0.0, le=1.0)]
    gas_fraction: Annotated[float, Field(ge=0.0, le=1.0)]
    absorbed_power_W: NonNegativeFloat | None = None

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
    electron_mobility_m2_V_s: PositiveFloat | None = None


class ExternalTableModel(StrictModel):
    kind: Literal["external_table"]
    time_reference: Literal["run_start", "step_start"] = "run_start"
    file: InputPath
    columns: dict[str, str] = Field(default_factory=dict)
    interpolation: Literal["linear", "previous"] = "linear"
    bounds: Literal["error", "hold"] = "error"
    power_scale: NonNegativeFloat = 1.0
    voltage_scale: NonNegativeFloat = 1.0
    current_scale: NonNegativeFloat = 1.0
    gap_m: PositiveFloat | None = None
    total_density_m3: PositiveFloat | None = None
    plasma_potential_V: float = 0.0


def _validate_rf_control(
    control: str, power: float | None, voltage: float | None
) -> None:
    unused = "voltage_rms_V" if control == "absorbed_power" else "absorbed_power_W"
    value = voltage if control == "absorbed_power" else power
    if value is not None:
        raise ValueError(f"control={control!r} does not use {unused}; remove it")


class ExperimentalRFEnvelopeModel(StrictModel):
    kind: Literal["experimental.rf_envelope"]
    role: Literal["source", "bias"] = "source"
    frequency_Hz: PositiveFloat = 13.56e6
    control: Literal["absorbed_power", "voltage"] = "absorbed_power"
    absorbed_power_W: NonNegativeFloat | None = None
    voltage_rms_V: NonNegativeFloat | None = None
    effective_impedance_ohm: PositiveFloat = 50.0
    coupling_efficiency: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    base_reduced_field_Td: NonNegativeFloat = 0.0
    reduced_field_per_sqrt_W_Td: NonNegativeFloat = 0.0
    self_bias_fraction: NonNegativeFloat = 0.35
    plasma_potential_offset_V: NonNegativeFloat = 0.0
    plasma_potential_per_sqrt_W: NonNegativeFloat = 0.0

    @model_validator(mode="after")
    def compatible_default(self) -> Self:
        _validate_rf_control(self.control, self.absorbed_power_W, self.voltage_rms_V)
        return self


class ExperimentalCCPModel(StrictModel):
    kind: Literal["experimental.ccp"]
    electrode_gap_m: PositiveFloat
    powered_area_m2: PositiveFloat
    grounded_area_m2: PositiveFloat
    dominant_ion: Identifier
    frequency_Hz: PositiveFloat = 2.0e6
    control: Literal["absorbed_power", "voltage"] = "absorbed_power"
    absorbed_power_W: NonNegativeFloat | None = None
    voltage_rms_V: NonNegativeFloat | None = None

    @model_validator(mode="after")
    def compatible_default(self) -> Self:
        _validate_rf_control(self.control, self.absorbed_power_W, self.voltage_rms_V)
        return self


class ExperimentalICPModel(StrictModel):
    kind: Literal["experimental.icp"]
    frequency_Hz: PositiveFloat = 13.56e6
    delivered_power_W: NonNegativeFloat | None = None
    downstream_zone_id: Identifier | None = None
    downstream_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0

    @model_validator(mode="after")
    def downstream_target(self) -> Self:
        if self.downstream_fraction > 0.0 and self.downstream_zone_id is None:
            raise ValueError("positive downstream_fraction requires downstream_zone_id")
        return self


PowerModel = Annotated[
    PrescribedPowerModel
    | DCSeriesModel
    | ExternalTableModel
    | ExperimentalRFEnvelopeModel
    | ExperimentalCCPModel
    | ExperimentalICPModel,
    Field(discriminator="kind"),
]


class ContinuousWaveform(StrictModel):
    kind: Literal["continuous"] = "continuous"


class SquarePulseWaveform(StrictModel):
    kind: Literal["square_pulse"]
    duty_cycle: Annotated[float, Field(ge=0.0, le=1.0)]
    repetition_Hz: PositiveFloat
    phase_s: NonNegativeFloat = 0.0
    time_reference: Literal["run_start", "step_start"] = "run_start"


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
    time_reference: Literal["run_start", "step_start"] | None = None
    file: InputPath | None = None
    columns: dict[str, str] | None = None
    power_scale: NonNegativeFloat | None = None
    voltage_scale: NonNegativeFloat | None = None
    current_scale: NonNegativeFloat | None = None
    time_offset_s: float = 0.0
    interpolation: Literal["linear", "previous"] | None = None
    bounds: Literal["error", "hold"] | None = None
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


def _validate_power_command_control(
    model: PowerModel, command: PowerPortCommand
) -> None:
    if isinstance(
        model, (ExperimentalRFEnvelopeModel, ExperimentalCCPModel)
    ) and isinstance(command, (ExperimentalRFEnvelopeCommand, ExperimentalCCPCommand)):
        _validate_rf_control(
            model.control, command.absorbed_power_W, command.voltage_rms_V
        )


_COMMAND_ADAPTER: TypeAdapter[PowerPortCommand] = TypeAdapter(PowerPortCommand)


def parse_power_command(
    model: PowerModel, values: Mapping[str, Any]
) -> PowerPortCommand:
    """Validate one command against its declared port without resolving clocks."""
    if "kind" in values:
        raise ValueError("command kind is determined by its declared power port")
    command = _COMMAND_ADAPTER.validate_python({"kind": model.kind, **values})
    _validate_power_command_control(model, command)
    return command
