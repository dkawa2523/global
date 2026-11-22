"""Resolve recipe power commands and their time discontinuities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, assert_never

from plasma_global.errors import CaseValidationError
from plasma_global.input._compile_power import compile_external_binding
from plasma_global.input._schema_power import (
    ContinuousWaveform,
    DCSeriesCommand,
    DCSeriesModel,
    ExperimentalCCPModel,
    ExperimentalICPModel,
    ExperimentalRFEnvelopeModel,
    ExternalTableCommand,
    ExternalTableModel,
    PowerPortCommand,
    PrescribedPowerModel,
    SquarePulseWaveform,
    Waveform,
    parse_power_command,
)
from plasma_global.models.external_table import ExternalTableBinding, ExternalTableStore
from plasma_global.models.power import CompiledPowerCommand

if TYPE_CHECKING:
    from plasma_global.input.schema import PowerDefinition, RunStep

MAX_COMPILED_SEGMENTS = 100_000


def parse_step_power_commands(
    step: RunStep, ports: Mapping[str, PowerDefinition]
) -> dict[str, PowerPortCommand]:
    """Parse public command values once before splitting the step."""
    return {
        port_id: parse_power_command(ports[port_id].model, values)
        for port_id, values in step.commands.power_ports.items()
    }


def power_step_boundaries(
    commands: dict[str, PowerPortCommand],
    external_bindings: dict[str, ExternalTableBinding],
    start_s: float,
    end_s: float,
    *,
    run_start_s: float,
) -> set[float]:
    """Return discontinuities introduced by power commands within one step."""

    boundaries: set[float] = set()
    for binding in external_bindings.values():
        binding.validate_interval(start_s, end_s)
        boundaries.update(_external_binding_boundaries(binding, start_s, end_s))
    for command in commands.values():
        if not isinstance(command, ExternalTableCommand):
            boundaries.update(
                _waveform_boundaries(
                    command.waveform, start_s, end_s, run_start_s=run_start_s
                )
            )
    return boundaries


def resolved_step_power_commands(
    commands: dict[str, PowerPortCommand],
    ports: Mapping[str, PowerDefinition],
    external_bindings: dict[str, ExternalTableBinding],
    midpoint_s: float,
    *,
    run_start_s: float,
    step_start_s: float,
    step_id: str,
) -> dict[str, CompiledPowerCommand]:
    """Bind every active port command at one constant-command interval."""

    resolved = {
        port_id: CompiledPowerCommand(
            kind="external_table",
            external_table=(
                binding.bind_previous(midpoint_s)
                if binding.interpolation == "previous"
                else binding
            ),
        )
        for port_id, binding in external_bindings.items()
    }
    for port_id, port in ports.items():
        if port_id in external_bindings:
            continue
        command = commands.get(port_id)
        if isinstance(command, ExternalTableCommand):
            raise CaseValidationError(
                f"power port {port_id!r} has no external table binding"
            )
        enabled = command is None or _waveform_is_on(
            command.waveform, midpoint_s, step_start_s, run_start_s=run_start_s
        )
        resolved[port_id] = _resolve_power_command(
            port_id, port, command, enabled=enabled, step_id=step_id
        )
    return resolved


def _resolve_power_command(
    port_id: str,
    port: PowerDefinition,
    command: PowerPortCommand | None,
    *,
    enabled: bool,
    step_id: str,
) -> CompiledPowerCommand:
    if not enabled:
        if isinstance(command, DCSeriesCommand):
            return CompiledPowerCommand(kind="voltage", voltage_V=command.off_voltage_V)
        return CompiledPowerCommand(kind="off")
    kind, value = _command_setpoint(port_id, port, command)
    if value is None:
        raise CaseValidationError(
            f"recipe step {step_id!r}, power port {port_id!r} requires a "
            f"{kind} command or a model default"
        )
    if value == 0.0 and isinstance(
        port.model,
        (ExperimentalRFEnvelopeModel, ExperimentalCCPModel, ExperimentalICPModel),
    ):
        return CompiledPowerCommand(kind="off")
    return (
        CompiledPowerCommand(kind="power", power_W=value)
        if kind == "power"
        else CompiledPowerCommand(kind="voltage", voltage_V=value)
    )


def _command_setpoint(
    port_id: str, port: PowerDefinition, command: PowerPortCommand | None
) -> tuple[str, float | None]:
    model = port.model
    if isinstance(model, PrescribedPowerModel):
        return "power", _setpoint(command, "absorbed_power_W", model.absorbed_power_W)
    if isinstance(model, DCSeriesModel):
        return "voltage", _setpoint(command, "source_voltage_V", model.source_voltage_V)
    if isinstance(model, ExperimentalICPModel):
        return "power", _setpoint(command, "delivered_power_W", model.delivered_power_W)
    if isinstance(model, (ExperimentalRFEnvelopeModel, ExperimentalCCPModel)):
        if model.control == "voltage":
            return "voltage", _setpoint(command, "voltage_rms_V", model.voltage_rms_V)
        return "power", _setpoint(command, "absorbed_power_W", model.absorbed_power_W)
    raise TypeError(f"port {port_id!r} requires an external-table binding")


def _setpoint(
    command: PowerPortCommand | None, field: str, default: float | None
) -> float | None:
    value = getattr(command, field, None)
    return default if value is None else float(value)


def _external_binding_boundaries(
    binding: ExternalTableBinding, start_s: float, end_s: float
) -> tuple[float, ...]:
    if binding.interpolation != "previous":
        return ()
    return tuple(
        float(knot_s + binding.time_offset_s)
        for knot_s in binding.data.time_s
        if start_s < knot_s + binding.time_offset_s < end_s
    )


def _waveform_boundaries(
    waveform: Waveform, start_s: float, end_s: float, *, run_start_s: float
) -> list[float]:
    if isinstance(waveform, ContinuousWaveform):
        return []
    if not isinstance(waveform, SquarePulseWaveform):
        assert_never(waveform)
    origin = start_s if waveform.time_reference == "step_start" else run_start_s
    phase = origin + waveform.phase_s
    if waveform.duty_cycle == 0.0:
        return []
    if waveform.duty_cycle == 1.0:
        return [phase] if start_s < phase < end_s else []
    return _square_pulse_boundaries(waveform, start_s, end_s, phase)


def _square_pulse_boundaries(
    waveform: SquarePulseWaveform,
    start_s: float,
    end_s: float,
    phase: float,
) -> list[float]:
    period = 1.0 / waveform.repetition_Hz
    values: list[float] = []
    cycle = max(0, int((start_s - phase) // period) - 1)
    while True:
        rising = phase + cycle * period
        falling = rising + waveform.duty_cycle * period
        if rising >= end_s and falling >= end_s:
            break
        if start_s < rising < end_s:
            values.append(rising)
        if start_s < falling < end_s:
            values.append(falling)
        cycle += 1
        if len(values) > MAX_COMPILED_SEGMENTS:
            raise CaseValidationError(
                "square-pulse recipe creates too many discontinuity segments"
            )
    return values


def _waveform_is_on(
    waveform: Waveform, time_s: float, step_start_s: float, *, run_start_s: float
) -> bool:
    if isinstance(waveform, ContinuousWaveform):
        return True
    if not isinstance(waveform, SquarePulseWaveform):
        assert_never(waveform)
    origin = step_start_s if waveform.time_reference == "step_start" else run_start_s
    relative = time_s - origin - waveform.phase_s
    if relative < 0.0 or waveform.duty_cycle == 0.0:
        return False
    if waveform.duty_cycle == 1.0:
        return True
    period = 1.0 / waveform.repetition_Hz
    return (relative % period) < waveform.duty_cycle * period


def bind_step_power_tables(
    commands: dict[str, PowerPortCommand],
    ports: Mapping[str, PowerDefinition],
    external_tables: ExternalTableStore,
    *,
    run_start_s: float,
    step_start_s: float,
) -> dict[str, ExternalTableBinding]:
    bindings: dict[str, ExternalTableBinding] = {}
    for port_id, port in ports.items():
        if isinstance(port.model, ExternalTableModel):
            command = commands.get(port_id)
            if command is not None and not isinstance(command, ExternalTableCommand):
                raise CaseValidationError(
                    f"power port {port_id!r} requires an external table command"
                )
            bindings[port_id] = compile_external_binding(
                port.model,
                external_tables,
                command,
                run_start_s=run_start_s,
                step_start_s=step_start_s,
            )
    return bindings
