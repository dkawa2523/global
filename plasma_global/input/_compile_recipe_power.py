"""Resolve recipe power commands and their time discontinuities."""

from __future__ import annotations

from typing import assert_never

from plasma_global.errors import CaseValidationError
from plasma_global.experimental.profile import PrescribedElectronProfile
from plasma_global.input._compile_power import compile_external_binding
from plasma_global.input.schema import (
    CaseSpec,
    ContinuousWaveform,
    DCSeriesModel,
    ExperimentalCCPCommand,
    ExperimentalCCPModel,
    ExperimentalICPCommand,
    ExperimentalICPModel,
    ExperimentalRFEnvelopeCommand,
    ExperimentalRFEnvelopeModel,
    ExternalTableCommand,
    ExternalTableModel,
    PowerPortCommand,
    PowerPortConfig,
    PrescribedPowerModel,
    RecipeStepConfig,
    SquarePulseWaveform,
    Waveform,
)
from plasma_global.input.schema import (
    DCSeriesCommand as DCSeriesCommandConfig,
)
from plasma_global.input.schema import (
    PrescribedPowerCommand as PrescribedPowerCommandConfig,
)
from plasma_global.models.external_table import ExternalTableBinding, ExternalTableStore
from plasma_global.models.power import CompiledPowerCommand

MAX_COMPILED_SEGMENTS = 100_000


def compiled_step_boundaries(
    step: RecipeStepConfig,
    ports: dict[str, PowerPortConfig],
    external_tables: ExternalTableStore,
    prescribed_electron_profile: PrescribedElectronProfile | None,
    start_s: float,
    end_s: float,
) -> tuple[list[float], dict[str, ExternalTableBinding]]:
    """Return every time boundary at which a step's commands can change."""

    boundaries = {start_s, end_s}
    boundaries.update(_profile_boundaries(prescribed_electron_profile, start_s, end_s))
    external_bindings, external_boundaries = _external_bindings_for_step(
        step,
        ports,
        external_tables,
        start_s,
        end_s,
    )
    boundaries.update(external_boundaries)
    for command in step.commands.power_ports.values():
        waveform = _command_waveform(command)
        if waveform is not None:
            boundaries.update(_waveform_boundaries(waveform, start_s, end_s))
    return sorted(boundaries), external_bindings


def resolved_step_power_commands(
    step: RecipeStepConfig,
    ports: dict[str, PowerPortConfig],
    external_bindings: dict[str, ExternalTableBinding],
    external_tables: ExternalTableStore,
    midpoint_s: float,
    step_start_s: float,
) -> dict[str, CompiledPowerCommand]:
    """Bind every active port command at one constant-command interval."""

    commands = {
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
    for port_id, command in step.commands.power_ports.items():
        if port_id in external_bindings:
            continue
        waveform = _command_waveform(command)
        enabled = waveform is None or _waveform_is_on(
            waveform,
            midpoint_s,
            step_start_s,
        )
        resolved_command = _resolved_port_command(
            ports[port_id],
            command,
            enabled=enabled,
            external_tables=external_tables,
        )
        if resolved_command is not None:
            commands[port_id] = resolved_command
    return commands


def prescribed_electron_density_at(
    case: CaseSpec,
    prescribed_electron_profile: PrescribedElectronProfile | None,
    time_s: float,
) -> dict[str, float]:
    """Sample a piecewise-constant prescribed electron profile."""

    if (
        prescribed_electron_profile is None
        or prescribed_electron_profile.interpolation != "previous"
    ):
        return {}
    return {
        zone.zone_id: prescribed_electron_profile.density(time_s, zone.zone_id)
        for zone in case.reactor.zones
    }


def _profile_boundaries(
    profile: PrescribedElectronProfile | None, start_s: float, end_s: float
) -> tuple[float, ...]:
    if profile is None or profile.interpolation != "previous":
        return ()
    return tuple(float(time_s) for time_s in profile.time_s if start_s < time_s < end_s)


def _command_waveform(command: PowerPortCommand) -> Waveform | None:
    """Return the waveform only for command variants that define one."""

    if isinstance(command, ExternalTableCommand):
        return None
    if isinstance(command, PrescribedPowerCommandConfig):
        return command.waveform
    if isinstance(command, DCSeriesCommandConfig):
        return command.waveform
    if isinstance(command, ExperimentalRFEnvelopeCommand):
        return command.waveform
    if isinstance(command, ExperimentalCCPCommand):
        return command.waveform
    if isinstance(command, ExperimentalICPCommand):
        return command.waveform
    assert_never(command)


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
    waveform: Waveform, start_s: float, end_s: float
) -> list[float]:
    if isinstance(waveform, ContinuousWaveform):
        return []
    if not isinstance(waveform, SquarePulseWaveform):
        assert_never(waveform)
    phase = start_s + waveform.phase_s
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
    cycle = 0
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
        if cycle > MAX_COMPILED_SEGMENTS:
            raise CaseValidationError(
                "square-pulse recipe creates too many discontinuity segments"
            )
    return values


def _waveform_is_on(waveform: Waveform, time_s: float, step_start_s: float) -> bool:
    if isinstance(waveform, ContinuousWaveform):
        return True
    if not isinstance(waveform, SquarePulseWaveform):
        assert_never(waveform)
    relative = time_s - step_start_s - waveform.phase_s
    if relative < 0.0 or waveform.duty_cycle == 0.0:
        return False
    if waveform.duty_cycle == 1.0:
        return True
    period = 1.0 / waveform.repetition_Hz
    return (relative % period) < waveform.duty_cycle * period


def _resolved_prescribed_power_command(
    port: PowerPortConfig,
    command: PowerPortCommand,
    *,
    enabled: bool,
) -> CompiledPowerCommand | None:
    if not isinstance(command, PrescribedPowerCommandConfig):
        raise CaseValidationError(
            f"power port {port.port_id!r} command does not match prescribed_power"
        )
    if not enabled:
        return CompiledPowerCommand(kind="off")
    if command.absorbed_power_W is None:
        return None
    return CompiledPowerCommand(kind="power", power_W=command.absorbed_power_W)


def _resolved_dc_series_command(
    port: PowerPortConfig,
    model: DCSeriesModel,
    command: PowerPortCommand,
    *,
    enabled: bool,
) -> CompiledPowerCommand | None:
    if not isinstance(command, DCSeriesCommandConfig):
        raise CaseValidationError(
            f"power port {port.port_id!r} command does not match dc_series"
        )
    value = command.source_voltage_V
    if value is None:
        value = model.source_voltage_V
    if not enabled:
        value = command.off_voltage_V
    if value is None:
        return None
    return CompiledPowerCommand(kind="voltage", voltage_V=value)


def _resolved_external_table_command(
    port: PowerPortConfig,
    model: ExternalTableModel,
    command: PowerPortCommand,
    external_tables: ExternalTableStore,
) -> CompiledPowerCommand:
    if not isinstance(command, ExternalTableCommand):
        raise CaseValidationError(
            f"power port {port.port_id!r} command does not match external_table"
        )
    return CompiledPowerCommand(
        kind="external_table",
        external_table=compile_external_binding(model, external_tables, command),
    )


def _resolved_experimental_icp_command(
    port: PowerPortConfig,
    command: PowerPortCommand,
    *,
    enabled: bool,
) -> CompiledPowerCommand | None:
    if not isinstance(command, ExperimentalICPCommand):
        raise CaseValidationError(
            f"power port {port.port_id!r} command does not match experimental.icp"
        )
    if not enabled:
        return CompiledPowerCommand(kind="off")
    if command.delivered_power_W is None:
        return None
    return CompiledPowerCommand(kind="power", power_W=command.delivered_power_W)


def _resolved_experimental_rf_command(
    port: PowerPortConfig,
    model: ExperimentalRFEnvelopeModel | ExperimentalCCPModel,
    command: PowerPortCommand,
    *,
    enabled: bool,
) -> CompiledPowerCommand | None:
    if isinstance(model, ExperimentalRFEnvelopeModel):
        if not isinstance(command, ExperimentalRFEnvelopeCommand):
            raise CaseValidationError(
                f"power port {port.port_id!r} command does not match "
                "experimental.rf_envelope"
            )
        voltage_rms_V = command.voltage_rms_V
        absorbed_power_W = command.absorbed_power_W
    else:
        if not isinstance(command, ExperimentalCCPCommand):
            raise CaseValidationError(
                f"power port {port.port_id!r} command does not match experimental.ccp"
            )
        voltage_rms_V = command.voltage_rms_V
        absorbed_power_W = command.absorbed_power_W
    if not enabled:
        return CompiledPowerCommand(kind="off")
    if model.control == "voltage":
        if voltage_rms_V is None:
            return None
        return CompiledPowerCommand(kind="voltage", voltage_V=voltage_rms_V)
    if absorbed_power_W is None:
        return None
    return CompiledPowerCommand(kind="power", power_W=absorbed_power_W)


def _resolved_port_command(
    port: PowerPortConfig,
    command: PowerPortCommand,
    *,
    enabled: bool,
    external_tables: ExternalTableStore,
) -> CompiledPowerCommand | None:
    """Resolve one schema command against its port's compiled command kind."""

    model = port.model
    if isinstance(model, PrescribedPowerModel):
        return _resolved_prescribed_power_command(port, command, enabled=enabled)
    if isinstance(model, DCSeriesModel):
        return _resolved_dc_series_command(port, model, command, enabled=enabled)
    if isinstance(model, ExternalTableModel):
        return _resolved_external_table_command(
            port,
            model,
            command,
            external_tables,
        )
    if isinstance(model, ExperimentalICPModel):
        return _resolved_experimental_icp_command(port, command, enabled=enabled)
    if isinstance(model, (ExperimentalRFEnvelopeModel, ExperimentalCCPModel)):
        return _resolved_experimental_rf_command(
            port,
            model,
            command,
            enabled=enabled,
        )
    assert_never(model)


def _external_bindings_for_step(
    step: RecipeStepConfig,
    ports: dict[str, PowerPortConfig],
    external_tables: ExternalTableStore,
    start_s: float,
    end_s: float,
) -> tuple[dict[str, ExternalTableBinding], set[float]]:
    bindings: dict[str, ExternalTableBinding] = {}
    boundaries: set[float] = set()
    for port_id, port in ports.items():
        model = port.model
        if not isinstance(model, ExternalTableModel):
            continue
        command = step.commands.power_ports.get(port_id)
        if command is not None and not isinstance(command, ExternalTableCommand):
            raise TypeError(
                f"external-table port {port_id!r} has an invalid command type"
            )
        binding = compile_external_binding(model, external_tables, command)
        bindings[port_id] = binding
        boundaries.update(_external_binding_boundaries(binding, start_s, end_s))
    return bindings, boundaries
