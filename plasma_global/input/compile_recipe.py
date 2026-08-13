"""Compile recipe steps into smooth, command-bound solver segments."""

from __future__ import annotations

from itertools import pairwise
from typing import assert_never

import numpy as np

from plasma_global.chemistry.compile import CompiledChemistry
from plasma_global.chemistry.data import ChemistryData
from plasma_global.core.domain import RecipeSegment
from plasma_global.core.transport import SCCM_TO_PARTICLES_PER_S, SegmentTransport
from plasma_global.errors import CaseValidationError
from plasma_global.experimental.profile import PrescribedElectronProfile
from plasma_global.input.compile_reactor import compile_external_binding
from plasma_global.input.schema import (
    CaseSpec,
    ContinuousWaveform,
    DCSeriesModel,
    EvolvedGasEnergy,
    ExperimentalCCPCommand,
    ExperimentalCCPModel,
    ExperimentalICPCommand,
    ExperimentalICPModel,
    ExperimentalRFEnvelopeCommand,
    ExperimentalRFEnvelopeModel,
    ExternalTableCommand,
    ExternalTableModel,
    FixedGasEnergy,
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
from plasma_global.models.external_table import (
    ExternalTableBinding,
    ExternalTableStore,
)
from plasma_global.models.power import (
    CompiledPowerCommand,
)

_BOLTZMANN_J_K = 1.380649e-23
_MAX_COMPILED_SEGMENTS = 100_000


def _profile_boundaries(
    profile: PrescribedElectronProfile | None, start_s: float, end_s: float
) -> tuple[float, ...]:
    if profile is None or profile.interpolation != "previous":
        return ()
    return tuple(float(time_s) for time_s in profile.time_s if start_s < time_s < end_s)


def _wall_temperatures_by_zone(
    case: CaseSpec, overrides: dict[str, float]
) -> dict[str, float]:
    values: dict[str, float] = {}
    for zone in case.reactor.zones:
        surfaces = [
            surface
            for surface in case.reactor.surfaces
            if surface.zone_id == zone.zone_id
        ]
        total_area = sum(surface.area_m2 for surface in surfaces)
        values[zone.zone_id] = (
            sum(
                surface.area_m2
                * overrides.get(surface.surface_id, surface.temperature_K)
                for surface in surfaces
            )
            / total_area
            if total_area > 0.0
            else zone.gas_temperature_K
        )
    return values


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


def _compile_segment_transport(
    case: CaseSpec,
    chemistry_data: ChemistryData,
    chemistry: CompiledChemistry,
    step: RecipeStepConfig,
) -> SegmentTransport:
    zones = tuple(case.reactor.zones)
    zone_index = {zone.zone_id: index for index, zone in enumerate(zones)}
    species_index = {
        species_id: index for index, species_id in enumerate(chemistry.species_ids)
    }
    cv_by_species = {
        item.id: item.cv_over_kb
        for item in chemistry_data.species
        if item.id in species_index and item.cv_over_kb is not None
    }
    sources = np.zeros((len(zones), len(chemistry.species_ids)))
    heavy_energy = np.zeros(len(zones))
    gas_energy = case.models.gas_energy
    if isinstance(gas_energy, EvolvedGasEnergy):
        evolves_gas_energy = True
    elif isinstance(gas_energy, FixedGasEnergy):
        evolves_gas_energy = False
    else:
        assert_never(gas_energy)
    commands = step.commands.gas_inlets
    for inlet in case.reactor.gas_inlets:
        flow = (
            commands[inlet.inlet_id].flow_sccm
            if inlet.inlet_id in commands
            else inlet.flow_sccm
        )
        zone_row = zone_index[inlet.zone_id]
        volume = zones[zone_row].volume_m3
        for species_id, flow_sccm in flow.items():
            if species_id not in species_index:
                raise CaseValidationError(
                    f"gas inlet {inlet.inlet_id!r} references unknown/non-heavy "
                    f"species {species_id!r}"
                )
            rate_m3_s = SCCM_TO_PARTICLES_PER_S * flow_sccm / volume
            sources[zone_row, species_index[species_id]] += rate_m3_s
            if evolves_gas_energy:
                if species_id not in cv_by_species:
                    raise CaseValidationError(
                        f"evolved gas energy requires cv_over_kb for inlet species "
                        f"{species_id!r}"
                    )
                heavy_energy[zone_row] += (
                    rate_m3_s
                    * (cv_by_species[species_id] + 1.0)
                    * _BOLTZMANN_J_K
                    * inlet.temperature_K
                )
    return SegmentTransport(sources, heavy_energy)


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
        if cycle > _MAX_COMPILED_SEGMENTS:
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


def _compiled_step_boundaries(
    step: RecipeStepConfig,
    ports: dict[str, PowerPortConfig],
    external_tables: ExternalTableStore,
    prescribed_electron_profile: PrescribedElectronProfile | None,
    start_s: float,
    end_s: float,
) -> tuple[list[float], dict[str, ExternalTableBinding]]:
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


def _step_temperatures(
    case: CaseSpec, step: RecipeStepConfig
) -> tuple[dict[str, float], dict[str, float]]:
    surface_temperatures = {
        surface_id: command.temperature_K
        for surface_id, command in step.commands.surfaces.items()
        if command.temperature_K is not None
    }
    return surface_temperatures, _wall_temperatures_by_zone(
        case,
        surface_temperatures,
    )


def _resolved_step_power_commands(
    step: RecipeStepConfig,
    ports: dict[str, PowerPortConfig],
    external_bindings: dict[str, ExternalTableBinding],
    external_tables: ExternalTableStore,
    midpoint_s: float,
    step_start_s: float,
) -> dict[str, CompiledPowerCommand]:
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


def _prescribed_electron_density_at(
    case: CaseSpec,
    prescribed_electron_profile: PrescribedElectronProfile | None,
    time_s: float,
) -> dict[str, float]:
    if (
        prescribed_electron_profile is None
        or prescribed_electron_profile.interpolation != "previous"
    ):
        return {}
    return {
        zone.zone_id: prescribed_electron_profile.density(time_s, zone.zone_id)
        for zone in case.reactor.zones
    }


def compile_recipe(
    case: CaseSpec,
    chemistry_data: ChemistryData,
    chemistry: CompiledChemistry,
    external_tables: ExternalTableStore,
    prescribed_electron_profile: PrescribedElectronProfile | None = None,
) -> tuple[RecipeSegment, ...]:
    """Split every discontinuity and bind constant transport/power commands."""

    ports = {port.port_id: port for port in case.reactor.power_ports}
    segments: list[RecipeSegment] = []
    for step, (start, end) in zip(
        case.recipe.steps,
        case.recipe.step_bounds(),
        strict=True,
    ):
        ordered, external_bindings = _compiled_step_boundaries(
            step,
            ports,
            external_tables,
            prescribed_electron_profile,
            start,
            end,
        )
        if len(ordered) - 1 + len(segments) > _MAX_COMPILED_SEGMENTS:
            raise CaseValidationError("case creates too many compiled recipe segments")
        transport = _compile_segment_transport(case, chemistry_data, chemistry, step)
        surface_temperatures, wall_temperatures = _step_temperatures(case, step)
        split_count = len(ordered) - 1
        for index, (left, right) in enumerate(pairwise(ordered), start=1):
            midpoint = left + 0.5 * (right - left)
            commands = _resolved_step_power_commands(
                step,
                ports,
                external_bindings,
                external_tables,
                midpoint,
                start,
            )
            segment_id = (
                step.step_id if split_count == 1 else f"{step.step_id}[{index:04d}]"
            )
            electron_density = _prescribed_electron_density_at(
                case,
                prescribed_electron_profile,
                midpoint,
            )
            segments.append(
                RecipeSegment(
                    segment_id=segment_id,
                    start_s=left,
                    end_s=right,
                    surface_temperature_K_by_surface=surface_temperatures,
                    wall_temperature_K_by_zone=wall_temperatures,
                    transport=transport,
                    port_commands=commands,
                    prescribed_electron_density_m3_by_zone=electron_density,
                )
            )
    return tuple(segments)


__all__ = ["compile_recipe"]
