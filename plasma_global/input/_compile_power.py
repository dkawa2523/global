"""Translate public power configuration into the existing numerical ports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, assert_never

from plasma_global.chemistry.compile import CompiledChemistry
from plasma_global.errors import CaseValidationError
from plasma_global.input._schema_power import (
    DCSeriesModel,
    ExperimentalCCPModel,
    ExperimentalICPModel,
    ExperimentalRFEnvelopeModel,
    ExternalTableCommand,
    ExternalTableModel,
    PrescribedPowerModel,
)
from plasma_global.models.external_table import ExternalTableBinding, ExternalTableStore
from plasma_global.models.power import (
    CompiledPowerCommand,
    DCSeriesPort,
    ExternalTablePowerPort,
    PowerCoordinator,
    PowerPort,
    PrescribedPowerPort,
)

if TYPE_CHECKING:
    from plasma_global.input.schema import CaseSpec, PowerDefinition


def compile_external_binding(
    model: ExternalTableModel,
    store: ExternalTableStore,
    command: ExternalTableCommand | None = None,
    *,
    run_start_s: float = 0.0,
    step_start_s: float = 0.0,
) -> ExternalTableBinding:
    """Bind the command clock and table override at one recipe step."""
    resolved = model
    if command is not None:
        overrides = command.model_dump(
            exclude={"kind", "time_offset_s"}, exclude_none=True
        )
        resolved = model.model_copy(update=overrides)
    origin = run_start_s if resolved.time_reference == "run_start" else step_start_s
    offset = 0.0 if command is None else command.time_offset_s
    return ExternalTableBinding(
        data=store.get(resolved.file, columns=resolved.columns),
        interpolation=resolved.interpolation,
        bounds=resolved.bounds,
        power_scale=resolved.power_scale,
        voltage_scale=resolved.voltage_scale,
        current_scale=resolved.current_scale,
        gap_m=resolved.gap_m,
        total_density_m3=resolved.total_density_m3,
        plasma_potential_V=resolved.plasma_potential_V,
        time_offset_s=origin + offset,
    )


def _compile_rf_envelope_port(
    port_id: str, port: PowerDefinition, model: ExperimentalRFEnvelopeModel
) -> PowerPort:
    from plasma_global.experimental.power import RFEnvelopePort

    return RFEnvelopePort(
        port_id=port_id,
        zone_id=port.zone_id,
        frequency_Hz=model.frequency_Hz,
        role=model.role,
        coupling_efficiency=model.coupling_efficiency,
        effective_impedance_ohm=model.effective_impedance_ohm,
        base_reduced_field_Td=model.base_reduced_field_Td,
        reduced_field_per_sqrt_W_Td=model.reduced_field_per_sqrt_W_Td,
        self_bias_fraction=model.self_bias_fraction,
        plasma_potential_offset_V=model.plasma_potential_offset_V,
        plasma_potential_per_sqrt_W=model.plasma_potential_per_sqrt_W,
        absorbed_power_W=model.absorbed_power_W,
        voltage_rms_V=model.voltage_rms_V,
    )


def _compile_ccp_port(
    case: CaseSpec,
    chemistry: CompiledChemistry,
    port_id: str,
    port: PowerDefinition,
    model: ExperimentalCCPModel,
) -> PowerPort:
    from plasma_global.experimental.power import CCPPowerPort

    zone = case.reactor.zones[port.zone_id]
    index = {name: i for i, name in enumerate(chemistry.species_ids)}.get(
        model.dominant_ion
    )
    if index is None or chemistry.charges[index] <= 0.0:
        raise CaseValidationError(
            f"reactor.power_ports.{port_id}.model.dominant_ion "
            "must identify a positive gas ion"
        )
    return CCPPowerPort(
        port_id=port_id,
        zone_id=port.zone_id,
        frequency_Hz=model.frequency_Hz,
        absorbed_power_W=model.absorbed_power_W,
        voltage_rms_V=model.voltage_rms_V,
        zone_volume_m3=zone.volume_m3,
        powered_area_m2=model.powered_area_m2,
        grounded_area_m2=model.grounded_area_m2,
        electrode_gap_m=model.electrode_gap_m,
        dominant_ion_mass_kg=float(chemistry.masses_kg[index]),
        gas_temperature_K=zone.initial.gas_temperature_K,
    )


def _compile_power_port(
    case: CaseSpec,
    chemistry: CompiledChemistry,
    external_tables: ExternalTableStore,
    port_id: str,
    port: PowerDefinition,
) -> PowerPort:
    model = port.model
    if isinstance(model, PrescribedPowerModel):
        return PrescribedPowerPort(
            port_id,
            port.zone_id,
            electron_fraction=model.electron_fraction,
            gas_fraction=model.gas_fraction,
            absorbed_power_W=model.absorbed_power_W,
        )
    if isinstance(model, DCSeriesModel):
        return DCSeriesPort(
            port_id,
            port.zone_id,
            ballast_resistance_ohm=model.ballast_resistance_ohm,
            gap_m=model.gap_m,
            electrode_area_m2=model.electrode_area_m2,
            absorption_fraction=model.power_absorption_fraction,
            electron_mobility_m2_V_s=model.electron_mobility_m2_V_s,
            source_voltage_V=model.source_voltage_V,
        )
    if isinstance(model, ExternalTableModel):
        binding = compile_external_binding(
            model,
            external_tables,
            run_start_s=case.run.start_time_s,
            step_start_s=case.run.start_time_s,
        )
        return ExternalTablePowerPort(port_id, port.zone_id, binding)
    if isinstance(model, ExperimentalRFEnvelopeModel):
        return _compile_rf_envelope_port(port_id, port, model)
    if isinstance(model, ExperimentalCCPModel):
        return _compile_ccp_port(case, chemistry, port_id, port, model)
    if isinstance(model, ExperimentalICPModel):
        from plasma_global.experimental.power import ICPPowerPort

        zone = case.reactor.zones[port.zone_id]
        return ICPPowerPort(
            port_id=port_id,
            zone_id=port.zone_id,
            frequency_Hz=model.frequency_Hz,
            gas_temperature_K=zone.initial.gas_temperature_K,
            delivered_power_W=model.delivered_power_W,
            downstream_zone_id=model.downstream_zone_id,
            downstream_fraction=model.downstream_fraction,
        )
    assert_never(model)


def compile_power_coordinator(
    case: CaseSpec,
    chemistry: CompiledChemistry,
    external_tables: ExternalTableStore,
    initial_commands: Mapping[str, CompiledPowerCommand] | None = None,
) -> PowerCoordinator | None:
    """Preserve configured port order and validate declared physical topology."""
    ports = tuple(
        _compile_power_port(case, chemistry, external_tables, port_id, port)
        for port_id, port in case.reactor.power_ports.items()
    )
    if not ports:
        return None
    return PowerCoordinator(
        ports=ports,
        zone_ids=tuple(case.reactor.zones),
        initial_commands=initial_commands,
    )


__all__ = ["compile_external_binding", "compile_power_coordinator"]
