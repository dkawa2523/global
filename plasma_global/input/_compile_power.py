"""Compile schema power ports into runtime power components.

This module is the single adapter between input power configuration and the
runtime power model. Reactor topology and recipe segmentation consume the
compiled values without knowing how individual port variants are constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

import numpy as np

from plasma_global.chemistry.compile import CompiledChemistry
from plasma_global.errors import CaseValidationError
from plasma_global.input._schema_reactor import _power_coupling_target_error
from plasma_global.input.schema import (
    CaseSpec,
    DCSeriesModel,
    ExperimentalCCPModel,
    ExperimentalICPModel,
    ExperimentalRFEnvelopeModel,
    ExternalTableCommand,
    ExternalTableModel,
    PowerPortConfig,
    PrescribedPowerModel,
    SurfaceConfig,
    ZoneConfig,
)
from plasma_global.models.external_table import ExternalTableBinding, ExternalTableStore
from plasma_global.models.power import (
    DCSeriesPort,
    ExternalTablePowerPort,
    PowerCoordinator,
    PowerPort,
    PrescribedPowerPort,
)


@dataclass(frozen=True, slots=True)
class _CCPGeometry:
    zone_volume_m3: float
    powered_area_m2: float
    grounded_area_m2: float
    electrode_gap_m: float
    dominant_ion_mass_kg: float
    gas_temperature_K: float


def compile_external_binding(
    model: ExternalTableModel,
    store: ExternalTableStore,
    command: ExternalTableCommand | None = None,
) -> ExternalTableBinding:
    """Resolve a static external-table model and one optional step override."""

    resolved = model
    if command is not None:
        overrides = command.model_dump(
            exclude={"kind", "time_offset_s"},
            exclude_none=True,
        )
        resolved = model.model_copy(update=overrides)
    return ExternalTableBinding(
        data=store.get(resolved.file),
        interpolation=resolved.interpolation,
        bounds=resolved.bounds_policy,
        power_scale=resolved.power_scale,
        voltage_scale=resolved.voltage_scale,
        current_scale=resolved.current_scale,
        gap_m=resolved.gap_m,
        total_density_m3=resolved.total_density_m3,
        plasma_potential_V=resolved.plasma_potential_V,
        time_offset_s=0.0 if command is None else command.time_offset_s,
    )


def _ccp_geometry(
    zone: ZoneConfig,
    surfaces: list[SurfaceConfig],
    port: PowerPortConfig,
    chemistry: CompiledChemistry,
) -> _CCPGeometry:
    if not surfaces:
        raise CaseValidationError(
            f"experimental CCP port {port.port_id!r} needs reactor surface areas"
        )
    total_area = sum(item.area_m2 for item in surfaces)
    powered_area = total_area / 2.0
    if port.coupling_target:
        target = next(
            item for item in surfaces if item.surface_id == port.coupling_target
        )
        powered_area = target.area_m2
    positive_masses = chemistry.masses_kg[chemistry.charges > 0.0]
    if positive_masses.size == 0:
        raise CaseValidationError(
            f"experimental CCP port {port.port_id!r} requires a positive ion species"
        )
    return _CCPGeometry(
        zone_volume_m3=zone.volume_m3,
        powered_area_m2=powered_area,
        grounded_area_m2=max(total_area - powered_area, powered_area),
        electrode_gap_m=zone.volume_m3 / total_area,
        dominant_ion_mass_kg=float(np.min(positive_masses)),
        gas_temperature_K=zone.gas_temperature_K,
    )


def _compile_rf_envelope_port(
    port: PowerPortConfig,
    model: ExperimentalRFEnvelopeModel,
) -> PowerPort:
    from plasma_global.experimental.power import RFEnvelopePort

    return RFEnvelopePort(
        port_id=port.port_id,
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
        default_power_W=(
            model.default_absorbed_power_W
            if model.control == "absorbed_power"
            else None
        ),
        default_voltage_V=(
            model.default_voltage_rms_V if model.control == "voltage" else None
        ),
    )


def _compile_ccp_port(
    case: CaseSpec,
    chemistry: CompiledChemistry,
    port: PowerPortConfig,
    model: ExperimentalCCPModel,
) -> PowerPort:
    from plasma_global.experimental.power import CCPPowerPort

    zone = next(item for item in case.reactor.zones if item.zone_id == port.zone_id)
    surfaces = [item for item in case.reactor.surfaces if item.zone_id == port.zone_id]
    geometry = _ccp_geometry(zone, surfaces, port, chemistry)
    return CCPPowerPort(
        port_id=port.port_id,
        zone_id=port.zone_id,
        frequency_Hz=model.frequency_Hz,
        default_power_W=(
            model.default_absorbed_power_W
            if model.control == "absorbed_power"
            else None
        ),
        default_voltage_V=(
            model.default_voltage_rms_V if model.control == "voltage" else None
        ),
        zone_volume_m3=geometry.zone_volume_m3,
        powered_area_m2=geometry.powered_area_m2,
        grounded_area_m2=geometry.grounded_area_m2,
        electrode_gap_m=geometry.electrode_gap_m,
        dominant_ion_mass_kg=geometry.dominant_ion_mass_kg,
        gas_temperature_K=geometry.gas_temperature_K,
    )


def _compile_power_port(
    case: CaseSpec,
    chemistry: CompiledChemistry,
    external_tables: ExternalTableStore,
    port: PowerPortConfig,
) -> PowerPort:
    model = port.model
    if isinstance(model, PrescribedPowerModel):
        return PrescribedPowerPort(
            port.port_id,
            port.zone_id,
            electron_fraction=model.electron_fraction,
            gas_fraction=model.gas_fraction,
            default_power_W=model.default_absorbed_power_W,
        )
    if isinstance(model, DCSeriesModel):
        return DCSeriesPort(
            port.port_id,
            port.zone_id,
            ballast_resistance_ohm=model.ballast_resistance_ohm,
            gap_m=model.gap_m,
            electrode_area_m2=model.electrode_area_m2,
            absorption_fraction=model.power_absorption_fraction,
            configured_mobility_m2_V_s=model.electron_mobility_m2_V_s,
            default_voltage_V=model.source_voltage_V,
        )
    if isinstance(model, ExternalTableModel):
        default = compile_external_binding(model, external_tables)
        return ExternalTablePowerPort(port.port_id, port.zone_id, default)
    if isinstance(model, ExperimentalRFEnvelopeModel):
        return _compile_rf_envelope_port(port, model)
    if isinstance(model, ExperimentalCCPModel):
        return _compile_ccp_port(case, chemistry, port, model)
    if isinstance(model, ExperimentalICPModel):
        from plasma_global.experimental.power import ICPPowerPort

        zone = next(item for item in case.reactor.zones if item.zone_id == port.zone_id)
        return ICPPowerPort(
            port_id=port.port_id,
            zone_id=port.zone_id,
            frequency_Hz=model.frequency_Hz,
            gas_temperature_K=zone.gas_temperature_K,
            default_power_W=model.default_delivered_power_W,
        )
    assert_never(model)


def compile_power_coordinator(
    case: CaseSpec,
    chemistry: CompiledChemistry,
    external_tables: ExternalTableStore,
) -> PowerCoordinator | None:
    """Compile all configured power ports and preserve reactor declaration order."""

    surface_zone_by_id = {
        surface.surface_id: surface.zone_id for surface in case.reactor.surfaces
    }
    for port in case.reactor.power_ports:
        if error := _power_coupling_target_error(port, surface_zone_by_id):
            raise CaseValidationError(error)
    ports = tuple(
        _compile_power_port(case, chemistry, external_tables, port)
        for port in case.reactor.power_ports
    )
    if not ports:
        return None
    return PowerCoordinator(
        ports=ports,
        zone_ids=tuple(zone.zone_id for zone in case.reactor.zones),
    )


__all__ = ["compile_external_binding", "compile_power_coordinator"]
