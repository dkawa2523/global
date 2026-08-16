"""Identity and field-capability rules shared by public power objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from plasma_global.errors import CaseValidationError

if TYPE_CHECKING:
    from plasma_global.models.power import CompiledPowerCommand, PowerPort


def normalized_identities(
    ports: tuple[PowerPort, ...], zone_ids: tuple[str, ...]
) -> tuple[tuple[PowerPort, ...], tuple[str, ...], tuple[str, ...]]:
    normalized_ports = tuple(ports)
    zones = tuple(str(value) for value in zone_ids)
    if not zones or len(set(zones)) != len(zones):
        raise CaseValidationError(
            "PowerCoordinator zone_ids must be unique and non-empty"
        )
    port_ids = tuple(str(port.port_id) for port in normalized_ports)
    if len(set(port_ids)) != len(port_ids) or any(not value for value in port_ids):
        raise CaseValidationError(
            "PowerCoordinator port IDs must be unique and non-empty"
        )
    unknown = sorted({str(port.zone_id) for port in normalized_ports} - set(zones))
    if unknown:
        raise CaseValidationError(f"Power ports reference unknown zones: {unknown}")
    return normalized_ports, zones, port_ids


def group_ports(
    ports: tuple[PowerPort, ...], zones: tuple[str, ...]
) -> dict[str, tuple[PowerPort, ...]]:
    grouped: dict[str, list[PowerPort]] = {zone_id: [] for zone_id in zones}
    for port in ports:
        grouped[str(port.zone_id)].append(port)
    return {zone_id: tuple(zone_ports) for zone_id, zone_ports in grouped.items()}


def command_field_capability(
    port: PowerPort, command: CompiledPowerCommand | None
) -> bool:
    """Resolve one port's effective E/N capability for the current command."""

    command_capability = None if command is None else command.produces_reduced_field
    return (
        port.produces_reduced_field
        if command_capability is None
        else command_capability
    )


def validate_field_sources(
    zone_id: str,
    ports: tuple[PowerPort, ...],
    commands: Mapping[str, CompiledPowerCommand],
) -> None:
    field_ports: list[str] = []
    for port in ports:
        command = commands.get(str(port.port_id))
        if command_field_capability(port, command):
            field_ports.append(str(port.port_id))
    if len(field_ports) > 1:
        raise CaseValidationError(
            f"Zone {zone_id!r} has multiple E/N-producing power ports: {field_ports}"
        )


def commands_are_time_dependent(
    ports: tuple[PowerPort, ...], commands: Mapping[str, CompiledPowerCommand]
) -> bool:
    for port in ports:
        if not port.time_dependent:
            continue
        command = commands.get(str(port.port_id))
        if command is not None and not command.time_dependent:
            continue
        return True
    return False
