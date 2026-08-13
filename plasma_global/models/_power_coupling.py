"""Pure validation and iteration helpers for zone-level power coupling."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Generic, Protocol, SupportsFloat, TypeVar

from plasma_global.errors import CaseValidationError, ModelDomainError
from plasma_global.models.kinetics import (
    ElectronKineticsResult,
    TabulatedElectronKinetics,
)


class _IdentifiedPort(Protocol):
    @property
    def port_id(self) -> str: ...

    @property
    def zone_id(self) -> str: ...

    @property
    def time_dependent(self) -> bool: ...

    @property
    def produces_reduced_field(self) -> bool: ...


class _CommandLike(Protocol):
    @property
    def time_dependent(self) -> bool: ...

    @property
    def produces_reduced_field(self) -> bool | None: ...


class _PowerStateLike(Protocol):
    @property
    def electron_mobility_m2_V_s(self) -> float | None: ...


class _PowerPortResultLike(Protocol):
    @property
    def port_id(self) -> str: ...

    @property
    def zone_id(self) -> str: ...

    @property
    def electron_power_W(self) -> float: ...

    @property
    def gas_power_W(self) -> float: ...

    @property
    def reduced_field_Td(self) -> float | None: ...


PortT = TypeVar("PortT", bound=_IdentifiedPort)
StateT = TypeVar("StateT", bound=_PowerStateLike)
ResultT = TypeVar("ResultT", bound=_PowerPortResultLike)


def normalize_id(value: object) -> str:
    return str(value)


def python_float(value: SupportsFloat) -> float:
    return float(value)


def normalized_identities(
    ports: tuple[PortT, ...], zone_ids: tuple[str, ...]
) -> tuple[tuple[PortT, ...], tuple[str, ...], tuple[str, ...]]:
    normalized_ports = tuple(ports)
    zones = tuple(normalize_id(value) for value in zone_ids)
    if not zones or len(set(zones)) != len(zones):
        raise CaseValidationError(
            "PowerCoordinator zone_ids must be unique and non-empty"
        )
    port_ids = tuple(normalize_id(port.port_id) for port in normalized_ports)
    if len(set(port_ids)) != len(port_ids) or any(not value for value in port_ids):
        raise CaseValidationError(
            "PowerCoordinator port IDs must be unique and non-empty"
        )
    unknown = sorted(
        {normalize_id(port.zone_id) for port in normalized_ports} - set(zones)
    )
    if unknown:
        raise CaseValidationError(f"Power ports reference unknown zones: {unknown}")
    return normalized_ports, zones, port_ids


def group_ports(
    ports: tuple[PortT, ...], zones: tuple[str, ...]
) -> dict[str, tuple[PortT, ...]]:
    grouped: dict[str, list[PortT]] = {zone_id: [] for zone_id in zones}
    for port in ports:
        grouped[normalize_id(port.zone_id)].append(port)
    return {zone_id: tuple(zone_ports) for zone_id, zone_ports in grouped.items()}


def command_field_capability(
    port: _IdentifiedPort, command: _CommandLike | None
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
    ports: tuple[_IdentifiedPort, ...],
    commands: Mapping[str, _CommandLike],
) -> None:
    field_ports: list[str] = []
    for port in ports:
        command = commands.get(normalize_id(port.port_id))
        if command_field_capability(port, command):
            field_ports.append(normalize_id(port.port_id))
    if len(field_ports) > 1:
        raise CaseValidationError(
            f"Zone {zone_id!r} has multiple E/N-producing power ports: {field_ports}"
        )


def commands_are_time_dependent(
    ports: tuple[_IdentifiedPort, ...], commands: Mapping[str, _CommandLike]
) -> bool:
    for port in ports:
        if not port.time_dependent:
            continue
        command = commands.get(normalize_id(port.port_id))
        if command is not None and not command.time_dependent:
            continue
        return True
    return False


@dataclass(frozen=True, slots=True)
class _ZonePowerInputs:
    electron_density_m3: float
    neutral_density_m3: float
    mean_energy_eV: float | None
    base_electron_power_W: float
    prescribed_field_Td: float | None
    kinetics: TabulatedElectronKinetics | None
    mean_energy_from_field: Callable[[float], float] | None


@dataclass(frozen=True, slots=True)
class _ZonePowerResult(Generic[StateT, ResultT]):
    electron_power_W: float
    gas_power_W: float
    reduced_field_Td: float | None
    power_state: StateT
    kinetics: ElectronKineticsResult | None
    port_results: tuple[ResultT, ...]
    iterations: int


def _effective_result_field(
    *,
    current_field_Td: float | None,
    next_field_Td: float | None,
    field_is_coupled: bool,
    kinetics: ElectronKineticsResult | None,
) -> float | None:
    """Return the field paired with the iteration's transport and rate data."""

    effective_field_Td = current_field_Td
    if not field_is_coupled and next_field_Td is not None:
        effective_field_Td = next_field_Td
    if effective_field_Td is None and kinetics is not None:
        effective_field_Td = python_float(kinetics.effective_field_Td)
    return effective_field_Td


@dataclass(frozen=True, slots=True)
class _ZonePowerIteration(Generic[StateT, ResultT]):
    power_state: StateT
    kinetics: ElectronKineticsResult | None
    port_results: tuple[ResultT, ...]
    electron_power_W: float
    next_field_Td: float | None
    field_is_coupled: bool

    def as_zone_result(
        self, current_field_Td: float | None, iterations: int
    ) -> _ZonePowerResult[StateT, ResultT]:
        # Coupled kinetics and ports were evaluated at ``current_field_Td``.
        # ``next_field_Td`` is only the next fixed-point proposal and must not
        # be paired with transport/rate data from the preceding iterate.
        effective_field_Td = _effective_result_field(
            current_field_Td=current_field_Td,
            next_field_Td=self.next_field_Td,
            field_is_coupled=self.field_is_coupled,
            kinetics=self.kinetics,
        )
        return _ZonePowerResult(
            electron_power_W=self.electron_power_W,
            gas_power_W=sum(item.gas_power_W for item in self.port_results),
            reduced_field_Td=effective_field_Td,
            power_state=self.power_state,
            kinetics=self.kinetics,
            port_results=self.port_results,
            iterations=iterations,
        )


def field_feedback(
    zone_id: str,
    results: tuple[ResultT, ...],
    prescribed_field_Td: float | None,
) -> tuple[float | None, bool]:
    candidates = [
        result.reduced_field_Td
        for result in results
        if result.reduced_field_Td is not None
    ]
    if len(candidates) > 1:
        field_ports = [
            result.port_id for result in results if result.reduced_field_Td is not None
        ]
        raise CaseValidationError(
            f"Zone {zone_id!r} has multiple E/N-producing power ports: {field_ports}"
        )
    if candidates:
        return candidates[0], True
    return prescribed_field_Td, False


def relative_change(new: float, old: float) -> float:
    return abs(new - old) / max(abs(new), abs(old), 1.0)


def advance_coupled_field(
    *,
    zone_id: str,
    current_field_Td: float | None,
    iteration: _ZonePowerIteration[StateT, ResultT],
    previous_mobility_m2_V_s: float | None,
    previous_electron_power_W: float | None,
) -> tuple[float, float]:
    next_field_Td = iteration.next_field_Td
    if current_field_Td is None or next_field_Td is None:
        raise ModelDomainError(f"Zone {zone_id!r} local-field coupling requires E/N")
    mobility_m2_V_s = iteration.power_state.electron_mobility_m2_V_s
    field_residual = relative_change(next_field_Td, current_field_Td)
    mobility_residual = (
        0.0
        if previous_mobility_m2_V_s is None or mobility_m2_V_s is None
        else relative_change(mobility_m2_V_s, previous_mobility_m2_V_s)
    )
    power_residual = (
        0.0
        if previous_electron_power_W is None
        else relative_change(iteration.electron_power_W, previous_electron_power_W)
    )
    return python_float(next_field_Td), max(
        field_residual, mobility_residual, power_residual
    )
