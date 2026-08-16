"""Stable public power API with calculations delegated to private modules.

Power ports return only quantities they actually calculate. In particular,
prescribed absorbed power does not manufacture a sheath voltage or E/N.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, cast

from plasma_global.errors import (
    CaseValidationError,
)
from plasma_global.models._power_coupling import (
    command_field_capability,
    commands_are_time_dependent,
    group_ports,
    normalized_identities,
    validate_field_sources,
)
from plasma_global.models._power_ports import E_CHARGE as E_CHARGE
from plasma_global.models._power_ports import TD_TO_V_M2 as TD_TO_V_M2
from plasma_global.models._power_ports import (
    _PortEvaluation,
    dc_series_evaluation,
    external_table_evaluation,
    prescribed_power_evaluation,
    solve_dc_series_local_field,
    validate_dc_series_port,
    validate_prescribed_power_port,
)
from plasma_global.models._power_types import (
    _empty_observables,
    finite_observables,
    frozen_mapping,
    validate_compiled_power_command,
    validate_power_port_result,
    validate_power_state,
)
from plasma_global.models.external_table import (
    ExternalTableBinding,
    ExternalTableSample,
)
from plasma_global.models.kinetics import (
    ElectronKineticsResult,
    TabulatedElectronKinetics,
)


@dataclass(frozen=True, slots=True)
class PowerState:
    """Validated plasma state supplied to one power port."""

    electron_density_m3: float
    neutral_density_m3: float
    electron_temperature_eV: float
    electron_mobility_m2_V_s: float | None = None

    def __post_init__(self) -> None:
        validate_power_state(
            self.electron_density_m3,
            self.neutral_density_m3,
            self.electron_temperature_eV,
            self.electron_mobility_m2_V_s,
        )


@dataclass(frozen=True, slots=True)
class CompiledPowerCommand:
    """One fully bound numeric command consumed by the power hot path."""

    kind: Literal["off", "power", "voltage", "external_table"]
    power_W: float | None = None
    voltage_V: float | None = None
    external_table: ExternalTableBinding | ExternalTableSample | None = None
    reduced_field_capability: bool | None = None

    def __post_init__(self) -> None:
        validate_compiled_power_command(
            self.kind,
            self.power_W,
            self.voltage_V,
            self.external_table,
            self.reduced_field_capability,
        )

    @property
    def time_dependent(self) -> bool:
        return self.kind == "external_table" and isinstance(
            self.external_table, ExternalTableBinding
        )

    @property
    def produces_reduced_field(self) -> bool | None:
        if self.kind == "off":
            return self.reduced_field_capability
        if self.kind != "external_table":
            return None
        table = cast(ExternalTableBinding | ExternalTableSample, self.external_table)
        return table.produces_reduced_field


@dataclass(frozen=True, slots=True)
class PowerPortResult:
    """Power and optional field contribution returned by one port."""

    port_id: str
    zone_id: str
    electron_power_W: float
    gas_power_W: float = 0.0
    reduced_field_Td: float | None = None
    observables: Mapping[str, float] = field(default_factory=_empty_observables)

    def __post_init__(self) -> None:
        validate_power_port_result(
            self.electron_power_W,
            self.gas_power_W,
            self.reduced_field_Td,
        )
        object.__setattr__(self, "observables", finite_observables(self.observables))


class PowerPort(Protocol):
    """Structural interface implemented by compiled power ports."""

    @property
    def port_id(self) -> str: ...

    @property
    def zone_id(self) -> str: ...

    @property
    def time_dependent(self) -> bool: ...

    @property
    def produces_reduced_field(self) -> bool: ...

    def evaluate(
        self, time_s: float, state: PowerState, command: CompiledPowerCommand | None
    ) -> PowerPortResult: ...


@dataclass(frozen=True, slots=True)
class PowerCouplingResult:
    """Zone aggregates and the converged table/port evaluations."""

    electron_power_W_by_zone: Mapping[str, float]
    gas_power_W_by_zone: Mapping[str, float]
    reduced_field_Td_by_zone: Mapping[str, float]
    power_state_by_zone: Mapping[str, PowerState]
    kinetics_by_zone: Mapping[str, ElectronKineticsResult]
    port_results: Mapping[str, PowerPortResult]
    iterations_by_zone: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "electron_power_W_by_zone",
            frozen_mapping(self.electron_power_W_by_zone),
        )
        object.__setattr__(
            self,
            "gas_power_W_by_zone",
            frozen_mapping(self.gas_power_W_by_zone),
        )
        object.__setattr__(
            self,
            "reduced_field_Td_by_zone",
            frozen_mapping(self.reduced_field_Td_by_zone),
        )
        object.__setattr__(
            self,
            "power_state_by_zone",
            frozen_mapping(self.power_state_by_zone),
        )
        object.__setattr__(
            self,
            "kinetics_by_zone",
            frozen_mapping(self.kinetics_by_zone),
        )
        object.__setattr__(
            self,
            "port_results",
            frozen_mapping(self.port_results),
        )
        object.__setattr__(
            self,
            "iterations_by_zone",
            frozen_mapping(self.iterations_by_zone),
        )


def _power_port_result(
    port_id: str, zone_id: str, evaluated: _PortEvaluation
) -> PowerPortResult:
    return PowerPortResult(
        port_id=port_id,
        zone_id=zone_id,
        electron_power_W=evaluated.electron_power_W,
        gas_power_W=evaluated.gas_power_W,
        reduced_field_Td=evaluated.reduced_field_Td,
        observables=evaluated.observables,
    )


@dataclass(frozen=True, slots=True)
class _ZonePowerDistributor:
    port_id: str
    source_zone_id: str
    static_target_zone_ids: tuple[str, ...] | None
    distribute: Callable[[PowerPortResult], Mapping[str, float]]


@dataclass(frozen=True, slots=True)
class PrescribedPowerPort:
    """Partition a prescribed power command between electrons and gas."""

    port_id: str
    zone_id: str
    electron_fraction: float = 1.0
    gas_fraction: float = 0.0
    default_power_W: float | None = None

    @property
    def time_dependent(self) -> bool:
        return False

    @property
    def produces_reduced_field(self) -> bool:
        return False

    def __post_init__(self) -> None:
        validate_prescribed_power_port(
            self.electron_fraction,
            self.gas_fraction,
            self.default_power_W,
        )

    def evaluate(
        self, time_s: float, state: PowerState, command: CompiledPowerCommand | None
    ) -> PowerPortResult:
        del time_s, state
        evaluated = prescribed_power_evaluation(
            port_id=self.port_id,
            electron_fraction=self.electron_fraction,
            gas_fraction=self.gas_fraction,
            default_power_W=self.default_power_W,
            command_kind=None if command is None else command.kind,
            command_power_W=None if command is None else command.power_W,
        )
        return _power_port_result(self.port_id, self.zone_id, evaluated)


@dataclass(frozen=True, slots=True)
class DCSeriesPort:
    """Series-circuit DC power port with optional local-field coupling."""

    port_id: str
    zone_id: str
    ballast_resistance_ohm: float
    gap_m: float
    electrode_area_m2: float
    absorption_fraction: float = 1.0
    configured_mobility_m2_V_s: float | None = None
    default_voltage_V: float | None = None

    @property
    def time_dependent(self) -> bool:
        return False

    @property
    def produces_reduced_field(self) -> bool:
        return True

    def __post_init__(self) -> None:
        validate_dc_series_port(
            ballast_resistance_ohm=self.ballast_resistance_ohm,
            gap_m=self.gap_m,
            electrode_area_m2=self.electrode_area_m2,
            absorption_fraction=self.absorption_fraction,
            configured_mobility_m2_V_s=self.configured_mobility_m2_V_s,
            default_voltage_V=self.default_voltage_V,
        )

    def evaluate(
        self, time_s: float, state: PowerState, command: CompiledPowerCommand | None
    ) -> PowerPortResult:
        del time_s
        evaluated = dc_series_evaluation(
            port_id=self.port_id,
            ballast_resistance_ohm=self.ballast_resistance_ohm,
            gap_m=self.gap_m,
            electrode_area_m2=self.electrode_area_m2,
            absorption_fraction=self.absorption_fraction,
            configured_mobility_m2_V_s=self.configured_mobility_m2_V_s,
            default_voltage_V=self.default_voltage_V,
            electron_density_m3=state.electron_density_m3,
            neutral_density_m3=state.neutral_density_m3,
            electron_mobility_m2_V_s=state.electron_mobility_m2_V_s,
            command_kind=None if command is None else command.kind,
            command_voltage_V=None if command is None else command.voltage_V,
        )
        return _power_port_result(self.port_id, self.zone_id, evaluated)

    def solve_local_field(
        self,
        *,
        electron_density_m3: float,
        neutral_density_m3: float,
        kinetics: TabulatedElectronKinetics,
        command: CompiledPowerCommand | None,
    ) -> float:
        """Solve the series-circuit E/N equation on a piecewise-linear table.

        Multiple circuit roots can exist when tabulated mobility is not
        monotone. We retain the stable root reached from the table's central
        node, which is the deterministic branch selected by the former fixed-
        point iteration, without requiring dozens of RHS-time iterations.
        """

        return solve_dc_series_local_field(
            port_id=self.port_id,
            ballast_resistance_ohm=self.ballast_resistance_ohm,
            gap_m=self.gap_m,
            electrode_area_m2=self.electrode_area_m2,
            default_voltage_V=self.default_voltage_V,
            electron_density_m3=electron_density_m3,
            neutral_density_m3=neutral_density_m3,
            kinetics=kinetics,
            command_kind=None if command is None else command.kind,
            command_voltage_V=None if command is None else command.voltage_V,
        )


@dataclass(frozen=True, slots=True)
class ExternalTablePowerPort:
    """Power port backed by one preloaded external table command."""

    port_id: str
    zone_id: str
    default: ExternalTableBinding
    time_dependent: bool = True

    @property
    def produces_reduced_field(self) -> bool:
        return self.default.produces_reduced_field

    def _table_for_command(
        self, command: CompiledPowerCommand | None
    ) -> ExternalTableBinding | ExternalTableSample:
        if command is None:
            return self.default
        if command.kind == "external_table" and command.external_table is not None:
            return command.external_table
        raise CaseValidationError(
            f"external-table port {self.port_id!r} received an invalid command"
        )

    def evaluate(
        self,
        time_s: float,
        state: PowerState,
        command: CompiledPowerCommand | None,
    ) -> PowerPortResult:
        evaluated = external_table_evaluation(
            table=self._table_for_command(command),
            time_s=time_s,
            neutral_density_m3=state.neutral_density_m3,
        )
        return _power_port_result(self.port_id, self.zone_id, evaluated)


@dataclass(frozen=True, slots=True)
class PowerCoordinator:
    """Evaluate and aggregate typed ports, including E/N-mobility coupling."""

    ports: tuple[PowerPort, ...]
    zone_ids: tuple[str, ...]
    coupling_rtol: float = 1.0e-6
    max_iterations: int = 12
    _ports_by_zone: Mapping[str, tuple[PowerPort, ...]] = field(init=False, repr=False)
    _port_ids: frozenset[str] = field(init=False, repr=False)
    _zone_power_distributors: tuple[_ZonePowerDistributor, ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        ports, zones, port_ids = normalized_identities(self.ports, self.zone_ids)
        if not math.isfinite(self.coupling_rtol) or self.coupling_rtol <= 0.0:
            raise CaseValidationError("coupling_rtol must be positive")
        if self.max_iterations <= 0:
            raise CaseValidationError("max_iterations must be positive")
        grouped = group_ports(ports, zones)
        for zone_id, zone_ports in grouped.items():
            validate_field_sources(zone_id, zone_ports, {})
        distributors: list[_ZonePowerDistributor] = []
        for port in ports:
            distribute = getattr(port, "power_by_zone", None)
            if callable(distribute):
                static_targets = getattr(port, "_static_power_target_zone_ids", None)
                distributors.append(
                    _ZonePowerDistributor(
                        port_id=str(port.port_id),
                        source_zone_id=str(port.zone_id),
                        static_target_zone_ids=(
                            None
                            if static_targets is None
                            else tuple(str(value) for value in static_targets)
                        ),
                        distribute=cast(
                            Callable[[PowerPortResult], Mapping[str, float]],
                            distribute,
                        ),
                    )
                )
        object.__setattr__(self, "ports", ports)
        object.__setattr__(self, "zone_ids", zones)
        object.__setattr__(self, "_ports_by_zone", frozen_mapping(grouped))
        object.__setattr__(self, "_port_ids", frozenset(port_ids))
        object.__setattr__(self, "_zone_power_distributors", tuple(distributors))

    def validate_commands(self, commands: Mapping[str, CompiledPowerCommand]) -> None:
        """Validate one compiled segment command set outside the ODE hot path."""

        unknown = set(commands) - self._port_ids
        if unknown:
            raise CaseValidationError(
                f"Recipe commands reference unknown ports: {sorted(unknown)}"
            )
        for zone_id, zone_ports in self._ports_by_zone.items():
            validate_field_sources(zone_id, zone_ports, commands)

    def commands_are_time_dependent(
        self, commands: Mapping[str, CompiledPowerCommand]
    ) -> bool:
        """Return explicit wall-time dependence for one compiled segment."""

        return commands_are_time_dependent(self.ports, commands)

    def has_reduced_field_source(
        self, zone_id: str, commands: Mapping[str, CompiledPowerCommand]
    ) -> bool:
        """Return whether one segment guarantees E/N for the requested zone."""

        return any(
            command_field_capability(port, commands.get(str(port.port_id)))
            for port in self._ports_by_zone[str(zone_id)]
        )

    def evaluate(
        self,
        *,
        time_s: float,
        commands: Mapping[str, CompiledPowerCommand],
        electron_density_m3_by_zone: Mapping[str, float],
        neutral_density_m3_by_zone: Mapping[str, float],
        mean_energy_eV_by_zone: Mapping[str, float | None],
        prescribed_electron_power_W_by_zone: Mapping[str, float] | None = None,
        prescribed_reduced_field_Td_by_zone: Mapping[str, float] | None = None,
        kinetics_by_zone: Mapping[str, TabulatedElectronKinetics] | None = None,
        mean_energy_from_field_by_zone: Mapping[str, Callable[[float], float]]
        | None = None,
    ) -> PowerCouplingResult:
        # Imported lazily so the private engine can construct these public
        # value types without a module-initialization cycle.
        from plasma_global.models._power_coordinator import evaluate_power_coupling

        return evaluate_power_coupling(
            self,
            time_s=time_s,
            commands=commands,
            electron_density_m3_by_zone=electron_density_m3_by_zone,
            neutral_density_m3_by_zone=neutral_density_m3_by_zone,
            mean_energy_eV_by_zone=mean_energy_eV_by_zone,
            prescribed_electron_power_W_by_zone=(prescribed_electron_power_W_by_zone),
            prescribed_reduced_field_Td_by_zone=(prescribed_reduced_field_Td_by_zone),
            kinetics_by_zone=kinetics_by_zone,
            mean_energy_from_field_by_zone=mean_energy_from_field_by_zone,
        )


__all__ = [
    "CompiledPowerCommand",
    "DCSeriesPort",
    "ExternalTablePowerPort",
    "PowerCoordinator",
    "PowerCouplingResult",
    "PowerPort",
    "PowerPortResult",
    "PowerState",
    "PrescribedPowerPort",
]
