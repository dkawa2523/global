"""Stable public power API with calculations delegated to private modules.

Power ports return only quantities they actually calculate. In particular,
prescribed absorbed power does not manufacture a sheath voltage or E/N.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from plasma_global.errors import (
    CaseValidationError,
    CouplingConvergenceError,
    ModelDomainError,
)
from plasma_global.models._power_coupling import (
    _ZonePowerInputs,
    _ZonePowerIteration,
    _ZonePowerResult,
    advance_coupled_field,
    command_field_capability,
    commands_are_time_dependent,
    field_feedback,
    group_ports,
    normalize_id,
    normalized_identities,
    python_float,
    validate_field_sources,
)
from plasma_global.models._power_ports import E_CHARGE as E_CHARGE
from plasma_global.models._power_ports import TD_TO_V_M2 as TD_TO_V_M2
from plasma_global.models._power_ports import (
    _PortEvaluation,
    _source_voltage,
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
        return _table_reduced_field_capability(self.external_table)


def _table_reduced_field_capability(
    table: ExternalTableBinding | ExternalTableSample | None,
) -> bool | None:
    return None if table is None else table.produces_reduced_field


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


def _kinetics_for_power_state(
    zone_id: str,
    inputs: _ZonePowerInputs,
    reduced_field_Td: float | None,
) -> ElectronKineticsResult | None:
    """Evaluate the configured kinetics table at the iteration coordinate."""

    kinetics = inputs.kinetics
    if kinetics is None:
        return None
    if kinetics.lookup == "local_field":
        if reduced_field_Td is None:
            raise ModelDomainError(f"Zone {zone_id!r} local-field table needs E/N")
        if reduced_field_Td == 0.0 and kinetics.axis[0] > 0.0:
            return kinetics.zero_field_result()
        return kinetics.evaluate(reduced_field_Td=reduced_field_Td)
    if kinetics.lookup == "mean_energy":
        if inputs.mean_energy_eV is None:
            raise ModelDomainError(
                f"Zone {zone_id!r} mean-energy table needs mean_energy_eV"
            )
        return kinetics.evaluate(mean_energy_eV=inputs.mean_energy_eV)
    raise CaseValidationError(f"Unsupported kinetics lookup {kinetics.lookup!r}")


def _mean_energy_from_inputs(
    zone_id: str,
    inputs: _ZonePowerInputs,
    reduced_field_Td: float | None,
) -> float:
    """Resolve direct or local-field mean energy when no table is configured."""

    if inputs.mean_energy_eV is not None:
        return python_float(inputs.mean_energy_eV)
    if reduced_field_Td is None or inputs.mean_energy_from_field is None:
        raise ModelDomainError(
            f"Zone {zone_id!r} needs a local-field mean-energy model"
        )
    return python_float(inputs.mean_energy_from_field(reduced_field_Td))


def _zone_iteration_is_terminal(
    exact_zero_field: bool,
    iteration: _ZonePowerIteration[PowerState, PowerPortResult],
) -> bool:
    return exact_zero_field or not iteration.field_is_coupled


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

    def _source_voltage(self, command: CompiledPowerCommand | None) -> float:
        return _source_voltage(
            port_id=self.port_id,
            default_voltage_V=self.default_voltage_V,
            command_kind=None if command is None else command.kind,
            command_voltage_V=None if command is None else command.voltage_V,
        )

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

    def __post_init__(self) -> None:
        ports, zones, port_ids = normalized_identities(self.ports, self.zone_ids)
        if not math.isfinite(self.coupling_rtol) or self.coupling_rtol <= 0.0:
            raise CaseValidationError("coupling_rtol must be positive")
        if self.max_iterations <= 0:
            raise CaseValidationError("max_iterations must be positive")
        grouped = group_ports(ports, zones)
        for zone_id, zone_ports in grouped.items():
            validate_field_sources(zone_id, zone_ports, {})
        object.__setattr__(self, "ports", ports)
        object.__setattr__(self, "zone_ids", zones)
        object.__setattr__(self, "_ports_by_zone", frozen_mapping(grouped))
        object.__setattr__(self, "_port_ids", frozenset(port_ids))

    def validate_commands(self, commands: Mapping[str, CompiledPowerCommand]) -> None:
        """Validate one compiled segment command set outside the ODE hot path."""

        unknown = set(commands) - self._port_ids
        if unknown:
            raise CaseValidationError(
                f"Recipe commands reference unknown ports: {sorted(unknown)}"
            )
        for zone_id, zone_ports in self._ports_by_zone.items():
            validate_field_sources(zone_id, zone_ports, commands)

    @property
    def is_time_dependent(self) -> bool:
        """Whether any configured port has explicit dependence on wall time."""

        return any(port.time_dependent for port in self.ports)

    def commands_are_time_dependent(
        self, commands: Mapping[str, CompiledPowerCommand]
    ) -> bool:
        """Return explicit wall-time dependence for one compiled segment."""

        return commands_are_time_dependent(self.ports, commands)

    def has_reduced_field_source(
        self, zone_id: str, commands: Mapping[str, CompiledPowerCommand]
    ) -> bool:
        """Return whether one segment guarantees E/N for the requested zone."""

        for port in self._ports_by_zone[normalize_id(zone_id)]:
            command = commands.get(normalize_id(port.port_id))
            if command_field_capability(port, command):
                return True
        return False

    @staticmethod
    def _evaluate_port(
        port: PowerPort,
        time_s: float,
        state: PowerState,
        command: CompiledPowerCommand | None,
    ) -> PowerPortResult:
        if command is not None and command.kind == "off":
            return PowerPortResult(
                port_id=normalize_id(port.port_id),
                zone_id=normalize_id(port.zone_id),
                electron_power_W=0.0,
                gas_power_W=0.0,
                reduced_field_Td=(
                    0.0 if command_field_capability(port, command) else None
                ),
            )
        return port.evaluate(time_s, state, command)

    def _initial_reduced_field(
        self,
        zone_id: str,
        commands: Mapping[str, CompiledPowerCommand],
        inputs: _ZonePowerInputs,
    ) -> float | None:
        field = inputs.prescribed_field_Td
        kinetics = inputs.kinetics
        if kinetics is None or kinetics.lookup != "local_field":
            return field
        if field is None:
            field = float(kinetics.axis[len(kinetics.axis) // 2])

        zone_ports = self._ports_by_zone[zone_id]
        if len(zone_ports) != 1 or not isinstance(zone_ports[0], DCSeriesPort):
            return field
        port = zone_ports[0]
        command = commands.get(normalize_id(port.port_id))
        if command is not None and command.kind == "off":
            return field
        return port.solve_local_field(
            electron_density_m3=inputs.electron_density_m3,
            neutral_density_m3=inputs.neutral_density_m3,
            kinetics=kinetics,
            command=command,
        )

    def _has_exact_zero_field_command(
        self, zone_id: str, commands: Mapping[str, CompiledPowerCommand]
    ) -> bool:
        """Return whether an off command fixes this zone's field at zero."""

        return any(
            (command := commands.get(normalize_id(port.port_id))) is not None
            and command.kind == "off"
            and command_field_capability(port, command)
            for port in self._ports_by_zone[zone_id]
        )

    @staticmethod
    def _require_energy_source(
        zone_id: str,
        inputs: _ZonePowerInputs,
        reduced_field_Td: float | None,
    ) -> None:
        if (
            inputs.mean_energy_eV is None
            and inputs.kinetics is None
            and reduced_field_Td is None
        ):
            raise ModelDomainError(
                f"Zone {zone_id!r} needs E/N or an electron-energy value "
                "for power coupling"
            )

    @staticmethod
    def _power_state_for_iteration(
        zone_id: str,
        inputs: _ZonePowerInputs,
        reduced_field_Td: float | None,
    ) -> tuple[PowerState, ElectronKineticsResult | None]:
        kinetics_result = _kinetics_for_power_state(zone_id, inputs, reduced_field_Td)
        if kinetics_result is not None:
            local_mean_energy_eV = python_float(kinetics_result.mean_energy_eV)
            mobility_m2_V_s = kinetics_result.mobility_m2_V_s
        else:
            local_mean_energy_eV = _mean_energy_from_inputs(
                zone_id, inputs, reduced_field_Td
            )
            mobility_m2_V_s = None

        return (
            PowerState(
                electron_density_m3=inputs.electron_density_m3,
                neutral_density_m3=inputs.neutral_density_m3,
                electron_temperature_eV=(2.0 / 3.0) * local_mean_energy_eV,
                electron_mobility_m2_V_s=mobility_m2_V_s,
            ),
            kinetics_result,
        )

    def _evaluate_zone_ports(
        self,
        zone_id: str,
        time_s: float,
        state: PowerState,
        commands: Mapping[str, CompiledPowerCommand],
    ) -> tuple[PowerPortResult, ...]:
        results: list[PowerPortResult] = []
        for port in self._ports_by_zone[zone_id]:
            result = self._evaluate_port(
                port,
                time_s,
                state,
                commands.get(normalize_id(port.port_id)),
            )
            if (
                result.port_id != normalize_id(port.port_id)
                or result.zone_id != zone_id
            ):
                raise ModelDomainError(
                    f"Port {port.port_id!r} returned mismatched identity "
                    f"({result.port_id!r}, {result.zone_id!r})"
                )
            results.append(result)
        return tuple(results)

    def _evaluate_iteration(
        self,
        zone_id: str,
        time_s: float,
        commands: Mapping[str, CompiledPowerCommand],
        inputs: _ZonePowerInputs,
        reduced_field_Td: float | None,
    ) -> _ZonePowerIteration[PowerState, PowerPortResult]:
        state, kinetics_result = self._power_state_for_iteration(
            zone_id, inputs, reduced_field_Td
        )
        port_results = self._evaluate_zone_ports(zone_id, time_s, state, commands)
        electron_power_W = inputs.base_electron_power_W + sum(
            result.electron_power_W for result in port_results
        )
        next_field, port_produced_field = field_feedback(
            zone_id, port_results, inputs.prescribed_field_Td
        )
        local_field_kinetics = (
            inputs.kinetics is not None and inputs.kinetics.lookup == "local_field"
        )
        if local_field_kinetics and next_field is None:
            raise ModelDomainError(
                f"Zone {zone_id!r} local-field kinetics has no E/N-producing "
                "port or prescribed E/N"
            )
        return _ZonePowerIteration(
            power_state=state,
            kinetics=kinetics_result,
            port_results=port_results,
            electron_power_W=electron_power_W,
            next_field_Td=next_field,
            field_is_coupled=port_produced_field and local_field_kinetics,
        )

    def _zone_iteration_start(
        self,
        zone_id: str,
        commands: Mapping[str, CompiledPowerCommand],
        inputs: _ZonePowerInputs,
    ) -> tuple[bool, float | None]:
        exact_zero_field = self._has_exact_zero_field_command(zone_id, commands)
        reduced_field_Td = (
            0.0
            if exact_zero_field
            else self._initial_reduced_field(zone_id, commands, inputs)
        )
        self._require_energy_source(zone_id, inputs, reduced_field_Td)
        return exact_zero_field, reduced_field_Td

    def _evaluate_zone(
        self,
        zone_id: str,
        time_s: float,
        commands: Mapping[str, CompiledPowerCommand],
        inputs: _ZonePowerInputs,
    ) -> _ZonePowerResult[PowerState, PowerPortResult]:
        exact_zero_field, reduced_field_Td = self._zone_iteration_start(
            zone_id, commands, inputs
        )

        previous_mobility: float | None = None
        previous_power: float | None = None
        last_residual = math.inf
        for iteration_count in range(1, self.max_iterations + 1):
            iteration = self._evaluate_iteration(
                zone_id, time_s, commands, inputs, reduced_field_Td
            )
            if _zone_iteration_is_terminal(exact_zero_field, iteration):
                break
            mobility = iteration.power_state.electron_mobility_m2_V_s
            reduced_field_Td, last_residual = advance_coupled_field(
                zone_id=zone_id,
                current_field_Td=reduced_field_Td,
                iteration=iteration,
                previous_mobility_m2_V_s=previous_mobility,
                previous_electron_power_W=previous_power,
            )
            if iteration_count > 1 and last_residual <= self.coupling_rtol:
                # Re-evaluate every coupled quantity at the field we return.
                # The converged iteration above was evaluated at the preceding
                # fixed-point proposal.
                iteration = self._evaluate_iteration(
                    zone_id, time_s, commands, inputs, reduced_field_Td
                )
                break
            previous_mobility = mobility
            previous_power = iteration.electron_power_W
        else:
            zone_port_ids = [port.port_id for port in self._ports_by_zone[zone_id]]
            raise CouplingConvergenceError(
                f"Power/EEDF coupling in zone {zone_id!r} did not converge after "
                f"{self.max_iterations} iterations for ports {zone_port_ids} "
                f"(relative residual={last_residual:.3e})"
            )

        return iteration.as_zone_result(reduced_field_Td, iteration_count)

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
        base_power = prescribed_electron_power_W_by_zone or {}
        prescribed_field = prescribed_reduced_field_Td_by_zone or {}
        kinetics_models = kinetics_by_zone or {}
        field_models = mean_energy_from_field_by_zone or {}
        electron_power: dict[str, float] = {}
        gas_power: dict[str, float] = {}
        effective_field: dict[str, float] = {}
        power_states: dict[str, PowerState] = {}
        port_results: dict[str, PowerPortResult] = {}
        iterations: dict[str, int] = {}
        final_kinetics: dict[str, ElectronKineticsResult] = {}

        for zone_id in self.zone_ids:
            inputs = _ZonePowerInputs(
                electron_density_m3=electron_density_m3_by_zone[zone_id],
                neutral_density_m3=python_float(neutral_density_m3_by_zone[zone_id]),
                mean_energy_eV=mean_energy_eV_by_zone.get(zone_id),
                base_electron_power_W=python_float(base_power.get(zone_id, 0.0)),
                prescribed_field_Td=prescribed_field.get(zone_id),
                kinetics=kinetics_models.get(zone_id),
                mean_energy_from_field=field_models.get(zone_id),
            )
            result = self._evaluate_zone(zone_id, time_s, commands, inputs)
            electron_power[zone_id] = result.electron_power_W
            gas_power[zone_id] = result.gas_power_W
            power_states[zone_id] = result.power_state
            iterations[zone_id] = result.iterations
            if result.reduced_field_Td is not None:
                effective_field[zone_id] = result.reduced_field_Td
            if result.kinetics is not None:
                final_kinetics[zone_id] = result.kinetics
            for port_result in result.port_results:
                port_results[port_result.port_id] = port_result

        return PowerCouplingResult(
            electron_power_W_by_zone=electron_power,
            gas_power_W_by_zone=gas_power,
            reduced_field_Td_by_zone=effective_field,
            power_state_by_zone=power_states,
            kinetics_by_zone=final_kinetics,
            port_results=port_results,
            iterations_by_zone=iterations,
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
