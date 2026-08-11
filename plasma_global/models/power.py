"""Typed stable power-port models.

Power ports return only quantities they actually calculate.  In particular,
prescribed absorbed power does not manufacture a sheath voltage or E/N.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol

import numpy as np

from plasma_global.errors import (
    CaseValidationError,
    CouplingConvergenceError,
    ModelDomainError,
)
from plasma_global.models.external_table import (
    ExternalTableBinding,
    ExternalTableSample,
)
from plasma_global.models.kinetics import (
    ElectronKineticsResult,
    TabulatedElectronKinetics,
)


def _empty_observables() -> Mapping[str, float]:
    return MappingProxyType({})


E_CHARGE = 1.602176634e-19
TD_TO_V_M2 = 1.0e-21


@dataclass(frozen=True, slots=True)
class PowerState:
    electron_density_m3: float
    neutral_density_m3: float
    electron_temperature_eV: float
    electron_mobility_m2_V_s: float | None = None

    def __post_init__(self) -> None:
        values = (
            self.electron_density_m3,
            self.neutral_density_m3,
            self.electron_temperature_eV,
        )
        if not np.isfinite(values).all() or min(values) < 0.0:
            raise ModelDomainError("power state values must be finite and nonnegative")
        if self.electron_mobility_m2_V_s is not None and (
            not math.isfinite(self.electron_mobility_m2_V_s)
            or self.electron_mobility_m2_V_s <= 0.0
        ):
            raise ModelDomainError("electron mobility must be finite and positive")


@dataclass(frozen=True, slots=True)
class CompiledPowerCommand:
    """One fully bound numeric command consumed by the power hot path."""

    kind: Literal["off", "power", "voltage", "external_table"]
    power_W: float | None = None
    voltage_V: float | None = None
    external_table: ExternalTableBinding | ExternalTableSample | None = None

    def __post_init__(self) -> None:
        if self.kind == "off":
            if any(
                value is not None
                for value in (self.power_W, self.voltage_V, self.external_table)
            ):
                raise CaseValidationError("off power commands must not carry a value")
            return
        if self.kind == "power":
            if (
                self.power_W is None
                or not math.isfinite(self.power_W)
                or self.power_W < 0.0
                or self.voltage_V is not None
                or self.external_table is not None
            ):
                raise CaseValidationError(
                    "compiled power commands require one finite nonnegative power_W"
                )
            return
        if self.kind == "voltage":
            if (
                self.voltage_V is None
                or not math.isfinite(self.voltage_V)
                or self.power_W is not None
                or self.external_table is not None
            ):
                raise CaseValidationError(
                    "compiled voltage commands require one finite voltage_V"
                )
            return
        if self.kind == "external_table":
            if (
                not isinstance(
                    self.external_table, (ExternalTableBinding, ExternalTableSample)
                )
                or self.power_W is not None
                or self.voltage_V is not None
            ):
                raise CaseValidationError(
                    "compiled external-table commands require one table binding"
                )
            return
        raise CaseValidationError(f"unsupported compiled power command {self.kind!r}")

    @property
    def time_dependent(self) -> bool:
        return self.kind == "external_table" and isinstance(
            self.external_table, ExternalTableBinding
        )

    @property
    def produces_reduced_field(self) -> bool | None:
        if self.kind != "external_table":
            return None
        assert self.external_table is not None
        return self.external_table.produces_reduced_field


@dataclass(frozen=True, slots=True)
class PowerPortResult:
    port_id: str
    zone_id: str
    electron_power_W: float
    gas_power_W: float = 0.0
    reduced_field_Td: float | None = None
    observables: Mapping[str, float] = field(default_factory=_empty_observables)

    def __post_init__(self) -> None:
        if (
            min(self.electron_power_W, self.gas_power_W) < 0.0
            or not np.isfinite([self.electron_power_W, self.gas_power_W]).all()
        ):
            raise ModelDomainError("power partitions must be finite and nonnegative")
        if self.reduced_field_Td is not None and (
            not math.isfinite(self.reduced_field_Td) or self.reduced_field_Td < 0.0
        ):
            raise ModelDomainError("reduced_field_Td must be finite and nonnegative")
        object.__setattr__(
            self, "observables", MappingProxyType(dict(self.observables))
        )


class PowerPort(Protocol):
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
class PrescribedPowerPort:
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
        fractions = self.electron_fraction, self.gas_fraction
        if not np.isfinite(fractions).all() or min(fractions) < 0.0:
            raise CaseValidationError("power fractions must be finite and nonnegative")
        if sum(fractions) > 1.0 + 1.0e-12:
            raise CaseValidationError(
                "electron_fraction + gas_fraction must not exceed one"
            )
        if self.default_power_W is not None and (
            not math.isfinite(self.default_power_W) or self.default_power_W < 0.0
        ):
            raise CaseValidationError("default_power_W must be finite and nonnegative")

    def evaluate(
        self, time_s: float, state: PowerState, command: CompiledPowerCommand | None
    ) -> PowerPortResult:
        del time_s, state
        if command is not None and command.kind == "power":
            assert command.power_W is not None
            power = command.power_W
        elif command is None and self.default_power_W is not None:
            power = self.default_power_W
        else:
            raise CaseValidationError(
                f"prescribed-power port {self.port_id!r} requires a power command"
            )
        return PowerPortResult(
            port_id=self.port_id,
            zone_id=self.zone_id,
            electron_power_W=power * self.electron_fraction,
            gas_power_W=power * self.gas_fraction,
            observables={"commanded_power_W": power},
        )


@dataclass(frozen=True, slots=True)
class DCSeriesPort:
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
        positive = {
            "ballast_resistance_ohm": self.ballast_resistance_ohm,
            "gap_m": self.gap_m,
            "electrode_area_m2": self.electrode_area_m2,
        }
        invalid = [
            name
            for name, value in positive.items()
            if not math.isfinite(value) or value <= 0.0
        ]
        if invalid:
            raise CaseValidationError(
                f"DC series parameters must be positive: {', '.join(invalid)}"
            )
        if (
            not math.isfinite(self.absorption_fraction)
            or not 0.0 <= self.absorption_fraction <= 1.0
        ):
            raise CaseValidationError(
                "absorption_fraction must be between zero and one"
            )
        if (
            self.configured_mobility_m2_V_s is not None
            and self.configured_mobility_m2_V_s <= 0.0
        ):
            raise CaseValidationError("configured mobility must be positive")

    def evaluate(
        self, time_s: float, state: PowerState, command: CompiledPowerCommand | None
    ) -> PowerPortResult:
        del time_s
        source_voltage = self._source_voltage(command)
        mobility = state.electron_mobility_m2_V_s or self.configured_mobility_m2_V_s
        if mobility is None:
            raise ModelDomainError(
                f"DC-series port {self.port_id!r} requires electron mobility"
            )
        conductivity = E_CHARGE * state.electron_density_m3 * mobility
        if conductivity <= 0.0:
            plasma_resistance = math.inf
            current = 0.0
            plasma_voltage = source_voltage
            absorbed = 0.0
        else:
            plasma_resistance = self.gap_m / (conductivity * self.electrode_area_m2)
            current = source_voltage / (self.ballast_resistance_ohm + plasma_resistance)
            plasma_voltage = current * plasma_resistance
            absorbed = current * current * plasma_resistance * self.absorption_fraction
        field = abs(plasma_voltage) / self.gap_m
        reduced_field = (
            field / state.neutral_density_m3 / TD_TO_V_M2
            if state.neutral_density_m3 > 0.0
            else 0.0
        )
        return PowerPortResult(
            port_id=self.port_id,
            zone_id=self.zone_id,
            electron_power_W=absorbed,
            reduced_field_Td=reduced_field,
            observables={
                "source_voltage_V": source_voltage,
                "plasma_voltage_V": plasma_voltage,
                "current_A": current,
                "plasma_resistance_ohm": plasma_resistance,
            },
        )

    def _source_voltage(self, command: CompiledPowerCommand | None) -> float:
        if command is not None and command.kind == "voltage":
            assert command.voltage_V is not None
            return command.voltage_V
        elif command is None and self.default_voltage_V is not None:
            return self.default_voltage_V
        raise CaseValidationError(
            f"DC-series port {self.port_id!r} requires a voltage command"
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
        monotone.  We retain the stable root reached from the table's central
        node, which is the deterministic branch selected by the former fixed-
        point iteration, without requiring dozens of RHS-time iterations.
        """

        axis = np.asarray(kinetics.axis, dtype=float)
        mobility = np.asarray(kinetics.mobility_m2_V_s, dtype=float)
        if mobility.shape != axis.shape:
            raise CaseValidationError(
                "local-field kinetics mobility must share the E/N axis"
            )
        if neutral_density_m3 <= 0.0:
            raise ModelDomainError(
                f"DC-series port {self.port_id!r} requires positive neutral density"
            )
        source_voltage = self._source_voltage(command)
        vacuum_field_Td = (
            abs(source_voltage) / self.gap_m / neutral_density_m3 / TD_TO_V_M2
        )
        conductance_factor = (
            self.ballast_resistance_ohm
            * E_CHARGE
            * max(electron_density_m3, 0.0)
            * self.electrode_area_m2
            / self.gap_m
        )
        slopes = np.diff(mobility) / np.diff(axis)
        intercepts = mobility[:-1] - slopes * axis[:-1]
        quadratic = conductance_factor * slopes
        linear = 1.0 + conductance_factor * intercepts
        constant = -vacuum_field_Td

        machine_epsilon = np.finfo(float).eps
        linear_intervals = np.abs(quadratic) <= machine_epsilon * np.maximum(
            np.abs(linear), 1.0
        )
        discriminant = linear * linear - 4.0 * quadratic * constant
        real_quadratic = ~linear_intervals & (discriminant >= 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            root_scale = np.sqrt(np.maximum(discriminant, 0.0))
            root_sets = (
                np.where(linear_intervals, -constant / linear, np.nan),
                np.where(
                    real_quadratic,
                    (-linear - root_scale) / (2.0 * quadratic),
                    np.nan,
                ),
                np.where(
                    real_quadratic,
                    (-linear + root_scale) / (2.0 * quadratic),
                    np.nan,
                ),
            )
        left = axis[:-1]
        right = axis[1:]
        interval_tolerance = (
            32.0
            * machine_epsilon
            * np.maximum.reduce((np.abs(left), np.abs(right), np.ones_like(left)))
        )
        candidates: list[float] = []
        for roots in root_sets:
            in_interval = (
                np.isfinite(roots)
                & (roots >= left - interval_tolerance)
                & (roots <= right + interval_tolerance)
            )
            clipped = np.clip(roots, left, right)
            local_mobility = slopes * clipped + intercepts
            derivative = np.abs(
                -vacuum_field_Td
                * conductance_factor
                * slopes
                / (1.0 + conductance_factor * local_mobility) ** 2
            )
            candidates.extend(
                float(value) for value in clipped[in_interval & (derivative < 1.0)]
            )

        if not candidates:
            raise CouplingConvergenceError(
                f"DC-series port {self.port_id!r} has no stable E/N root in "
                f"[{axis[0]:g}, {axis[-1]:g}] Td"
            )

        selector = float(axis[len(axis) // 2])
        for _ in range(3):
            selector_mobility = float(np.interp(selector, axis, mobility))
            selector = vacuum_field_Td / (1.0 + conductance_factor * selector_mobility)
            selector = float(np.clip(selector, axis[0], axis[-1]))

        def distance_from_selector(value: float) -> float:
            return abs(value - selector)

        return min(candidates, key=distance_from_selector)


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

    def evaluate(
        self,
        time_s: float,
        state: PowerState,
        command: CompiledPowerCommand | None,
    ) -> PowerPortResult:
        if command is None:
            table: ExternalTableBinding | ExternalTableSample = self.default
        elif command.kind == "external_table" and command.external_table is not None:
            table = command.external_table
        else:
            raise CaseValidationError(
                f"external-table port {self.port_id!r} received an invalid command"
            )
        if isinstance(table, ExternalTableBinding):
            data = table.data
            electron_power = table.power_scale * table._at(
                data.electron_power_W, time_s
            )
            gas_power = table.power_scale * table._at(data.gas_power_W, time_s)
            voltage = (
                None
                if data.voltage_V is None
                else table.voltage_scale * table._at(data.voltage_V, time_s)
            )
            current = (
                None
                if data.current_A is None
                else table.current_scale * table._at(data.current_A, time_s)
            )
            reduced_field = (
                None
                if data.reduced_field_Td is None
                else table._at(data.reduced_field_Td, time_s)
            )
            gap_m = table.gap_m
            total_density_m3 = table.total_density_m3
            plasma_potential_V = table.plasma_potential_V
        else:
            electron_power = table.electron_power_W
            gas_power = table.gas_power_W
            voltage = table.voltage_V
            current = table.current_A
            reduced_field = table.reduced_field_Td
            gap_m = table.gap_m
            total_density_m3 = table.total_density_m3
            plasma_potential_V = table.plasma_potential_V
        if reduced_field is None and voltage is not None and gap_m is not None:
            density = (
                state.neutral_density_m3
                if total_density_m3 is None
                else total_density_m3
            )
            reduced_field = (
                abs(voltage) / gap_m / density / TD_TO_V_M2 if density > 0.0 else 0.0
            )
        observables = {
            "absorbed_power_W": electron_power + gas_power,
            "plasma_potential_V": plasma_potential_V,
        }
        if voltage is not None:
            observables["source_voltage_V"] = voltage
        if current is not None:
            observables["current_A"] = current
        return PowerPortResult(
            port_id=self.port_id,
            zone_id=self.zone_id,
            electron_power_W=electron_power,
            gas_power_W=gas_power,
            reduced_field_Td=reduced_field,
            observables=observables,
        )


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
            MappingProxyType(dict(self.electron_power_W_by_zone)),
        )
        object.__setattr__(
            self,
            "gas_power_W_by_zone",
            MappingProxyType(dict(self.gas_power_W_by_zone)),
        )
        object.__setattr__(
            self,
            "reduced_field_Td_by_zone",
            MappingProxyType(dict(self.reduced_field_Td_by_zone)),
        )
        object.__setattr__(
            self,
            "power_state_by_zone",
            MappingProxyType(dict(self.power_state_by_zone)),
        )
        object.__setattr__(
            self,
            "kinetics_by_zone",
            MappingProxyType(dict(self.kinetics_by_zone)),
        )
        object.__setattr__(
            self, "port_results", MappingProxyType(dict(self.port_results))
        )
        object.__setattr__(
            self,
            "iterations_by_zone",
            MappingProxyType(dict(self.iterations_by_zone)),
        )


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
        ports = tuple(self.ports)
        zones = tuple(str(value) for value in self.zone_ids)
        if not zones or len(set(zones)) != len(zones):
            raise CaseValidationError(
                "PowerCoordinator zone_ids must be unique and non-empty"
            )
        port_ids = [str(port.port_id) for port in ports]
        if len(set(port_ids)) != len(port_ids) or any(not value for value in port_ids):
            raise CaseValidationError(
                "PowerCoordinator port IDs must be unique and non-empty"
            )
        unknown = sorted({str(port.zone_id) for port in ports} - set(zones))
        if unknown:
            raise CaseValidationError(f"Power ports reference unknown zones: {unknown}")
        if not math.isfinite(self.coupling_rtol) or self.coupling_rtol <= 0.0:
            raise CaseValidationError("coupling_rtol must be positive")
        if self.max_iterations <= 0:
            raise CaseValidationError("max_iterations must be positive")
        grouped = {
            zone_id: tuple(port for port in ports if str(port.zone_id) == zone_id)
            for zone_id in zones
        }
        for zone_id, zone_ports in grouped.items():
            field_ports = [
                str(port.port_id) for port in zone_ports if port.produces_reduced_field
            ]
            if len(field_ports) > 1:
                raise CaseValidationError(
                    f"Zone {zone_id!r} has multiple E/N-producing power ports: "
                    f"{field_ports}"
                )
        object.__setattr__(self, "ports", ports)
        object.__setattr__(self, "zone_ids", zones)
        object.__setattr__(self, "_ports_by_zone", MappingProxyType(grouped))
        object.__setattr__(self, "_port_ids", frozenset(port_ids))

    def validate_commands(self, commands: Mapping[str, CompiledPowerCommand]) -> None:
        """Validate one compiled segment command set outside the ODE hot path."""

        unknown = set(commands) - self._port_ids
        if unknown:
            raise CaseValidationError(
                f"Recipe commands reference unknown ports: {sorted(unknown)}"
            )
        for zone_id, zone_ports in self._ports_by_zone.items():
            field_ports: list[str] = []
            for port in zone_ports:
                command = commands.get(str(port.port_id))
                produces_field = (
                    None if command is None else command.produces_reduced_field
                )
                if produces_field is None:
                    produces_field = port.produces_reduced_field
                if produces_field:
                    field_ports.append(str(port.port_id))
            if len(field_ports) > 1:
                raise CaseValidationError(
                    f"Zone {zone_id!r} has multiple E/N-producing power ports: "
                    f"{field_ports}"
                )

    @property
    def is_time_dependent(self) -> bool:
        """Whether any configured port has explicit dependence on wall time."""

        return any(port.time_dependent for port in self.ports)

    def commands_are_time_dependent(
        self, commands: Mapping[str, CompiledPowerCommand]
    ) -> bool:
        """Return explicit wall-time dependence for one compiled segment."""

        for port in self.ports:
            if not port.time_dependent:
                continue
            command = commands.get(str(port.port_id))
            if command is not None and not command.time_dependent:
                continue
            return True
        return False

    @staticmethod
    def _relative_change(new: float, old: float) -> float:
        return abs(new - old) / max(abs(new), abs(old), 1.0)

    @staticmethod
    def _evaluate_port(
        port: PowerPort,
        time_s: float,
        state: PowerState,
        command: CompiledPowerCommand | None,
    ) -> PowerPortResult:
        if command is not None and command.kind == "off":
            return PowerPortResult(
                port_id=str(port.port_id),
                zone_id=str(port.zone_id),
                electron_power_W=0.0,
                gas_power_W=0.0,
                reduced_field_Td=(0.0 if port.produces_reduced_field else None),
            )
        return port.evaluate(time_s, state, command)

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
        final_power_state: dict[str, PowerState] = {}
        final_kinetics: dict[str, ElectronKineticsResult] = {}
        all_port_results: dict[str, PowerPortResult] = {}
        iterations: dict[str, int] = {}

        for zone_id in self.zone_ids:
            mean_energy = mean_energy_eV_by_zone.get(zone_id)
            table = kinetics_models.get(zone_id)
            field = prescribed_field.get(zone_id)
            if table is not None and table.lookup == "local_field" and field is None:
                field = float(table.axis[len(table.axis) // 2])
            zone_ports = self._ports_by_zone[zone_id]
            if (
                table is not None
                and table.lookup == "local_field"
                and len(zone_ports) == 1
                and isinstance(zone_ports[0], DCSeriesPort)
            ):
                dc_port = zone_ports[0]
                dc_command = commands.get(str(dc_port.port_id))
                if dc_command is None or dc_command.kind != "off":
                    field = dc_port.solve_local_field(
                        electron_density_m3=float(electron_density_m3_by_zone[zone_id]),
                        neutral_density_m3=float(neutral_density_m3_by_zone[zone_id]),
                        kinetics=table,
                        command=dc_command,
                    )
            if mean_energy is None and table is None and field is None:
                raise ModelDomainError(
                    f"Zone {zone_id!r} needs E/N or an electron-energy value for power coupling"
                )

            previous_mobility: float | None = None
            previous_power: float | None = None
            last_residual = math.inf
            zone_results: tuple[PowerPortResult, ...] = ()
            zone_kinetics: ElectronKineticsResult | None = None
            zone_state: PowerState | None = None
            for iteration in range(1, self.max_iterations + 1):
                if table is not None:
                    if table.lookup == "local_field":
                        if field is None:
                            raise ModelDomainError(
                                f"Zone {zone_id!r} local-field table needs E/N"
                            )
                        zone_kinetics = table.evaluate(reduced_field_Td=field)
                    elif table.lookup == "mean_energy":
                        if mean_energy is None:
                            raise ModelDomainError(
                                f"Zone {zone_id!r} mean-energy table needs mean_energy_eV"
                            )
                        zone_kinetics = table.evaluate(mean_energy_eV=mean_energy)
                    else:
                        raise CaseValidationError(
                            f"Unsupported kinetics lookup {table.lookup!r}"
                        )
                    local_mean_energy = float(zone_kinetics.mean_energy_eV)
                    mobility = float(zone_kinetics.mobility_m2_V_s)
                elif mean_energy is not None:
                    local_mean_energy = float(mean_energy)
                    mobility = None
                else:
                    if field is None or zone_id not in field_models:
                        raise ModelDomainError(
                            f"Zone {zone_id!r} needs a local-field mean-energy model"
                        )
                    local_mean_energy = float(field_models[zone_id](field))
                    mobility = None

                zone_state = PowerState(
                    electron_density_m3=float(electron_density_m3_by_zone[zone_id]),
                    neutral_density_m3=float(neutral_density_m3_by_zone[zone_id]),
                    electron_temperature_eV=(2.0 / 3.0) * local_mean_energy,
                    electron_mobility_m2_V_s=mobility,
                )
                evaluated_ports: list[PowerPortResult] = []
                for port in self._ports_by_zone[zone_id]:
                    result = self._evaluate_port(
                        port,
                        time_s,
                        zone_state,
                        commands.get(str(port.port_id)),
                    )
                    if result.port_id != str(port.port_id) or result.zone_id != zone_id:
                        raise ModelDomainError(
                            f"Port {port.port_id!r} returned mismatched identity "
                            f"({result.port_id!r}, {result.zone_id!r})"
                        )
                    evaluated_ports.append(result)
                zone_results = tuple(evaluated_ports)
                zone_electron_power = float(base_power.get(zone_id, 0.0)) + sum(
                    result.electron_power_W for result in zone_results
                )
                field_candidates = [
                    float(result.reduced_field_Td)
                    for result in zone_results
                    if result.reduced_field_Td is not None
                ]
                if len(field_candidates) > 1:
                    field_ports = [
                        result.port_id
                        for result in zone_results
                        if result.reduced_field_Td is not None
                    ]
                    raise CaseValidationError(
                        f"Zone {zone_id!r} has multiple E/N-producing power ports: "
                        f"{field_ports}"
                    )
                next_field = (
                    field_candidates[0]
                    if field_candidates
                    else prescribed_field.get(zone_id)
                )
                field_is_coupled = (
                    bool(field_candidates)
                    and table is not None
                    and table.lookup == "local_field"
                )
                if (
                    table is not None
                    and table.lookup == "local_field"
                    and next_field is None
                ):
                    raise ModelDomainError(
                        f"Zone {zone_id!r} local-field kinetics has no E/N-producing port or prescribed E/N"
                    )
                if not field_is_coupled:
                    if next_field is not None:
                        field = float(next_field)
                    iterations[zone_id] = iteration
                    break

                assert field is not None and next_field is not None
                field_residual = self._relative_change(float(next_field), field)
                mobility_residual = (
                    0.0
                    if previous_mobility is None or mobility is None
                    else self._relative_change(mobility, previous_mobility)
                )
                power_residual = (
                    0.0
                    if previous_power is None
                    else self._relative_change(zone_electron_power, previous_power)
                )
                last_residual = max(field_residual, mobility_residual, power_residual)
                if iteration > 1 and last_residual <= self.coupling_rtol:
                    field = float(next_field)
                    iterations[zone_id] = iteration
                    break

                previous_mobility = mobility
                previous_power = zone_electron_power
                field = float(next_field)
            else:
                zone_port_ids = [
                    str(port.port_id) for port in self._ports_by_zone[zone_id]
                ]
                raise CouplingConvergenceError(
                    f"Power/EEDF coupling in zone {zone_id!r} did not converge after "
                    f"{self.max_iterations} iterations for ports {zone_port_ids} "
                    f"(relative residual={last_residual:.3e})"
                )

            assert zone_state is not None
            zone_electron_power = float(base_power.get(zone_id, 0.0)) + sum(
                result.electron_power_W for result in zone_results
            )
            electron_power[zone_id] = zone_electron_power
            gas_power[zone_id] = sum(result.gas_power_W for result in zone_results)
            if field is not None:
                effective_field[zone_id] = float(field)
            elif zone_kinetics is not None:
                effective_field[zone_id] = float(zone_kinetics.effective_field_Td)
            final_power_state[zone_id] = zone_state
            if zone_kinetics is not None:
                final_kinetics[zone_id] = zone_kinetics
            all_port_results.update({result.port_id: result for result in zone_results})

        return PowerCouplingResult(
            electron_power_W_by_zone=electron_power,
            gas_power_W_by_zone=gas_power,
            reduced_field_Td_by_zone=effective_field,
            power_state_by_zone=final_power_state,
            kinetics_by_zone=final_kinetics,
            port_results=all_port_results,
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
