"""Zone-level iteration and aggregation for the public power coordinator."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from plasma_global.errors import (
    CaseValidationError,
    CouplingConvergenceError,
    ModelDomainError,
)
from plasma_global.models._power_coupling import (
    command_field_capability,
)
from plasma_global.models.kinetics import (
    ElectronKineticsResult,
    TabulatedElectronKinetics,
)
from plasma_global.models.power import (
    CompiledPowerCommand,
    DCSeriesPort,
    PowerCoordinator,
    PowerCouplingResult,
    PowerPort,
    PowerPortResult,
    PowerState,
)


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
class _ZonePowerResult:
    electron_power_W: float
    gas_power_W: float
    reduced_field_Td: float | None
    power_state: PowerState
    kinetics: ElectronKineticsResult | None
    port_results: tuple[PowerPortResult, ...]
    iterations: int


@dataclass(frozen=True, slots=True)
class _ZonePowerIteration:
    power_state: PowerState
    kinetics: ElectronKineticsResult | None
    port_results: tuple[PowerPortResult, ...]
    electron_power_W: float
    next_field_Td: float | None
    field_is_coupled: bool

    def as_zone_result(
        self, current_field_Td: float | None, iterations: int
    ) -> _ZonePowerResult:
        # Coupled kinetics and ports were evaluated at ``current_field_Td``.
        # ``next_field_Td`` is only the next fixed-point proposal and must not
        # be paired with transport/rate data from the preceding iterate.
        effective_field_Td = current_field_Td
        if not self.field_is_coupled and self.next_field_Td is not None:
            effective_field_Td = self.next_field_Td
        if effective_field_Td is None and self.kinetics is not None:
            effective_field_Td = float(self.kinetics.effective_field_Td)
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
    results: tuple[PowerPortResult, ...],
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


def _relative_change(new: float, old: float) -> float:
    return abs(new - old) / max(abs(new), abs(old), 1.0)


def _advance_coupled_field(
    *,
    zone_id: str,
    current_field_Td: float | None,
    iteration: _ZonePowerIteration,
    previous_mobility_m2_V_s: float | None,
    previous_electron_power_W: float | None,
) -> tuple[float, float]:
    next_field_Td = iteration.next_field_Td
    if current_field_Td is None or next_field_Td is None:
        raise ModelDomainError(f"Zone {zone_id!r} local-field coupling requires E/N")
    mobility_m2_V_s = iteration.power_state.electron_mobility_m2_V_s
    field_residual = _relative_change(next_field_Td, current_field_Td)
    mobility_residual = (
        0.0
        if previous_mobility_m2_V_s is None or mobility_m2_V_s is None
        else _relative_change(mobility_m2_V_s, previous_mobility_m2_V_s)
    )
    power_residual = (
        0.0
        if previous_electron_power_W is None
        else _relative_change(iteration.electron_power_W, previous_electron_power_W)
    )
    return float(next_field_Td), max(field_residual, mobility_residual, power_residual)


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
            return kinetics.zero_field_result(
                neutral_density_m3=inputs.neutral_density_m3
            )
        return kinetics.evaluate(
            reduced_field_Td=reduced_field_Td,
            neutral_density_m3=inputs.neutral_density_m3,
        )
    if kinetics.lookup == "mean_energy":
        if inputs.mean_energy_eV is None:
            raise ModelDomainError(
                f"Zone {zone_id!r} mean-energy table needs mean_energy_eV"
            )
        return kinetics.evaluate(
            mean_energy_eV=inputs.mean_energy_eV,
            neutral_density_m3=inputs.neutral_density_m3,
        )
    raise CaseValidationError(f"Unsupported kinetics lookup {kinetics.lookup!r}")


def _mean_energy_from_inputs(
    zone_id: str,
    inputs: _ZonePowerInputs,
    reduced_field_Td: float | None,
) -> float:
    """Resolve direct or local-field mean energy when no table is configured."""

    if inputs.mean_energy_eV is not None:
        return float(inputs.mean_energy_eV)
    if reduced_field_Td is None or inputs.mean_energy_from_field is None:
        raise ModelDomainError(
            f"Zone {zone_id!r} needs a local-field mean-energy model"
        )
    return float(inputs.mean_energy_from_field(reduced_field_Td))


@dataclass(frozen=True, slots=True)
class _PowerCoordinatorEngine:
    """Run one coordinator evaluation without expanding its public API class."""

    coordinator: PowerCoordinator

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

        zone_ports = self.coordinator._ports_by_zone[zone_id]
        if len(zone_ports) != 1 or not isinstance(zone_ports[0], DCSeriesPort):
            return field
        port = zone_ports[0]
        command = commands.get(str(port.port_id))
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
            (command := commands.get(str(port.port_id))) is not None
            and command.kind == "off"
            and command_field_capability(port, command)
            for port in self.coordinator._ports_by_zone[zone_id]
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
            local_mean_energy_eV = float(kinetics_result.mean_energy_eV)
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
        for port in self.coordinator._ports_by_zone[zone_id]:
            result = self._evaluate_port(
                port,
                time_s,
                state,
                commands.get(str(port.port_id)),
            )
            if result.port_id != str(port.port_id) or result.zone_id != zone_id:
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
    ) -> _ZonePowerIteration:
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
    ) -> _ZonePowerResult:
        exact_zero_field, reduced_field_Td = self._zone_iteration_start(
            zone_id, commands, inputs
        )

        previous_mobility: float | None = None
        previous_power: float | None = None
        last_residual = math.inf
        for iteration_count in range(1, self.coordinator.max_iterations + 1):
            iteration = self._evaluate_iteration(
                zone_id, time_s, commands, inputs, reduced_field_Td
            )
            if exact_zero_field or not iteration.field_is_coupled:
                break
            mobility = iteration.power_state.electron_mobility_m2_V_s
            reduced_field_Td, last_residual = _advance_coupled_field(
                zone_id=zone_id,
                current_field_Td=reduced_field_Td,
                iteration=iteration,
                previous_mobility_m2_V_s=previous_mobility,
                previous_electron_power_W=previous_power,
            )
            if iteration_count > 1 and last_residual <= self.coordinator.coupling_rtol:
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
            zone_port_ids = [
                port.port_id for port in self.coordinator._ports_by_zone[zone_id]
            ]
            raise CouplingConvergenceError(
                f"Power/EEDF coupling in zone {zone_id!r} did not converge after "
                f"{self.coordinator.max_iterations} iterations for ports "
                f"{zone_port_ids} (relative residual={last_residual:.3e})"
            )

        return iteration.as_zone_result(reduced_field_Td, iteration_count)

    def _distribute_zone_power(
        self,
        electron_power_W_by_zone: dict[str, float],
        port_results: Mapping[str, PowerPortResult],
    ) -> None:
        """Add optional cross-zone port deposition after all zones are evaluated."""

        for distributor in self.coordinator._zone_power_distributors:
            result = port_results[distributor.port_id]
            for raw_zone_id, raw_power_W in distributor.distribute(result).items():
                zone_id = str(raw_zone_id)
                local_power_W = (
                    result.electron_power_W if zone_id == result.zone_id else 0.0
                )
                additional_power_W = float(raw_power_W) - local_power_W
                if additional_power_W == 0.0:
                    continue
                if zone_id not in electron_power_W_by_zone:
                    raise ModelDomainError(
                        f"Port {distributor.port_id!r} distributes power to "
                        f"unknown zone {zone_id!r}"
                    )
                electron_power_W_by_zone[zone_id] += additional_power_W

    def evaluate(
        self,
        *,
        time_s: float,
        commands: Mapping[str, CompiledPowerCommand],
        electron_density_m3_by_zone: Mapping[str, float],
        neutral_density_m3_by_zone: Mapping[str, float],
        mean_energy_eV_by_zone: Mapping[str, float | None],
        prescribed_electron_power_W_by_zone: Mapping[str, float] | None,
        prescribed_reduced_field_Td_by_zone: Mapping[str, float] | None,
        kinetics_by_zone: Mapping[str, TabulatedElectronKinetics] | None,
        mean_energy_from_field_by_zone: Mapping[str, Callable[[float], float]] | None,
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

        for zone_id in self.coordinator.zone_ids:
            inputs = _ZonePowerInputs(
                electron_density_m3=electron_density_m3_by_zone[zone_id],
                neutral_density_m3=float(neutral_density_m3_by_zone[zone_id]),
                mean_energy_eV=mean_energy_eV_by_zone.get(zone_id),
                base_electron_power_W=float(base_power.get(zone_id, 0.0)),
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

        self._distribute_zone_power(electron_power, port_results)

        return PowerCouplingResult(
            electron_power_W_by_zone=electron_power,
            gas_power_W_by_zone=gas_power,
            reduced_field_Td_by_zone=effective_field,
            power_state_by_zone=power_states,
            kinetics_by_zone=final_kinetics,
            port_results=port_results,
            iterations_by_zone=iterations,
        )


def evaluate_power_coupling(
    coordinator: PowerCoordinator,
    *,
    time_s: float,
    commands: Mapping[str, CompiledPowerCommand],
    electron_density_m3_by_zone: Mapping[str, float],
    neutral_density_m3_by_zone: Mapping[str, float],
    mean_energy_eV_by_zone: Mapping[str, float | None],
    prescribed_electron_power_W_by_zone: Mapping[str, float] | None,
    prescribed_reduced_field_Td_by_zone: Mapping[str, float] | None,
    kinetics_by_zone: Mapping[str, TabulatedElectronKinetics] | None,
    mean_energy_from_field_by_zone: Mapping[str, Callable[[float], float]] | None,
) -> PowerCouplingResult:
    """Evaluate all zones while returning only public power value types."""

    return _PowerCoordinatorEngine(coordinator).evaluate(
        time_s=time_s,
        commands=commands,
        electron_density_m3_by_zone=electron_density_m3_by_zone,
        neutral_density_m3_by_zone=neutral_density_m3_by_zone,
        mean_energy_eV_by_zone=mean_energy_eV_by_zone,
        prescribed_electron_power_W_by_zone=prescribed_electron_power_W_by_zone,
        prescribed_reduced_field_Td_by_zone=prescribed_reduced_field_Td_by_zone,
        kinetics_by_zone=kinetics_by_zone,
        mean_energy_from_field_by_zone=mean_energy_from_field_by_zone,
    )


__all__ = ["evaluate_power_coupling"]
