"""Resolve runtime power, electron kinetics, and electron closure state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from plasma_global.core._runtime_assembly import (
    PowerEvaluation,
    PreparedEvaluationState,
)
from plasma_global.core.domain import RecipeSegment
from plasma_global.errors import StateDomainError
from plasma_global.models.electrons import (
    ElectronState,
    LocalFieldClosure,
)
from plasma_global.models.kinetics import (
    ElectronKineticsResult,
)

if TYPE_CHECKING:
    from plasma_global.core.compiled import CompiledGlobalModel


@dataclass(frozen=True, slots=True)
class RuntimeElectronEvaluator:
    """Evaluate power coupling and the electron state seen by source terms."""

    model: CompiledGlobalModel

    def evaluate_power(
        self,
        time_s: float,
        segment: RecipeSegment,
        prepared: PreparedEvaluationState,
    ) -> PowerEvaluation:
        coordinator = self.model.power_coordinator
        if coordinator is None:
            raise RuntimeError("Compiled model requires a power coordinator")
        coupling = coordinator.evaluate(
            time_s=time_s,
            commands=segment.port_commands,
            electron_density_m3_by_zone=prepared.electron_density_by_zone,
            neutral_density_m3_by_zone=prepared.neutral_density_by_zone,
            mean_energy_eV_by_zone=prepared.mean_energy_for_power,
            kinetics_by_zone=self.model.electron_kinetics_by_zone,
        )
        electron_power = coupling.electron_power_W_by_zone
        gas_power = coupling.gas_power_W_by_zone
        reduced_field = coupling.reduced_field_Td_by_zone
        kinetics = coupling.kinetics_by_zone
        electron_states = self._evaluate_electron_states(
            prepared, reduced_field, kinetics
        )
        return PowerEvaluation(
            coupling=coupling,
            kinetics_by_zone=kinetics,
            electron_power_W_by_zone=electron_power,
            gas_power_W_by_zone=gas_power,
            reduced_field_Td_by_zone=reduced_field,
            electron_states=electron_states,
        )

    def _evaluate_electron_states(
        self,
        prepared: PreparedEvaluationState,
        reduced_field_by_zone: Mapping[str, float],
        kinetics_by_zone: Mapping[str, ElectronKineticsResult],
    ) -> dict[str, ElectronState]:
        states: dict[str, ElectronState] = {}
        for zone in self.model.zones:
            field_value = reduced_field_by_zone.get(zone.zone_id)
            kinetics = kinetics_by_zone.get(zone.zone_id)
            electron_density = prepared.electron_density_by_zone[zone.zone_id]
            closure = self.model.electron_closure
            if isinstance(closure, LocalFieldClosure):
                if kinetics is None:
                    raise StateDomainError(
                        f"Zone {zone.zone_id!r} local-field closure "
                        "needs its kinetics table"
                    )
                states[zone.zone_id] = closure.evaluate(
                    electron_density_m3=electron_density,
                    reduced_field_Td=field_value,
                    kinetics=kinetics,
                )
                continue
            prepared_state = prepared.electron_state_by_zone[zone.zone_id]
            if prepared_state is None:
                raise RuntimeError("Electron-energy state was not prepared")
            states[zone.zone_id] = (
                prepared_state
                if prepared_state.reduced_field_Td == field_value
                else replace(prepared_state, reduced_field_Td=field_value)
            )
        return states


__all__ = ["RuntimeElectronEvaluator"]
