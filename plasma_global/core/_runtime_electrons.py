"""Resolve runtime power, electron kinetics, and electron closure state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from plasma_global.core._runtime_assembly import (
    PowerEvaluation,
    PreparedEvaluationState,
)
from plasma_global.core.domain import RecipeSegment
from plasma_global.errors import StateDomainError
from plasma_global.models.electrons import (
    ELEMENTARY_CHARGE_C,
    ElectronEnergyClosure,
    ElectronState,
)
from plasma_global.models.kinetics import (
    ElectronKineticsResult,
    TabulatedElectronKinetics,
)

if TYPE_CHECKING:
    from plasma_global.core.compiled import CompiledGlobalModel


def _evaluate_uncoordinated_table(
    table: TabulatedElectronKinetics,
    *,
    zone_id: str,
    electron_density_m3: float,
    neutral_density_m3: float,
    mean_energy_eV: float | None,
    reduced_field_Td: float | None,
) -> tuple[ElectronKineticsResult | None, float | None]:
    """Evaluate one table without a power port and report any inferred field."""

    if table.lookup == "mean_energy":
        if mean_energy_eV is None:
            raise StateDomainError(
                f"Zone {zone_id!r} has no mean energy for table lookup"
            )
        if electron_density_m3 == 0.0 and mean_energy_eV == 0.0:
            # No electron property or electron-driven rate is defined or needed.
            return None, None
        result = table.evaluate(
            mean_energy_eV=mean_energy_eV,
            neutral_density_m3=neutral_density_m3,
        )
        return result, result.effective_field_Td
    if reduced_field_Td is None:
        raise StateDomainError(
            f"Zone {zone_id!r} has no E/N for local-field table lookup"
        )
    result = (
        table.zero_field_result(neutral_density_m3=neutral_density_m3)
        if reduced_field_Td == 0.0 and table.axis[0] > 0.0
        else table.evaluate(
            reduced_field_Td=reduced_field_Td,
            neutral_density_m3=neutral_density_m3,
        )
    )
    return result, None


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
        if self.model.power_coordinator is None:
            coupling = None
            electron_power = segment.absorbed_power_W_by_zone
            gas_power: Mapping[str, float] = MappingProxyType({})
            reduced_field = dict(segment.reduced_field_Td_by_zone)
            kinetics = self._uncoordinated_kinetics(prepared, reduced_field)
        else:
            field_models: dict[str, Any] = {}
            field_model = self.model.electron_closure.mean_energy_from_field
            if callable(field_model):
                field_models = {zone.zone_id: field_model for zone in self.model.zones}
            coupling = self.model.power_coordinator.evaluate(
                time_s=time_s,
                commands=segment.port_commands,
                electron_density_m3_by_zone=prepared.electron_density_by_zone,
                neutral_density_m3_by_zone=prepared.neutral_density_by_zone,
                mean_energy_eV_by_zone=prepared.mean_energy_for_power,
                prescribed_electron_power_W_by_zone=segment.absorbed_power_W_by_zone,
                prescribed_reduced_field_Td_by_zone=segment.reduced_field_Td_by_zone,
                kinetics_by_zone=self.model.electron_kinetics_by_zone,
                mean_energy_from_field_by_zone=field_models,
            )
            electron_power = coupling.electron_power_W_by_zone
            gas_power = coupling.gas_power_W_by_zone
            reduced_field = dict(coupling.reduced_field_Td_by_zone)
            kinetics = dict(coupling.kinetics_by_zone)
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

    def _uncoordinated_kinetics(
        self,
        prepared: PreparedEvaluationState,
        reduced_field_by_zone: dict[str, float],
    ) -> dict[str, ElectronKineticsResult]:
        results: dict[str, ElectronKineticsResult] = {}
        for zone in self.model.zones:
            table = self.model.electron_kinetics_by_zone.get(zone.zone_id)
            if table is None:
                continue
            result, inferred_field = _evaluate_uncoordinated_table(
                table,
                zone_id=zone.zone_id,
                electron_density_m3=prepared.electron_density_by_zone[zone.zone_id],
                neutral_density_m3=prepared.neutral_density_by_zone[zone.zone_id],
                mean_energy_eV=prepared.mean_energy_for_power[zone.zone_id],
                reduced_field_Td=reduced_field_by_zone.get(zone.zone_id),
            )
            if result is None:
                continue
            if inferred_field is not None:
                reduced_field_by_zone.setdefault(zone.zone_id, inferred_field)
            results[zone.zone_id] = result
        return results

    def _evaluate_electron_states(
        self,
        prepared: PreparedEvaluationState,
        reduced_field_by_zone: Mapping[str, float],
        kinetics_by_zone: Mapping[str, ElectronKineticsResult],
    ) -> dict[str, ElectronState]:
        states: dict[str, ElectronState] = {}
        for zone_index, zone in enumerate(self.model.zones):
            field_value = reduced_field_by_zone.get(zone.zone_id)
            kinetics = kinetics_by_zone.get(zone.zone_id)
            electron_density = prepared.electron_density_by_zone[zone.zone_id]
            if (
                self.model.electron_closure.mode == "local_field"
                and kinetics is not None
            ):
                states[zone.zone_id] = self._local_field_electron_state(
                    electron_density, field_value, kinetics
                )
                continue
            prepared_state = prepared.electron_state_by_zone[zone.zone_id]
            if (
                type(self.model.electron_closure) is ElectronEnergyClosure
                and prepared_state is not None
            ):
                states[zone.zone_id] = (
                    prepared_state
                    if prepared_state.reduced_field_Td == field_value
                    else replace(prepared_state, reduced_field_Td=field_value)
                )
                continue
            energy_density = (
                None
                if prepared.closure_electron_energy_by_zone is None
                else float(prepared.closure_electron_energy_by_zone[zone_index])
            )
            states[zone.zone_id] = self.model.electron_closure.evaluate(
                net_heavy_charge_density_m3=prepared.net_charge_by_zone[zone.zone_id],
                energy_density_J_m3=energy_density,
                reduced_field_Td=field_value,
                electron_density_m3=electron_density,
            )
        return states

    @staticmethod
    def _local_field_electron_state(
        electron_density: float,
        field_value: float | None,
        kinetics: ElectronKineticsResult,
    ) -> ElectronState:
        mean_energy = kinetics.mean_energy_eV
        return ElectronState(
            density_m3=electron_density,
            mean_energy_eV=mean_energy,
            temperature_eV=kinetics.electron_temperature_eV,
            energy_density_J_m3=(electron_density * ELEMENTARY_CHARGE_C * mean_energy),
            reduced_field_Td=(
                field_value if field_value is not None else kinetics.effective_field_Td
            ),
        )


__all__ = ["RuntimeElectronEvaluator"]
