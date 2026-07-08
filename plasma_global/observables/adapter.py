from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from plasma_global.chemistry.models import E_CHARGE
from plasma_global.observables.budgets import reaction_budget_observable_fields
from plasma_global.observables.fields import electrical_port_observable_fields, extra_state_observable_fields, table_lookup_observable_fields


def compute_observables(system: Any, t: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    return _ObservableComputer(system).compute(t, y)


def compute_observable_record(system: Any, time_s: float, state: np.ndarray) -> dict[str, Any]:
    return _ObservableComputer(system).record(time_s, state)


@dataclass
class _ObservableComputer:
    system: Any

    def compute(self, t: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for i, time_s in enumerate(t):
            records.append(self.record(float(time_s), y[:, i]))
        return records

    def record(self, time_s: float, raw_state: np.ndarray) -> dict[str, Any]:
        system = self.system
        state = system.project_state(raw_state)
        step = system.current_step(time_s)
        coupled = system.electrical_adapter.evaluate(time_s, state, step)
        rec = self._global_fields(time_s, step, coupled)
        for zone_id in system.zone_ids:
            rec.update(self._zone_fields(zone_id, coupled))
        for port_id, snapshot in coupled.power.port_observables.items():
            rec.update(electrical_port_observable_fields(snapshot, port_id))
        rec.update(self._surface_fields(state, coupled))
        rec.update(extra_state_observable_fields(system, state))
        rec.update(self._budget_fields(coupled))
        return rec

    def _budget_fields(self, coupled: Any) -> dict[str, float]:
        budgets = getattr(self.system.run_config.outputs, 'budgets', None)
        if not bool(getattr(budgets, 'enabled', False)):
            return {}
        return reaction_budget_observable_fields(self.system.gas_core, coupled)

    def _global_fields(self, time_s: float, step: Any, coupled: Any) -> dict[str, Any]:
        system = self.system
        total_volume = sum(system.chamber.zone_by_id[z].volume_m3 for z in system.zone_ids)
        electron_count = sum(coupled.ne_by_zone[z] * system.chamber.zone_by_id[z].volume_m3 for z in system.zone_ids)
        electron_energy_total = sum(
            coupled.electron_energy[system.zone_index[z]] * system.chamber.zone_by_id[z].volume_m3
            for z in system.zone_ids
        )
        return {
            'time_s': float(time_s),
            'step_id': step.step_id,
            'self_bias_V': float(coupled.power.self_bias_V),
            'plasma_potential_V': float(coupled.power.plasma_potential_V),
            'total_absorbed_power_W': float(sum(coupled.power.absorbed_power_W_by_zone.values())),
            'electron_density_m3': float(electron_count / max(total_volume, 1.0e-30)),
            'mean_electron_energy_eV': float(electron_energy_total / max(electron_count, system.floor_density * total_volume) / E_CHARGE),
        }

    def _zone_fields(self, zone_id: str, coupled: Any) -> dict[str, Any]:
        system = self.system
        z = system.zone_index[zone_id]
        eedf = coupled.eedf_by_zone[zone_id]
        fields: dict[str, Any] = {
            f'ne_{zone_id}_m3': float(coupled.ne_by_zone[zone_id]),
            f'mean_energy_{zone_id}_eV': float(coupled.mean_e_by_zone[zone_id]),
            f'pabs_{zone_id}_W': float(coupled.power.absorbed_power_W_by_zone.get(zone_id, 0.0)),
            f'Tg_{zone_id}_K': float(coupled.gas_temperature[z]),
            f'pressure_{zone_id}_Pa': float(coupled.pressure_by_zone[zone_id]),
            f'EoverN_{zone_id}_Td': float(
                coupled.power.zone_reduced_field_Td.get(
                    zone_id,
                    eedf.transport.effective_field_Td,
                )
            ),
        }
        lookup_info = eedf.metadata.get('table_lookup')
        if lookup_info is not None:
            fields.update(table_lookup_observable_fields(lookup_info, zone_id))
        return fields

    def _surface_fields(self, state: np.ndarray, coupled: Any) -> dict[str, Any]:
        system = self.system
        fields: dict[str, Any] = {}
        for surface_id in system.surface_ids:
            surface = system.chamber.surface_by_id[surface_id]
            z = system.zone_index[surface.zone_id]
            fields[f'ion_flux_{surface_id}_m2_s'] = float(
                system.surface_core.surface_ion_flux_m2_s(surface_id, surface.zone_id, coupled.gas[z], coupled)
            )
        for surface_id, idx in system.state_layout.film_index.items():
            fields[f'film_{surface_id}_m'] = float(state[idx])
        return fields
