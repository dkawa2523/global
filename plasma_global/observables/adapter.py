from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from plasma_global.chemistry.models import E_CHARGE
from plasma_global.diagnostics.budgets import flatten_reaction_budget


@dataclass
class ObservablesAdapter:
    system: Any

    def compute(self, t: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
        sys = self.system
        records: list[dict[str, Any]] = []
        for i, time_s in enumerate(t):
            state = sys.project_state(y[:, i])
            step = sys.current_step(float(time_s))
            coupled = sys.electrical_adapter.evaluate(float(time_s), state, step)
            gas = coupled.gas
            total_volume = sum(sys.chamber.zone_by_id[z].volume_m3 for z in sys.zone_ids)
            electron_count = sum(coupled.ne_by_zone[z] * sys.chamber.zone_by_id[z].volume_m3 for z in sys.zone_ids)
            electron_energy_total = sum(coupled.electron_energy[sys.zone_index[z]] * sys.chamber.zone_by_id[z].volume_m3 for z in sys.zone_ids)
            rec: dict[str, Any] = {
                'time_s': float(time_s),
                'step_id': step.step_id,
                'self_bias_V': float(coupled.power.self_bias_V),
                'plasma_potential_V': float(coupled.power.plasma_potential_V),
                'total_absorbed_power_W': float(sum(coupled.power.absorbed_power_W_by_zone.values())),
                'electron_density_m3': float(electron_count / max(total_volume, 1.0e-30)),
                'mean_electron_energy_eV': float(electron_energy_total / max(electron_count, sys.floor_density * total_volume) / E_CHARGE),
            }
            for zone_id in sys.zone_ids:
                z = sys.zone_index[zone_id]
                eedf = coupled.eedf_by_zone[zone_id]
                rec[f'ne_{zone_id}_m3'] = float(coupled.ne_by_zone[zone_id])
                rec[f'mean_energy_{zone_id}_eV'] = float(coupled.mean_e_by_zone[zone_id])
                rec[f'pabs_{zone_id}_W'] = float(coupled.power.absorbed_power_W_by_zone.get(zone_id, 0.0))
                rec[f'Tg_{zone_id}_K'] = float(coupled.gas_temperature[sys.zone_index[zone_id]])
                rec[f'pressure_{zone_id}_Pa'] = float(coupled.pressure_by_zone[zone_id])
                rec[f'EoverN_{zone_id}_Td'] = float(
                    coupled.power.zone_reduced_field_Td.get(
                        zone_id,
                        coupled.eedf_by_zone[zone_id].transport.effective_field_Td,
                    )
                )
                if eedf.diagnostics is not None:
                    rec.update(eedf.diagnostics.to_observable_fields(zone_id))

            for port_id, snapshot in coupled.power.port_observables.items():
                rec.update(snapshot.to_observable_fields(port_id))

            for surface_id in sys.surface_ids:
                surface = sys.chamber.surface_by_id[surface_id]
                z = sys.zone_index[surface.zone_id]
                gas_row = gas[z]
                rec[f'ion_flux_{surface_id}_m2_s'] = float(
                    sys.surface_core.surface_ion_flux_m2_s(surface_id, surface.zone_id, gas_row, coupled)
                )

            for surface_id, idx in sys.state_layout.film_index.items():
                rec[f'film_{surface_id}_m'] = float(state[idx])
            if bool(getattr(sys.run_config.outputs.diagnostics, 'budgets', False)):
                rec.update(flatten_reaction_budget(sys.gas_core.reaction_source_loss_budget(coupled)))
            records.append(rec)
        return records
