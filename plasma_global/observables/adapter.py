from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from plasma_global.chemistry.models import E_CHARGE
from plasma_global.electrical.sheath import debye_length_m


PORT_DETAIL_TEXT_KEYS = {'backend', 'zone_id', 'role', 'file', 'surface_id', 'mode_label'}


def _observable_key_token(value: str) -> str:
    chars = []
    for ch in str(value):
        chars.append(ch if ch.isalnum() else '_')
    token = ''.join(chars).strip('_')
    while '__' in token:
        token = token.replace('__', '_')
    return token or 'unknown'


def _finite_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _add_power_port_observables(rec: dict[str, Any], power: Any) -> None:
    port_details = ((power.metadata or {}).get('port_details', {}) or {})
    for port_id, detail in port_details.items():
        if not isinstance(detail, dict):
            continue
        port_token = _observable_key_token(port_id)
        if port_id in power.port_power_W:
            rec[f'port_{port_token}_absorbed_power_W'] = float(power.port_power_W[port_id])
        for key, value in detail.items():
            if key in PORT_DETAIL_TEXT_KEYS:
                continue
            scalar = _finite_float_or_none(value)
            if scalar is None:
                continue
            metric_token = _observable_key_token(key)
            rec[f'port_{port_token}_{metric_token}'] = scalar


@dataclass
class ObservablesAdapter:
    system: Any

    def compute(self, t: np.ndarray, y: np.ndarray) -> list[dict[str, float]]:
        sys = self.system
        records: list[dict[str, float]] = []
        for i, time_s in enumerate(t):
            state = sys.project_state(y[:, i])
            step = sys.current_step(float(time_s))
            coupled = sys.electrical_adapter.evaluate(float(time_s), state, step)
            gas = coupled.gas
            surface_net_fluxes = sys.surface_core.surface_net_gas_fluxes(state, step, coupled)
            total_volume = sum(sys.chamber.zone_by_id[z].volume_m3 for z in sys.zone_ids)
            electron_count = sum(coupled.ne_by_zone[z] * sys.chamber.zone_by_id[z].volume_m3 for z in sys.zone_ids)
            electron_energy_total = sum(coupled.electron_energy[sys.zone_index[z]] * sys.chamber.zone_by_id[z].volume_m3 for z in sys.zone_ids)
            rec: dict[str, float | str] = {
                'time_s': float(time_s),
                'step_id': step.step_id,
                'self_bias_V': float(coupled.power.self_bias_V),
                'plasma_potential_V': float(coupled.power.plasma_potential_V),
                'total_absorbed_power_W': float(sum(coupled.power.absorbed_power_W_by_zone.values())),
                'electron_density_m3': float(electron_count / max(total_volume, 1.0e-30)),
                'mean_electron_energy_eV': float(electron_energy_total / max(electron_count, sys.floor_density * total_volume) / E_CHARGE),
            }
            _add_power_port_observables(rec, coupled.power)
            for zone_id in sys.zone_ids:
                z = sys.zone_index[zone_id]
                gas_row = gas[z]
                neg = sys.gas_core.negative_ion_density_from_state_row(gas_row)
                ne = max(coupled.ne_by_zone[zone_id], sys.floor_density)
                pressure = coupled.pressure_by_zone[zone_id]
                alpha = neg / ne
                rec[f'ne_{zone_id}_m3'] = float(coupled.ne_by_zone[zone_id])
                rec[f'mean_energy_{zone_id}_eV'] = float(coupled.mean_e_by_zone[zone_id])
                rec[f'pabs_{zone_id}_W'] = float(coupled.power.absorbed_power_W_by_zone.get(zone_id, 0.0))
                rec[f'Tg_{zone_id}_K'] = float(coupled.gas_temperature[sys.zone_index[zone_id]])
                rec[f'pressure_{zone_id}_Pa'] = float(pressure)
                rec[f'total_density_{zone_id}_m3'] = float(np.sum(np.clip(gas_row, 0.0, None)))
                rec[f'electronegativity_{zone_id}'] = float(alpha)
                rec[f'residence_time_{zone_id}_s'] = float(sys.zone_residence_time_s.get(zone_id, float('inf')))
                rec[f'debye_length_{zone_id}_m'] = float(debye_length_m(coupled.ne_by_zone[zone_id], coupled.mean_e_by_zone[zone_id]))
                rec[f'EoverN_{zone_id}_Td'] = float((coupled.power.metadata or {}).get('zone_reduced_field_Td', {}).get(zone_id, coupled.eedf_by_zone[zone_id].transport.get('effective_field_Td', 0.0)))
                rec[f'ion_loss_family_{zone_id}'] = str(sys.zone_ion_loss_family.get(zone_id, 'disabled'))
                rec[f'ion_loss_area_{zone_id}_m2'] = float(sys.zone_ion_loss_area.get(zone_id, 0.0))
                rec[f'ion_loss_h_factor_{zone_id}'] = float(sys.zone_ion_loss_h_factor.get(zone_id, 0.0))
                rec[f'ion_loss_characteristic_length_{zone_id}_m'] = float(sys.zone_ion_loss_characteristic_length_m.get(zone_id, 0.0))
                rec[f'ambipolar_loss_rate_{zone_id}_s'] = float(sys.zone_ambipolar_loss_rate_s.get(zone_id, 0.0))
                wall_loss = sys.gas_core.ion_wall_loss_diagnostics(zone_id, gas_row, coupled.mean_e_by_zone[zone_id])
                rec[f'ion_wall_loss_frequency_{zone_id}_s'] = float(wall_loss['frequency_s'])
                rec[f'ion_wall_loss_source_{zone_id}_m3_s'] = float(wall_loss['source_m3_s'])
                rec[f'ion_wall_flux_{zone_id}_m2_s'] = float(wall_loss['flux_m2_s'])

            for surface_id in sys.surface_ids:
                surface = sys.chamber.surface_by_id[surface_id]
                z = sys.zone_index[surface.zone_id]
                gas_row = gas[z]
                ied = ((coupled.power.metadata or {}).get('surface_ied', {}) or {}).get(surface_id, {})
                if ied:
                    rec[f'ion_flux_{surface_id}_m2_s'] = float(ied.get('ion_flux_m2_s', 0.0))
                    rec[f'mean_ion_energy_{surface_id}_eV'] = float(ied.get('mean_ion_energy_eV', 0.0))
                    rec[f'ied_width_{surface_id}_eV'] = float(ied.get('width_eV', 0.0))
                    rec[f'ied_collisionality_{surface_id}'] = float(ied.get('collisionality', 0.0))
                    rec[f'angle_spread_{surface_id}_deg'] = float(ied.get('angle_spread_deg', 0.0))
                    rec[f'sheath_voltage_{surface_id}_V'] = float(ied.get('sheath_voltage_V', 0.0))
                    rec[f'sheath_thickness_{surface_id}_m'] = float(ied.get('sheath_thickness_m', 0.0))
                    for ion_id, payload in (ied.get('ion_species', {}) or {}).items():
                        rec[f'ion_flux_{surface_id}_{ion_id}_m2_s'] = float(payload.get('ion_flux_m2_s', 0.0))
                        rec[f'mean_ion_energy_{surface_id}_{ion_id}_eV'] = float(payload.get('mean_ion_energy_eV', 0.0))
                        rec[f'ied_width_{surface_id}_{ion_id}_eV'] = float(payload.get('width_eV', 0.0))
                        rec[f'charge_fraction_{surface_id}_{ion_id}'] = float(payload.get('charge_fraction', 0.0))
                else:
                    rec[f'ion_flux_{surface_id}_m2_s'] = float(
                        sys.surface_core.surface_ion_flux_m2_s(surface_id, surface.zone_id, gas_row, coupled)
                    )
                site_metrics = sys.surface_core.surface_site_metrics(surface_id, state)
                rec[f'site_fill_{surface_id}'] = float(site_metrics['total_fraction'])
                rec[f'occupied_site_fraction_{surface_id}'] = float(site_metrics['occupied_fraction'])
                rec[f'free_site_fraction_{surface_id}'] = float(site_metrics['free_fraction'])

                radical_incident = 0.0
                halogen_atom_flux = 0.0
                carbon_atom_flux = 0.0
                oxygen_atom_flux = 0.0
                for local_idx in sys.radical_local_indices:
                    sp_id = sys.gas_species_ids[local_idx]
                    flux = sys.gas_core.incident_neutral_flux_m2_s(gas_row, float(coupled.gas_temperature[z]), local_idx)
                    if flux <= 0.0:
                        continue
                    rec[f'incident_flux_{surface_id}_{sp_id}_m2_s'] = float(flux)
                    radical_incident += flux
                    elements = sys.mechanism.species_by_id[sp_id].elements
                    halogen_atom_flux += sum(float(elements.get(el, 0.0)) for el in sys.halogen_elements) * flux
                    carbon_atom_flux += float(elements.get('C', 0.0)) * flux
                    oxygen_atom_flux += float(elements.get('O', 0.0)) * flux
                ion_flux = float(rec.get(f'ion_flux_{surface_id}_m2_s', 0.0) or 0.0)
                rec[f'radical_incident_flux_{surface_id}_m2_s'] = float(radical_incident)
                rec[f'halogen_atom_flux_{surface_id}_m2_s'] = float(halogen_atom_flux)
                rec[f'carbon_atom_flux_{surface_id}_m2_s'] = float(carbon_atom_flux)
                rec[f'oxygen_atom_flux_{surface_id}_m2_s'] = float(oxygen_atom_flux)
                rec[f'radical_to_ion_flux_ratio_{surface_id}'] = float(radical_incident / max(ion_flux, 1.0e-30)) if ion_flux > 0.0 else 0.0
                ratio_hc = float(halogen_atom_flux / max(carbon_atom_flux, 1.0e-30)) if carbon_atom_flux > 0.0 else 0.0
                rec[f'halogen_to_C_radical_flux_ratio_{surface_id}'] = ratio_hc
                rec[f'F_to_C_radical_flux_ratio_{surface_id}'] = ratio_hc
                rec[f'etch_deposition_balance_{surface_id}'] = float(halogen_atom_flux - carbon_atom_flux)

                for sp_id, flux in surface_net_fluxes.get(surface_id, {}).items():
                    rec[f'net_flux_{surface_id}_{sp_id}_m2_s'] = float(flux)
                rec[f'net_total_flux_{surface_id}_m2_s'] = float(sum(surface_net_fluxes.get(surface_id, {}).values()))

            for surface_id, idx in sys.state_layout.film_index.items():
                rec[f'film_{surface_id}_m'] = float(state[idx])
            records.append(rec)
        return records
