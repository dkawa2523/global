from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from plasma_global.eedf.base import EEDFRequest
from plasma_global.electrical.base import PowerRequest
from plasma_global.physics.types import CoupledPlasmaEvaluation


@dataclass
class ElectricalCouplingAdapter:
    system: Any
    last_power: Any = None
    last_zone_mean_energy_eV: dict[str, float] = field(default_factory=dict)
    last_zone_electron_density_m3: dict[str, float] = field(default_factory=dict)
    last_zone_electron_mobility_m2_V_s: dict[str, float] = field(default_factory=dict)

    def _needs_transport_coupling(self, step: Any) -> bool:
        sys = self.system
        for port_id, step_cfg in step.power_ports.items():
            port = sys.chamber.power_port_by_id.get(port_id)
            cfg = dict(getattr(port, 'parameters', {}) or {})
            cfg.update(step_cfg or {})
            if str(cfg.get('mobility_source', '')).lower() in {'table', 'transport', 'eedf'}:
                return True
        return False

    def _transport_coupling_iterations(self, step: Any) -> int:
        count = 1
        sys = self.system
        for port_id, step_cfg in step.power_ports.items():
            port = sys.chamber.power_port_by_id.get(port_id)
            cfg = dict(getattr(port, 'parameters', {}) or {})
            cfg.update(step_cfg or {})
            if str(cfg.get('mobility_source', '')).lower() in {'table', 'transport', 'eedf'}:
                count = max(count, int(cfg.get('transport_coupling_iterations', 4)))
        return max(count, 1)

    def evaluate(self, time_s: float, y: np.ndarray, step: Any | None = None) -> CoupledPlasmaEvaluation:
        sys = self.system
        step = step or sys.current_step(time_s)
        gas = y[sys.state_layout.slice('gas_densities')].reshape(sys.n_zones, sys.n_gas_species)
        We = y[sys.state_layout.slice('electron_energy')]
        if 'gas_temperature' in sys.state_layout.slices:
            Tg = y[sys.state_layout.slice('gas_temperature')]
        else:
            Tg = np.array([sys.chamber.zone_by_id[z].gas_temperature_K for z in sys.zone_ids], dtype=float)

        ne_by_zone, mean_e_by_zone, pos_by_zone, ion_mass_by_zone = sys.gas_core.zone_state_meta(gas, We)
        prescribed_ne = sys.prescribed_electron_density_by_zone(time_s)
        if prescribed_ne is not None:
            for zone_id, ne in prescribed_ne.items():
                z = sys.zone_index[zone_id]
                ne_by_zone[zone_id] = max(float(ne), sys.floor_density)
                mean_e_by_zone[zone_id] = sys.gas_core.mean_energy_eV(float(We[z]), ne_by_zone[zone_id])
        pressure_by_zone: dict[str, float] = {}
        ion_species_by_zone: dict[str, dict[str, dict[str, float]]] = {}
        total_density_by_zone: dict[str, float] = {}
        for z_idx, zone_id in enumerate(sys.zone_ids):
            pressure_by_zone[zone_id] = sys.gas_core.zone_pressure_from_state_row(gas[z_idx], float(Tg[z_idx]))
            total_density_by_zone[zone_id] = max(float(np.sum(np.clip(gas[z_idx], 0.0, None))), sys.floor_density)
            ion_species_by_zone[zone_id] = sys.gas_core.ion_species_payload_from_state_row(gas[z_idx])

        metadata: dict[str, Any] = {
            'zone_electron_density_m3': ne_by_zone,
            'zone_mean_energy_eV': mean_e_by_zone,
            'zone_positive_ion_density_m3': pos_by_zone,
            'zone_dominant_ion_mass_kg': ion_mass_by_zone,
            'zone_pressure_Pa': pressure_by_zone,
            'zone_gas_temperature_K': {zone_id: float(Tg[sys.zone_index[zone_id]]) for zone_id in sys.zone_ids},
            'zone_total_density_m3': total_density_by_zone,
            'zone_positive_ion_species': ion_species_by_zone,
            'electron_density_closure': sys.electron_density_closure,
        }
        if prescribed_ne is not None and sys.prescribed_electron_profile is not None:
            metadata['prescribed_electron_profile'] = sys.prescribed_electron_profile.provenance()
        if self.last_zone_electron_mobility_m2_V_s:
            metadata['zone_electron_mobility_m2_V_s'] = dict(self.last_zone_electron_mobility_m2_V_s)

        needs_transport_coupling = self._needs_transport_coupling(step)
        iterations = self._transport_coupling_iterations(step) if needs_transport_coupling else 1
        power = None
        eedf_by_zone = {}
        mobility_by_zone: dict[str, float] = {}

        for _ in range(iterations):
            power = sys.electrical_backend.evaluate(
                PowerRequest(
                    time_s=time_s,
                    state_vector=y,
                    recipe_step=step,
                    chamber=sys.chamber,
                    metadata=metadata,
                )
            )

            red_field_map = (power.metadata or {}).get('zone_reduced_field_Td', {})
            eedf_by_zone = {}
            mobility_by_zone = {}
            for z_idx, zone_id in enumerate(sys.zone_ids):
                req = EEDFRequest(
                    time_s=time_s,
                    zone_id=zone_id,
                    composition={sp_id: gas[z_idx, sys.gas_species_index[sp_id]] for sp_id in sys.gas_species_ids},
                    electron_density_m3=ne_by_zone[zone_id],
                    mean_energy_eV=mean_e_by_zone[zone_id],
                    reduced_field_Td=float(red_field_map[zone_id]) if zone_id in red_field_map else None,
                    gas_temperature_K=float(Tg[z_idx]),
                    pressure_Pa=pressure_by_zone[zone_id],
                    metadata={'absorbed_power_W': power.absorbed_power_W_by_zone.get(zone_id, 0.0)},
                )
                eedf = sys.eedf_backend.evaluate(req)
                eedf_by_zone[zone_id] = eedf
                mu = eedf.transport.get('mobility_m2_V_s')
                if mu is not None:
                    mobility_by_zone[zone_id] = float(mu)

            if not needs_transport_coupling or not mobility_by_zone:
                break
            previous = metadata.get('zone_electron_mobility_m2_V_s') or {}
            metadata['zone_electron_mobility_m2_V_s'] = mobility_by_zone
            converged = all(
                abs(float(previous.get(zone_id, mu)) - mu) <= max(abs(mu), 1.0e-30) * 1.0e-3
                for zone_id, mu in mobility_by_zone.items()
            )
            if converged:
                break

        assert power is not None
        if needs_transport_coupling and mobility_by_zone:
            power = sys.electrical_backend.evaluate(
                PowerRequest(
                    time_s=time_s,
                    state_vector=y,
                    recipe_step=step,
                    chamber=sys.chamber,
                    metadata=metadata,
                )
            )
            red_field_map = (power.metadata or {}).get('zone_reduced_field_Td', {})
            eedf_by_zone = {}
            for z_idx, zone_id in enumerate(sys.zone_ids):
                req = EEDFRequest(
                    time_s=time_s,
                    zone_id=zone_id,
                    composition={sp_id: gas[z_idx, sys.gas_species_index[sp_id]] for sp_id in sys.gas_species_ids},
                    electron_density_m3=ne_by_zone[zone_id],
                    mean_energy_eV=mean_e_by_zone[zone_id],
                    reduced_field_Td=float(red_field_map[zone_id]) if zone_id in red_field_map else None,
                    gas_temperature_K=float(Tg[z_idx]),
                    pressure_Pa=pressure_by_zone[zone_id],
                    metadata={'absorbed_power_W': power.absorbed_power_W_by_zone.get(zone_id, 0.0)},
                )
                eedf = sys.eedf_backend.evaluate(req)
                eedf_by_zone[zone_id] = eedf
                mu = eedf.transport.get('mobility_m2_V_s')
                if mu is not None:
                    mobility_by_zone[zone_id] = float(mu)

        self.last_zone_electron_mobility_m2_V_s = mobility_by_zone

        self.last_power = power
        self.last_zone_electron_density_m3 = ne_by_zone
        self.last_zone_mean_energy_eV = mean_e_by_zone
        return CoupledPlasmaEvaluation(
            gas=gas,
            electron_energy=We,
            gas_temperature=Tg,
            power=power,
            eedf_by_zone=eedf_by_zone,
            ne_by_zone=ne_by_zone,
            mean_e_by_zone=mean_e_by_zone,
            pos_by_zone=pos_by_zone,
            ion_mass_by_zone=ion_mass_by_zone,
            pressure_by_zone=pressure_by_zone,
            ion_species_by_zone=ion_species_by_zone,
            total_density_by_zone=total_density_by_zone,
        )
