from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from plasma_global.eedf.base import EEDFRequest
from plasma_global.electrical.base import PowerRequest, PowerResult, ZoneElectricalState
from plasma_global.physics.gas_closure import mean_energy_eV, zone_pressure_from_state_row, zone_state_meta
from plasma_global.physics.types import CoupledPlasmaEvaluation


@dataclass
class _ElectricalStateView:
    gas: np.ndarray
    electron_energy: np.ndarray
    gas_temperature: np.ndarray
    zone_state: dict[str, ZoneElectricalState]
    ne_by_zone: dict[str, float]
    mean_e_by_zone: dict[str, float]
    pos_by_zone: dict[str, float]
    ion_mass_by_zone: dict[str, float]
    pressure_by_zone: dict[str, float]
    total_density_by_zone: dict[str, float]


@dataclass
class ElectricalCouplingAdapter:
    system: Any
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

    def _evaluate_eedf_by_zone(
        self,
        *,
        time_s: float,
        gas: np.ndarray,
        Tg: np.ndarray,
        ne_by_zone: dict[str, float],
        mean_e_by_zone: dict[str, float],
        pressure_by_zone: dict[str, float],
        power: PowerResult,
    ) -> tuple[dict[str, Any], dict[str, float]]:
        sys = self.system
        red_field_map = power.zone_reduced_field_Td
        eedf_by_zone = {}
        mobility_by_zone: dict[str, float] = {}
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
            )
            eedf = sys.eedf_backend.evaluate(req)
            eedf_by_zone[zone_id] = eedf
            mobility_by_zone[zone_id] = float(eedf.transport.mobility_m2_V_s)
        return eedf_by_zone, mobility_by_zone

    def _state_view(self, time_s: float, y: np.ndarray) -> _ElectricalStateView:
        sys = self.system
        gas = y[sys.state_layout.slice('gas_densities')].reshape(sys.n_zones, sys.n_gas_species)
        We = y[sys.state_layout.slice('electron_energy')]
        if 'gas_temperature' in sys.state_layout.slices:
            Tg = y[sys.state_layout.slice('gas_temperature')]
        else:
            Tg = np.array([sys.chamber.zone_by_id[z].gas_temperature_K for z in sys.zone_ids], dtype=float)

        ne_by_zone, mean_e_by_zone, pos_by_zone, ion_mass_by_zone = zone_state_meta(sys, gas, We)
        prescribed_ne = sys.prescribed_electron_density_by_zone(time_s)
        if prescribed_ne is not None:
            for zone_id, ne in prescribed_ne.items():
                z = sys.zone_index[zone_id]
                ne_by_zone[zone_id] = max(float(ne), sys.floor_density)
                mean_e_by_zone[zone_id] = mean_energy_eV(sys, float(We[z]), ne_by_zone[zone_id])
        pressure_by_zone: dict[str, float] = {}
        total_density_by_zone: dict[str, float] = {}
        for z_idx, zone_id in enumerate(sys.zone_ids):
            pressure_by_zone[zone_id] = zone_pressure_from_state_row(sys, gas[z_idx], float(Tg[z_idx]))
            total_density_by_zone[zone_id] = max(float(np.sum(np.clip(gas[z_idx], 0.0, None))), sys.floor_density)

        zone_state: dict[str, ZoneElectricalState] = {
            zone_id: ZoneElectricalState(
                electron_density_m3=ne_by_zone[zone_id],
                mean_energy_eV=mean_e_by_zone[zone_id],
                positive_ion_density_m3=pos_by_zone[zone_id],
                dominant_ion_mass_kg=ion_mass_by_zone[zone_id],
                pressure_Pa=pressure_by_zone[zone_id],
                gas_temperature_K=float(Tg[sys.zone_index[zone_id]]),
                total_density_m3=total_density_by_zone[zone_id],
                electron_mobility_m2_V_s=self.last_zone_electron_mobility_m2_V_s.get(zone_id),
            )
            for zone_id in sys.zone_ids
        }

        return _ElectricalStateView(
            gas=gas,
            electron_energy=We,
            gas_temperature=Tg,
            zone_state=zone_state,
            ne_by_zone=ne_by_zone,
            mean_e_by_zone=mean_e_by_zone,
            pos_by_zone=pos_by_zone,
            ion_mass_by_zone=ion_mass_by_zone,
            pressure_by_zone=pressure_by_zone,
            total_density_by_zone=total_density_by_zone,
        )

    def _evaluate_power(
        self,
        *,
        time_s: float,
        y: np.ndarray,
        step: Any,
        zone_state: dict[str, ZoneElectricalState],
    ) -> PowerResult:
        sys = self.system
        return sys.electrical_backend.evaluate(
            PowerRequest(
                time_s=time_s,
                state_vector=y,
                recipe_step=step,
                chamber=sys.chamber,
                zone_state=zone_state,
            )
        )

    def evaluate(self, time_s: float, y: np.ndarray, step: Any | None = None) -> CoupledPlasmaEvaluation:
        sys = self.system
        step = step or sys.current_step(time_s)
        state = self._state_view(time_s, y)

        needs_transport_coupling = self._needs_transport_coupling(step)
        iterations = self._transport_coupling_iterations(step) if needs_transport_coupling else 1
        power = None
        eedf_by_zone = {}
        mobility_by_zone: dict[str, float] = {}

        for _ in range(iterations):
            power = self._evaluate_power(time_s=time_s, y=y, step=step, zone_state=state.zone_state)

            eedf_by_zone, mobility_by_zone = self._evaluate_eedf_by_zone(
                time_s=time_s,
                gas=state.gas,
                Tg=state.gas_temperature,
                ne_by_zone=state.ne_by_zone,
                mean_e_by_zone=state.mean_e_by_zone,
                pressure_by_zone=state.pressure_by_zone,
                power=power,
            )

            if not needs_transport_coupling or not mobility_by_zone:
                break
            previous = {zone_id: zone.electron_mobility_m2_V_s for zone_id, zone in state.zone_state.items()}
            for zone_id, mobility in mobility_by_zone.items():
                state.zone_state[zone_id].electron_mobility_m2_V_s = mobility
            converged = all(
                previous.get(zone_id) is not None
                and abs(float(previous[zone_id]) - mu) <= max(abs(mu), 1.0e-30) * 1.0e-3
                for zone_id, mu in mobility_by_zone.items()
            )
            if converged:
                break

        assert power is not None
        if needs_transport_coupling and mobility_by_zone:
            power = self._evaluate_power(time_s=time_s, y=y, step=step, zone_state=state.zone_state)
            eedf_by_zone, mobility_by_zone = self._evaluate_eedf_by_zone(
                time_s=time_s,
                gas=state.gas,
                Tg=state.gas_temperature,
                ne_by_zone=state.ne_by_zone,
                mean_e_by_zone=state.mean_e_by_zone,
                pressure_by_zone=state.pressure_by_zone,
                power=power,
            )

        self.last_zone_electron_mobility_m2_V_s = mobility_by_zone

        return CoupledPlasmaEvaluation(
            gas=state.gas,
            electron_energy=state.electron_energy,
            gas_temperature=state.gas_temperature,
            power=power,
            eedf_by_zone=eedf_by_zone,
            ne_by_zone=state.ne_by_zone,
            mean_e_by_zone=state.mean_e_by_zone,
            pos_by_zone=state.pos_by_zone,
            ion_mass_by_zone=state.ion_mass_by_zone,
            pressure_by_zone=state.pressure_by_zone,
            total_density_by_zone=state.total_density_by_zone,
        )
