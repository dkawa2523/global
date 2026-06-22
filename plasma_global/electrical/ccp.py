from __future__ import annotations

import math
from typing import Any

from plasma_global.electrical.base import PowerRequest, PowerResult, SurfaceIED, ZoneElectricalState
from plasma_global.electrical.circuit_models import waveform_multiplier
from plasma_global.electrical.direct_power import DirectPowerBackend
from plasma_global.electrical.sheath import build_surface_ied, debye_length_m

EPS0 = 8.8541878128e-12


class CCPBackend(DirectPowerBackend):
    def prepare(self, chamber: Any, recipe: Any, run_config: Any, resolved_paths: Any) -> None:
        super().prepare(chamber=chamber, recipe=recipe, run_config=run_config, resolved_paths=resolved_paths)
        self._zone_ground_area: dict[str, float] = {}
        for zone in chamber.zones:
            self._zone_ground_area[zone.zone_id] = sum(s.area_m2 for s in chamber.surfaces_by_zone.get(zone.zone_id, []))

    def _zone_meta(self, request: PowerRequest, zone_id: str) -> ZoneElectricalState:
        zone = self.chamber.zone_by_id[zone_id]
        return request.zone_state.get(
            zone_id,
            ZoneElectricalState(
                electron_density_m3=1.0e15,
                mean_energy_eV=3.0,
                positive_ion_density_m3=1.0e15,
                dominant_ion_mass_kg=6.63e-26,
                pressure_Pa=zone.pressure_Pa,
                gas_temperature_K=zone.gas_temperature_K,
                total_density_m3=0.0,
            ),
        )

    def _surface_area_for_port(self, port_id: str) -> tuple[str | None, float, float]:
        port = self.chamber.power_port_by_id[port_id]
        target = port.coupling_target
        if target in self.chamber.surface_by_id:
            surf = self.chamber.surface_by_id[target]
            total = self._zone_ground_area.get(surf.zone_id, surf.area_m2)
            return surf.surface_id, surf.area_m2, max(total - surf.area_m2, 1.0e-6)
        zone_id = port.zone_id
        zone_surfaces = self.chamber.surfaces_by_zone.get(zone_id, [])
        if not zone_surfaces:
            return None, 0.03, 0.10
        powered = zone_surfaces[0].area_m2
        grounded = max(sum(s.area_m2 for s in zone_surfaces) - powered, 1.0e-6)
        return None, powered, grounded

    def _bulk_resistance_ohm(self, ne_m3: float, frequency_Hz: float, zone_volume_m3: float, area_m2: float) -> float:
        ne_norm = max(ne_m3 / 1.0e16, 1.0e-6)
        f_norm = max(frequency_Hz / 13.56e6, 1.0e-3)
        geom = max(zone_volume_m3 / max(area_m2, 1.0e-6), 1.0e-4)
        return 5.0 * geom / math.sqrt(ne_norm) * math.sqrt(1.0 / f_norm)

    def _sheath_capacitance_F(self, area_m2: float, ne_m3: float, te_eV: float, voltage_V: float) -> float:
        ld = debye_length_m(ne_m3, te_eV)
        s = max(4.0 * ld * math.sqrt(1.0 + max(voltage_V, 1.0) / max(te_eV, 0.1)), 5.0e-5)
        return EPS0 * max(area_m2, 1.0e-8) / s

    def _bias_port_result(self, request: PowerRequest, port_id: str, cfg: dict[str, Any]) -> dict[str, Any]:
        port = self.chamber.power_port_by_id[port_id]
        zone_id = cfg.get('zone_id') or port.zone_id
        zone = self.chamber.zone_by_id[zone_id]
        state = self._zone_meta(request, zone_id)
        ne = state.electron_density_m3
        te = state.mean_energy_eV
        surface_id, area_p, area_g = self._surface_area_for_port(port_id)
        area_ratio = max(area_g / max(area_p, 1.0e-8), 1.0)
        frequency_Hz = float(cfg.get('frequency_Hz') or cfg.get('carrier_frequency_Hz') or port.parameters.get('frequency_Hz') or port.parameters.get('carrier_frequency_Hz') or 2.0e6)
        mult = waveform_multiplier(request.time_s, cfg)
        mode = str(cfg.get('mode') or port.parameters.get('control_mode') or 'absorbed_power').lower()
        Rb = self._bulk_resistance_ohm(ne, frequency_Hz, zone.volume_m3, area_p)
        if 'voltage' in mode:
            vrms = float(cfg.get('value_V', cfg.get('value', 0.0))) * mult
            delivered = vrms * vrms / max(Rb, 1.0e-6)
        else:
            delivered = float(cfg.get('value_W', cfg.get('value', 0.0))) * mult
            vrms = math.sqrt(max(delivered, 0.0) * max(Rb, 1.0e-9))
        Csp = self._sheath_capacitance_F(area_p, ne, te, max(vrms, 1.0))
        Csg = self._sheath_capacitance_F(area_g, ne, te, max(vrms, 1.0))
        Xs = 0.0
        if frequency_Hz > 0.0:
            omega = 2.0 * math.pi * frequency_Hz
            Xs = abs(1.0 / max(omega * Csp, 1.0e-30)) + abs(1.0 / max(omega * Csg, 1.0e-30))
        Zmag = math.sqrt(Rb * Rb + Xs * Xs)
        if 'voltage' in mode:
            current = vrms / max(Zmag, 1.0e-9)
            absorbed = current * current * Rb
        else:
            absorbed = delivered * (Rb * Rb / max(Zmag * Zmag, 1.0e-30))
            current = math.sqrt(max(absorbed, 0.0) / max(Rb, 1.0e-12))
            vrms = current * Zmag
        q_exp = 1.35
        asym = area_ratio ** q_exp
        eta = (asym - 1.0) / max(asym + 1.0, 1.0e-12)
        v_rf_pk = math.sqrt(2.0) * vrms
        self_bias = -eta * v_rf_pk
        sheath_powered_V = max(0.0, 0.5 * v_rf_pk * (1.0 + eta))
        sheath_ground_V = max(0.0, 0.5 * v_rf_pk * (1.0 - eta))
        plasma_potential = max(3.0 * te, 8.0) + 0.25 * (sheath_powered_V + sheath_ground_V)
        surface_ied = build_surface_ied(
            ne_m3=ne,
            te_eV=te,
            ion_mass_kg=state.dominant_ion_mass_kg,
            sheath_voltage_V=sheath_powered_V,
            pressure_Pa=state.pressure_Pa,
            gas_temperature_K=state.gas_temperature_K,
            area_m2=area_p,
        )
        return {
            'zone_id': zone_id,
            'frequency_Hz': frequency_Hz,
            'delivered_power_W': delivered,
            'absorbed_power_W': absorbed,
            'rf_voltage_rms_V': vrms,
            'rf_current_rms_A': current,
            'bulk_resistance_Ohm': Rb,
            'sheath_reactance_Ohm': Xs,
            'self_bias_V': self_bias,
            'plasma_potential_V': plasma_potential,
            'sheath_voltage_powered_V': sheath_powered_V,
            'sheath_voltage_ground_V': sheath_ground_V,
            'pressure_Pa': state.pressure_Pa,
            'gas_temperature_K': state.gas_temperature_K,
            'surface_id': surface_id,
            'ied': surface_ied,
        }

    def evaluate(self, request: PowerRequest) -> PowerResult:
        p_zone: dict[str, float] = {z.zone_id: 0.0 for z in self.chamber.zones}
        p_port: dict[str, float] = {}
        surface_ied: dict[str, SurfaceIED] = {}
        bias_contrib: list[tuple[float, float]] = []
        plasma_potentials: list[tuple[float, float]] = []

        for port_id, cfg in request.recipe_step.power_ports.items():
            port = self.chamber.power_port_by_id[port_id]
            kind = (port.kind or '').lower()
            if 'bias' in kind or kind.startswith('ccp'):
                detail = self._bias_port_result(request, port_id, cfg)
                zone_id = detail['zone_id']
                absorbed = detail['absorbed_power_W']
                p_zone[zone_id] = p_zone.get(zone_id, 0.0) + absorbed
                p_port[port_id] = absorbed
                if detail.get('surface_id') is not None:
                    ied = detail['ied']
                    surface_ied[detail['surface_id']] = ied
                weight = abs(detail.get('delivered_power_W', absorbed))
                bias_contrib.append((weight, detail['self_bias_V']))
                plasma_potentials.append((weight, detail['plasma_potential_V']))
            else:
                zone_id = cfg.get('zone_id') or port.zone_id
                base = float(cfg.get('value_W', cfg.get('value', 0.0)))
                value = base * waveform_multiplier(request.time_s, cfg)
                p_zone[zone_id] = p_zone.get(zone_id, 0.0) + value
                p_port[port_id] = value
        total_w = sum(w for w, _ in bias_contrib)
        self_bias = sum(w * v for w, v in bias_contrib) / max(total_w, 1.0) if bias_contrib else 0.0
        total_pp = sum(w for w, _ in plasma_potentials)
        plasma_potential = sum(w * v for w, v in plasma_potentials) / max(total_pp, 1.0) if plasma_potentials else 0.0
        return PowerResult(
            absorbed_power_W_by_zone=p_zone,
            port_power_W=p_port,
            self_bias_V=self_bias,
            plasma_potential_V=plasma_potential,
            zone_reduced_field_Td={z: 35.0 + 0.02 * p for z, p in p_zone.items()},
            surface_ied=surface_ied,
        )
