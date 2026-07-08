from __future__ import annotations

import math
from typing import Any

from plasma_global.electrical.base import (
    PowerRequest,
    PowerResult,
    add_zone_value,
    iter_power_port_configs,
    set_zone_max,
    zone_value_map,
)
from plasma_global.electrical.circuit_models import waveform_multiplier
from plasma_global.electrical.ccp import CCPBackend


class ICPBackend(CCPBackend):
    def _edge_outflow_factor(self, zone_id: str) -> float:
        total = 0.0
        outgoing = 0.0
        for edge in self.chamber.edges:
            if edge.from_zone == zone_id:
                outgoing += edge.conductance_m3_s
            if edge.from_zone == zone_id or edge.to_zone == zone_id:
                total += edge.conductance_m3_s
        if total <= 0.0:
            return 0.0
        return min(max(outgoing / total, 0.0), 1.0)

    def _source_port_result(self, request: PowerRequest, port_id: str, cfg: dict[str, Any]) -> dict[str, Any]:
        port = self.chamber.power_port_by_id[port_id]
        zone_id = cfg.get('zone_id') or port.zone_id
        state = self._zone_meta(request, zone_id)
        ne = state.electron_density_m3
        te = state.mean_energy_eV
        pressure = state.pressure_Pa
        f_Hz = float(cfg.get('frequency_Hz') or 13.56e6)
        p_in = float(cfg.get('value_W', cfg.get('value', 0.0))) * waveform_multiplier(request.time_s, cfg)
        if p_in <= 0.0:
            return {
                'zone_id': zone_id,
                'frequency_Hz': f_Hz,
                'absorbed_power_W': 0.0,
                'delivered_power_W': 0.0,
                'coupling_efficiency': 0.0,
                'plasma_resistance_Ohm': 1.0,
                'coil_voltage_rms_V': 0.0,
                'coil_current_rms_A': 0.0,
                'E_over_H_mode_index': 0.0,
                'mode_label': 'off',
                'downstream_fraction': 0.0,
                'effective_field_Td': 0.0,
            }
        ne_norm = ne / 1.0e17
        press_norm = max(pressure / 10.0, 1.0e-3)
        freq_norm = max(f_Hz / 13.56e6, 0.1)
        density_factor = 1.0 - math.exp(-2.2 * ne_norm)
        pressure_factor = 0.65 + 0.20 * math.tanh((press_norm - 0.3) / 0.6)
        freq_factor = 0.85 + 0.08 * math.log10(1.0 + freq_norm)
        eh_index = density_factor * math.sqrt(max(p_in, 0.0) / 150.0)
        mode_label = 'H' if eh_index >= 0.55 else 'E'
        eta_min = 0.18 if mode_label == 'E' else 0.40
        eta_max = 0.55 if mode_label == 'E' else 0.88
        coupling = eta_min + (eta_max - eta_min) * density_factor * pressure_factor * freq_factor
        coupling = min(max(coupling, 0.05), 0.95)
        absorbed = coupling * p_in
        r_plasma = 2.5 + 12.0 / max(math.sqrt(ne / 1.0e16), 0.2)
        i_rms = math.sqrt(absorbed / max(r_plasma, 1.0e-12))
        v_rms = i_rms * r_plasma
        downstream_fraction = 0.05 + 0.25 * self._edge_outflow_factor(zone_id)
        downstream_fraction = min(max(downstream_fraction, 0.0), 0.35)
        reduced_field = 25.0 + 30.0 * math.sqrt(max(te, 0.1)) + 0.01 * absorbed
        return {
            'zone_id': zone_id,
            'frequency_Hz': f_Hz,
            'absorbed_power_W': absorbed,
            'delivered_power_W': p_in,
            'coupling_efficiency': coupling,
            'plasma_resistance_Ohm': r_plasma,
            'coil_voltage_rms_V': v_rms,
            'coil_current_rms_A': i_rms,
            'E_over_H_mode_index': eh_index,
            'mode_label': mode_label,
            'downstream_fraction': downstream_fraction,
            'effective_field_Td': reduced_field,
        }

    def evaluate(self, request: PowerRequest) -> PowerResult:
        p_zone = zone_value_map(self.chamber)
        p_port: dict[str, float] = {}
        zone_reduced_field = zone_value_map(self.chamber, 25.0)
        source_plasma_potential = 0.0

        # First handle ICP / source ports.
        for port_id, port, cfg in iter_power_port_configs(request):
            kind = (port.kind or '').lower()
            if 'icp' not in kind and 'source' not in kind:
                continue
            detail = self._source_port_result(request, port_id, cfg)
            zone_id = detail['zone_id']
            absorbed = detail['absorbed_power_W']
            add_zone_value(p_zone, zone_id, absorbed * (1.0 - detail['downstream_fraction']))
            # Simple downstream deposition of source power into the first outgoing edge.
            downstream_edges = [e for e in self.chamber.edges if e.from_zone == zone_id]
            if downstream_edges:
                share = absorbed * detail['downstream_fraction'] / len(downstream_edges)
                for edge in downstream_edges:
                    add_zone_value(p_zone, edge.to_zone, share)
                    set_zone_max(zone_reduced_field, edge.to_zone, 0.65 * detail['effective_field_Td'])
            p_port[port_id] = absorbed
            set_zone_max(zone_reduced_field, zone_id, detail['effective_field_Td'])
            source_plasma_potential = max(source_plasma_potential, 5.0 + 0.015 * absorbed + 2.0 * detail['coupling_efficiency'])

        # Then reuse CCP handling for any bias ports or generic direct-power ports.
        bias_power_ports: dict[str, Any] = {}
        for port_id, port, cfg in iter_power_port_configs(request):
            kind = (port.kind or '').lower()
            if 'bias' in kind or kind.startswith('ccp') or ('icp' not in kind and 'source' not in kind):
                bias_power_ports[port_id] = cfg
        bias_result = super().evaluate(request.with_power_ports(bias_power_ports))
        for zone_id, val in bias_result.absorbed_power_W_by_zone.items():
            add_zone_value(p_zone, zone_id, val)
        p_port.update(bias_result.port_power_W)
        for zone_id, red in bias_result.zone_reduced_field_Td.items():
            set_zone_max(zone_reduced_field, zone_id, red)

        self_bias = bias_result.self_bias_V
        plasma_potential = max(source_plasma_potential, bias_result.plasma_potential_V)
        return PowerResult(
            absorbed_power_W_by_zone=p_zone,
            port_power_W=p_port,
            self_bias_V=self_bias,
            plasma_potential_V=plasma_potential,
            zone_reduced_field_Td=zone_reduced_field,
            surface_ied=bias_result.surface_ied,
        )
