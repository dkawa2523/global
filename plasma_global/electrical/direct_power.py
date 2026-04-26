from __future__ import annotations

import math

from plasma_global.electrical.base import ElectricalBackend, PowerRequest, PowerResult


class DirectPowerBackend(ElectricalBackend):
    def _waveform_multiplier(self, t: float, port_cfg: dict) -> float:
        waveform = str(port_cfg.get('waveform', 'cw')).lower()
        if waveform in {'off', 'none'}:
            return 0.0
        if waveform == 'cw':
            return 1.0
        if waveform == 'pulsed_square':
            duty = float(port_cfg.get('duty_cycle', 0.5))
            f = float(port_cfg.get('repetition_Hz', 1.0))
            if f <= 0.0:
                return duty
            phase = (t * f) % 1.0
            return 1.0 if phase < duty else 0.0
        return 1.0

    def evaluate(self, request: PowerRequest) -> PowerResult:
        p_zone: dict[str, float] = {z.zone_id: 0.0 for z in self.chamber.zones}
        p_port: dict[str, float] = {}
        bias_power = 0.0
        for port_id, cfg in request.recipe_step.power_ports.items():
            zone_id = cfg.get('zone_id') or self.chamber.power_port_by_id[port_id].zone_id
            base = float(cfg.get('value_W', 0.0))
            value = base * self._waveform_multiplier(request.time_s, cfg)
            p_zone[zone_id] = p_zone.get(zone_id, 0.0) + value
            p_port[port_id] = value
            if 'bias' in port_id.lower() or 'ccp' in str(self.chamber.power_port_by_id[port_id].kind).lower():
                bias_power += value
        self_bias = -math.sqrt(max(bias_power, 0.0)) * 4.0
        plasma_potential = max(p_zone.values()) * 0.05 if p_zone else 0.0
        return PowerResult(
            absorbed_power_W_by_zone=p_zone,
            port_power_W=p_port,
            self_bias_V=self_bias,
            plasma_potential_V=plasma_potential,
            metadata={
                'port_details': {pid: {'zone_id': (request.recipe_step.power_ports.get(pid, {}) or {}).get('zone_id') or self.chamber.power_port_by_id[pid].zone_id, 'absorbed_power_W': p} for pid, p in p_port.items()},
                'surface_ied': {},
                'zone_reduced_field_Td': {z: 20.0 + 0.03 * p for z, p in p_zone.items()},
            },
        )
