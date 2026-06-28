from __future__ import annotations

import math

from plasma_global.electrical.base import ElectricalBackend, ElectricalPortSnapshot, PowerRequest, PowerResult
from plasma_global.electrical.circuit_models import waveform_multiplier


class DirectPowerBackend(ElectricalBackend):
    def evaluate(self, request: PowerRequest) -> PowerResult:
        p_zone: dict[str, float] = {z.zone_id: 0.0 for z in self.chamber.zones}
        p_port: dict[str, float] = {}
        port_observables: dict[str, ElectricalPortSnapshot] = {}
        bias_power = 0.0
        for port_id, cfg in request.recipe_step.power_ports.items():
            zone_id = cfg.get('zone_id') or self.chamber.power_port_by_id[port_id].zone_id
            base = float(cfg.get('value_W', 0.0))
            value = base * waveform_multiplier(request.time_s, cfg)
            p_zone[zone_id] = p_zone.get(zone_id, 0.0) + value
            p_port[port_id] = value
            port_observables[port_id] = ElectricalPortSnapshot({
                'absorbed_power_W': float(value),
                'delivered_power_W': float(value),
            })
            if 'bias' in port_id.lower() or 'ccp' in str(self.chamber.power_port_by_id[port_id].kind).lower():
                bias_power += value
        self_bias = -math.sqrt(max(bias_power, 0.0)) * 4.0
        plasma_potential = max(p_zone.values()) * 0.05 if p_zone else 0.0
        return PowerResult(
            absorbed_power_W_by_zone=p_zone,
            port_power_W=p_port,
            port_observables=port_observables,
            self_bias_V=self_bias,
            plasma_potential_V=plasma_potential,
            zone_reduced_field_Td={z: 20.0 + 0.03 * p for z, p in p_zone.items()},
        )
