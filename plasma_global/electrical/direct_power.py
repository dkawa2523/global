from __future__ import annotations

import math

from plasma_global.electrical.base import (
    ElectricalBackend,
    PowerRequest,
    PowerResult,
    add_port_power,
    iter_power_port_configs,
    zone_value_map,
)
from plasma_global.electrical.circuit_models import waveform_multiplier


class DirectPowerBackend(ElectricalBackend):
    def evaluate(self, request: PowerRequest) -> PowerResult:
        p_zone = zone_value_map(self.chamber)
        p_port: dict[str, float] = {}
        port_observables: dict[str, dict[str, float]] = {}
        bias_power = 0.0
        for port_id, port, cfg in iter_power_port_configs(request):
            zone_id = cfg.get('zone_id') or port.zone_id
            base = float(cfg.get('value_W', 0.0))
            value = base * waveform_multiplier(request.time_s, cfg)
            add_port_power(p_zone, p_port, port_id=port_id, zone_id=zone_id, absorbed_power_W=value)
            port_observables[port_id] = {
                'absorbed_power_W': float(value),
                'delivered_power_W': float(value),
            }
            if 'bias' in port_id.lower() or 'ccp' in str(port.kind).lower():
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
