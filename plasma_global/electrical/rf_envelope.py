from __future__ import annotations

import math
from typing import Any

from plasma_global.electrical.base import ElectricalBackend, ElectricalPortSnapshot, PowerRequest, PowerResult
from plasma_global.electrical.circuit_models import first_present, waveform_multiplier


POWER_KEYS = ('absorbed_power_W', 'power_W', 'value_W')
VOLTAGE_KEYS = ('voltage_rms_V', 'voltage_V', 'value_V')


def rf_frequency_Hz(cfg: dict[str, Any]) -> float:
    return first_present(cfg, ('frequency_Hz', 'carrier_frequency_Hz'))


def validate_rf_envelope_port(cfg: dict[str, Any]) -> None:
    rf_frequency_Hz(cfg)
    has_power = any(k in cfg and cfg[k] is not None for k in POWER_KEYS)
    has_voltage = any(k in cfg and cfg[k] is not None for k in VOLTAGE_KEYS)
    mode = str(cfg.get('control_mode') or cfg.get('mode') or '').lower()
    if not has_power and 'voltage' not in mode and 'value' in cfg:
        has_power = True
    if not has_voltage and 'voltage' in mode and 'value' in cfg:
        has_voltage = True
    if not has_power and not has_voltage:
        raise ValueError('rf_envelope needs absorbed_power_W/power_W/value_W or voltage_rms_V/voltage_V/value_V.')
    if has_voltage and not has_power and 'effective_impedance_ohm' not in cfg:
        raise ValueError('rf_envelope voltage-driven ports require effective_impedance_ohm.')
    if 'coupling_efficiency' in cfg:
        coupling = float(cfg['coupling_efficiency'])
        if not 0.0 <= coupling <= 1.0:
            raise ValueError('rf_envelope coupling_efficiency must be between 0 and 1.')
    if 'self_bias_fraction' in cfg and float(cfg['self_bias_fraction']) < 0.0:
        raise ValueError('rf_envelope self_bias_fraction must be non-negative.')
    if 'effective_impedance_ohm' in cfg and float(cfg['effective_impedance_ohm']) <= 0.0:
        raise ValueError('rf_envelope effective_impedance_ohm must be positive.')
    field_coeff = cfg.get('reduced_field_per_sqrt_W_Td', cfg.get('reduced_field_per_sqrt_W'))
    if field_coeff is not None and float(field_coeff) < 0.0:
        raise ValueError('rf_envelope reduced_field_per_sqrt_W_Td must be non-negative.')


def _role(port: Any, cfg: dict[str, Any]) -> str:
    return str(cfg.get('role') or port.kind or '').lower()


def _command_power_and_voltage(cfg: dict[str, Any]) -> tuple[float, float, float]:
    mode = str(cfg.get('control_mode') or cfg.get('mode') or '').lower()
    voltage = None
    for key in VOLTAGE_KEYS:
        if cfg.get(key) is not None:
            voltage = float(cfg[key])
            break
    if voltage is None and 'voltage' in mode and cfg.get('value') is not None:
        voltage = float(cfg['value'])

    power = None
    for key in POWER_KEYS:
        if cfg.get(key) is not None:
            power = float(cfg[key])
            break
    if power is None and 'voltage' not in mode and cfg.get('value') is not None:
        power = float(cfg['value'])

    impedance = float(cfg.get('effective_impedance_ohm', 50.0))
    if power is None:
        if voltage is None:
            raise ValueError('Missing RF power or voltage command.')
        power = voltage * voltage / max(impedance, 1.0e-30)
    if voltage is None:
        voltage = math.sqrt(max(power, 0.0) * max(impedance, 1.0e-30))
    return power, voltage, impedance


class RFEnvelopeBackend(ElectricalBackend):
    """Cycle-averaged HF/LF power and bias envelope backend."""

    def evaluate(self, request: PowerRequest) -> PowerResult:
        p_zone: dict[str, float] = {z.zone_id: 0.0 for z in self.chamber.zones}
        p_port: dict[str, float] = {}
        port_observables: dict[str, ElectricalPortSnapshot] = {}
        zone_reduced_field: dict[str, float] = {z.zone_id: 0.0 for z in self.chamber.zones}
        bias_terms: list[tuple[float, float]] = []
        plasma_terms: list[tuple[float, float]] = []

        for port_id, step_cfg in request.recipe_step.power_ports.items():
            port = self.chamber.power_port_by_id[port_id]
            cfg = dict(port.parameters or {})
            cfg.update(step_cfg or {})
            validate_rf_envelope_port(cfg)
            zone_id = str(cfg.get('zone_id') or port.zone_id)
            role = _role(port, cfg)
            mult = waveform_multiplier(request.time_s, cfg)
            commanded_power, voltage_rms, _impedance = _command_power_and_voltage(cfg)
            delivered = max(commanded_power * mult, 0.0)
            voltage_rms *= mult
            coupling = min(max(float(cfg.get('coupling_efficiency', 1.0)), 0.0), 1.0)
            absorbed = coupling * delivered

            base_field = float(cfg.get('base_reduced_field_Td', cfg.get('reduced_field_Td', 0.0)))
            field_coeff = float(cfg.get('reduced_field_per_sqrt_W_Td', cfg.get('reduced_field_per_sqrt_W', 0.0)))
            reduced_field = max(base_field * mult + field_coeff * math.sqrt(absorbed), 0.0)
            zone_reduced_field[zone_id] = max(zone_reduced_field.get(zone_id, 0.0), reduced_field)

            self_bias = 0.0
            if 'bias' in role or 'lf' in role:
                bias_fraction = float(cfg.get('self_bias_fraction', 0.35))
                self_bias = -bias_fraction * math.sqrt(2.0) * abs(voltage_rms)
                bias_terms.append((max(absorbed, 1.0e-30), self_bias))

            plasma_potential = float(cfg.get('plasma_potential_offset_V', 0.0))
            plasma_potential += float(cfg.get('plasma_potential_per_sqrt_W', 0.0)) * math.sqrt(absorbed)
            plasma_potential += float(cfg.get('plasma_potential_from_bias_fraction', 0.0)) * abs(self_bias)
            if plasma_potential > 0.0:
                plasma_terms.append((max(absorbed, 1.0e-30), plasma_potential))

            p_zone[zone_id] = p_zone.get(zone_id, 0.0) + absorbed
            p_port[port_id] = absorbed
            port_observables[port_id] = ElectricalPortSnapshot({
                'voltage_rms_V': float(voltage_rms),
                'absorbed_power_W': float(absorbed),
                'delivered_power_W': float(delivered),
                'reduced_field_Td': float(reduced_field),
            })

        total_bias_weight = sum(w for w, _ in bias_terms)
        self_bias_V = sum(w * v for w, v in bias_terms) / max(total_bias_weight, 1.0e-30) if bias_terms else 0.0
        total_plasma_weight = sum(w for w, _ in plasma_terms)
        plasma_potential_V = (
            sum(w * v for w, v in plasma_terms) / max(total_plasma_weight, 1.0e-30) if plasma_terms else 0.0
        )

        return PowerResult(
            absorbed_power_W_by_zone=p_zone,
            port_power_W=p_port,
            port_observables=port_observables,
            self_bias_V=self_bias_V,
            plasma_potential_V=plasma_potential_V,
            zone_reduced_field_Td=zone_reduced_field,
        )
