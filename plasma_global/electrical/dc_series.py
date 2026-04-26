from __future__ import annotations

from typing import Any

from plasma_global.electrical.base import ElectricalBackend, PowerRequest, PowerResult
from plasma_global.electrical.circuit_models import (
    DCSeriesCircuitConfig,
    DCSeriesCircuitModel,
    PlasmaLoadState,
    dc_series_config_mapping_at_time,
    waveform_multiplier,
)


class DCSeriesCircuitBackend(ElectricalBackend):
    """Voltage-source plus ballast-resistor backend for DC and pulsed DC cases."""

    def prepare(self, chamber: Any, recipe: Any, run_config: Any) -> None:
        super().prepare(chamber=chamber, recipe=recipe, run_config=run_config)
        self.model = DCSeriesCircuitModel()

    def _zone_load(self, request: PowerRequest, zone_id: str) -> PlasmaLoadState:
        md = request.metadata or {}
        zone = self.chamber.zone_by_id[zone_id]
        return PlasmaLoadState(
            electron_density_m3=float((md.get('zone_electron_density_m3') or {}).get(zone_id, 1.0e12)),
            mean_energy_eV=float((md.get('zone_mean_energy_eV') or {}).get(zone_id, 3.0)),
            pressure_Pa=float((md.get('zone_pressure_Pa') or {}).get(zone_id, zone.pressure_Pa)),
            gas_temperature_K=float((md.get('zone_gas_temperature_K') or {}).get(zone_id, zone.gas_temperature_K)),
            total_density_m3=float((md.get('zone_total_density_m3') or {}).get(zone_id, 0.0)) or None,
            electron_mobility_m2_V_s=(md.get('zone_electron_mobility_m2_V_s') or {}).get(zone_id),
        )

    def evaluate(self, request: PowerRequest) -> PowerResult:
        p_zone: dict[str, float] = {z.zone_id: 0.0 for z in self.chamber.zones}
        p_port: dict[str, float] = {}
        port_details: dict[str, dict[str, float | str]] = {}
        zone_reduced_field: dict[str, float] = {z.zone_id: 0.0 for z in self.chamber.zones}
        plasma_potential = 0.0

        for port_id, step_cfg in request.recipe_step.power_ports.items():
            port = self.chamber.power_port_by_id[port_id]
            cfg = dict(port.parameters or {})
            cfg.update(step_cfg or {})
            zone_id = cfg.get('zone_id') or port.zone_id
            nested_voltage = isinstance(cfg.get('voltage'), dict)
            source_mult = 1.0 if nested_voltage else waveform_multiplier(request.time_s, cfg)
            circuit_cfg = DCSeriesCircuitConfig.from_mapping(dc_series_config_mapping_at_time(cfg, request.time_s))
            if not nested_voltage:
                circuit_cfg.source_voltage_V *= source_mult
            solution = self.model.solve(circuit_cfg, self._zone_load(request, zone_id))
            p_zone[zone_id] = p_zone.get(zone_id, 0.0) + solution.absorbed_power_W
            p_port[port_id] = solution.absorbed_power_W
            zone_reduced_field[zone_id] = max(zone_reduced_field.get(zone_id, 0.0), solution.reduced_field_Td)
            plasma_potential = max(plasma_potential, 0.05 * abs(solution.gap_voltage_V))
            port_details[port_id] = {
                'backend': 'dc_series_circuit',
                'zone_id': zone_id,
                'source_voltage_V': solution.source_voltage_V,
                'gap_voltage_V': solution.gap_voltage_V,
                'current_A': solution.current_A,
                'absorbed_power_W': solution.absorbed_power_W,
                'delivered_power_W': solution.delivered_power_W,
                'ballast_resistance_ohm': circuit_cfg.ballast_resistance_ohm,
                'plasma_resistance_ohm': solution.plasma_resistance_ohm,
                'plasma_conductance_S': solution.plasma_conductance_S,
                'conductance_multiplier': circuit_cfg.conductance_multiplier,
                'electron_mobility_m2_V_s': solution.electron_mobility_m2_V_s,
                'electric_field_V_m': solution.electric_field_V_m,
                'reduced_field_Td': solution.reduced_field_Td,
                'gap_m': circuit_cfg.gap_m,
                'electrode_area_m2': circuit_cfg.electrode_area_m2,
                'waveform_multiplier': source_mult,
            }

        return PowerResult(
            absorbed_power_W_by_zone=p_zone,
            port_power_W=p_port,
            self_bias_V=0.0,
            plasma_potential_V=plasma_potential,
            metadata={
                'port_details': port_details,
                'surface_ied': {},
                'zone_reduced_field_Td': zone_reduced_field,
                'circuit_interface': {
                    'kind': 'internal_reduced_circuit',
                    'model': 'dc_series_circuit',
                    'version': 1,
                    'external_circuit_ready': True,
                },
            },
        )
