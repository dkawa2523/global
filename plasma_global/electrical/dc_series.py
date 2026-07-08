from __future__ import annotations

from typing import Any

from plasma_global.electrical.base import (
    ElectricalBackend,
    PowerRequest,
    PowerResult,
    ZoneElectricalState,
    add_port_power,
    iter_power_port_configs,
    set_zone_max,
    zone_value_map,
)
from plasma_global.electrical.circuit_models import (
    DCSeriesCircuitConfig,
    DCSeriesCircuitModel,
    PlasmaLoadState,
    dc_series_config_mapping_at_time,
    waveform_multiplier,
)


def validate_dc_series_port_config(step: Any, _port_id: str, cfg: dict[str, Any], _resolved_paths: Any) -> None:
    DCSeriesCircuitConfig.from_mapping(dc_series_config_mapping_at_time(cfg, float(step.t_start_s)))


class DCSeriesCircuitBackend(ElectricalBackend):
    """Voltage-source plus ballast-resistor backend for DC and pulsed DC cases."""

    def prepare(self, chamber: Any, recipe: Any, run_config: Any, resolved_paths: Any) -> None:
        super().prepare(chamber=chamber, recipe=recipe, run_config=run_config, resolved_paths=resolved_paths)
        self.model = DCSeriesCircuitModel()

    def _zone_load(self, request: PowerRequest, zone_id: str) -> PlasmaLoadState:
        zone = self.chamber.zone_by_id[zone_id]
        state = request.zone_state.get(
            zone_id,
            ZoneElectricalState(
                electron_density_m3=1.0e12,
                mean_energy_eV=3.0,
                positive_ion_density_m3=1.0e12,
                dominant_ion_mass_kg=6.63e-26,
                pressure_Pa=zone.pressure_Pa,
                gas_temperature_K=zone.gas_temperature_K,
                total_density_m3=0.0,
            ),
        )
        return PlasmaLoadState(
            electron_density_m3=float(state.electron_density_m3),
            mean_energy_eV=float(state.mean_energy_eV),
            pressure_Pa=float(state.pressure_Pa),
            gas_temperature_K=float(state.gas_temperature_K),
            total_density_m3=float(state.total_density_m3) or None,
            electron_mobility_m2_V_s=state.electron_mobility_m2_V_s,
        )

    def evaluate(self, request: PowerRequest) -> PowerResult:
        p_zone = zone_value_map(self.chamber)
        p_port: dict[str, float] = {}
        port_observables: dict[str, dict[str, float]] = {}
        zone_reduced_field = zone_value_map(self.chamber)
        plasma_potential = 0.0

        for port_id, port, cfg in iter_power_port_configs(request):
            zone_id = cfg.get('zone_id') or port.zone_id
            nested_voltage = isinstance(cfg.get('voltage'), dict)
            source_mult = 1.0 if nested_voltage else waveform_multiplier(request.time_s, cfg)
            circuit_cfg = DCSeriesCircuitConfig.from_mapping(dc_series_config_mapping_at_time(cfg, request.time_s))
            if not nested_voltage:
                circuit_cfg.source_voltage_V *= source_mult
            solution = self.model.solve(circuit_cfg, self._zone_load(request, zone_id))
            add_port_power(
                p_zone,
                p_port,
                port_id=port_id,
                zone_id=zone_id,
                absorbed_power_W=solution.absorbed_power_W,
            )
            port_observables[port_id] = {
                'source_voltage_V': float(solution.source_voltage_V),
                'gap_voltage_V': float(solution.gap_voltage_V),
                'current_A': float(solution.current_A),
                'absorbed_power_W': float(solution.absorbed_power_W),
                'delivered_power_W': float(solution.delivered_power_W),
                'plasma_resistance_ohm': float(solution.plasma_resistance_ohm),
                'plasma_conductance_S': float(solution.plasma_conductance_S),
                'electron_mobility_m2_V_s': float(solution.electron_mobility_m2_V_s),
                'electric_field_V_m': float(solution.electric_field_V_m),
                'reduced_field_Td': float(solution.reduced_field_Td),
            }
            set_zone_max(zone_reduced_field, zone_id, solution.reduced_field_Td)
            plasma_potential = max(plasma_potential, 0.05 * abs(solution.gap_voltage_V))

        return PowerResult(
            absorbed_power_W_by_zone=p_zone,
            port_power_W=p_port,
            port_observables=port_observables,
            plasma_potential_V=plasma_potential,
            zone_reduced_field_Td=zone_reduced_field,
        )
