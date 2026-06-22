from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plasma_global.chemistry.models import E_CHARGE, K_B

TD_TO_VM2 = 1.0e-21


def waveform_multiplier(time_s: float, cfg: dict[str, Any]) -> float:
    waveform = str(cfg.get('waveform', 'cw')).lower()
    if waveform in {'off', 'none'}:
        return 0.0
    if waveform == 'cw':
        return 1.0
    if waveform == 'pulsed_square':
        duty = float(cfg.get('duty_cycle', 0.5))
        f = float(cfg.get('repetition_Hz', 1.0))
        if f <= 0.0:
            return min(max(duty, 0.0), 1.0)
        phase = (time_s * f) % 1.0
        return 1.0 if phase < duty else 0.0
    return 1.0


def pulsed_voltage_value(time_s: float, cfg: dict[str, Any]) -> float:
    voltage_cfg = cfg.get('voltage')
    if not isinstance(voltage_cfg, dict):
        return first_present(cfg, ('source_voltage_V', 'voltage_V', 'value_V', 'value'))
    waveform = str(voltage_cfg.get('waveform', cfg.get('waveform', 'cw'))).lower()
    if waveform in {'off', 'none'}:
        return float(voltage_cfg.get('low_V', voltage_cfg.get('off_V', 0.0)))
    if waveform in {'pulsed_square', 'square'}:
        high = first_present(voltage_cfg, ('high_V', 'on_V', 'source_voltage_V', 'voltage_V', 'value_V', 'value'))
        low = float(voltage_cfg.get('low_V', voltage_cfg.get('off_V', 0.0)))
        duty = float(voltage_cfg.get('duty_cycle', cfg.get('duty_cycle', 0.5)))
        f = float(voltage_cfg.get('frequency_Hz', voltage_cfg.get('repetition_Hz', cfg.get('repetition_Hz', 1.0))))
        if f <= 0.0:
            return high if duty > 0.0 else low
        phase = (time_s * f) % 1.0
        return high if phase < min(max(duty, 0.0), 1.0) else low
    return first_present(voltage_cfg, ('source_voltage_V', 'voltage_V', 'value_V', 'value', 'high_V', 'on_V'))


def dc_series_config_mapping_at_time(cfg: dict[str, Any], time_s: float) -> dict[str, Any]:
    out = dict(cfg)
    out['source_voltage_V'] = pulsed_voltage_value(time_s, cfg)
    out.pop('voltage', None)
    return out


def first_present(cfg: dict[str, Any], keys: tuple[str, ...], *, required: bool = True, default: float | None = None) -> float:
    for key in keys:
        if key in cfg and cfg[key] is not None:
            return float(cfg[key])
    if required:
        names = ', '.join(keys)
        raise ValueError(f'Missing required circuit parameter; provide one of: {names}')
    if default is None:
        raise ValueError('Internal error: optional circuit parameter default is None')
    return float(default)


@dataclass
class PlasmaLoadState:
    electron_density_m3: float
    mean_energy_eV: float
    pressure_Pa: float
    gas_temperature_K: float
    total_density_m3: float | None = None
    electron_mobility_m2_V_s: float | None = None


@dataclass
class DCSeriesCircuitConfig:
    source_voltage_V: float
    ballast_resistance_ohm: float
    gap_m: float
    electrode_area_m2: float
    power_absorption_fraction: float = 1.0
    electron_mobility_m2_V_s: float | None = None
    mobility_source: str = 'config'
    conductance_multiplier: float = 1.0
    mobility_ref_m2_V_s: float = 0.10
    mobility_ref_pressure_Pa: float = 133.322
    min_plasma_resistance_ohm: float = 1.0e-6
    max_plasma_resistance_ohm: float = 1.0e12

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any]) -> 'DCSeriesCircuitConfig':
        mobility = cfg.get('electron_mobility_m2_V_s')
        return cls(
            source_voltage_V=first_present(cfg, ('source_voltage_V', 'voltage_V', 'value_V', 'value')),
            ballast_resistance_ohm=first_present(cfg, ('ballast_resistance_ohm', 'series_resistance_ohm')),
            gap_m=first_present(cfg, ('gap_m', 'electrode_gap_m')),
            electrode_area_m2=first_present(cfg, ('electrode_area_m2', 'area_m2')),
            power_absorption_fraction=first_present(cfg, ('power_absorption_fraction',), required=False, default=1.0),
            electron_mobility_m2_V_s=float(mobility) if mobility is not None else None,
            mobility_source=str(cfg.get('mobility_source', 'config')).lower(),
            conductance_multiplier=first_present(cfg, ('conductance_multiplier', 'plasma_conductance_multiplier'), required=False, default=1.0),
            mobility_ref_m2_V_s=first_present(cfg, ('mobility_ref_m2_V_s',), required=False, default=0.10),
            mobility_ref_pressure_Pa=first_present(cfg, ('mobility_ref_pressure_Pa',), required=False, default=133.322),
            min_plasma_resistance_ohm=first_present(cfg, ('min_plasma_resistance_ohm',), required=False, default=1.0e-6),
            max_plasma_resistance_ohm=first_present(cfg, ('max_plasma_resistance_ohm',), required=False, default=1.0e12),
        )


@dataclass
class CircuitSolution:
    source_voltage_V: float
    gap_voltage_V: float
    current_A: float
    absorbed_power_W: float
    delivered_power_W: float
    plasma_resistance_ohm: float
    plasma_conductance_S: float
    electron_mobility_m2_V_s: float
    electric_field_V_m: float
    reduced_field_Td: float


class DCSeriesCircuitModel:
    """Reduced DC/pulsed series circuit with a conductive plasma load.

    The model is intentionally independent of the global ODE system. Electrical
    backends translate typed zone electrical state into `PlasmaLoadState`, and
    this model returns circuit quantities. That keeps future RLC or
    external-circuit models isolated from the chemistry and transport code.
    """

    def electron_mobility(self, cfg: DCSeriesCircuitConfig, load: PlasmaLoadState) -> float:
        if cfg.mobility_source in {'table', 'transport', 'eedf'} and load.electron_mobility_m2_V_s is not None:
            return max(float(load.electron_mobility_m2_V_s), 0.0)
        if cfg.electron_mobility_m2_V_s is not None:
            return max(float(cfg.electron_mobility_m2_V_s), 0.0)
        pressure = max(float(load.pressure_Pa), 1.0e-12)
        return max(cfg.mobility_ref_m2_V_s * cfg.mobility_ref_pressure_Pa / pressure, 1.0e-8)

    def solve(self, cfg: DCSeriesCircuitConfig, load: PlasmaLoadState) -> CircuitSolution:
        mu = self.electron_mobility(cfg, load)
        ne = max(float(load.electron_density_m3), 0.0)
        conductivity = E_CHARGE * ne * mu
        conductance = (
            max(cfg.conductance_multiplier, 0.0)
            * conductivity
            * max(cfg.electrode_area_m2, 1.0e-30)
            / max(cfg.gap_m, 1.0e-30)
        )
        if conductance > 0.0:
            plasma_r = 1.0 / conductance
        else:
            plasma_r = cfg.max_plasma_resistance_ohm
        plasma_r = min(max(plasma_r, cfg.min_plasma_resistance_ohm), cfg.max_plasma_resistance_ohm)
        total_r = max(cfg.ballast_resistance_ohm + plasma_r, 1.0e-30)
        current = cfg.source_voltage_V / total_r
        gap_voltage = current * plasma_r
        delivered = gap_voltage * current
        absorbed = max(cfg.power_absorption_fraction, 0.0) * max(delivered, 0.0)
        electric_field = gap_voltage / max(cfg.gap_m, 1.0e-30)
        if load.total_density_m3 is not None and load.total_density_m3 > 0.0:
            n_gas = float(load.total_density_m3)
        else:
            n_gas = max(load.pressure_Pa, 0.0) / (K_B * max(load.gas_temperature_K, 1.0))
        reduced_field = abs(electric_field) / max(n_gas, 1.0e-30) / TD_TO_VM2
        return CircuitSolution(
            source_voltage_V=cfg.source_voltage_V,
            gap_voltage_V=gap_voltage,
            current_A=current,
            absorbed_power_W=absorbed,
            delivered_power_W=max(delivered, 0.0),
            plasma_resistance_ohm=plasma_r,
            plasma_conductance_S=1.0 / plasma_r,
            electron_mobility_m2_V_s=mu,
            electric_field_V_m=electric_field,
            reduced_field_Td=reduced_field,
        )
